import paddle
import copy
import math
import numpy as np
import commons
import modules
from modules import LayerNorm


class Encoder(paddle.nn.Layer):

    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers,
        kernel_size=1, p_dropout=0.0, window_size=None, block_length=None,
        **kwargs):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.block_length = block_length
        self.drop = paddle.nn.Dropout(p=p_dropout)
        self.attn_layers = paddle.nn.LayerList()
        self.norm_layers_1 = paddle.nn.LayerList()
        self.ffn_layers = paddle.nn.LayerList()
        self.norm_layers_2 = paddle.nn.LayerList()
        for i in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttention(hidden_channels,
                hidden_channels, n_heads, window_size=window_size,
                p_dropout=p_dropout, block_length=block_length))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(FFN(hidden_channels, hidden_channels,
                filter_channels, kernel_size, p_dropout=p_dropout))
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(axis=2) * x_mask.unsqueeze(axis=-1)
        for i in range(self.n_layers):
            x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


class CouplingBlock(paddle.nn.Layer):

    def __init__(self, in_channels, hidden_channels, kernel_size,
        dilation_rate, n_layers, gin_channels=0, p_dropout=0, sigmoid_scale
        =False):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        self.sigmoid_scale = sigmoid_scale
        start = paddle.nn.Conv1D(in_channels=in_channels // 2, out_channels
            =hidden_channels, kernel_size=1)
        start = paddle.nn.utils.weight_norm(layer=start)
        self.start = start
        end = paddle.nn.Conv1D(in_channels=hidden_channels, out_channels=
            in_channels, kernel_size=1)
        end.weight.set_value(paddle.zeros_like(end.weight))
        end.bias.set_value(paddle.zeros_like(end.bias))
        self.end = end
        self.wn = modules.WN(in_channels, hidden_channels, kernel_size,
            dilation_rate, n_layers, gin_channels, p_dropout)

    def forward(self, x, x_mask=None, reverse=False, g=None, **kwargs):
        b, c, t = x.shape
        if x_mask is None:
            x_mask = 1
        x_0, x_1 = x[:, :self.in_channels // 2], x[:, self.in_channels // 2:]
        x = self.start(x_0) * x_mask
        x = self.wn(x, x_mask, g)
        out = self.end(x)
        z_0 = x_0
        m = out[:, :self.in_channels // 2, :]
        logs = out[:, self.in_channels // 2:, :]
        if self.sigmoid_scale:
            logs = paddle.log(x=1e-06 + paddle.nn.functional.sigmoid(x=logs +
                2))
        if reverse:
            z_1 = (x_1 - m) * paddle.exp(x=-logs) * x_mask
            logdet = None
        else:
            z_1 = (m + paddle.exp(x=logs) * x_1) * x_mask
            logdet = paddle.sum(x=logs * x_mask, axis=[1, 2])
        z = paddle.concat(x=[z_0, z_1], axis=1)
        return z, logdet

    def store_inverse(self):
        self.wn.remove_weight_norm()


class MultiHeadAttention(paddle.nn.Layer):

    def __init__(self, channels, out_channels, n_heads, window_size=None,
        heads_share=True, p_dropout=0.0, block_length=None, proximal_bias=
        False, proximal_init=False):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.p_dropout = p_dropout
        self.attn = None
        self.k_channels = channels // n_heads
        self.conv_q = paddle.nn.Conv1D(in_channels=channels, out_channels=
            channels, kernel_size=1)
        self.conv_k = paddle.nn.Conv1D(in_channels=channels, out_channels=
            channels, kernel_size=1)
        self.conv_v = paddle.nn.Conv1D(in_channels=channels, out_channels=
            channels, kernel_size=1)
        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels ** -0.5
            out_0 = paddle.create_parameter(shape=(paddle.randn(shape=[
                n_heads_rel, window_size * 2 + 1, self.k_channels]) *
                rel_stddev).shape, dtype=(paddle.randn(shape=[n_heads_rel, 
                window_size * 2 + 1, self.k_channels]) * rel_stddev).numpy(
                ).dtype, default_initializer=paddle.nn.initializer.Assign(
                paddle.randn(shape=[n_heads_rel, window_size * 2 + 1, self.
                k_channels]) * rel_stddev))
            out_0.stop_gradient = not True
            self.emb_rel_k = out_0
            out_1 = paddle.create_parameter(shape=(paddle.randn(shape=[
                n_heads_rel, window_size * 2 + 1, self.k_channels]) *
                rel_stddev).shape, dtype=(paddle.randn(shape=[n_heads_rel, 
                window_size * 2 + 1, self.k_channels]) * rel_stddev).numpy(
                ).dtype, default_initializer=paddle.nn.initializer.Assign(
                paddle.randn(shape=[n_heads_rel, window_size * 2 + 1, self.
                k_channels]) * rel_stddev))
            out_1.stop_gradient = not True
            self.emb_rel_v = out_1
        self.conv_o = paddle.nn.Conv1D(in_channels=channels, out_channels=
            out_channels, kernel_size=1)
        self.drop = paddle.nn.Dropout(p=p_dropout)
        init_XavierUniform = paddle.nn.initializer.XavierUniform()
        init_XavierUniform(self.conv_q.weight)
        init_XavierUniform = paddle.nn.initializer.XavierUniform()
        init_XavierUniform(self.conv_k.weight)
        if proximal_init:
            paddle.assign(self.conv_q.weight.data, output=self.conv_k.
                weight.data)
            paddle.assign(self.conv_q.bias.data, output=self.conv_k.bias.data)
        init_XavierUniform = paddle.nn.initializer.XavierUniform()
        init_XavierUniform(self.conv_v.weight)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        x, self.attn = self.attention(q, k, v, mask=attn_mask)
        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        b, d, t_s, t_t = *key.shape, query.shape[2]
        
        query = query.reshape([b, self.n_heads, self.k_channels, t_t])
        query = paddle.transpose(x=query, perm=[0, 1, 3, 2])

        
        key = key.reshape([b, self.n_heads, self.k_channels, t_s])
        key = paddle.transpose(x=key, perm=[0, 1, 3, 2])

        value = value.reshape([b, self.n_heads, self.k_channels, t_s])
        value = paddle.transpose(x=value, perm=[0, 1, 3, 2])        
        
        scores = paddle.matmul(query, paddle.transpose(key, perm=[0, 1, 3, 2])) / math.sqrt(self.k_channels)
        
        if self.window_size is not None:
            assert t_s == t_t, 'Relative attention is only available for self-attention.'
            key_relative_embeddings = self._get_relative_embeddings(self.
                emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(query,
                key_relative_embeddings)
            rel_logits = self._relative_position_to_absolute_position(
                rel_logits)
            scores_local = rel_logits / math.sqrt(self.k_channels)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, 'Proximal bias is only available for self-attention.'
            scores = scores + self._attention_bias_proximal(t_s).to(device=
                scores.place, dtype=scores.dtype)
        if mask is not None:
            scores = paddle.where(~(mask == 0), scores, paddle.to_tensor(-10000.0, dtype='float32'))
            if self.block_length is not None:
                block_mask = paddle.tril(paddle.triu(paddle.ones_like(x=
                    scores), diagonal=-self.block_length), diagonal=self.
                    block_length)
                scores = scores * block_mask + -10000.0 * (1 - block_mask)
        p_attn = paddle.nn.functional.softmax(x=scores, axis=-1)
        p_attn = self.drop(p_attn)
        output = paddle.matmul(x=p_attn, y=value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(
                p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.
                emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(
                relative_weights, value_relative_embeddings)
        
        output = paddle.transpose(x=output, perm=[0, 1, 3, 2])
        output = output.reshape([b, d, t_t])
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        """
    x: [b, h, l, m]
    y: [h or 1, m, d]
    ret: [b, h, l, d]
    """
        ret = paddle.matmul(x=x, y=y.unsqueeze(axis=0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        """
    x: [b, h, l, d]
    y: [h or 1, m, d]
    ret: [b, h, l, m]
    """
        ret = paddle.matmul(x, paddle.transpose(y.unsqueeze(0), perm=[0, 1, 3, 2]))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length):
        max_relative_position = 2 * self.window_size + 1
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max(self.window_size + 1 - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
           padded_relative_embeddings = paddle.nn.functional.pad(
                relative_embeddings, commons.convert_pad_shape([[0, 0], [
                pad_length, pad_length], [0, 0]]), mode='constant', value=0)
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[:,
            slice_start_position:slice_end_position]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        """
    x: [b, h, l, 2*l-1]
    ret: [b, h, l, l]
    """
        batch, heads, length, _ = x.shape
        x = paddle.nn.functional.pad(x, commons.convert_pad_shape([[0, 1], [
            0, 0], [0, 0], [0, 0]]), mode='constant', value=0)
        x_flat = x.reshape([batch, heads, length * 2 * length])
        x_flat = paddle.nn.functional.pad(x_flat, commons.convert_pad_shape(
            [[0, length - 1], [0, 0], [0, 0]]), mode='constant', value=0)
        
        x_final = x_flat.reshape([batch, heads, length + 1, 2 * length - 1])[:,
            :, :length, length - 1:]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        """
    x: [b, h, l, l]
    ret: [b, h, l, 2*l-1]
    """
        batch, heads, length, _ = x.shape
        x = paddle.nn.functional.pad(x, commons.convert_pad_shape([[0, length - 1], [
            0, 0], [0, 0], [0, 0]]))
        
        x_flat = x.reshape([batch, heads, length ** 2 + length * (length - 1)])
        x_flat = paddle.nn.functional.pad(x_flat, commons.convert_pad_shape(
            [[length, 0], [0, 0], [0, 0]]))
        
        x_final = x_flat.reshape([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length):
        """Bias for self-attention to encourage attention to close positions.
    Args:
      length: an integer scalar.
    Returns:
      a Tensor with shape [1, 1, length, length]
    """
        r = paddle.arange(dtype='float32', end=length)
        diff = paddle.unsqueeze(x=r, axis=0) - paddle.unsqueeze(x=r, axis=1)
        return paddle.unsqueeze(x=paddle.unsqueeze(x=-paddle.log1p(x=paddle
            .abs(x=diff)), axis=0), axis=0)


class FFN(paddle.nn.Layer):

    def __init__(self, in_channels, out_channels, filter_channels,
        kernel_size, p_dropout=0.0, activation=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.conv_1 = paddle.nn.Conv1D(in_channels=in_channels,
            out_channels=filter_channels, kernel_size=kernel_size, padding=
            kernel_size // 2)
        self.conv_2 = paddle.nn.Conv1D(in_channels=filter_channels,
            out_channels=out_channels, kernel_size=kernel_size, padding=
            kernel_size // 2)
        self.drop = paddle.nn.Dropout(p=p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        if self.activation == 'gelu':
            x = x * paddle.nn.functional.sigmoid(x=1.702 * x)
        else:
            x = paddle.nn.functional.relu(x=x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask
