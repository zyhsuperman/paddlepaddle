import paddle
from collections import OrderedDict
import numpy as np
from .SubLayers import MultiHeadAttention, PositionwiseFeedForward


class FFTBlock(paddle.nn.Layer):
    """FFT Block"""

    def __init__(self, d_model, n_head, d_k, d_v, d_inner, kernel_size,
        dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v,
            dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner,
            kernel_size, dropout=dropout)

    def forward(self, enc_input, mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input,
            enc_input, mask=slf_attn_mask)
        enc_output = paddle.where(~mask.unsqueeze(axis=-1), enc_output,paddle.to_tensor(0.0, dtype=enc_output.dtype))
        enc_output = self.pos_ffn(enc_output)
        enc_output = paddle.where(~mask.unsqueeze(axis=-1), enc_output, paddle.to_tensor(0.0, dtype=enc_output.dtype))
        return enc_output, enc_slf_attn


class ConvNorm(paddle.nn.Layer):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
        padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = paddle.nn.Conv1D(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, stride=stride, padding=
            padding, dilation=dilation, bias_attr=bias)

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class PostNet(paddle.nn.Layer):
    """
    PostNet: Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, n_mel_channels=80, postnet_embedding_dim=512,
        postnet_kernel_size=5, postnet_n_convolutions=5):
        super(PostNet, self).__init__()
        self.convolutions = paddle.nn.LayerList()
        self.convolutions.append(paddle.nn.Sequential(ConvNorm(
            n_mel_channels, postnet_embedding_dim, kernel_size=
            postnet_kernel_size, stride=1, padding=int((postnet_kernel_size -
            1) / 2), dilation=1, w_init_gain='tanh'), paddle.nn.BatchNorm1D
            (num_features=postnet_embedding_dim, use_global_stats=True,
            weight_attr=None, bias_attr=None)))
        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(paddle.nn.Sequential(ConvNorm(
                postnet_embedding_dim, postnet_embedding_dim, kernel_size=
                postnet_kernel_size, stride=1, padding=int((
                postnet_kernel_size - 1) / 2), dilation=1, w_init_gain=
                'tanh'), paddle.nn.BatchNorm1D(num_features=
                postnet_embedding_dim, use_global_stats=True, weight_attr=
                None, bias_attr=None)))
        self.convolutions.append(paddle.nn.Sequential(ConvNorm(
            postnet_embedding_dim, n_mel_channels, kernel_size=
            postnet_kernel_size, stride=1, padding=int((postnet_kernel_size -
            1) / 2), dilation=1, w_init_gain='linear'), paddle.nn.
            BatchNorm1D(num_features=n_mel_channels, use_global_stats=True,
            weight_attr=None, bias_attr=None)))

    def forward(self, x):
        x = x
        perm_3 = list(range(x.ndim))
        perm_3[1] = 2
        perm_3[2] = 1
        x = x.transpose(perm=perm_3)
        for i in range(len(self.convolutions) - 1):
            x = paddle.nn.functional.dropout(x=paddle.tanh(x=self.
                convolutions[i](x)), p=0.5, training=self.training)
        x = paddle.nn.functional.dropout(x=self.convolutions[-1](x), p=0.5,
            training=self.training)
        x = x
        perm_4 = list(range(x.ndim))
        perm_4[1] = 2
        perm_4[2] = 1
        x = x.transpose(perm=perm_4)
        return x
