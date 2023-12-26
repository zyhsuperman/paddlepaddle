import paddle
import copy
import math
import numpy as np
import scipy
import commons


class LayerNorm(paddle.nn.Layer):

    def __init__(self, channels, eps=0.0001):
        super().__init__()
        self.channels = channels
        self.eps = eps
        out_2 = paddle.create_parameter(shape=paddle.ones(shape=channels).
            shape, dtype=paddle.ones(shape=channels).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.ones(
            shape=channels)))
        out_2.stop_gradient = not True
        self.gamma = out_2
        out_3 = paddle.create_parameter(shape=paddle.zeros(shape=channels).
            shape, dtype=paddle.zeros(shape=channels).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.zeros(
            shape=channels)))
        out_3.stop_gradient = not True
        self.beta = out_3

    def forward(self, x):
        n_dims = len(x.shape)
        mean = paddle.mean(x=x, axis=1, keepdim=True)
        variance = paddle.mean(x=(x - mean) ** 2, axis=1, keepdim=True)
        x = (x - mean) * paddle.rsqrt(x=variance + self.eps)
        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.reshape(shape) + self.beta.reshape(shape)
        return x


class ConvReluNorm(paddle.nn.Layer):

    def __init__(self, in_channels, hidden_channels, out_channels,
        kernel_size, n_layers, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        assert n_layers > 1, 'Number of layers should be larger than 0.'
        self.conv_layers = paddle.nn.LayerList()
        self.norm_layers = paddle.nn.LayerList()
        self.conv_layers.append(paddle.nn.Conv1D(in_channels=in_channels,
            out_channels=hidden_channels, kernel_size=kernel_size, padding=
            kernel_size // 2))
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = paddle.nn.Sequential(paddle.nn.ReLU(), paddle.nn.
            Dropout(p=p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(paddle.nn.Conv1D(in_channels=
                hidden_channels, out_channels=hidden_channels, kernel_size=
                kernel_size, padding=kernel_size // 2))
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = paddle.nn.Conv1D(in_channels=hidden_channels,
            out_channels=out_channels, kernel_size=1)
        self.proj.weight.set_value(paddle.zeros_like(self.proj.weight))
        self.proj.bias.set_value(paddle.zeros_like(self.proj.bias))

    def forward(self, x, x_mask):
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask


class WN(paddle.nn.Layer):

    def __init__(self, in_channels, hidden_channels, kernel_size,
        dilation_rate, n_layers, gin_channels=0, p_dropout=0):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        assert hidden_channels % 2 == 0
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size,
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        self.in_layers = paddle.nn.LayerList()
        self.res_skip_layers = paddle.nn.LayerList()
        self.drop = paddle.nn.Dropout(p=p_dropout)
        if gin_channels != 0:
            cond_layer = paddle.nn.Conv1D(in_channels=gin_channels,
                out_channels=2 * hidden_channels * n_layers, kernel_size=1)
            self.cond_layer = paddle.nn.utils.weight_norm(layer=cond_layer,
                name='weight')
        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = paddle.nn.Conv1D(in_channels=hidden_channels,
                out_channels=2 * hidden_channels, kernel_size=kernel_size,
                dilation=dilation, padding=padding)
            in_layer = paddle.nn.utils.weight_norm(layer=in_layer, name=
                'weight')
            self.in_layers.append(in_layer)
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels
            res_skip_layer = paddle.nn.Conv1D(in_channels=hidden_channels,
                out_channels=res_skip_channels, kernel_size=1)
            res_skip_layer = paddle.nn.utils.weight_norm(layer=
                res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask=None, g=None, **kwargs):
        output = paddle.zeros_like(x=x)
        n_channels_tensor = paddle.to_tensor(data=[self.hidden_channels],
            dtype='int32')
        if g is not None:
            g = self.cond_layer(g)
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            x_in = self.drop(x_in)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset + 2 * self.
                    hidden_channels, :]
            else:
                g_l = paddle.zeros_like(x=x_in)
            acts = commons.fused_add_tanh_sigmoid_multiply(x_in, g_l,
                n_channels_tensor)
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = (x + res_skip_acts[:, :self.hidden_channels, :]) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            paddle.nn.utils.remove_weight_norm(layer=self.cond_layer)
        for l in self.in_layers:
            paddle.nn.utils.remove_weight_norm(layer=l)
        for l in self.res_skip_layers:
            paddle.nn.utils.remove_weight_norm(layer=l)


class ActNorm(paddle.nn.Layer):

    def __init__(self, channels, ddi=False, **kwargs):
        super().__init__()
        self.channels = channels
        self.initialized = not ddi
        out_4 = paddle.create_parameter(shape=paddle.zeros(shape=[1,
            channels, 1]).shape, dtype=paddle.zeros(shape=[1, channels, 1])
            .numpy().dtype, default_initializer=paddle.nn.initializer.
            Assign(paddle.zeros(shape=[1, channels, 1])))
        out_4.stop_gradient = not True
        self.logs = out_4
        out_5 = paddle.create_parameter(shape=paddle.zeros(shape=[1,
            channels, 1]).shape, dtype=paddle.zeros(shape=[1, channels, 1])
            .numpy().dtype, default_initializer=paddle.nn.initializer.
            Assign(paddle.zeros(shape=[1, channels, 1])))
        out_5.stop_gradient = not True
        self.bias = out_5

    def forward(self, x, x_mask=None, reverse=False, **kwargs):
        if x_mask is None:
            x_mask = paddle.ones(shape=[x.shape[0], 1, x.shape[2]]).to(device
                =x.place, dtype=x.dtype)
        x_len = paddle.sum(x=x_mask, axis=[1, 2])
        if not self.initialized:
            self.initialize(x, x_mask)
            self.initialized = True
        if reverse:
            z = (x - self.bias) * paddle.exp(x=-self.logs) * x_mask
            logdet = None
        else:
            z = (self.bias + paddle.exp(x=self.logs) * x) * x_mask
            logdet = paddle.sum(x=self.logs) * x_len
        return z, logdet

    def store_inverse(self):
        pass

    def set_ddi(self, ddi):
        self.initialized = not ddi

    def initialize(self, x, x_mask):
        with paddle.no_grad():
            denom = paddle.sum(x=x_mask, axis=[0, 2])
            m = paddle.sum(x=x * x_mask, axis=[0, 2]) / denom
            m_sq = paddle.sum(x=x * x * x_mask, axis=[0, 2]) / denom
            v = m_sq - m ** 2
            logs = 0.5 * paddle.log(x=paddle.clip(x=v, min=1e-06))
            
            bias_init = (-m * paddle.exp(x=-logs)).reshape(self.bias.shape).astype(
                dtype=self.bias.dtype)
            
            logs_init = (-logs).reshape(self.logs.shape).astype(dtype=self.logs.dtype)
            paddle.assign(bias_init, output=self.bias.data)
            paddle.assign(logs_init, output=self.logs.data)


class InvConvNear(paddle.nn.Layer):

    def __init__(self, channels, n_split=4, no_jacobian=False, **kwargs):
        super().__init__()
        assert n_split % 2 == 0
        self.channels = channels
        self.n_split = n_split
        self.no_jacobian = no_jacobian
        
        # 创建一个正态分布的随机矩阵，并进行 QR 分解
        random_matrix_np = np.random.normal(size=(self.n_split, self.n_split))
        q_np, _ = np.linalg.qr(random_matrix_np)

        # 转换为 PaddlePaddle 张量
        w_init = paddle.to_tensor(q_np, dtype='float32')

        # 检查行列式，如果小于 0 则调整第一列
        if np.linalg.det(q_np) < 0:
            w_init[:, 0] = -1 * w_init[:, 0]
        self.weight = paddle.create_parameter(
            shape=w_init.shape, 
            dtype=str(w_init.numpy().dtype),
            default_initializer=paddle.nn.initializer.Assign(w_init)
        )

    def forward(self, x, x_mask=None, reverse=False, **kwargs):
        b, c, t = x.shape
        assert c % self.n_split == 0
        if x_mask is None:
            x_mask = 1
            x_len = paddle.ones(shape=(b,), dtype=x.dtype) * t
        else:
            x_len = paddle.sum(x=x_mask, axis=[1, 2])
        
        x = x.reshape([b, 2, c // self.n_split, self.n_split // 2, t])
        x = x.transpose(perm=[0, 1, 3, 2, 4]).reshape([b, self.n_split, c //self.n_split, t])
        if reverse:
            if hasattr(self, 'weight_inv'):
                weight = self.weight_inv
            else:
                weight = paddle.linalg.inv(x=self.weight.astype(dtype=
                    'float32')).astype(dtype=self.weight.dtype)
            logdet = None
        else:
            weight = self.weight
            if self.no_jacobian:
                logdet = 0
            else:
                logdet = paddle.log(paddle.linalg.det(self.weight)) * (c /
                    self.n_split) * x_len
        
        weight = weight.reshape([self.n_split, self.n_split, 1, 1])
        z = paddle.nn.functional.conv2d(x=x, weight=weight)
        
        z = z.reshape([b, 2, self.n_split // 2, c // self.n_split, t])
        
        z = z.transpose(perm=[0, 1, 3, 2, 4]).reshape([b, c, t]) * x_mask
        return z, logdet

    def store_inverse(self):
        self.weight_inv = paddle.linalg.inv(x=self.weight.astype(dtype=
            'float32')).astype(dtype=self.weight.dtype)
