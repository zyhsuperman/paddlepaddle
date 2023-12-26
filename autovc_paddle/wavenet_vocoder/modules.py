from __future__ import with_statement, print_function, absolute_import
import paddle
import paddle.nn as nn
import math
import numpy as np
from wavenet_vocoder import conv

def split(x, num_or_sections, axis=0):
    if isinstance(num_or_sections, int):
        return paddle.split(x, x.shape[axis]//num_or_sections, axis)
    else:
        return paddle.split(x, num_or_sections, axis)


def Conv1d(in_channels, out_channels, kernel_size, dropout=0, std_mul=4.0,
    **kwargs):
    m = conv.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt(std_mul * (1.0 - dropout) / (m._kernel_size[0] *
        in_channels))
    # 创建新的权重张量并初始化
    new_weight = paddle.normal(mean=0, std=std, shape=m.weight.shape)
    m.weight.set_value(new_weight)

    # 创建新的偏置张量并初始化为0
    new_bias = paddle.zeros(shape=m.bias.shape)
    m.bias.set_value(new_bias)
    return paddle.nn.utils.weight_norm(layer=m)


def Embedding(num_embeddings, embedding_dim, padding_idx, std=0.01):
    m = paddle.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=
        embedding_dim, padding_idx=padding_idx)
    new_weight = paddle.normal(mean=0, std=std, shape=m.weight.shape)
    m.weight.set_value(new_weight)
    return m


def ConvTranspose2d(in_channels, out_channels, kernel_size,
    weight_normalization=True, **kwargs):
    freq_axis_kernel_size = kernel_size[0]
    m = nn.Conv2DTranspose(in_channels, out_channels, kernel_size, **kwargs)
    # 创建新的权重张量并初始化
    new_weight = paddle.full(shape=m.weight.shape, fill_value=1.0 / freq_axis_kernel_size)
    m.weight.set_value(new_weight)

    # 创建新的偏置张量并初始化为0
    new_bias = paddle.zeros(shape=m.bias.shape)
    m.bias.set_value(new_bias)
    if weight_normalization:
        return paddle.nn.utils.weight_norm(layer=m)
    else:
        return m


def Conv1d1x1(in_channels, out_channels, bias=True, weight_normalization=True):
    """1-by-1 convolution layer
    """
    if weight_normalization:
        assert bias
        return Conv1d(in_channels, out_channels, kernel_size=1, padding=0,
            dilation=1, std_mul=1.0)
    else:
        return conv.Conv1d(in_channels, out_channels, kernel_size=1,
            padding=0, dilation=1)


def _conv1x1_forward(conv, x, is_incremental):
    """Conv1x1 forward
    """
    if is_incremental:
        x = conv.incremental_forward(x)
    else:
        x = conv(x)
    return x


class ResidualConv1dGLU(paddle.nn.Layer):
    """Residual dilated conv1d + Gated linear unit

    Args:
        residual_channels (int): Residual input / output channels
        gate_channels (int): Gated activation channels.
        kernel_size (int): Kernel size of convolution layers.
        skip_out_channels (int): Skip connection channels. If None, set to same
          as ``residual_channels``.
        cin_channels (int): Local conditioning channels. If negative value is
          set, local conditioning is disabled.
        gin_channels (int): Global conditioning channels. If negative value is
          set, global conditioning is disabled.
        dropout (float): Dropout probability.
        padding (int): Padding for convolution layers. If None, proper padding
          is computed depends on dilation and kernel_size.
        dilation (int): Dilation factor.
        weight_normalization (bool): If True, DeepVoice3-style weight
          normalization is applied.
    """

    def __init__(self, residual_channels, gate_channels, kernel_size,
        skip_out_channels=None, cin_channels=-1, gin_channels=-1, dropout=1 -
        0.95, padding=None, dilation=1, causal=True, bias=True,
        weight_normalization=True, *args, **kwargs):
        super(ResidualConv1dGLU, self).__init__()
        self.dropout = dropout
        if skip_out_channels is None:
            skip_out_channels = residual_channels
        if padding is None:
            if causal:
                padding = (kernel_size - 1) * dilation
            else:
                padding = (kernel_size - 1) // 2 * dilation
        self.causal = causal
        if weight_normalization:
            assert bias
            self.conv = Conv1d(residual_channels, gate_channels,
                kernel_size, *args, padding=padding, dilation=dilation,
                std_mul=1.0, **kwargs)
        else:
            self.conv = conv.Conv1d(residual_channels, gate_channels,
                kernel_size, *args, padding=padding, dilation=dilation,
                 **kwargs)
        if cin_channels > 0:
            self.conv1x1c = Conv1d1x1(cin_channels, gate_channels, bias=
                bias, weight_normalization=weight_normalization)
        else:
            self.conv1x1c = None
        if gin_channels > 0:
            self.conv1x1g = Conv1d1x1(gin_channels, gate_channels, bias=
                bias, weight_normalization=weight_normalization)
        else:
            self.conv1x1g = None
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = Conv1d1x1(gate_out_channels, residual_channels,
            bias=bias, weight_normalization=weight_normalization)
        self.conv1x1_skip = Conv1d1x1(gate_out_channels, skip_out_channels,
            bias=bias, weight_normalization=weight_normalization)

    def forward(self, x, c=None, g=None):
        return self._forward(x, c, g, False)

    def incremental_forward(self, x, c=None, g=None):
        return self._forward(x, c, g, True)

    def _forward(self, x, c, g, is_incremental):
        """Forward

        Args:
            x (Tensor): B x C x T
            c (Tensor): B x C x T, Local conditioning features
            g (Tensor): B x C x T, Expanded global conditioning features
            is_incremental (Bool) : Whether incremental mode or not

        Returns:
            Tensor: output
        """
        residual = x
        x = paddle.nn.functional.dropout(x=x, p=self.dropout, training=self
            .training)
        if is_incremental:
            splitdim = -1
            x = self.conv.incremental_forward(x)
        else:
            splitdim = 1
            x = self.conv(x)
            x = x[:, :, :residual.shape[-1]] if self.causal else x
        a, b = split(x, x.shape[splitdim] // 2, axis=splitdim)
        if c is not None:
            assert self.conv1x1c is not None
            c = _conv1x1_forward(self.conv1x1c, c, is_incremental)
            ca, cb = split(c, c.shape[splitdim] // 2, axis=splitdim)
            a, b = a + ca, b + cb
        if g is not None:
            assert self.conv1x1g is not None
            g = _conv1x1_forward(self.conv1x1g, g, is_incremental)
            ga, gb = split(g, g.shape[splitdim] // 2, dim=splitdim)
            a, b = a + ga, b + gb
        x = paddle.nn.functional.tanh(x=a) * paddle.nn.functional.sigmoid(x=b)
        s = _conv1x1_forward(self.conv1x1_skip, x, is_incremental)
        x = _conv1x1_forward(self.conv1x1_out, x, is_incremental)
        x = (x + residual) * math.sqrt(0.5)
        return x, s

    def clear_buffer(self):
        for c in [self.conv, self.conv1x1_out, self.conv1x1_skip, self.
            conv1x1c, self.conv1x1g]:
            if c is not None:
                c.clear_buffer()
