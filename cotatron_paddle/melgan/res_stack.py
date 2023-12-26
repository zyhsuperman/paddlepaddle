import paddle
import numpy as np


class ResStack(paddle.nn.Layer):

    def __init__(self, channel):
        super(ResStack, self).__init__()
        self.blocks = paddle.nn.LayerList(sublayers=[paddle.nn.Sequential(
            paddle.nn.LeakyReLU(negative_slope=0.2), paddle.nn.Pad1D(
            padding=3 ** i, mode='reflect'), paddle.nn.utils.weight_norm(
            layer=paddle.nn.Conv1D(in_channels=channel, out_channels=
            channel, kernel_size=3, dilation=3 ** i)), paddle.nn.LeakyReLU(
            negative_slope=0.2), paddle.nn.utils.weight_norm(layer=paddle.
            nn.Conv1D(in_channels=channel, out_channels=channel,
            kernel_size=1))) for i in range(3)])
        self.shortcuts = paddle.nn.LayerList(sublayers=[paddle.nn.utils.
            weight_norm(layer=paddle.nn.Conv1D(in_channels=channel,
            out_channels=channel, kernel_size=1)) for i in range(3)])

    def forward(self, x):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = shortcut(x) + block(x)
        return x

    def remove_weight_norm(self):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            paddle.nn.utils.remove_weight_norm(layer=block[2])
            paddle.nn.utils.remove_weight_norm(layer=block[4])
            paddle.nn.utils.remove_weight_norm(layer=shortcut)
