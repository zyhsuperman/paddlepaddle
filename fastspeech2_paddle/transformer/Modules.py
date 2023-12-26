import paddle
import paddle.nn as nn
import numpy as np

class ScaledDotProductAttention(nn.Layer):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(axis=2)

    def forward(self, q, k, v, mask=None):

        attn = paddle.bmm(q, paddle.transpose(k, [0, 2, 1]))
        attn = attn / self.temperature

        if mask is not None:
            attn = paddle.where(~mask, attn, paddle.to_tensor(-np.inf, dtype='float32'))

        attn = self.softmax(attn)
        output = paddle.bmm(attn, v)

        return output, attn