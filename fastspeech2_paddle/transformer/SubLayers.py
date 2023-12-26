import paddle
import numpy as np
from .Modules import ScaledDotProductAttention


class MultiHeadAttention(paddle.nn.Layer):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = paddle.nn.Linear(in_features=d_model, out_features=
            n_head * d_k)
        self.w_ks = paddle.nn.Linear(in_features=d_model, out_features=
            n_head * d_k)
        self.w_vs = paddle.nn.Linear(in_features=d_model, out_features=
            n_head * d_v)
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k,
            0.5))
        self.layer_norm = paddle.nn.LayerNorm(normalized_shape=d_model)
        self.fc = paddle.nn.Linear(in_features=n_head * d_v, out_features=
            d_model)
        self.dropout = paddle.nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.shape
        sz_b, len_k, _ = k.shape
        sz_b, len_v, _ = v.shape
        residual = q

        q = self.w_qs(q).reshape([sz_b, len_q, n_head, d_k])
        
        k = self.w_ks(k).reshape([sz_b, len_k, n_head, d_k])
        
        v = self.w_vs(v).reshape([sz_b, len_v, n_head, d_v])
        
        q = q.transpose(perm=[2, 0, 1, 3]).reshape([-1, len_q, d_k])
        
        k = k.transpose(perm=[2, 0, 1, 3]).reshape([-1, len_k, d_k])
        
        v = v.transpose(perm=[2, 0, 1, 3]).reshape([-1, len_v, d_v])
        mask = mask.tile(repeat_times=[n_head, 1, 1])
        output, attn = self.attention(q, k, v, mask=mask)
        
        output = output.reshape([n_head, sz_b, len_q, d_v])
        
        output = output.transpose(perm=[1, 2, 0, 3]).reshape([sz_b, len_q, -1])
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, attn


class PositionwiseFeedForward(paddle.nn.Layer):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
        super().__init__()
        self.w_1 = paddle.nn.Conv1D(in_channels=d_in, out_channels=d_hid,
            kernel_size=kernel_size[0], padding=(kernel_size[0] - 1) // 2)
        self.w_2 = paddle.nn.Conv1D(in_channels=d_hid, out_channels=d_in,
            kernel_size=kernel_size[1], padding=(kernel_size[1] - 1) // 2)
        self.layer_norm = paddle.nn.LayerNorm(normalized_shape=d_in)
        self.dropout = paddle.nn.Dropout(p=dropout)

    def forward(self, x):
        residual = x
        x = x
        perm_1 = list(range(x.ndim))
        perm_1[1] = 2
        perm_1[2] = 1
        output = x.transpose(perm=perm_1)
        output = self.w_2(paddle.nn.functional.relu(x=self.w_1(output)))
        x = output
        perm_2 = list(range(x.ndim))
        perm_2[1] = 2
        perm_2[2] = 1
        output = x.transpose(perm=perm_2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output
