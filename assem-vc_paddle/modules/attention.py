import paddle
import math
import numpy as np
from scipy.special import gamma


class StaticFilter(paddle.nn.Layer):

    def __init__(self, channels, kernel_size, out_dim):
        super().__init__()
        assert kernel_size % 2 == 1, 'kernel size of StaticFilter must be odd, got %d' % kernel_size
        padding = (kernel_size - 1) // 2
        self.conv = paddle.nn.Conv1D(in_channels=1, out_channels=channels,
            kernel_size=kernel_size, padding=padding)
        self.fc = paddle.nn.Linear(in_features=channels, out_features=
            out_dim, bias_attr=False)

    def forward(self, prev_attn):
        x = prev_attn.unsqueeze(axis=1)
        x = self.conv(x)
        x = x.transpose(perm=[0, 2, 1])
        x = self.fc(x)
        return x


class DynamicFilter(paddle.nn.Layer):

    def __init__(self, channels, kernel_size, attn_rnn_dim, hypernet_dim,
        out_dim):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        assert kernel_size % 2 == 1, 'kernel size of DynamicFilter must be odd, god %d' % kernel_size
        self.padding = (kernel_size - 1) // 2
        self.hypernet = paddle.nn.Sequential(paddle.nn.Linear(in_features=
            attn_rnn_dim, out_features=hypernet_dim), paddle.nn.Tanh(),
            paddle.nn.Linear(in_features=hypernet_dim, out_features=
            channels * kernel_size))
        self.fc = paddle.nn.Linear(in_features=channels, out_features=out_dim)

    def forward(self, query, prev_attn):
        B, T = prev_attn.shape
        convweight = self.hypernet(query)
        convweight = convweight.reshape([B, self.channels, self.kernel_size])
        convweight = convweight.reshape([B * self.channels, 1, self.kernel_size])
        prev_attn = prev_attn.unsqueeze(axis=0)

        x = paddle.nn.functional.conv1d(x=prev_attn, weight=convweight,
            padding=self.padding, groups=B)
        
        x = x.reshape([B, self.channels, T])
        x = x.transpose(perm=[0, 2, 1])
        x = self.fc(x)
        return x


class PriorFilter(paddle.nn.Layer):

    def __init__(self, causal_n, alpha, beta):
        super().__init__()
        self.causal_n = causal_n
        self.alpha = alpha
        self.beta = beta

        def beta_func(x, y):
            return gamma(x) * gamma(y) / gamma(x + y)

        def p(n, k, alpha, beta):

            def nCr(n, r):
                f = math.factorial
                return f(n) / (f(r) * f(n - r))
            return nCr(n, k) * beta_func(k + alpha, n - k + beta) / beta_func(
                alpha, beta)
        self.prior = np.array([p(self.causal_n - 1, i, self.alpha, self.
            beta) for i in range(self.causal_n)[::-1]]).astype(np.float32)
        self.prior = paddle.to_tensor(data=self.prior)
        self.prior = self.prior.reshape([1, 1, -1])
        self.register_buffer(name='prior_filter', tensor=self.prior)

    def forward(self, prev_attn):
        prev_attn = prev_attn.unsqueeze(axis=1)
        energies = paddle.nn.functional.conv1d(x=paddle.nn.functional.pad(
            prev_attn, [0, 0, 0, 0, self.causal_n - 1, 0]), weight=self.prior_filter)
        energies = energies.squeeze(axis=1)
        energies = paddle.log(x=paddle.clip(x=energies, min=1e-08))
        return energies


class Attention(paddle.nn.Layer):

    def __init__(self, attn_rnn_dim, attn_dim, static_channels,
        static_kernel_size, dynamic_channels, dynamic_kernel_size, causal_n,
        causal_alpha, causal_beta):
        super().__init__()
        self.v = paddle.nn.Linear(in_features=attn_dim, out_features=1,
            bias_attr=False)
        self.static_filter = StaticFilter(static_channels,
            static_kernel_size, attn_dim)
        self.dynamic_filter = DynamicFilter(dynamic_channels,
            dynamic_kernel_size, attn_rnn_dim, attn_dim, attn_dim)
        self.prior_filter = PriorFilter(causal_n, causal_alpha, causal_beta)
        self.score_mask_value = -float('inf')

    def get_alignment_energies(self, query, prev_attn):
        static_result = self.static_filter(prev_attn)
        dynamic_result = self.dynamic_filter(query, prev_attn)
        prior_result = self.prior_filter(prev_attn)
        energies = self.v(paddle.nn.functional.tanh(x=static_result +
            dynamic_result)).squeeze(axis=-1) + prior_result
        return energies

    def forward(self, attn_hidden, memory, prev_attn, mask):
        alignment = self.get_alignment_energies(attn_hidden, prev_attn)
        if mask is not None:
            alignment = paddle.where(mask, paddle.full_like(alignment, self.score_mask_value), alignment)
        attn_weights = paddle.nn.functional.softmax(x=alignment, axis=1)
        context = paddle.bmm(x=attn_weights.unsqueeze(axis=1), y=memory)
        context = context.squeeze(axis=1)
        return context, attn_weights
