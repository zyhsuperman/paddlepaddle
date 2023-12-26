import sys
import paddle
import paddle.nn as nn
from paddle.nn.utils import weight_norm, remove_weight_norm

import copy
import math
import commons
import modules
import attentions
import monotonic_align
from commons import init_weights, get_padding


def split(x, num_or_sections, axis=0):
    if isinstance(num_or_sections, int):
        return paddle.split(x, x.shape[axis]//num_or_sections, axis)
    else:
        return paddle.split(x, num_or_sections, axis)

def sequence_mask(lengths, maxlen, dtype=paddle.bool):
    return paddle.arange(maxlen, dtype=lengths.dtype).expand([len(lengths), maxlen]) < lengths.unsqueeze(-1)

class StochasticDurationPredictor(paddle.nn.Layer):

    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout,
        n_flows=4, gin_channels=0):
        super().__init__()
        filter_channels = in_channels
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels
        self.log_flow = modules.Log()
        self.flows = paddle.nn.LayerList()
        self.flows.append(modules.ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(modules.ConvFlow(2, filter_channels,
                kernel_size, n_layers=3))
            self.flows.append(modules.Flip())
        self.post_pre = paddle.nn.Conv1D(in_channels=1, out_channels=
            filter_channels, kernel_size=1)
        self.post_proj = paddle.nn.Conv1D(in_channels=filter_channels,
            out_channels=filter_channels, kernel_size=1)
        self.post_convs = modules.DDSConv(filter_channels, kernel_size,
            n_layers=3, p_dropout=p_dropout)
        self.post_flows = paddle.nn.LayerList()
        self.post_flows.append(modules.ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(modules.ConvFlow(2, filter_channels,
                kernel_size, n_layers=3))
            self.post_flows.append(modules.Flip())
        self.pre = paddle.nn.Conv1D(in_channels=in_channels, out_channels=
            filter_channels, kernel_size=1)
        self.proj = paddle.nn.Conv1D(in_channels=filter_channels,
            out_channels=filter_channels, kernel_size=1)
        self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers
            =3, p_dropout=p_dropout)
        if gin_channels != 0:
            self.cond = paddle.nn.Conv1D(in_channels=gin_channels,
                out_channels=filter_channels, kernel_size=1)

    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0
        ):
        # import pdb
        # pdb.set_trace()
        x = x.detach()
        x = self.pre(x)
        if g is not None:
            g = g.detach()
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask
        if not reverse:
            flows = self.flows
            assert w is not None
            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = paddle.randn(shape=[w.shape[0], 2, w.shape[2]])
            e_q = paddle.cast(e_q, dtype=x.dtype) * x_mask
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=x + h_w)
                logdet_tot_q += logdet_q
            z_u, z1 = split(x=z_q, num_or_sections=[1, 1], axis=1)
            u = paddle.nn.functional.sigmoid(x=z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += paddle.sum(x=(paddle.nn.functional.log_sigmoid(
                x=z_u) + paddle.nn.functional.log_sigmoid(x=-z_u)) * x_mask,
                axis=[1, 2])
            logq = paddle.sum(x=-0.5 * (math.log(2 * math.pi) + e_q ** 2) *
                x_mask, axis=[1, 2]) - logdet_tot_q
            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = paddle.concat(x=[z0, z1], axis=1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = paddle.sum(x=0.5 * (math.log(2 * math.pi) + z ** 2) *
                x_mask, axis=[1, 2]) - logdet_tot
            return nll + logq
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]
            z = paddle.randn([x.shape[0], 2, x.shape[2]], dtype=x.dtype) * noise_scale
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = split(x=z, num_or_sections=[1, 1], axis=1)
            logw = z0
            return logw


class DurationPredictor(paddle.nn.Layer):

    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout,
        gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels
        self.drop = paddle.nn.Dropout(p=p_dropout)
        self.conv_1 = paddle.nn.Conv1D(in_channels=in_channels,
            out_channels=filter_channels, kernel_size=kernel_size, padding=
            kernel_size // 2)
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = paddle.nn.Conv1D(in_channels=filter_channels,
            out_channels=filter_channels, kernel_size=kernel_size, padding=
            kernel_size // 2)
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = paddle.nn.Conv1D(in_channels=filter_channels,
            out_channels=1, kernel_size=1)
        if gin_channels != 0:
            self.cond = paddle.nn.Conv1D(in_channels=gin_channels,
                out_channels=in_channels, kernel_size=1)

    def forward(self, x, x_mask, g=None):
        x = x.detach()
        if g is not None:
            g = g.detach()
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = paddle.nn.functional.relu(x=x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = paddle.nn.functional.relu(x=x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class TextEncoder(paddle.nn.Layer):

    def __init__(self, n_vocab, out_channels, hidden_channels,
        filter_channels, n_heads, n_layers, kernel_size, p_dropout):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.emb = paddle.nn.Embedding(num_embeddings=n_vocab,
            embedding_dim=hidden_channels)
        init_Normal = paddle.nn.initializer.Normal(mean=0.0, std=
            hidden_channels ** -0.5)
        init_Normal(self.emb.weight)
        self.encoder = attentions.Encoder(hidden_channels, filter_channels,
            n_heads, n_layers, kernel_size, p_dropout)
        self.proj = paddle.nn.Conv1D(in_channels=hidden_channels,
            out_channels=out_channels * 2, kernel_size=1)

    def forward(self, x, x_lengths):
        x = self.emb(x) * math.sqrt(self.hidden_channels)
        x = paddle.transpose(x=x, perm=[0, 2, 1])

        x_mask = paddle.unsqueeze(sequence_mask(x_lengths, x.shape[2], dtype=paddle.bool), 1).astype(x.dtype)
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = split(x=stats, num_or_sections=self.
            out_channels, axis=1)
        return x, m, logs, x_mask


class ResidualCouplingBlock(paddle.nn.Layer):

    def __init__(self, channels, hidden_channels, kernel_size,
        dilation_rate, n_layers, n_flows=4, gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels
        self.flows = paddle.nn.LayerList()
        for i in range(n_flows):
            self.flows.append(modules.ResidualCouplingLayer(channels,
                hidden_channels, kernel_size, dilation_rate, n_layers,
                gin_channels=gin_channels, mean_only=True))
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(paddle.nn.Layer):

    def __init__(self, in_channels, out_channels, hidden_channels,
        kernel_size, dilation_rate, n_layers, gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.pre = paddle.nn.Conv1D(in_channels=in_channels, out_channels=
            hidden_channels, kernel_size=1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate,
            n_layers, gin_channels=gin_channels)
        self.proj = paddle.nn.Conv1D(in_channels=hidden_channels,
            out_channels=out_channels * 2, kernel_size=1)

    def forward(self, x, x_lengths, g=None):
        # x_mask = paddle.unsqueeze(x=commons.sequence_mask(x_lengths, x.
        #     shape[2]), axis=1).to(x.dtype)
        x_mask = paddle.unsqueeze(x=commons.sequence_mask(x_lengths, x.shape[2]), axis=1)
        x_mask = paddle.cast(x_mask, dtype=x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = split(x=stats, num_or_sections=self.
            out_channels, axis=1)
        z = (m + paddle.randn(shape=m.shape, dtype=m.dtype) * paddle.exp(x=
            logs)) * x_mask
        return z, m, logs, x_mask


class Generator(paddle.nn.Layer):

    def __init__(self, initial_channel, resblock, resblock_kernel_sizes,
        resblock_dilation_sizes, upsample_rates, upsample_initial_channel,
        upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = paddle.nn.Conv1D(in_channels=initial_channel,
            out_channels=upsample_initial_channel, kernel_size=7, stride=1,
            padding=3)
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2
        self.ups = paddle.nn.LayerList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(paddle.nn.utils.weight_norm(layer=paddle.nn.
                Conv1DTranspose(in_channels=upsample_initial_channel // 2 **
                i, out_channels=upsample_initial_channel // 2 ** (i + 1),
                kernel_size=k, stride=u, padding=(k - u) // 2)))
        self.resblocks = paddle.nn.LayerList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // 2 ** (i + 1)
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes,
                resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))
        self.conv_post = paddle.nn.Conv1D(in_channels=ch, out_channels=1,
            kernel_size=7, stride=1, padding=3, bias_attr=False)
        self.ups.apply(init_weights)
        if gin_channels != 0:
            self.cond = paddle.nn.Conv1D(in_channels=gin_channels,
                out_channels=upsample_initial_channel, kernel_size=1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        for i in range(self.num_upsamples):
            x = paddle.nn.functional.leaky_relu(x=x, negative_slope=modules
                .LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = paddle.nn.functional.leaky_relu(x=x)
        x = self.conv_post(x)
        x = paddle.nn.functional.tanh(x=x)
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            paddle.nn.utils.remove_weight_norm(layer=l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(paddle.nn.Layer):

    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False
        ):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else nn.SpectralNorm
        self.convs = paddle.nn.LayerList(sublayers=[norm_f(paddle.nn.Conv2D
            (in_channels=1, out_channels=32, kernel_size=(kernel_size, 1),
            stride=(stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(paddle.nn.Conv2D(in_channels=32, out_channels=128,
            kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(
            get_padding(kernel_size, 1), 0))), norm_f(paddle.nn.Conv2D(
            in_channels=128, out_channels=512, kernel_size=(kernel_size, 1),
            stride=(stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(paddle.nn.Conv2D(in_channels=512, out_channels=1024,
            kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(
            get_padding(kernel_size, 1), 0))), norm_f(paddle.nn.Conv2D(
            in_channels=1024, out_channels=1024, kernel_size=(kernel_size, 
            1), stride=1, padding=(get_padding(kernel_size, 1), 0)))])
        self.conv_post = norm_f(paddle.nn.Conv2D(in_channels=1024,
            out_channels=1, kernel_size=(3, 1), stride=1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - t % self.period
            x = paddle.nn.functional.pad(x, pad=(0, n_pad), mode='reflect', data_format='NCL')
            t = t + n_pad
        x = x.reshape([b, c, t // self.period, self.period])
        for l in self.convs:
            x = l(x)
            x = paddle.nn.functional.leaky_relu(x=x, negative_slope=modules
                .LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = paddle.flatten(x=x, start_axis=1, stop_axis=-1)
        return x, fmap


class DiscriminatorS(paddle.nn.Layer):

    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else nn.SpectralNorm
        self.convs = paddle.nn.LayerList(sublayers=[norm_f(paddle.nn.Conv1D
            (in_channels=1, out_channels=16, kernel_size=15, stride=1,
            padding=7)), norm_f(paddle.nn.Conv1D(in_channels=16,
            out_channels=64, kernel_size=41, stride=4, groups=4, padding=20
            )), norm_f(paddle.nn.Conv1D(in_channels=64, out_channels=256,
            kernel_size=41, stride=4, groups=16, padding=20)), norm_f(
            paddle.nn.Conv1D(in_channels=256, out_channels=1024,
            kernel_size=41, stride=4, groups=64, padding=20)), norm_f(
            paddle.nn.Conv1D(in_channels=1024, out_channels=1024,
            kernel_size=41, stride=4, groups=256, padding=20)), norm_f(
            paddle.nn.Conv1D(in_channels=1024, out_channels=1024,
            kernel_size=5, stride=1, padding=2))])
        self.conv_post = norm_f(paddle.nn.Conv1D(in_channels=1024,
            out_channels=1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = paddle.nn.functional.leaky_relu(x=x, negative_slope=modules
                .LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = paddle.flatten(x=x, start_axis=1, stop_axis=-1)
        return x, fmap


class MultiPeriodDiscriminator(paddle.nn.Layer):

    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]
        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=
            use_spectral_norm) for i in periods]
        self.discriminators = paddle.nn.LayerList(sublayers=discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SynthesizerTrn(paddle.nn.Layer):
    """
  Synthesizer for Training
  """

    def __init__(self, n_vocab, spec_channels, segment_size, inter_channels,
        hidden_channels, filter_channels, n_heads, n_layers, kernel_size,
        p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes,
        upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
        n_speakers=0, gin_channels=0, use_sdp=True, **kwargs):
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.use_sdp = use_sdp
        self.enc_p = TextEncoder(n_vocab, inter_channels, hidden_channels,
            filter_channels, n_heads, n_layers, kernel_size, p_dropout)
        self.dec = Generator(inter_channels, resblock,
            resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
            upsample_initial_channel, upsample_kernel_sizes, gin_channels=
            gin_channels)
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels,
            hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 
            5, 1, 4, gin_channels=gin_channels)
        if use_sdp:
            self.dp = StochasticDurationPredictor(hidden_channels, 192, 3, 
                0.5, 4, gin_channels=gin_channels)
        else:
            self.dp = DurationPredictor(hidden_channels, 256, 3, 0.5,
                gin_channels=gin_channels)
        if n_speakers > 1:
            self.emb_g = paddle.nn.Embedding(num_embeddings=n_speakers,
                embedding_dim=gin_channels)

    def forward(self, x, x_lengths, y, y_lengths, sid=None):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        with paddle.no_grad():
            # negative cross-entropy
            s_p_sq_r = paddle.exp(-2 * logs_p)
            neg_cent1 = paddle.sum(-0.5 * math.log(2 * math.pi) - logs_p, axis=1, keepdim=True)
            neg_cent2 = paddle.matmul(-0.5 * paddle.transpose(z_p ** 2, [0, 2, 1]), s_p_sq_r)
            neg_cent3 = paddle.matmul(paddle.transpose(z_p, [0, 2, 1]), m_p * s_p_sq_r)
            neg_cent4 = paddle.sum(-0.5 * (m_p ** 2) * s_p_sq_r, axis=1, keepdim=True)
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            attn_mask = paddle.unsqueeze(x_mask, 2) * paddle.unsqueeze(y_mask, -1)
            attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

        w = paddle.sum(attn, axis=2)
        if self.use_sdp:
            l_length = self.dp(x, x_mask, w, g=g)
            l_length = l_length / paddle.sum(x_mask)
        else:
            logw_ = paddle.log(w + 1e-6) * x_mask
            logw = self.dp(x, x_mask, g=g)
            l_length = paddle.sum((logw - logw_)**2, axis=[1, 2]) / paddle.sum(x_mask)

        m_p = paddle.matmul(attn.squeeze(1), paddle.transpose(m_p, [0, 2, 1])).transpose([0, 2, 1])
        logs_p = paddle.matmul(attn.squeeze(1), paddle.transpose(logs_p, [0, 2, 1])).transpose([0, 2, 1])

        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)

        return o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, x, x_lengths, sid=None, noise_scale=1, length_scale=1,
        noise_scale_w=1.0, max_len=None):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)

        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(axis=-1)
        else:
            g = None
        if self.use_sdp:
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=
                noise_scale_w)
        else:
            logw = self.dp(x, x_mask, g=g)
        # w = paddle.exp(x=logw) * x_mask * length_scale
        # w_ceil = paddle.ceil(x=w)
        # y_lengths = paddle.clip(x=paddle.sum(x=w_ceil, axis=[1, 2]), min=1
        #     ).astype(dtype='int64')
        # y_mask = paddle.unsqueeze(x=commons.sequence_mask(y_lengths, None),
        #     axis=1).to(x_mask.dtype)
        # attn_mask = paddle.unsqueeze(x=x_mask, axis=2) * paddle.unsqueeze(x
        #     =y_mask, axis=-1)
        # attn = commons.generate_path(w_ceil, attn_mask)
        # x = m_p
        # perm_13 = list(range(x.ndim))
        # perm_13[1] = 2
        # perm_13[2] = 1
        # x = paddle.matmul(x=attn.squeeze(axis=1), y=x.transpose(perm=perm_13))
        # perm_14 = list(range(x.ndim))
        # perm_14[1] = 2
        # perm_14[2] = 1
        # m_p = x.transpose(perm=perm_14)
        # x = logs_p
        # perm_15 = list(range(x.ndim))
        # perm_15[1] = 2
        # perm_15[2] = 1
        # x = paddle.matmul(x=attn.squeeze(axis=1), y=x.transpose(perm=perm_15))
        # perm_16 = list(range(x.ndim))
        # perm_16[1] = 2
        # perm_16[2] = 1
        # logs_p = x.transpose(perm=perm_16)
        # z_p = m_p + paddle.randn(shape=m_p.shape, dtype=m_p.dtype
        #     ) * paddle.exp(x=logs_p) * noise_scale

        w = paddle.exp(logw) * x_mask * length_scale
        w_ceil = paddle.ceil(w)
        y_lengths = paddle.clip(paddle.sum(w_ceil, axis=[1, 2]), min=1).astype('int64')
        y_mask = paddle.unsqueeze(commons.sequence_mask(y_lengths, None), 1).astype(x_mask.dtype)
        attn_mask = paddle.unsqueeze(x_mask, 2) * paddle.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = paddle.matmul(paddle.squeeze(attn, axis=1), paddle.transpose(m_p, [0, 2, 1])).transpose([0, 2, 1])
        logs_p = paddle.matmul(paddle.squeeze(attn, axis=1), paddle.transpose(logs_p, [0, 2, 1])).transpose([0, 2, 1])

        z_p = m_p + paddle.normal(mean=0, std=1, shape=m_p.shape).astype(m_p.dtype) * paddle.exp(logs_p) * noise_scale

        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)

    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
        assert self.n_speakers > 0, 'n_speakers have to be larger than 0.'
        g_src = self.emb_g(sid_src).unsqueeze(axis=-1)
        g_tgt = self.emb_g(sid_tgt).unsqueeze(axis=-1)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.dec(z_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)
