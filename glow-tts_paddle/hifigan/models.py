import paddle
from paddle.nn.utils import spectral_norm, weight_norm, remove_weight_norm
LRELU_SLOPE = 0.1



def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean, std)



def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

class ResBlock1(paddle.nn.Layer):

    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = paddle.nn.LayerList(sublayers=[paddle.nn.utils.
            weight_norm(layer=paddle.nn.Conv1D(in_channels=channels,
            out_channels=channels, kernel_size=kernel_size, stride=1,
            dilation=dilation[0], padding=get_padding(kernel_size, dilation
            [0]))), paddle.nn.utils.weight_norm(layer=paddle.nn.Conv1D(
            in_channels=channels, out_channels=channels, kernel_size=
            kernel_size, stride=1, dilation=dilation[1], padding=
            get_padding(kernel_size, dilation[1]))), paddle.nn.utils.
            weight_norm(layer=paddle.nn.Conv1D(in_channels=channels,
            out_channels=channels, kernel_size=kernel_size, stride=1,
            dilation=dilation[2], padding=get_padding(kernel_size, dilation
            [2])))])
        self.convs1.apply(init_weights)
        self.convs2 = paddle.nn.LayerList(sublayers=[paddle.nn.utils.
            weight_norm(layer=paddle.nn.Conv1D(in_channels=channels,
            out_channels=channels, kernel_size=kernel_size, stride=1,
            dilation=1, padding=get_padding(kernel_size, 1))), paddle.nn.
            utils.weight_norm(layer=paddle.nn.Conv1D(in_channels=channels,
            out_channels=channels, kernel_size=kernel_size, stride=1,
            dilation=1, padding=get_padding(kernel_size, 1))), paddle.nn.
            utils.weight_norm(layer=paddle.nn.Conv1D(in_channels=channels,
            out_channels=channels, kernel_size=kernel_size, stride=1,
            dilation=1, padding=get_padding(kernel_size, 1)))])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = paddle.nn.functional.leaky_relu(x=x, negative_slope=
                LRELU_SLOPE)
            xt = c1(xt)
            xt = paddle.nn.functional.leaky_relu(x=xt, negative_slope=
                LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            paddle.nn.utils.remove_weight_norm(layer=l)
        for l in self.convs2:
            paddle.nn.utils.remove_weight_norm(layer=l)


class ResBlock2(paddle.nn.Layer):

    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = paddle.nn.LayerList(sublayers=[paddle.nn.utils.
            weight_norm(layer=paddle.nn.Conv1D(in_channels=channels,
            out_channels=channels, kernel_size=kernel_size, stride=1,
            dilation=dilation[0], padding=get_padding(kernel_size, dilation
            [0]))), paddle.nn.utils.weight_norm(layer=paddle.nn.Conv1D(
            in_channels=channels, out_channels=channels, kernel_size=
            kernel_size, stride=1, dilation=dilation[1], padding=
            get_padding(kernel_size, dilation[1])))])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = paddle.nn.functional.leaky_relu(x=x, negative_slope=
                LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            paddle.nn.utils.remove_weight_norm(layer=l)


class Generator(paddle.nn.Layer):

    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = paddle.nn.utils.weight_norm(layer=paddle.nn.Conv1D(
            in_channels=80, out_channels=h.upsample_initial_channel,
            kernel_size=7, stride=1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2
        self.ups = paddle.nn.LayerList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.
            upsample_kernel_sizes)):
            self.ups.append(paddle.nn.utils.weight_norm(layer=paddle.nn.
                Conv1DTranspose(in_channels=h.upsample_initial_channel // 2 **
                i, out_channels=h.upsample_initial_channel // 2 ** (i + 1),
                kernel_size=k, stride=u, padding=(k - u) // 2)))
        self.resblocks = paddle.nn.LayerList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // 2 ** (i + 1)
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.
                resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))
        self.conv_post = paddle.nn.utils.weight_norm(layer=paddle.nn.Conv1D
            (in_channels=ch, out_channels=1, kernel_size=7, stride=1,
            padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = paddle.nn.functional.leaky_relu(x=x, negative_slope=LRELU_SLOPE
                )
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
        paddle.nn.utils.remove_weight_norm(layer=self.conv_pre)
        paddle.nn.utils.remove_weight_norm(layer=self.conv_post)


class DiscriminatorP(paddle.nn.Layer):

    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False
        ):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = paddle.nn.LayerList(sublayers=[norm_f(paddle.nn.Conv2D
            (in_channels=1, out_channels=32, kernel_size=(kernel_size, 1),
            stride=(stride, 1), padding=(get_padding(5, 1), 0))), norm_f(
            paddle.nn.Conv2D(in_channels=32, out_channels=128, kernel_size=
            (kernel_size, 1), stride=(stride, 1), padding=(get_padding(5, 1
            ), 0))), norm_f(paddle.nn.Conv2D(in_channels=128, out_channels=
            512, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=
            (get_padding(5, 1), 0))), norm_f(paddle.nn.Conv2D(in_channels=
            512, out_channels=1024, kernel_size=(kernel_size, 1), stride=(
            stride, 1), padding=(get_padding(5, 1), 0))), norm_f(paddle.nn.
            Conv2D(in_channels=1024, out_channels=1024, kernel_size=(
            kernel_size, 1), stride=1, padding=(2, 0)))])
        self.conv_post = norm_f(paddle.nn.Conv2D(in_channels=1024,
            out_channels=1, kernel_size=(3, 1), stride=1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - t % self.period
            x = paddle.nn.functional.pad(x, (0, n_pad), 'reflect')
            t = t + n_pad
        
        x = x.reshape([b, c, t // self.period, self.period])
        for l in self.convs:
            x = l(x)
            x = paddle.nn.functional.leaky_relu(x=x, negative_slope=LRELU_SLOPE
                )
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = paddle.flatten(x=x, start_axis=1, stop_axis=-1)
        return x, fmap


class MultiPeriodDiscriminator(paddle.nn.Layer):

    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = paddle.nn.LayerList(sublayers=[DiscriminatorP
            (2), DiscriminatorP(3), DiscriminatorP(5), DiscriminatorP(7),
            DiscriminatorP(11)])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(paddle.nn.Layer):

    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = paddle.nn.LayerList(sublayers=[norm_f(paddle.nn.Conv1D
            (in_channels=1, out_channels=128, kernel_size=15, stride=1,
            padding=7)), norm_f(paddle.nn.Conv1D(in_channels=128,
            out_channels=128, kernel_size=41, stride=2, groups=4, padding=
            20)), norm_f(paddle.nn.Conv1D(in_channels=128, out_channels=256,
            kernel_size=41, stride=2, groups=16, padding=20)), norm_f(
            paddle.nn.Conv1D(in_channels=256, out_channels=512, kernel_size
            =41, stride=4, groups=16, padding=20)), norm_f(paddle.nn.Conv1D
            (in_channels=512, out_channels=1024, kernel_size=41, stride=4,
            groups=16, padding=20)), norm_f(paddle.nn.Conv1D(in_channels=
            1024, out_channels=1024, kernel_size=41, stride=1, groups=16,
            padding=20)), norm_f(paddle.nn.Conv1D(in_channels=1024,
            out_channels=1024, kernel_size=5, stride=1, padding=2))])
        self.conv_post = norm_f(paddle.nn.Conv1D(in_channels=1024,
            out_channels=1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = paddle.nn.functional.leaky_relu(x=x, negative_slope=LRELU_SLOPE
                )
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = paddle.flatten(x=x, start_axis=1, stop_axis=-1)
        return x, fmap


class MultiScaleDiscriminator(paddle.nn.Layer):

    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = paddle.nn.LayerList(sublayers=[DiscriminatorS
            (use_spectral_norm=True), DiscriminatorS(), DiscriminatorS()])
        self.meanpools = paddle.nn.LayerList(sublayers=[paddle.nn.AvgPool1D
            (kernel_size=4, stride=2, padding=2, exclusive=False), paddle.
            nn.AvgPool1D(kernel_size=4, stride=2, padding=2, exclusive=False)])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += paddle.mean(x=paddle.abs(x=rl - gl))
    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = paddle.mean(x=(1 - dr) ** 2)
        g_loss = paddle.mean(x=dg ** 2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = paddle.mean(x=(1 - dg) ** 2)
        gen_losses.append(l)
        loss += l
    return loss, gen_losses
