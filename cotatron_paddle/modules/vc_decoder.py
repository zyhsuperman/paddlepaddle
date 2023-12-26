import paddle
from .cond_bn import ConditionalBatchNorm1d


class GBlock(paddle.nn.Layer):

    def __init__(self, in_channels, out_channels, condition_dim):
        super().__init__()
        self.cond_bn = paddle.nn.LayerList(sublayers=[
            ConditionalBatchNorm1d(in_channels if i == 0 else out_channels,
            condition_dim) for i in range(4)])
        self.leaky_relu = paddle.nn.LeakyReLU(negative_slope=0.2)
        self.cnn = paddle.nn.LayerList(sublayers=[paddle.nn.Conv1D(
            in_channels=in_channels if i == 0 else out_channels,
            out_channels=out_channels, kernel_size=3, dilation=2 ** i,
            padding=2 ** i) for i in range(4)])
        self.shortcut = paddle.nn.Conv1D(in_channels=in_channels,
            out_channels=out_channels, kernel_size=1)

    def forward(self, x, z, mask=None):
        identity = x
        x = self.cnn[0](self.leaky_relu(self.cond_bn[0](x, z)))
        if mask is not None:
            # x.masked_fill_(mask, 0.0)
            x = paddle.where(mask, paddle.to_tensor(0.0, dtype=x.dtype), x)
        x = self.cnn[1](self.leaky_relu(self.cond_bn[1](x, z)))
        if mask is not None:
            # x.masked_fill_(mask, 0.0)
            x = paddle.where(mask, paddle.to_tensor(0.0, dtype=x.dtype), x)
        x = x + self.shortcut(identity)
        if mask is not None:
            # x.masked_fill_(mask, 0.0)
            x = paddle.where(mask, paddle.to_tensor(0.0, dtype=x.dtype), x)
        identity = x
        x = self.cnn[2](self.leaky_relu(self.cond_bn[2](x, z)))
        if mask is not None:
            # x.masked_fill_(mask, 0.0)
            x = paddle.where(mask, paddle.to_tensor(0.0, dtype=x.dtype), x)
        x = self.cnn[3](self.leaky_relu(self.cond_bn[3](x, z)))
        if mask is not None:
            # x.masked_fill_(mask, 0.0)
            x = paddle.where(mask, paddle.to_tensor(0.0, dtype=x.dtype), x)
        x = x + identity
        return x


class VCDecoder(paddle.nn.Layer):

    def __init__(self, hp):
        super().__init__()
        self.stem = paddle.nn.Conv1D(in_channels=hp.chn.encoder + hp.chn.
            residual_out, out_channels=hp.chn.gblock[0], kernel_size=7,
            padding=3)
        self.gblock = paddle.nn.LayerList(sublayers=[GBlock(in_channels,
            out_channels, hp.chn.speaker.token) for in_channels,
            out_channels in zip(list(hp.chn.gblock)[:-1], hp.chn.gblock[1:])])
        self.final = paddle.nn.Conv1D(in_channels=hp.chn.gblock[-1],
            out_channels=hp.audio.n_mel_channels, kernel_size=1)

    def forward(self, x, speaker_emb, mask=None):
        x = self.stem(x)
        if mask is not None:
            # x.masked_fill_(mask, 0.0)
            x = paddle.where(mask, paddle.to_tensor(0.0, dtype=x.dtype), x)
        for gblock in self.gblock:
            x = gblock(x, speaker_emb, mask)
        x = self.final(x)
        if mask is not None:
            # x.masked_fill_(mask, 0.0)
            x = paddle.where(mask, paddle.to_tensor(0.0, dtype=x.dtype), x)
        return x
