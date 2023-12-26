import paddle
from .padded_instancenorm import PaddedInstanceNorm1d
import math

def paddle_hann_window(win_size):
    return 0.5 - 0.5 * paddle.cos(2 * math.pi * paddle.arange(win_size) / (win_size - 1))


class ResidualEncoder(paddle.nn.Layer):

    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.relu = paddle.nn.ReLU()
        self.stem = paddle.nn.Conv2D(in_channels=1, out_channels=hp.chn.
            residual[0], kernel_size=(7, 7), padding=(3, 3), stride=(2, 1))
        self.conv_layers = paddle.nn.LayerList(sublayers=[paddle.nn.Conv2D(
            in_channels=in_channels, out_channels=out_channels, kernel_size
            =(3, 3), padding=(1, 1), stride=(2, 1)) for in_channels,
            out_channels in zip(list(hp.chn.residual)[:-1], hp.chn.residual
            [1:])])
        self.bn_layers = paddle.nn.LayerList(sublayers=[paddle.nn.
            BatchNorm2D(num_features=channels) for channels in hp.chn.residual]
            )
        self.avgpool = paddle.nn.AdaptiveAvgPool2D(output_size=(1, None))
        self.fc = paddle.nn.Conv1D(in_channels=hp.chn.residual[-1],
            out_channels=hp.chn.residual_out, kernel_size=1)
        assert hp.ker.hann_window % 2 == 1, 'hp.ker.hann_window must be odd'
        hann_window = paddle_hann_window(win_size=hp.ker.hann_window)
        
        hann_window = hann_window.reshape([1, 1, -1]) * (2.0 / (hp.ker.
            hann_window - 1))
        self.register_buffer(name='hann', tensor=hann_window)
        self.padded_norm = PaddedInstanceNorm1d(hp.chn.residual_out)
        self.norm = paddle.nn.InstanceNorm1D(num_features=hp.chn.
            residual_out, momentum=1 - 0.1)

    def forward(self, mel, mask, lengths):
        if mask is not None:
            mask = mask.unsqueeze(axis=1)
        x = mel.unsqueeze(axis=1)
        x = self.stem(x)
        for cnn, bn in zip(self.conv_layers, self.bn_layers):
            x = bn(x)
            x = self.relu(x)
            x = cnn(x)
            if mask is not None:
                x = paddle.where(mask, paddle.to_tensor(0.0, dtype=x.dtype), x)

        x = self.avgpool(x)  # [B, C, 1, T]
        x = x.squeeze(axis=2)  # [B, C, T]
        x = self.fc(x)  # [B, chn.residual_out, T]

        if mask is not None:
            x = paddle.where(mask.squeeze(1), paddle.to_tensor(0.0, dtype=x.dtype), x)
            assert lengths is not None
            x = self.padded_norm(x, lengths)
            x = paddle.where(mask.squeeze(1), paddle.to_tensor(0.0, dtype=x.dtype), x)
        else:
            x = self.norm(x)
        x = paddle.nn.functional.tanh(x=x)
        x = x.reshape([-1, 1, x.shape[2]])
        x = paddle.nn.functional.conv1d(x=x, weight=self.hann, padding=(
            self.hp.ker.hann_window - 1) // 2)
        x = x.reshape([-1, self.hp.chn.residual_out, x.shape[2]])
        return x

    def inference(self, mel):
        return self.forward(mel, None, None)
