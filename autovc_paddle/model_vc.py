import paddle
import numpy as np
import math

class LinearNorm(paddle.nn.Layer):

    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = paddle.nn.Linear(in_features=in_dim,
            out_features=out_dim, bias_attr=bias)
        


        # 使用 Xavier 均匀分布初始化权重
        initializer = paddle.nn.initializer.XavierUniform()
        init_weight = paddle.create_parameter(shape=self.linear_layer.weight.shape, 
                                dtype='float32', 
                                default_initializer=initializer)
        self.linear_layer.weight.set_value(init_weight)

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(paddle.nn.Layer):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
        padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = paddle.nn.Conv1D(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, stride=stride, padding=
            padding, dilation=dilation, bias_attr=bias)
        
        initializer = paddle.nn.initializer.XavierUniform()
        init_weight = paddle.create_parameter(shape=self.conv.weight.shape, 
                                      dtype='float32', 
                                      default_initializer=initializer)
        self.conv.weight.set_value(init_weight)


    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Encoder(paddle.nn.Layer):
    """Encoder module:
    """

    def __init__(self, dim_neck, dim_emb, freq):
        super(Encoder, self).__init__()
        self.dim_neck = dim_neck
        self.freq = freq
        convolutions = []
        for i in range(3):
            conv_layer = paddle.nn.Sequential(ConvNorm(80 + dim_emb if i ==
                0 else 512, 512, kernel_size=5, stride=1, padding=2,
                dilation=1, w_init_gain='relu'), paddle.nn.BatchNorm1D(
                num_features=512))
            convolutions.append(conv_layer)
        self.convolutions = paddle.nn.LayerList(sublayers=convolutions)
        self.lstm = paddle.nn.LSTM(input_size=512, hidden_size=dim_neck,
            num_layers=2, time_major=not True, direction='bidirect')

    def forward(self, x, c_org):
        x = x.squeeze(axis=1)
        x = paddle.transpose(x, perm=[0, 2, 1])
        c_org = c_org.unsqueeze(axis=-1).expand(shape=[-1, -1, x.shape[-1]])
        x = paddle.concat(x=(x, c_org), axis=1)
        for conv in self.convolutions:
            x = paddle.nn.functional.relu(x=conv(x))

        x = paddle.transpose(x, perm=[0, 2, 1])
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]
        codes = []
        for i in range(0, outputs.shape[1], self.freq):
            codes.append(paddle.concat(x=(out_forward[:, (i + self.freq - 1
                ), :], out_backward[:, (i), :]), axis=-1))
        return codes


class Decoder(paddle.nn.Layer):
    """Decoder module:
    """

    def __init__(self, dim_neck, dim_emb, dim_pre):
        super(Decoder, self).__init__()
        self.lstm1 = paddle.nn.LSTM(input_size=dim_neck * 2 + dim_emb,
            hidden_size=dim_pre, num_layers=1, time_major=not True,
            direction='forward')
        convolutions = []
        for i in range(3):
            conv_layer = paddle.nn.Sequential(ConvNorm(dim_pre, dim_pre,
                kernel_size=5, stride=1, padding=2, dilation=1, w_init_gain
                ='relu'), paddle.nn.BatchNorm1D(num_features=dim_pre))
            convolutions.append(conv_layer)
        self.convolutions = paddle.nn.LayerList(sublayers=convolutions)
        self.lstm2 = paddle.nn.LSTM(input_size=dim_pre, hidden_size=1024,
            num_layers=2, time_major=not True, direction='forward')
        self.linear_projection = LinearNorm(1024, 80)

    def forward(self, x):
        x, _ = self.lstm1(x)

        x = paddle.transpose(x, perm=[0, 2, 1])
        for conv in self.convolutions:
            x = paddle.nn.functional.relu(x=conv(x))

        x = paddle.transpose(x, perm=[0, 2, 1])
        outputs, _ = self.lstm2(x)
        decoder_output = self.linear_projection(outputs)
        return decoder_output


class Postnet(paddle.nn.Layer):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = paddle.nn.LayerList()
        self.convolutions.append(paddle.nn.Sequential(ConvNorm(80, 512,
            kernel_size=5, stride=1, padding=2, dilation=1, w_init_gain=
            'tanh'), paddle.nn.BatchNorm1D(num_features=512)))
        for i in range(1, 5 - 1):
            self.convolutions.append(paddle.nn.Sequential(ConvNorm(512, 512,
                kernel_size=5, stride=1, padding=2, dilation=1, w_init_gain
                ='tanh'), paddle.nn.BatchNorm1D(num_features=512)))
        self.convolutions.append(paddle.nn.Sequential(ConvNorm(512, 80,
            kernel_size=5, stride=1, padding=2, dilation=1, w_init_gain=
            'linear'), paddle.nn.BatchNorm1D(num_features=80)))

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = paddle.nn.functional.tanh(x=self.convolutions[i](x))
        x = self.convolutions[-1](x)
        return x


class Generator(paddle.nn.Layer):
    """Generator network."""

    def __init__(self, dim_neck, dim_emb, dim_pre, freq):
        super(Generator, self).__init__()
        self.encoder = Encoder(dim_neck, dim_emb, freq)
        self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
        self.postnet = Postnet()

    def forward(self, x, c_org, c_trg):
        codes = self.encoder(x, c_org)
        if c_trg is None:
            return paddle.concat(x=codes, axis=-1)
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(axis=1).expand(shape=[-1, int(x.shape
                [1] / len(codes)), -1]))
        code_exp = paddle.concat(x=tmp, axis=1)
        encoder_outputs = paddle.concat(x=(code_exp, c_trg.unsqueeze(axis=1
            ).expand(shape=[-1, x.shape[1], -1])), axis=-1)
        mel_outputs = self.decoder(encoder_outputs)

        mel_outputs_postnet = self.postnet(paddle.transpose(mel_outputs, perm=[0, 2, 1]))

        mel_outputs_postnet = mel_outputs + paddle.transpose(mel_outputs_postnet, perm=[0, 2, 1])
        mel_outputs = mel_outputs.unsqueeze(axis=1)
        mel_outputs_postnet = mel_outputs_postnet.unsqueeze(axis=1)
        return mel_outputs, mel_outputs_postnet, paddle.concat(x=codes, axis=-1)
