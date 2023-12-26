import paddle


class TextEncoder(paddle.nn.Layer):

    def __init__(self, channels, kernel_size, depth, n_symbols):
        super().__init__()
        self.embedding = paddle.nn.Embedding(num_embeddings=n_symbols,
            embedding_dim=channels)
        padding = (kernel_size - 1) // 2
        self.cnn = list()
        for _ in range(depth):
            self.cnn.append(paddle.nn.Sequential(paddle.nn.Conv1D(
                in_channels=channels, out_channels=channels, kernel_size=
                kernel_size, padding=padding), paddle.nn.BatchNorm1D(
                num_features=channels), paddle.nn.ReLU(), paddle.nn.Dropout
                (p=0.5)))
        self.cnn = paddle.nn.Sequential(*self.cnn)
        self.lstm = paddle.nn.LSTM(input_size=channels, hidden_size=
            channels // 2, num_layers=1, time_major=not True, direction='bidirectional')

    def forward(self, x, input_lengths):
        x = self.embedding(x)
        x = x.transpose([0 ,2, 1])
        x = self.cnn(x)
        x = x.transpose([0 ,2, 1])
        x, (_, _) = self.lstm(x, sequence_length=input_lengths)
        return x

    def inference(self, x):
        x = self.embedding(x)
        x = x.transpose([0, 2, 1])
        x = self.cnn(x)
        x = x.transpose([0, 2, 1])
        # self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x


class SpeakerEncoder(paddle.nn.Layer):

    def __init__(self, hp):
        super().__init__()
        self.relu = paddle.nn.ReLU()
        self.stem = paddle.nn.Conv2D(in_channels=1, out_channels=hp.chn.
            speaker.cnn[0], kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))
        self.cnn = paddle.nn.LayerList(sublayers=[paddle.nn.Conv2D(
            in_channels=in_channels, out_channels=out_channels, kernel_size
            =(3, 3), padding=(1, 1), stride=(2, 2)) for in_channels,
            out_channels in zip(list(hp.chn.speaker.cnn)[:-1], hp.chn.
            speaker.cnn[1:])])
        self.bn = paddle.nn.LayerList(sublayers=[paddle.nn.BatchNorm2D(
            num_features=channels) for channels in hp.chn.speaker.cnn])
        self.gru = paddle.nn.GRU(input_size=hp.chn.speaker.cnn[-1] * 2,
            hidden_size=hp.chn.speaker.token, time_major=not True,
            direction='forward')

    def forward(self, x, input_lengths):
        x = x.unsqueeze(axis=1)
        x = self.stem(x)
        input_lengths = (input_lengths + 1) // 2
        for cnn, bn in zip(self.cnn, self.bn):
            x = bn(x)
            x = self.relu(x)
            x = cnn(x)
            input_lengths = (input_lengths + 1) // 2
        x = x.reshape([x.shape[0], -1, x.shape[-1]])

        x = x.transpose([0, 2, 1])

        _, x = self.gru(x, sequence_length=input_lengths)

        x = x.squeeze(0)
        return x
    def inference(self, x):
        x = x.unsqueeze(1)
        x = self.stem(x)

        for cnn, bn in zip(self.cnn, self.bn):
            x = bn(x)
            x = self.relu(x)
            x = cnn(x)

        x = x.reshape([x.shape[0], -1, x.shape[-1]])
        x = x.transpose([0, 2, 1])

        # 在推理时，不需要处理可变长度序列
        _, x = self.gru(x)
        x = x.squeeze(0)
        x = x.squeeze(1)
        return x
