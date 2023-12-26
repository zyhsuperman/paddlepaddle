import paddle
from paddle import nn


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
            channels // 2, num_layers=1, time_major=not True, 
            direction='bidirectional')

    def forward(self, x, input_lengths):
        x = self.embedding(x)
        x = x.transpose([0 ,2, 1])
        x = self.cnn(x)
        x = x.transpose([0 ,2, 1])
# >>>>>>        x = torch.nn.utils.rnn.pack_padded_sequence(x, input_lengths,
#             batch_first=True)
        # self.lstm.flatten_parameters()
        x, (_, _) = self.lstm(x, sequence_length=input_lengths)
# >>>>>>        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return x

    def inference(self, x):
        x = self.embedding(x)
        x = x.transpose([0, 2, 1])
        x = self.cnn(x)
        x = x.transpose([0, 2, 1])
        # self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x


# class SpeakerEncoder(paddle.nn.Layer):

#     def __init__(self, hp):
#         super().__init__()
#         self.relu = paddle.nn.ReLU()
#         self.stem = paddle.nn.Conv2D(in_channels=1, out_channels=hp.chn.
#             speaker.cnn[0], kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))
#         self.cnn = paddle.nn.LayerList(sublayers=[paddle.nn.Conv2D(
#             in_channels=in_channels, out_channels=out_channels, kernel_size
#             =(3, 3), padding=(1, 1), stride=(2, 2)) for in_channels,
#             out_channels in zip(list(hp.chn.speaker.cnn)[:-1], hp.chn.
#             speaker.cnn[1:])])
#         self.bn = paddle.nn.LayerList(sublayers=[paddle.nn.BatchNorm2D(
#             num_features=channels) for channels in hp.chn.speaker.cnn])
#         self.gru = paddle.nn.GRU(input_size=hp.chn.speaker.cnn[-1] * 2,
#             hidden_size=hp.chn.speaker.token, time_major=not True,
#             direction='forward')

#     def forward(self, x, input_lengths):
#         x = x.unsqueeze(axis=1)
#         x = self.stem(x)
#         input_lengths = (input_lengths + 1) // 2
#         for cnn, bn in zip(self.cnn, self.bn):
#             x = bn(x)
#             x = self.relu(x)
#             x = cnn(x)
#             input_lengths = (input_lengths + 1) // 2
        
#         x = x.reshape([x.shape[0], -1, x.shape[-1]])
#         x = x
#         perm_13 = list(range(x.ndim))
#         perm_13[1] = 2
#         perm_13[2] = 1
#         x = x.transpose(perm=perm_13)
#         input_lengths, indices = paddle.sort(descending=True, x=input_lengths
#             ), paddle.argsort(descending=True, x=input_lengths)
#         x = paddle.index_select(x=x, axis=0, index=indices)
#         input_lengths = input_lengths.cpu().numpy()
# >>>>>>        x = torch.nn.utils.rnn.pack_padded_sequence(x, input_lengths,
#             batch_first=True)
#         self.gru.flatten_parameters()
#         _, x = self.gru(x)
#         x = paddle.index_select(x=x[0], axis=0, index=(paddle.sort(x=
#             indices), paddle.argsort(x=indices))[1])
#         return x

#     def inference(self, x):
#         x = x.unsqueeze(axis=1)
#         x = self.stem(x)
#         for cnn, bn in zip(self.cnn, self.bn):
#             x = bn(x)
#             x = self.relu(x)
#             x = cnn(x)
#         """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
# >>>>>>        x = x.view(x.shape[0], -1, x.shape[-1])
#         x = x
#         perm_14 = list(range(x.ndim))
#         perm_14[1] = 2
#         perm_14[2] = 1
#         x = x.transpose(perm=perm_14)
#         self.gru.flatten_parameters()
#         _, x = self.gru(x)
#         x = x.squeeze(axis=1)
#         return x


class SpeakerEncoder(nn.Layer):
    def __init__(self, hp):
        super().__init__()
        self.relu = nn.ReLU()
        self.stem = nn.Conv2D(
            1, hp.chn.speaker.cnn[0], kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))
        self.cnn = nn.LayerList([
            nn.Conv2D(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))
            for in_channels, out_channels in zip(list(hp.chn.speaker.cnn)[:-1], hp.chn.speaker.cnn[1:])
        ]) # 80 - 40 - 20 - 10 - 5 - 3 - 2
        self.bn = nn.LayerList([
            nn.BatchNorm2D(channels) for channels in hp.chn.speaker.cnn
        ])
        self.gru = nn.GRU(hp.chn.speaker.cnn[-1]*2, hp.chn.speaker.token,
                          direction='forward')

    def forward(self, x, input_lengths):
        x = x.unsqueeze(1)
        x = self.stem(x)
        input_lengths = (input_lengths + 1) // 2

        for cnn, bn in zip(self.cnn, self.bn):
            x = bn(x)
            x = self.relu(x)
            x = cnn(x)
            input_lengths = (input_lengths + 1) // 2

        x = x.reshape([x.shape[0], -1, x.shape[-1]])  # [B, chn.speaker.cnn[-1]*2, T]
        x = x.transpose([0, 2, 1])  # [B, T, chn.speaker.cnn[-1]*2]

        _, x = self.gru(x, sequence_length=input_lengths)

        x = x.squeeze(0)

        return x

    # def forward(self, x, input_lengths):
    #     # x: [B, mel, T]
    #     x = x.unsqueeze(1)  # [B, 1, mel, T]
    #     x = self.stem(x)
    #     input_lengths = (input_lengths + 1) // 2

    #     for cnn, bn in zip(self.cnn, self.bn):
    #         x = bn(x)
    #         x = self.relu(x)
    #         x = cnn(x)
    #         input_lengths = (input_lengths + 1) // 2

    #     x = x.reshape([x.shape[0], -1, x.shape[-1]])  # [B, chn.speaker.cnn[-1]*2, T]
    #     x = x.transpose([0, 2, 1])  # [B, T, chn.speaker.cnn[-1]*2]

    #     # 排序 input_lengths 并获取排序后的索引
    #     sorted_indices = paddle.argsort(input_lengths, axis=0, descending=True)
    #     sorted_input_lengths = paddle.gather(input_lengths, sorted_indices, axis=0)
    #     x = paddle.gather(x, sorted_indices, axis=0)

    #     # 使用 PaddlePaddle 的 pack_padded_sequence
    #     x = pack_padded_sequence(x, sorted_input_lengths, batch_first=True)

    #     # GRU层
    #     _, x = self.gru(x)

    #     # 重新排序 GRU 的输出
    #     inverse_indices = paddle.argsort(sorted_indices, axis=0)
    #     x = paddle.gather(x[0], inverse_indices, axis=0)

    #     return x

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
