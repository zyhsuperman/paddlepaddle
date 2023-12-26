import paddle
import random
import numpy as np
from .attention import Attention
from .zoneout import ZoneoutLSTMCell


class PreNet(paddle.nn.Layer):

    def __init__(self, channels, in_dim, depth):
        super().__init__()
        sizes = [in_dim] + [channels] * depth
        self.layers = paddle.nn.LayerList(sublayers=[paddle.nn.Linear(
            in_features=in_size, out_features=out_size) for in_size,
            out_size in zip(sizes[:-1], sizes[1:])])

    def forward(self, x, prenet_dropout):
        for linear in self.layers:
            x = paddle.nn.functional.dropout(x=paddle.nn.functional.relu(x=
                linear(x)), p=prenet_dropout, training=True)
        return x


class PostNet(paddle.nn.Layer):

    def __init__(self, channels, kernel_size, n_mel_channels, depth):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.cnn = list()
        self.cnn.append(paddle.nn.Sequential(paddle.nn.Conv1D(in_channels=
            n_mel_channels, out_channels=channels, kernel_size=kernel_size,
            padding=padding), paddle.nn.BatchNorm1D(num_features=channels),
            paddle.nn.Tanh(), paddle.nn.Dropout(p=0.5)))
        for i in range(1, depth - 1):
            self.cnn.append(paddle.nn.Sequential(paddle.nn.Conv1D(
                in_channels=channels, out_channels=channels, kernel_size=
                kernel_size, padding=padding), paddle.nn.BatchNorm1D(
                num_features=channels), paddle.nn.Tanh(), paddle.nn.Dropout
                (p=0.5)))
        self.cnn.append(paddle.nn.Sequential(paddle.nn.Conv1D(in_channels=
            channels, out_channels=n_mel_channels, kernel_size=kernel_size,
            padding=padding)))
        self.cnn = paddle.nn.Sequential(*self.cnn)

    def forward(self, x):
        return self.cnn(x)


class TTSDecoder(paddle.nn.Layer):

    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        out_0 = paddle.create_parameter(shape=paddle.randn(shape=[1, hp.
            audio.n_mel_channels]).shape, dtype=paddle.randn(shape=[1, hp.
            audio.n_mel_channels]).numpy().dtype, default_initializer=
            paddle.nn.initializer.Assign(paddle.randn(shape=[1, hp.audio.
            n_mel_channels])))
        out_0.stop_gradient = not True
        self.go_frame = out_0
        self.prenet = PreNet(hp.chn.prenet, in_dim=hp.audio.n_mel_channels,
            depth=hp.depth.encoder)
        self.postnet = PostNet(hp.chn.postnet, hp.ker.postnet, hp.audio.
            n_mel_channels, hp.depth.postnet)
        self.attention_rnn = ZoneoutLSTMCell(hp.chn.prenet + hp.chn.encoder +
            hp.chn.speaker.token, hp.chn.attention_rnn, zoneout_prob=0.1)
        self.attention_layer = Attention(hp.chn.attention_rnn, hp.chn.
            attention, hp.chn.static, hp.ker.static, hp.chn.dynamic, hp.ker
            .dynamic, hp.ker.causal, hp.ker.alpha, hp.ker.beta)
        self.decoder_rnn = ZoneoutLSTMCell(hp.chn.attention_rnn + hp.chn.
            encoder + hp.chn.speaker.token, hp.chn.decoder_rnn,
            zoneout_prob=0.1)
        self.mel_fc = paddle.nn.Linear(in_features=hp.chn.decoder_rnn + hp.
            chn.encoder + hp.chn.speaker.token, out_features=hp.audio.
            n_mel_channels)

    def get_go_frame(self, memory):
        return self.go_frame.expand(shape=[memory.shape[0], self.hp.audio.
            n_mel_channels])

    def initialize(self, memory, mask):
        B, T, _ = memory.shape
        self.memory = memory
        self.mask = mask
        device = memory.place
        attn_h = paddle.zeros(shape=[B, self.hp.chn.attention_rnn])
        attn_c = paddle.zeros(shape=[B, self.hp.chn.attention_rnn])
        dec_h = paddle.zeros(shape=[B, self.hp.chn.decoder_rnn])
        dec_c = paddle.zeros(shape=[B, self.hp.chn.decoder_rnn])
        prev_attn = paddle.zeros(shape=[B, T])
        prev_attn[:, (0)] = 1.0
        context = paddle.zeros(shape=[B, self.hp.chn.encoder + self.hp.chn.
            speaker.token])
        return attn_h, attn_c, dec_h, dec_c, prev_attn, context

    def decode(self, x, attn_h, attn_c, dec_h, dec_c, prev_attn, context):
        x = paddle.concat(x=(x, context), axis=-1)

        attn_h, attn_c = self.attention_rnn(x, (attn_h, attn_c))
        context, prev_attn = self.attention_layer(attn_h, self.memory,
            prev_attn, self.mask)
        x = paddle.concat(x=(attn_h, context), axis=-1)
        dec_h, dec_c = self.decoder_rnn(x, (dec_h, dec_c))
        x = paddle.concat(x=(dec_h, context), axis=-1)
        mel_out = self.mel_fc(x)
        return mel_out, attn_h, attn_c, dec_h, dec_c, prev_attn, context

    def parse_decoder_outputs(self, mel_outputs, alignments):
        x = paddle.stack(x=mel_outputs, axis=0)
        perm_2 = list(range(x.ndim))
        perm_2[0] = 1
        perm_2[1] = 0
        mel_outputs = x.transpose(perm=perm_2)
        x = mel_outputs
        perm_3 = list(range(x.ndim))
        perm_3[1] = 2
        perm_3[2] = 1
        mel_outputs = x.transpose(perm=perm_3)
        x = paddle.stack(x=alignments, axis=0)
        perm_4 = list(range(x.ndim))
        perm_4[0] = 1
        perm_4[1] = 0
        alignments = x.transpose(perm=perm_4)
        return mel_outputs, alignments

    def forward(self, x, memory, memory_lengths, output_lengths,
        max_input_len, prenet_dropout=0.5, no_mask=False, tfrate=True):
        go_frame = self.get_go_frame(memory).unsqueeze(axis=0)
        x = x
        perm_5 = list(range(x.ndim))
        perm_5[1] = 2
        perm_5[2] = 1
        x = x.transpose(perm=perm_5)
        perm_6 = list(range(x.ndim))
        perm_6[0] = 1
        perm_6[1] = 0
        x = x.transpose(perm=perm_6)
        x = paddle.concat(x=(go_frame, x), axis=0)
        x = self.prenet(x, prenet_dropout)
        attn_h, attn_c, dec_h, dec_c, prev_attn, context = self.initialize(
            memory, mask=None if no_mask else ~self.get_mask_from_lengths(
            memory_lengths))
        mel_outputs, alignments = [], []
        decoder_input = x[0]
        while len(mel_outputs) < x.shape[0] - 1:
            mel_out, attn_h, attn_c, dec_h, dec_c, prev_attn, context = (self
                .decode(decoder_input, attn_h, attn_c, dec_h, dec_c,
                prev_attn, context))
            mel_outputs.append(mel_out)
            alignments.append(prev_attn)
            if tfrate and self.hp.train.teacher_force.rate < random.random():
                decoder_input = self.prenet(mel_out, prenet_dropout)
            else:
                decoder_input = x[len(mel_outputs)]
        mel_outputs, alignments = self.parse_decoder_outputs(mel_outputs,
            alignments)
        mel_postnet = mel_outputs + self.postnet(mel_outputs)
        alignments = alignments.unsqueeze(axis=0)
        
        alignments = paddle.nn.functional.pad(alignments, (0, 0, 0, 0, 0, 0, 0, max_input_len[
            0] - alignments.shape[-1]), 'constant', 0)
        alignments = alignments.squeeze(axis=0)
        mel_outputs, mel_postnet, alignments = self.mask_output(mel_outputs,
            mel_postnet, alignments, output_lengths)
        return mel_outputs, mel_postnet, alignments

    def get_mask_from_lengths(self, lengths, max_len=None):
        if max_len is None:
            max_len = paddle.max(x=lengths).item()
        ids = paddle.assign(paddle.arange(start=0, end=max_len), output=
            paddle.to_tensor(data=max_len, dtype='int64'))
        mask = ids < lengths.unsqueeze(axis=1)
        return mask

    def mask_output(self, mel_outputs, mel_postnet, alignments,
        output_lengths=None):
        if self.hp.train.mask_padding and output_lengths is not None:
            mask = ~self.get_mask_from_lengths(output_lengths, max_len=
                mel_outputs.shape[-1])
            mask = mask.unsqueeze(axis=1)
            mel_outputs = paddle.where(mask, paddle.to_tensor(0.0, dtype=mel_outputs.dtype), mel_outputs)
            mel_postnet = paddle.where(mask, paddle.to_tensor(0.0, dtype=mel_postnet.dtype), mel_postnet)
        return mel_outputs, mel_postnet, alignments
