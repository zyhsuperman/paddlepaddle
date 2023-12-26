import paddle
import os
import random
import numpy as np
from omegaconf import OmegaConf
from cotatron import Cotatron
from modules import VCDecoder, ResidualEncoder
from datasets import TextMelDataset, text_mel_collate


class Synthesizer(paddle.nn.Layer):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        hp_global = OmegaConf.load(hparams.config[0])
        hp_vc = OmegaConf.load(hparams.config[1])
        hp = OmegaConf.merge(hp_global, hp_vc)
        self.hp = hp
        self.num_speakers = len(self.hp.data.speakers)
        self.cotatron = Cotatron(hparams)
        self.residual_encoder = ResidualEncoder(hp)
        self.decoder = VCDecoder(hp)
        self.speaker = paddle.nn.Embedding(num_embeddings=self.num_speakers,
            embedding_dim=hp.chn.speaker.token)

    def load_cotatron(self, checkpoint_path):
        checkpoint = paddle.load(path=checkpoint_path)
        self.cotatron.set_state_dict(state_dict=checkpoint['state_dict'])
        self.cotatron.eval()
        self.cotatron.freeze()

    def train(self, mode=True):
        super().train()  # 调用基类的 train 方法以设置 self.training

        if mode:
            # 设置为训练模式
            for module in self.children():
                if module is not self.cotatron:
                    module.train()
        else:
            # 设置为评估模式
            for module in self.children():
                module.eval()

        # 即使在训练模式下，cotatron 保持为评估模式
        self.cotatron.eval()
        if hasattr(self.cotatron, 'freeze'):
            self.cotatron.freeze()

        return self

    def forward(self, text, mel_source, input_lengths, output_lengths,
        max_input_len):
        z_s_aligner = self.cotatron.speaker(mel_source, output_lengths)
        text_encoding = self.cotatron.encoder(text, input_lengths)
        z_s_repeated = z_s_aligner.unsqueeze(axis=1).expand(shape=[-1,
            text_encoding.shape[1], -1])
        decoder_input = paddle.concat(x=(text_encoding, z_s_repeated), axis=2)
        _, _, alignment = self.cotatron.decoder(mel_source, decoder_input,
            input_lengths, output_lengths, max_input_len, prenet_dropout=
            0.0, tfrate=False)
        linguistic = paddle.bmm(x=alignment, y=text_encoding)
        x = linguistic
        perm_0 = list(range(x.ndim))
        perm_0[1] = 2
        perm_0[2] = 1
        linguistic = x.transpose(perm=perm_0)
        return linguistic, alignment

    def inference(self, text, mel_source, target_speaker):
        device = text.place
        in_len = paddle.to_tensor(data=[text.shape[1]], dtype='int64')
        out_len = paddle.to_tensor(data=[mel_source.shape[2]], dtype='int64')
        z_s = self.cotatron.speaker.inference(mel_source)
        text_encoding = self.cotatron.encoder.inference(text)
        z_s_repeated = z_s.unsqueeze(axis=1).expand(shape=[-1,
            text_encoding.shape[1], -1])
        decoder_input = paddle.concat(x=(text_encoding, z_s_repeated), axis=2)
        _, _, alignment = self.cotatron.decoder(mel_source, decoder_input,
            in_len, out_len, in_len, prenet_dropout=0.0, no_mask=True,
            tfrate=False)
        ling_s = paddle.bmm(x=alignment, y=text_encoding)
        x = ling_s
        perm_1 = list(range(x.ndim))
        perm_1[1] = 2
        perm_1[2] = 1
        ling_s = x.transpose(perm=perm_1)
        residual = self.residual_encoder.inference(mel_source)
        ling_s = paddle.concat(x=(ling_s, residual), axis=1)
        z_t = self.speaker(target_speaker)
        mel_s_t = self.decoder(ling_s, z_t)
        return mel_s_t, alignment, residual

    def get_cnn_mask(self, lengths):
        max_len = paddle.max(x=lengths).item()
        ids = paddle.assign(paddle.arange(start=0, end=max_len), output=
            paddle.to_tensor(data=max_len, dtype='int64'))
        mask = ids >= lengths.unsqueeze(axis=1)
        mask = mask.unsqueeze(axis=1)
        return mask
