import paddle
import os
import random
import numpy as np
from omegaconf import OmegaConf
from modules import TextEncoder, TTSDecoder, SpeakerEncoder, SpkClassifier
from datasets import TextMelDataset, text_mel_collate
from datasets.text import Language
from modules.alignment_loss import GuidedAttentionLoss


class Cotatron(paddle.nn.Layer):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        hp_global = OmegaConf.load(hparams.config[0])
        hp_cota = OmegaConf.load(hparams.config[1])
        hp = OmegaConf.merge(hp_global, hp_cota)
        self.hp = hp
        self.symbols = Language(hp.data.lang, hp.data.text_cleaners
            ).get_symbols()
        self.symbols = ['"{}"'.format(symbol) for symbol in self.symbols]
        self.encoder = TextEncoder(hp.chn.encoder, hp.ker.encoder, hp.depth
            .encoder, len(self.symbols))
        self.speaker = SpeakerEncoder(hp)
        self.classifier = SpkClassifier(hp)
        self.decoder = TTSDecoder(hp)
        self.is_val_first = True
        self.use_attn_loss = hp.train.use_attn_loss
        if self.use_attn_loss:
            self.attn_loss = GuidedAttentionLoss(20000, 0.25, 1.00025)
        else:
            self.attn_loss = None

    def forward(self, text, mel_target, speakers, input_lengths,
        output_lengths, max_input_len, prenet_dropout=0.5, no_mask=False,
        tfrate=True):
        text_encoding = self.encoder(text, input_lengths)
        speaker_emb = self.speaker(mel_target, output_lengths)
        speaker_emb_rep = speaker_emb.unsqueeze(axis=1).expand(shape=[-1,
            text_encoding.shape[1], -1])
        decoder_input = paddle.concat(x=(text_encoding, speaker_emb_rep),
            axis=2)
        mel_pred, mel_postnet, alignment = self.decoder(mel_target,
            decoder_input, input_lengths, output_lengths, max_input_len,
            prenet_dropout, no_mask, tfrate)
        return speaker_emb, mel_pred, mel_postnet, alignment

    def inference(self, text, mel_target):
        device = text.place
        in_len = paddle.to_tensor(data=[text.shape[1]], dtype='int64').to(
            device)
        out_len = paddle.to_tensor(data=[mel_target.shape[2]], dtype='int64'
            ).to(device)
        text_encoding = self.encoder.inference(text)
        speaker_emb = self.speaker.inference(mel_target)
        speaker_emb_rep = speaker_emb.unsqueeze(axis=1).expand(shape=[-1,
            text_encoding.shape[1], -1])
        decoder_input = paddle.concat(x=(text_encoding, speaker_emb_rep),
            axis=2)
        _, mel_postnet, alignment = self.decoder(mel_target, decoder_input,
            in_len, out_len, in_len, prenet_dropout=0.5, no_mask=True,
            tfrate=False)
        return mel_postnet, alignment

    def freeze(self):
        for param in self.parameters():
            param.stop_gradient = not False
