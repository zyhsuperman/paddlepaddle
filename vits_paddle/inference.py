import matplotlib.pyplot as plt
import IPython.display as ipd
import soundfile as sf

import os
import json
import math
import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.io import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write

paddle.set_device('gpu')

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = paddle.to_tensor(text_norm)
    return text_norm

hps = utils.get_hparams_from_file("./configs/ljs_base.json")

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model)
_ = net_g.eval()

_ = utils.load_checkpoint("/home1/zhaoyh/paddlemodel/vits_paddle/checkpoints/pretrained_ljs.pdparams", net_g, None)

stn_tst = get_text("What the Fuck!", hps)
with paddle.no_grad():
    x_tst = stn_tst.unsqueeze(0)
    x_tst_lengths =paddle.to_tensor([stn_tst.shape[0]], dtype='int64')
    audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0, 0].cpu().numpy()


# ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))

# 假设 audio 是您的音频数据数组，hps.data.sampling_rate 是采样率
sf.write('output_audio.wav', audio, hps.data.sampling_rate)
