import paddle
import os
import gdown
import librosa
import argparse
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from matplotlib.colors import Normalize
import soundfile as sf
os.sys.path.append('./cotatron')
from synthesizer import Synthesizer
from datasets.text import Language
from melgan.generator import Generator
hp_path = ['checkpoints/libritts_vctk_global.yaml', 'checkpoints/libritts_vctk_vc.yaml']
hp_global = OmegaConf.load(hp_path[0])
hp_vc = OmegaConf.load(hp_path[1])
hp = OmegaConf.merge(hp_global, hp_vc)
parser = argparse.ArgumentParser()
parser.add_argument('--config')
hparams = parser.parse_args(['--config', hp_path])
paddle.set_device('gpu')
checkpoint = paddle.load(path=
    '/cotatron_paddle/checkpoints/pretrained_decoder_libritts_vctk_epoch652_15388cc.pdparams')
model = Synthesizer(hparams)
model.set_state_dict(state_dict=checkpoint['state_dict'])
model.eval()
lang = Language(hp.data.lang, hp.data.text_cleaners)
text = 'sometimes you get them sometimes you dont'
source_wavpath = '/cotatron/VCTK-Corpus/p293/p293_148-22k.wav'
text_norm = paddle.to_tensor(data=lang.text_to_sequence(text, hp.data.
    text_cleaners), dtype='int64')
text_norm = text_norm.unsqueeze(axis=0)
wav_source_original, sr = librosa.load(source_wavpath, sr=None, mono=True)
wav_source_original *= 0.99 / np.max(np.abs(wav_source_original))
assert sr == hp.audio.sampling_rate

wav_source = paddle.to_tensor(data=wav_source_original).reshape([1, 1, -1])
mel_source = model.cotatron.audio2mel(wav_source)
target_speaker = paddle.to_tensor(data=[hp.data.speakers.index('p234')],
    dtype='int64')
with paddle.no_grad():
    mel_s_t, alignment, residual = model.inference(text_norm, mel_source,
        target_speaker)
ms = mel_source[0].cpu().detach().numpy()
mst = mel_s_t[0].cpu().detach().numpy()
a = alignment[0].cpu().detach().numpy()
r = residual[0][0].cpu().detach().numpy()


def show_spectrogram(mel):
    plt.figure(figsize=(15, 4))
    plt.imshow(mel, aspect='auto', origin='lower', interpolation='none')
    plt.xlabel('time frames')
    plt.ylabel('mel')
    plt.subplots_adjust(bottom=0.1, right=0.88, top=0.9)
    cax = plt.axes([0.9, 0.1, 0.02, 0.8])
    plt.colorbar(cax=cax)
    plt.show()


plt.figure(figsize=(15, 8))
plt.imshow(a.T, aspect='auto', origin='lower', interpolation='none', norm=
    Normalize(vmin=0.0, vmax=1.0))
plt.xlabel('Decoder timestep')
plt.ylabel('Encoder timestep')
plt.subplots_adjust(bottom=0.1, right=0.88, top=0.9)
cax = plt.axes([0.9, 0.1, 0.02, 0.8])
plt.colorbar(cax=cax)
plt.show()
show_spectrogram(ms)
show_spectrogram(mst)
melgan = Generator(80)
melgan_ckpt = paddle.load(path='/cotatron_paddle/checkpoints/melgan_libritts_g_only.pdparams')
melgan.set_state_dict(state_dict=melgan_ckpt['model_g'])
melgan.eval()
with paddle.no_grad():
    audio_s_t = melgan(mel_s_t).squeeze().cpu().detach().numpy()
sf.write('output.wav', wav_source_original, samplerate=22050)
