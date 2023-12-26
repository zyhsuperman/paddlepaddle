import paddle
import math
import os
import random
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-05):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-05):
    return paddle.log(x=paddle.clip(x=x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return paddle.exp(x=x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

def paddle_hann_window(win_size):
    return 0.5 - 0.5 * paddle.cos(2 * math.pi * paddle.arange(win_size) / (win_size - 1))

mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size,
    fmin, fmax, center=False):
    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels,
            fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + '_' + str(y.place)] = paddle.to_tensor(data=mel
            ).astype(dtype='float32')
        hann_window[str(y.place)] = paddle_hann_window(win_size)
    y = paddle.nn.functional.pad(y.unsqueeze(axis=1), (int((n_fft - hop_size
        ) / 2), int((n_fft - hop_size) / 2)), mode='reflect', data_format='NCL')
    y = y.squeeze(axis=1)
    spec = paddle.signal.stft(y, n_fft, hop_length=hop_size, win_length=win_size,
        window=hann_window[str(y.place)], center=center, pad_mode='reflect',
        normalized=False, onesided=True)
    spec = paddle.abs(spec)
    spec = paddle.matmul(x=mel_basis[str(fmax) + '_' + str(y.place)], y=spec)
    spec = spectral_normalize_torch(spec)
    return spec
