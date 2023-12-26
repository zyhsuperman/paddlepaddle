"""
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts
"""
from __future__ import print_function, division
from .hyperparams import Hyperparams as hp
import numpy as np
import librosa
import copy
from scipy import signal
import os


def _mel_to_linear_matrix(sr, n_fft, n_mels):
    m = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    m_t = np.transpose(m)
    p = np.matmul(m, m_t)
    d = [(1.0 / x if np.abs(x) > 1e-08 else x) for x in np.sum(p, axis=0)]
    return np.matmul(m_t, np.diag(d))


def get_spectrograms(fpath):
    """Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
    Args:
      sound_file: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) <- Transposed
      mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
    """
    y, sr = librosa.load(fpath, sr=hp.sr)
    y, _ = librosa.effects.trim(y, top_db=hp.top_db)
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])
    linear = librosa.stft(y=y, n_fft=hp.n_fft, hop_length=hp.hop_length,
        win_length=hp.win_length)
    mag = np.abs(linear)
    mel_basis = librosa.filters.mel(sr=hp.sr, n_fft=hp.n_fft, n_mels=hp.n_mels)
    mel = np.dot(mel_basis, mag)
    mel = 20 * np.log10(np.maximum(1e-05, mel))
    mag = 20 * np.log10(np.maximum(1e-05, mag))
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-08, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-08, 1)
    mel = mel.T.astype(np.float32)
    mag = mag.T.astype(np.float32)
    return mel, mag


def melspectrogram2wav(mel):
    """# Generate wave file from spectrogram"""
    mel = mel.T
    mel = np.clip(mel, 0, 1) * hp.max_db - hp.max_db + hp.ref_db
    mel = np.power(10.0, mel * 0.05)
    m = _mel_to_linear_matrix(hp.sr, hp.n_fft, hp.n_mels)
    mag = np.dot(m, mel)
    wav = griffin_lim(mag)
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)
    wav, _ = librosa.effects.trim(wav)
    return wav.astype(np.float32)


def spectrogram2wav(mag):
    """# Generate wave file from spectrogram"""
    mag = mag.T
    mag = np.clip(mag, 0, 1) * hp.max_db - hp.max_db + hp.ref_db
    mag = np.power(10.0, mag * 0.05)
    wav = griffin_lim(mag)
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)
    wav, _ = librosa.effects.trim(wav)
    return wav.astype(np.float32)


def griffin_lim(spectrogram):
    """Applies Griffin-Lim's raw.
    """
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(y=X_t, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.
            win_length)
        phase = est / np.maximum(1e-08, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)
    return y


def invert_spectrogram(spectrogram):
    """
    spectrogram: [f, t]
    """
    return librosa.istft(stft_matrix=spectrogram, hop_length=hp.hop_length, win_length=hp.
        win_length, window='hann')


def load_spectrograms(fpath):
    fname = os.path.basename(fpath)
    mel, mag = get_spectrograms(fpath)
    t = mel.shape[0]
    num_paddings = hp.r - t % hp.r if t % hp.r != 0 else 0
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode='constant')
    mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode='constant')
    return fname, mel.reshape((-1, hp.n_mels * hp.r)), mag
