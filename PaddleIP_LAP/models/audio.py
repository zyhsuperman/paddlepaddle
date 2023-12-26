import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile
import lws


class HParams:

    def __init__(self, **kwargs):
        self.data = {}
        for key, value in kwargs.items():
            self.data[key] = value

    def __getattr__(self, key):
        if key not in self.data:
            raise AttributeError("'HParams' object has no attribute %s" % key)
        return self.data[key]

    def set_hparam(self, key, value):
        self.data[key] = value


hp = HParams(num_mels=80, rescale=True, rescaling_max=0.9, use_lws=False,
    n_fft=800, hop_size=200, win_size=800, sample_rate=16000,
    frame_shift_ms=None, signal_normalization=True,
    allow_clipping_in_normalization=True, symmetric_mels=True,
    max_abs_value=4.0, preemphasize=True, preemphasis=0.97, min_level_db=-
    100, ref_level_db=20, fmin=55, fmax=7600, img_size=288, fps=25,
    batch_size=8, initial_learning_rate=0.0001, nepochs=200000000000000000,
    num_workers=4, checkpoint_interval=6000, eval_interval=6000,
    save_optimizer_state=True, syncnet_wt=0.0, syncnet_batch_size=128,
    syncnet_lr=0.0001, syncnet_eval_interval=4500,
    syncnet_checkpoint_interval=4500, disc_wt=0.07,
    disc_initial_learning_rate=0.0001)


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, sr, wav.astype(np.int16))


def save_wavenet_wav(wav, path, sr):
    librosa.output.write_wav(path, wav, sr=sr)


def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav


def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav


def get_hop_size():
    hop_size = hp.hop_size
    if hop_size is None:
        assert hp.frame_shift_ms is not None
        hop_size = int(hp.frame_shift_ms / 1000 * hp.sample_rate)
    return hop_size


def linearspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(np.abs(D)) - hp.ref_level_db
    if hp.signal_normalization:
        return _normalize(S)
    return S


def melspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hp.ref_level_db
    if hp.signal_normalization:
        return _normalize(S)
    return S


def _lws_processor():
    return lws.lws(hp.n_fft, get_hop_size(), fftsize=hp.win_size, mode='speech'
        )


def _stft(y):
    if hp.use_lws:
        """Class Method: *.stft, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
        return _lws_processor(hp).stft(y).T
    else:
        return librosa.stft(y=y, n_fft=hp.n_fft, hop_length=get_hop_size(),
            win_length=hp.win_size)


def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
    """
    pad = fsize - fshift
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = fsize - fshift
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r


def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]


_mel_basis = None


def _linear_to_mel(spectogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)


def _build_mel_basis():
    assert hp.fmax <= hp.sample_rate // 2
    return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,
        fmin=hp.fmin, fmax=hp.fmax)


def _amp_to_db(x):
    min_level = np.exp(hp.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _normalize(S):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return np.clip(2 * hp.max_abs_value * ((S - hp.min_level_db) / 
                -hp.min_level_db) - hp.max_abs_value, -hp.max_abs_value, hp
                .max_abs_value)
        else:
            return np.clip(hp.max_abs_value * ((S - hp.min_level_db) / -hp.
                min_level_db), 0, hp.max_abs_value)
    assert S.max() <= 0 and S.min() - hp.min_level_db >= 0
    if hp.symmetric_mels:
        return 2 * hp.max_abs_value * ((S - hp.min_level_db) / -hp.min_level_db
            ) - hp.max_abs_value
    else:
        return hp.max_abs_value * ((S - hp.min_level_db) / -hp.min_level_db)


def _denormalize(D):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return (np.clip(D, -hp.max_abs_value, hp.max_abs_value) + hp.
                max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value
                ) + hp.min_level_db
        else:
            return np.clip(D, 0, hp.max_abs_value
                ) * -hp.min_level_db / hp.max_abs_value + hp.min_level_db
    if hp.symmetric_mels:
        return (D + hp.max_abs_value) * -hp.min_level_db / (2 * hp.
            max_abs_value) + hp.min_level_db
    else:
        return D * -hp.min_level_db / hp.max_abs_value + hp.min_level_db
