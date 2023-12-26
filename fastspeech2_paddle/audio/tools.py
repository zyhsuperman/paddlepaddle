import paddle
import numpy as np
from scipy.io.wavfile import write
from audio.audio_processing import griffin_lim


def get_mel_from_wav(audio, _stft):
    audio = paddle.clip(x=paddle.to_tensor(data=audio, dtype='float32').
        unsqueeze(axis=0), min=-1, max=1)
    audio = paddle.to_tensor(audio, stop_gradient=True)
    melspec, energy = _stft.mel_spectrogram(audio)
    melspec = paddle.squeeze(x=melspec, axis=0).numpy().astype(np.float32)
    energy = paddle.squeeze(x=energy, axis=0).numpy().astype(np.float32)
    return melspec, energy


def inv_mel_spec(mel, out_filename, _stft, griffin_iters=60):
    mel = paddle.stack(x=[mel])
    mel_decompress = _stft.spectral_de_normalize(mel)
    mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
    spec_from_mel_scaling = 1000
    spec_from_mel = paddle.mm(input=mel_decompress[0], mat2=_stft.mel_basis)
    x = spec_from_mel
    perm_5 = list(range(x.ndim))
    perm_5[0] = 1
    perm_5[1] = 0
    spec_from_mel = x.transpose(perm=perm_5).unsqueeze(axis=0)
    spec_from_mel = spec_from_mel * spec_from_mel_scaling

    audio = griffin_lim(
        spec_from_mel[:, :, :-1], 
        _stft._stft_fn,  # 替换为相应的 PaddlePaddle STFT 函数
        griffin_iters
    )
    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio_path = out_filename
    write(audio_path, _stft.sampling_rate, audio)
