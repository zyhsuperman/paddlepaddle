import paddle
import librosa
from librosa.filters import mel as librosa_mel_fn
import math

def paddle_hann_window(win_size):
    return 0.5 - 0.5 * paddle.cos(2 * math.pi * paddle.arange(win_size) / (win_size - 1))


class Audio2Mel(paddle.nn.Layer):

    def __init__(self, n_fft, hop_length, win_length, sampling_rate,
        n_mel_channels, mel_fmin, mel_fmax):
        super().__init__()
        window = paddle_hann_window(win_length).astype(dtype='float32')
        mel_basis = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=
            n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)
        mel_basis = paddle.to_tensor(data=mel_basis).astype(dtype='float32')
        self.register_buffer(name='mel_basis', tensor=mel_basis)
        self.register_buffer(name='window', tensor=window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio):
        p = (self.n_fft - self.hop_length) // 2
        audio = paddle.nn.functional.pad(audio, (p, p), 'reflect', data_format='NCL').squeeze(axis=1)
        fft = paddle.signal.stft(audio, n_fft=self.n_fft, hop_length=self.
            hop_length, win_length=self.win_length, window=self.window,
            center=False)
        magnitude = paddle.abs(fft)
        mel_output = paddle.matmul(x=self.mel_basis, y=magnitude)
        log_mel_spec = paddle.log10(x=paddle.clip(x=mel_output, min=1e-05))
        return log_mel_spec


if __name__ == '__main__':
    filename = librosa.util.example_audio_file()
    y, sr = librosa.load(filename, sr=22050)
    y = y[:163840]
    
    y = paddle.to_tensor(data=y).reshape([1, 1, -1])
    print(y.shape)
    audio2mel = Audio2Mel(1024, 256, 1024, 22050, 80, 70.0, 7600.0)
    mel = audio2mel(y)
    print(mel.shape)
    print(paddle.min(x=mel))
    print(paddle.max(x=mel))
