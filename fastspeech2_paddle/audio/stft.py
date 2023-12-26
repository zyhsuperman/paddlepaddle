import paddle
import numpy as np
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from librosa.filters import mel as librosa_mel_fn
from audio.audio_processing import dynamic_range_compression, dynamic_range_decompression, window_sumsquare


class STFT(paddle.nn.Layer):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    def __init__(self, filter_length, hop_length, win_length, window='hann'):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        cutoff = int(self.filter_length / 2 + 1)
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]), np.
            imag(fourier_basis[:cutoff, :])])
        forward_basis = paddle.to_tensor(data=fourier_basis[:, (None), :],
            dtype='float32')
        inverse_basis = paddle.to_tensor(data=np.linalg.pinv(scale *
            fourier_basis).T[:, (None), :], dtype='float32')
        if window is not None:
            assert filter_length >= win_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(data = fft_window, size=filter_length)
            fft_window = paddle.to_tensor(data=fft_window).astype(dtype=
                'float32')
            forward_basis *= fft_window
            inverse_basis *= fft_window
        self.register_buffer(name='forward_basis', tensor=forward_basis.
            astype(dtype='float32'))
        self.register_buffer(name='inverse_basis', tensor=inverse_basis.
            astype(dtype='float32'))

    def transform(self, input_data):
        num_batches = input_data.shape[0]
        num_samples = input_data.shape[1]
        self.num_samples = num_samples
        input_data = input_data.reshape([num_batches, 1, num_samples])
        input_data = paddle.nn.functional.pad(
            input_data.unsqueeze(1),
            pad=[int(self.filter_length / 2), int(self.filter_length / 2), 0, 0],
            mode='reflect',
            data_format="NCHW"
        )
        input_data = input_data.squeeze(axis=1)
        forward_transform = paddle.nn.functional.conv1d(
            input_data,
            paddle.to_tensor(self.forward_basis, stop_gradient=True),
            stride=self.hop_length,
            padding=0
        )
        cutoff = int(self.filter_length / 2 + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]
        magnitude = paddle.sqrt(x=real_part ** 2 + imag_part ** 2)
        phase = paddle.atan2(imag_part, real_part)
        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = paddle.concat(x=[magnitude * paddle.cos
            (x=phase), magnitude * paddle.sin(x=phase)], axis=1)
        inverse_transform = paddle.nn.functional.conv1d_transpose(
            recombine_magnitude_phase,
            paddle.to_tensor(self.inverse_basis, stop_gradient=True),
            stride=self.hop_length,
            padding=0
        )
        if self.window is not None:
            window_sum = window_sumsquare(self.window, magnitude.shape[-1],
                hop_length=self.hop_length, win_length=self.win_length,
                n_fft=self.filter_length, dtype=np.float32)
            approx_nonzero_indices = paddle.to_tensor(data=np.where(
                window_sum > tiny(window_sum))[0])
            window_sum = paddle.to_tensor(window_sum, stop_gradient=True)
            window_sum = window_sum if 'gpu' in str(magnitude.place
                ) else window_sum
            inverse_transform[:, :, (approx_nonzero_indices)] /= window_sum[
                approx_nonzero_indices]
            inverse_transform *= float(self.filter_length) / self.hop_length
        inverse_transform = inverse_transform[:, :, int(self.filter_length /
            2):]
        inverse_transform = inverse_transform[:, :, :-int(self.
            filter_length / 2)]
        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class TacotronSTFT(paddle.nn.Layer):

    def __init__(self, filter_length, hop_length, win_length,
        n_mel_channels, sampling_rate, mel_fmin, mel_fmax):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(sr=sampling_rate, n_fft=filter_length,
            n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)
        mel_basis = paddle.to_tensor(data=mel_basis).astype(dtype='float32')
        self.register_buffer(name='mel_basis', tensor=mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert paddle.min(x=y.data) >= -1
        assert paddle.max(x=y.data) <= 1
        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = paddle.matmul(x=self.mel_basis, y=magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        energy = paddle.linalg.norm(x=magnitudes, axis=1)
        return mel_output, energy
