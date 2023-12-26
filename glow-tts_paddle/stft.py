import paddle
"""
BSD 3-Clause License

Copyright (c) 2017, Prem Seetharaman
All rights reserved.

* Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import numpy as np
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from librosa import stft, istft
from audio_processing import window_sumsquare

paddle.set_device('gpu')

class STFT(paddle.nn.Layer):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    def __init__(self, filter_length=800, hop_length=200, win_length=800,
        window='hann'):
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
            fft_window = pad_center(data=fft_window, size=filter_length)
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
        if isinstance(input_data.place, paddle.CUDAPlace):
            
            input_data = input_data.reshape([num_batches, 1, num_samples])
            input_data = paddle.nn.functional.pad(input_data.unsqueeze(axis=
                1), (int(self.filter_length / 2), int(self.filter_length / 
                2), 0, 0), mode='reflect')
            input_data = input_data.squeeze(axis=1)
            forward_transform = paddle.nn.functional.conv1d(x=input_data,
                weight=self.forward_basis, stride=self.hop_length, padding=0)
            cutoff = int(self.filter_length / 2 + 1)
            real_part = forward_transform[:, :cutoff, :]
            imag_part = forward_transform[:, cutoff:, :]
        else:
            x = input_data.detach().numpy()
            real_part = []
            imag_part = []
            for y in x:
                y_ = stft(y=y, n_fft=self.filter_length, hop_length=self.
                    hop_length, win_length=self.win_length, window=self.window)
                
                real_part.append(y_.real[(None), :, :])
                imag_part.append(y_.imag[(None), :, :])
            real_part = np.concatenate(real_part, 0)
            imag_part = np.concatenate(imag_part, 0)
            real_part = paddle.to_tensor(data=real_part, dtype=input_data.dtype)
            imag_part = paddle.to_tensor(data=imag_part, dtype=input_data.dtype)
        magnitude = paddle.sqrt(x=real_part ** 2 + imag_part ** 2)
        phase = paddle.atan2(x=imag_part.data, y=real_part.data)
        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = paddle.concat(x=[magnitude * paddle.cos
            (x=phase), magnitude * paddle.sin(x=phase)], axis=1)
        
        if isinstance(magnitude.place, paddle.CUDAPlace):
            inverse_transform = paddle.nn.functional.conv1d_transpose(x=
                recombine_magnitude_phase, weight=self.inverse_basis,
                stride=self.hop_length, padding=0)
            if self.window is not None:
                window_sum = window_sumsquare(self.window, magnitude.shape[
                    -1], hop_length=self.hop_length, win_length=self.
                    win_length, n_fft=self.filter_length, dtype=np.float32)
                approx_nonzero_indices = paddle.to_tensor(data=np.where(
                    window_sum > tiny(window_sum))[0])
                window_sum = paddle.to_tensor(data=window_sum).to(
                    inverse_transform.place)
                inverse_transform[:, :, (approx_nonzero_indices)
                    ] /= window_sum[approx_nonzero_indices]
                inverse_transform *= float(self.filter_length
                    ) / self.hop_length
            inverse_transform = inverse_transform[:, :, int(self.
                filter_length / 2):]
            inverse_transform = inverse_transform[:, :, :-int(self.
                filter_length / 2)]
            inverse_transform = inverse_transform.squeeze(axis=1)
        else:
            x_org = recombine_magnitude_phase.detach().numpy()
            n_b, n_f, n_t = x_org.shape
            x = np.empty([n_b, n_f // 2, n_t], dtype=np.complex64)
            x.real = x_org[:, :n_f // 2]
            x.imag = x_org[:, n_f // 2:]
            inverse_transform = []
            for y in x:
                y_ = istft(y, self.hop_length, self.win_length, self.window)
                inverse_transform.append(y_[(None), :])
            inverse_transform = np.concatenate(inverse_transform, 0)
            inverse_transform = paddle.to_tensor(data=inverse_transform).to(
                recombine_magnitude_phase.dtype)
        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction
