import sys
import paddle
"""
Synthesis waveform from trained WaveNet.

Modified from https://github.com/r9y9/wavenet_vocoder
"""
from tqdm import tqdm
import librosa
from hparams import hparams
from wavenet_vocoder import WaveNet
use_cuda = paddle.device.cuda.device_count() >= 1
device = str('cuda' if use_cuda else 'cpu').replace('cuda', 'gpu')


def build_model():
    model = WaveNet(out_channels=hparams.out_channels, layers=hparams.
        layers, stacks=hparams.stacks, residual_channels=hparams.
        residual_channels, gate_channels=hparams.gate_channels,
        skip_out_channels=hparams.skip_out_channels, cin_channels=hparams.
        cin_channels, gin_channels=hparams.gin_channels,
        weight_normalization=hparams.weight_normalization, n_speakers=
        hparams.n_speakers, dropout=hparams.dropout, kernel_size=hparams.
        kernel_size, upsample_conditional_features=hparams.
        upsample_conditional_features, upsample_scales=hparams.
        upsample_scales, freq_axis_kernel_size=hparams.
        freq_axis_kernel_size, scalar_input=True, legacy=hparams.legacy)
    _ = model.eval()
    
    return model


def wavegen(model, c=None, tqdm=tqdm):
    """Generate waveform samples by WaveNet.
    
    """
    model.eval()
    model.make_generation_fast_()
    Tc = c.shape[0]
    upsample_factor = hparams.hop_size
    length = Tc * upsample_factor
    c = paddle.to_tensor(data=c.T, dtype='float32').unsqueeze(axis=0)
    initial_input = paddle.zeros(shape=[1, 1, 1]).fill_(value=0.0)
    initial_input = initial_input
    c = None if c is None else c
    with paddle.no_grad():
        y_hat = model.incremental_forward(initial_input, c=c, g=None, T=
            length, tqdm=tqdm, softmax=True, quantize=True, log_scale_min=
            hparams.log_scale_min)
    y_hat = y_hat.reshape([-1]).cpu().data.numpy()
    return y_hat
