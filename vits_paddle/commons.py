import sys
import paddle
import math
import numpy as np


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def kl_divergence(m_p, logs_p, m_q, logs_q):
    """KL(P||Q)"""
    kl = logs_q - logs_p - 0.5
    kl += 0.5 * (paddle.exp(x=2.0 * logs_p) + (m_p - m_q) ** 2) * paddle.exp(x
        =-2.0 * logs_q)
    return kl


def rand_gumbel(shape):
    """Sample from the Gumbel distribution, protect from overflows."""
    uniform_samples = paddle.rand(shape=shape) * 0.99998 + 1e-05
    return -paddle.log(x=-paddle.log(x=uniform_samples))


def rand_gumbel_like(x):
    g = rand_gumbel(x.shape).to(dtype=x.dtype, device=x.place)
    return g


def slice_segments(x, ids_str, segment_size=4):
    ret = paddle.zeros_like(x=x[:, :, :segment_size])
    for i in range(x.shape[0]):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[(i), :, idx_str:idx_end]
    return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4):
    b, d, t = x.shape
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (paddle.rand(shape=[b]) * ids_str_max)
    ids_str = paddle.cast(ids_str, dtype='int64')
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale
    =10000.0):
    position = paddle.arange(dtype='float32', end=length)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(
        min_timescale)) / (num_timescales - 1)
    inv_timescales = min_timescale * paddle.exp(x=paddle.arange(dtype=
        'float32', end=num_timescales) * -log_timescale_increment)
    scaled_time = position.unsqueeze(axis=0) * inv_timescales.unsqueeze(axis=1)
    signal = paddle.concat(x=[paddle.sin(x=scaled_time), paddle.cos(x=
        scaled_time)], axis=0)
    signal = paddle.nn.functional.pad(signal, pad=[0, 0, 0, channels % 2], mode='constant', value=0)
    signal = signal.reshape([1, channels, length])
    return signal


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=10000.0):
    b, channels, length = x.shape
    signal = get_timing_signal_1d(length, channels, min_timescale,
        max_timescale)
    return x + signal.to(dtype=x.dtype, device=x.place)


def cat_timing_signal_1d(x, min_timescale=1.0, max_timescale=10000.0, axis=1):
    b, channels, length = x.shape
    signal = get_timing_signal_1d(length, channels, min_timescale,
        max_timescale)
    return paddle.concat(x=[x, signal.to(dtype=x.dtype, device=x.place)],
        axis=axis)


def subsequent_mask(length):
    mask = paddle.tril(x=paddle.ones(shape=[length, length])).unsqueeze(axis=0
        ).unsqueeze(axis=0)
    return mask


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = paddle.nn.functional.tanh(x=in_act[:, :n_channels_int, :])
    s_act = paddle.nn.functional.sigmoid(x=in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def shift_1d(x):
    x = paddle.nn.functional.pad(x, pad=convert_pad_shape([[1, 0], [0, 0], [0, 0]])
        )[:, :, :-1]
    return x


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = paddle.arange(dtype=length.dtype, end=max_length)
    return x.unsqueeze(axis=0) < length.unsqueeze(axis=1)


def generate_path(duration, mask):
    """
  duration: [b, 1, t_x]
  mask: [b, 1, t_y, t_x]
  """
    device = duration.place
    b, _, t_y, t_x = mask.shape
    cum_duration = paddle.cumsum(x=duration, axis=-1)
    cum_duration_flat = cum_duration.reshape([b * t_x])
    path = sequence_mask(cum_duration_flat, t_y).astype(mask.dtype)
    path = path.reshape([b, t_x, t_y])
    path = path - paddle.nn.functional.pad(path, pad=convert_pad_shape([[0, 0],
        [1, 0], [0, 0]]))[:, :-1]
    x = path.unsqueeze(axis=1)
    perm_19 = list(range(x.ndim))
    perm_19[2] = 3
    perm_19[3] = 2
    path = x.transpose(perm=perm_19) * mask
    return path


def clip_grad_value_(parameters, clip_value, norm_type=2):
    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm
