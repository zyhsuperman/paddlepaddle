import sys
import paddle
import os
import json
import copy
import math
from collections import OrderedDict
import numpy as np
from utils.tools import get_mask_from_lengths, pad
device = str('cuda' if paddle.device.cuda.device_count() >= 1 else 'cpu'
    ).replace('cuda', 'gpu')


class VarianceAdaptor(paddle.nn.Layer):
    """Variance Adaptor"""

    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)
        self.pitch_feature_level = preprocess_config['preprocessing']['pitch'][
            'feature']
        self.energy_feature_level = preprocess_config['preprocessing']['energy'
            ]['feature']
        assert self.pitch_feature_level in ['phoneme_level', 'frame_level']
        assert self.energy_feature_level in ['phoneme_level', 'frame_level']
        pitch_quantization = model_config['variance_embedding'][
            'pitch_quantization']
        energy_quantization = model_config['variance_embedding'][
            'energy_quantization']
        n_bins = model_config['variance_embedding']['n_bins']
        assert pitch_quantization in ['linear', 'log']
        assert energy_quantization in ['linear', 'log']
        with open(os.path.join(preprocess_config['path'][
            'preprocessed_path'], 'stats.json')) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats['pitch'][:2]
            energy_min, energy_max = stats['energy'][:2]
        if pitch_quantization == 'log':
            out_0 = paddle.create_parameter(shape=paddle.exp(x=paddle.
                linspace(start=np.log(pitch_min), stop=np.log(pitch_max),
                num=n_bins - 1)).shape, dtype=paddle.exp(x=paddle.linspace(
                start=np.log(pitch_min), stop=np.log(pitch_max), num=n_bins -
                1)).numpy().dtype, default_initializer=paddle.nn.
                initializer.Assign(paddle.exp(x=paddle.linspace(start=np.
                log(pitch_min), stop=np.log(pitch_max), num=n_bins - 1))))
            out_0.stop_gradient = not False
            self.pitch_bins = out_0
        else:
            out_1 = paddle.create_parameter(shape=paddle.linspace(start=
                pitch_min, stop=pitch_max, num=n_bins - 1).shape, dtype=
                paddle.linspace(start=pitch_min, stop=pitch_max, num=n_bins -
                1).numpy().dtype, default_initializer=paddle.nn.initializer
                .Assign(paddle.linspace(start=pitch_min, stop=pitch_max,
                num=n_bins - 1)))
            out_1.stop_gradient = not False
            self.pitch_bins = out_1
        if energy_quantization == 'log':
            out_2 = paddle.create_parameter(shape=paddle.exp(x=paddle.
                linspace(start=np.log(energy_min), stop=np.log(energy_max),
                num=n_bins - 1)).shape, dtype=paddle.exp(x=paddle.linspace(
                start=np.log(energy_min), stop=np.log(energy_max), num=
                n_bins - 1)).numpy().dtype, default_initializer=paddle.nn.
                initializer.Assign(paddle.exp(x=paddle.linspace(start=np.
                log(energy_min), stop=np.log(energy_max), num=n_bins - 1))))
            out_2.stop_gradient = not False
            self.energy_bins = out_2
        else:
            out_3 = paddle.create_parameter(shape=paddle.linspace(start=
                energy_min, stop=energy_max, num=n_bins - 1).shape, dtype=
                paddle.linspace(start=energy_min, stop=energy_max, num=
                n_bins - 1).numpy().dtype, default_initializer=paddle.nn.
                initializer.Assign(paddle.linspace(start=energy_min, stop=
                energy_max, num=n_bins - 1)))
            out_3.stop_gradient = not False
            self.energy_bins = out_3
        self.pitch_embedding = paddle.nn.Embedding(num_embeddings=n_bins,
            embedding_dim=model_config['transformer']['encoder_hidden'])
        self.energy_embedding = paddle.nn.Embedding(num_embeddings=n_bins,
            embedding_dim=model_config['transformer']['encoder_hidden'])

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(paddle.bucketize(x=target,
                sorted_sequence=self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(paddle.bucketize(x=prediction,
                sorted_sequence=self.pitch_bins))
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(paddle.bucketize(x=target,
                sorted_sequence=self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(paddle.bucketize(x=prediction,
                sorted_sequence=self.energy_bins))
        return prediction, embedding

    def forward(self, x, src_mask, mel_mask=None, max_len=None,
        pitch_target=None, energy_target=None, duration_target=None,
        p_control=1.0, e_control=1.0, d_control=1.0):
        log_duration_prediction = self.duration_predictor(x, src_mask)

        if self.pitch_feature_level == 'phoneme_level':
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(x,
                pitch_target, src_mask, p_control)
            x = x + pitch_embedding
        if self.energy_feature_level == 'phoneme_level':
            energy_prediction, energy_embedding = self.get_energy_embedding(x,
                energy_target, src_mask, p_control)
            x = x + energy_embedding
        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = paddle.clip(x=paddle.round(paddle.exp(x=
                log_duration_prediction) - 1) * d_control, min=0)
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)
        if self.pitch_feature_level == 'frame_level':
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(x,
                pitch_target, mel_mask, p_control)
            x = x + pitch_embedding
        if self.energy_feature_level == 'frame_level':
            energy_prediction, energy_embedding = self.get_energy_embedding(x,
                energy_target, mel_mask, p_control)
            x = x + energy_embedding
        return (x, pitch_prediction, energy_prediction,
            log_duration_prediction, duration_rounded, mel_len, mel_mask)


class LengthRegulator(paddle.nn.Layer):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])
        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)
        return output, paddle.to_tensor(data=mel_len, dtype='int64')

    # def expand(self, batch, predicted):
    #     out = list()
    #     for i, vec in enumerate(batch):
    #         expand_size = predicted[i].item()
    #         out.append(vec.expand(shape=[max(int(expand_size), 0), -1]))
    #     out = paddle.concat(x=out, axis=0)
    #     return out
    def expand(self, batch, predicted):
        out = list()
        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            if expand_size > 0:
                out.append(vec.expand(shape=[int(expand_size), -1]))
        out = paddle.concat(x=out, axis=0)
        return out


    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(paddle.nn.Layer):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()
        self.input_size = model_config['transformer']['encoder_hidden']
        self.filter_size = model_config['variance_predictor']['filter_size']
        self.kernel = model_config['variance_predictor']['kernel_size']
        self.conv_output_size = model_config['variance_predictor'][
            'filter_size']
        self.dropout = model_config['variance_predictor']['dropout']
        self.conv_layer = paddle.nn.Sequential(*[('conv1d_1', Conv(self.
            input_size, self.filter_size, kernel_size=self.kernel, padding=
            (self.kernel - 1) // 2)), ('relu_1', paddle.nn.ReLU()), (
            'layer_norm_1', paddle.nn.LayerNorm(normalized_shape=self.
            filter_size)), ('dropout_1', paddle.nn.Dropout(p=self.dropout)),
            ('conv1d_2', Conv(self.filter_size, self.filter_size,
            kernel_size=self.kernel, padding=1)), ('relu_2', paddle.nn.ReLU
            ()), ('layer_norm_2', paddle.nn.LayerNorm(normalized_shape=self
            .filter_size)), ('dropout_2', paddle.nn.Dropout(p=self.dropout))])
        self.linear_layer = paddle.nn.Linear(in_features=self.
            conv_output_size, out_features=1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(axis=-1)
        if mask is not None:
            out = paddle.where(~mask, out, paddle.to_tensor(0.0, dtype=out.dtype))
        return out


class Conv(paddle.nn.Layer):
    """
    Convolution Module
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
        padding=0, dilation=1, bias=True, w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()
        self.conv = paddle.nn.Conv1D(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, stride=stride, padding=
            padding, dilation=dilation, bias_attr=bias)

    def forward(self, x):
        x = x
        perm_0 = list(range(x.ndim))
        perm_0[1] = 2
        perm_0[2] = 1
        x = x.transpose(perm=perm_0)
        x = self.conv(x)
        x = x
        perm_1 = list(range(x.ndim))
        perm_1[1] = 2
        perm_1[2] = 1
        x = x.transpose(perm=perm_1)
        return x
