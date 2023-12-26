import paddle
import numpy as np
import sys
import os
import yaml
import pickle
from model import AE
from utils import *
from functools import reduce
import json
from collections import defaultdict
from argparse import ArgumentParser, Namespace
from scipy.io.wavfile import write
import random
from preprocess.tacotron.utils import melspectrogram2wav
from preprocess.tacotron.utils import get_spectrograms
import librosa


class Inferencer(object):

    def __init__(self, config, args):
        self.config = config
        print(config)
        self.args = args
        print(self.args)
        self.build_model()
        self.load_model()
        with open(self.args.attr, 'rb') as f:
            self.attr = pickle.load(f)

    def load_model(self):
        print(f'Load model from {self.args.model}')
        self.model.set_state_dict(state_dict=paddle.load(path=
            f'{self.args.model}'))
        return

    def build_model(self):
        self.model = cc(AE(self.config))
        print(self.model)
        self.model.eval()
        return

    def utt_make_frames(self, x):
        frame_size = self.config['data_loader']['frame_size']
        remains = x.shape[0] % frame_size

        if remains != 0:
            x = x.unsqueeze(0)
            x = paddle.nn.functional.pad(x, [0, 0, 0, 0, 0, remains], mode='constant', value=0)
            x = x.squeeze(0)
        
        x = x.reshape([1, x.shape[0] // frame_size, frame_size * x.shape[1]])
        perm_0 = list(range(x.ndim))
        perm_0[1] = 2
        perm_0[2] = 1
        out = x.transpose(perm=perm_0)
        return out

    def inference_one_utterance(self, x, x_cond):
        x = self.utt_make_frames(x)
        x_cond = self.utt_make_frames(x_cond)
        dec = self.model.inference(x, x_cond)
        x = dec
        perm_1 = list(range(x.ndim))
        perm_1[1] = 2
        perm_1[2] = 1
        dec = x.transpose(perm=perm_1).squeeze(axis=0)
        dec = dec.detach().cpu().numpy()
        dec = self.denormalize(dec)
        wav_data = melspectrogram2wav(dec)
        return wav_data, dec

    def denormalize(self, x):
        m, s = self.attr['mean'], self.attr['std']
        ret = x * s + m
        return ret

    def normalize(self, x):
        m, s = self.attr['mean'], self.attr['std']
        ret = (x - m) / s
        return ret

    def write_wav_to_file(self, wav_data, output_path):
        write(output_path, rate=self.args.sample_rate, data=wav_data)
        return

    def inference_from_path(self):
        src_mel, _ = get_spectrograms(self.args.source)
        tar_mel, _ = get_spectrograms(self.args.target)
        src_mel = paddle.to_tensor(data=self.normalize(src_mel))
        tar_mel = paddle.to_tensor(data=self.normalize(tar_mel))
        conv_wav, conv_mel = self.inference_one_utterance(src_mel, tar_mel)
        self.write_wav_to_file(conv_wav, self.args.output)
        return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-attr', '-a', help='attr file path', default='attr.pkl')
    parser.add_argument('-config', '-c', help='config file path', default='vctk_model.config.yaml')
    parser.add_argument('-model', '-m', help='model path', default='vctk_model.pdparams')
    parser.add_argument('-source', '-s', help='source wav path', default='p225_002.wav')
    parser.add_argument('-target', '-t', help='target wav path', default='p227_001.wav')
    parser.add_argument('-output', '-o', help='output wav path', default='result.wav')
    parser.add_argument('-sample_rate', '-sr', help='sample rate', default=24000, type=int)
    args = parser.parse_args()
    paddle.set_device('gpu')
    with open(args.config) as f:
        config = yaml.load(f, yaml.SafeLoader)
    inferencer = Inferencer(config=config, args=args)
    inferencer.inference_from_path()
