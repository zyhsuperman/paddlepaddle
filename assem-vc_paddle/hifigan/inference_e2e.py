from __future__ import absolute_import, division, print_function, unicode_literals
import paddle
import glob
import os
import numpy as np
import argparse
import json
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import MAX_WAV_VALUE
from models import Generator
h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = paddle.load(path=filepath)
    print('Complete.')
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    generator = Generator(h)
    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.set_state_dict(state_dict=state_dict_g['generator'])
    filelist = os.listdir(a.input_mels_dir)
    os.makedirs(a.output_dir, exist_ok=True)
    generator.eval()
    generator.remove_weight_norm()
    with paddle.no_grad():
        for i, filname in enumerate(filelist):
            x = np.load(os.path.join(a.input_mels_dir, filname))
            x = paddle.to_tensor(data=x, dtype='float32')
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')
            output_file = os.path.join(a.output_dir, os.path.splitext(
                filname)[0] + '_generated_e2e.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)


def main():
    print('Initializing Inference Process..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mels_dir', default='hifigan/test_mel_files')
    parser.add_argument('--output_dir', default='generated_files_from_mel')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()
    config_file = os.path.join(os.path.split(a.checkpoint_file)[0],
        'config.json')
    with open(config_file) as f:
        data = f.read()
    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)
    paddle.seed(seed=h.seed)
    global device
    if paddle.device.cuda.device_count() >= 1:
        paddle.seed(seed=h.seed)
        device = str('cuda').replace('cuda', 'gpu')
    else:
        device = str('cpu').replace('cuda', 'gpu')
    inference(a)


if __name__ == '__main__':
    main()
