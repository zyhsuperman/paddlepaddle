import sys
sys.path.append('/home1/zhaoyh/paddlemodel/assem-vc_paddle/utils')
import paddle_aux
import paddle
from __future__ import absolute_import, division, print_function, unicode_literals
import glob
import os
import argparse
import json
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import Generator
h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = paddle.load(path=filepath)
    print('Complete.')
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.
        hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    generator = Generator(h).to(device)
    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.set_state_dict(state_dict=state_dict_g['generator'])
    filelist = os.listdir(a.input_wavs_dir)
    os.makedirs(a.output_dir, exist_ok=True)
    generator.eval()
    generator.remove_weight_norm()
    with paddle.no_grad():
        for i, filname in enumerate(filelist):
            wav, sr = load_wav(os.path.join(a.input_wavs_dir, filname))
            wav = wav / MAX_WAV_VALUE
            wav = paddle.to_tensor(data=wav, dtype='float32').to(device)
            x = get_mel(wav.unsqueeze(axis=0))
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')
            output_file = os.path.join(a.output_dir, os.path.splitext(
                filname)[0] + '_generated.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)


def main():
    print('Initializing Inference Process..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='test_files')
    parser.add_argument('--output_dir', default='generated_files')
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
