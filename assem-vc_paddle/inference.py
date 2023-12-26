import paddle
import os
import librosa
import argparse
from omegaconf import OmegaConf
import soundfile as sf
os.sys.path.append('./cotatron')
from synthesizer import Synthesizer
import sys
sys.path.append('hifi-gan/')
import os
import numpy as np
import json
import librosa
import random
from omegaconf import OmegaConf
from datasets import TextMelDataset, text_mel_collate
from hifigan.models import Generator
import soundfile as sf
MAX_WAV_VALUE = 32768.0


class AttrDict(dict):

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

paddle.set_device('gpu')

hp_path = ['config/global/config.yaml', 'config/vc/config.yaml']
hp_global = OmegaConf.load(hp_path[0])
hp_vc = OmegaConf.load(hp_path[1])
hp = OmegaConf.merge(hp_global, hp_vc)
parser = argparse.ArgumentParser()
parser.add_argument('--config')
hparams = parser.parse_args(['--config', hp_path])
model = Synthesizer(hparams)

synthesizer_path = 'checkpoints/assem-vc_pretrained.pdparams'
generator_path = 'checkpoints/hifi-gan_vctk_g_02600000.pdparams'
config_path = 'hifigan/config_v1.json'
with open(config_path) as f:
    data = f.read()
sync_ckpt = paddle.load(path=synthesizer_path)
model.set_state_dict(state_dict=sync_ckpt['state_dict'])
model.eval()

json_config = json.loads(data)
h = AttrDict(json_config)
paddle.seed(seed=h.seed)
generator = Generator(h)
state_dict_g = paddle.load(path=generator_path)
generator.set_state_dict(state_dict=state_dict_g['generator'])
generator.remove_weight_norm()
generator.eval()

sourceloader = TextMelDataset(hp, 'datasets/inference_source',
    'metadata_g2p.txt', train=False, use_f0s=True)
target_root, target_dir = 'datasets/inference_target', 'metadata_g2p.txt'
targetloader = TextMelDataset(hp, target_root, target_dir, train=False,
    use_f0s=True)
print('length of the source metadata is : ', len(sourceloader))
print('length of the target metadata is : ', len(targetloader))

source_idx = 0
audio_path, text, _ = sourceloader.meta[source_idx]
x = sourceloader.__getitem__(source_idx)
batch = text_mel_collate([x])
print(text)

x, sr = librosa.load(os.path.join('datasets/inference_source', audio_path))
speaker_list = list(hp.data.speakers)
with open('f0s.txt', 'r', encoding='utf-8') as g:
    pitches = g.readlines()
speaker_list_2 = [x.split('|')[0].strip() for x in pitches]
means = [float(x.split('|')[1].strip()) for x in pitches]
variences = [float(x.split('|')[2].strip()) for x in pitches]

if speaker_list == speaker_list_2:
    print('ok')
else:
    print('error')
file_idx = random.randrange(len(targetloader))
target_audio_path, _, _ = targetloader.meta[file_idx]
x = targetloader.__getitem__(file_idx)
target_batch = text_mel_collate([x])

x, sr = librosa.load(os.path.join(target_root, target_audio_path))
with paddle.no_grad():
    (text, mel_source, speakers, f0_padded, input_lengths, output_lengths,
        max_input_len, _) = batch
    _, mel_reference, _, _, _, _, _, _ = target_batch
    text = text
    mel_source = mel_source
    mel_reference = mel_reference
    f0_padded = f0_padded
    mel_predicted, alignment, residual = model.inference(text, mel_source,
        mel_reference, f0_padded)
with paddle.no_grad():
    y_g_hat = generator(mel_predicted.detach())
    audio = y_g_hat.squeeze()
    audio = audio * MAX_WAV_VALUE
    audio = audio.detach().cpu().numpy().astype('int16')
sf.write('output.wav', audio, hp.audio.sampling_rate)
