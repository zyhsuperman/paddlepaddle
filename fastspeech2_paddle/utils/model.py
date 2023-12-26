import sys
import paddle
import os
import json
import numpy as np
import hifigan
from model import FastSpeech2, ScheduledOptim


def get_model(args, configs, device, train=False):
    preprocess_config, model_config, train_config = configs
    model = FastSpeech2(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(train_config['path']['ckpt_path'],
            '{}.pdparams'.format(args.restore_step))
        ckpt = paddle.load(path=ckpt_path)
        model.set_state_dict(state_dict=ckpt['model'])
    if train:
        scheduled_optim = ScheduledOptim(model, train_config, model_config,
            args.restore_step)
        if args.restore_step:
            scheduled_optim.set_state_dict(state_dict=ckpt['optimizer'])
        model.train()
        return model, scheduled_optim
    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.size for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config['vocoder']['model']
    speaker = config['vocoder']['speaker']
    if name == 'MelGAN':
        if speaker == 'LJSpeech':
            vocoder = paddle.hub.load(repo_dir='descriptinc/melgan-neurips',
                model='load_melgan', source='linda_johnson')
        elif speaker == 'universal':
            vocoder = paddle.hub.load(repo_dir='descriptinc/melgan-neurips',
                model='load_melgan', source='multi_speaker')
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == 'HiFi-GAN':
        with open('hifigan/config.json', 'r') as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == 'LJSpeech':
            ckpt = paddle.load(path='hifigan/generator_LJSpeech.pth.tar')
        elif speaker == 'universal':
            ckpt = paddle.load(path='hifigan/paddle_vocoder_universal.pdparams')
        vocoder.set_state_dict(state_dict=ckpt['generator'])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)
    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None
    ):
    name = model_config['vocoder']['model']
    with paddle.no_grad():
        if name == 'MelGAN':
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == 'HiFi-GAN':
            wavs = vocoder(mels).squeeze(axis=1)
    wavs = (wavs.cpu().numpy() * preprocess_config['preprocessing']['audio'
        ]['max_wav_value']).astype('int16')
    wavs = [wav for wav in wavs]
    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][:lengths[i]]
    return wavs
