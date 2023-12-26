import paddle
import os
import re
import random
import librosa
import numpy as np
from collections import Counter
from .text import Language
from .text.cmudict import CMUDict
from modules.mel import Audio2Mel


class TextMelDataset(paddle.io.Dataset):

    def __init__(self, hp, data_dir, metadata_path, train=True, norm=False):
        super().__init__()
        self.hp = hp
        self.lang = Language(hp.data.lang, hp.data.text_cleaners)
        self.train = train
        self.norm = norm
        self.data_dir = data_dir
        self.meta = self.load_metadata(metadata_path)
        self.speaker_dict = {speaker: idx for idx, speaker in enumerate(hp.
            data.speakers)}
        if train:
            speaker_counter = Counter(spk_id for audiopath, text, spk_id in
                self.meta)
            weights = [(1.0 / speaker_counter[spk_id]) for audiopath, text,
                spk_id in self.meta]
            self.mapping_weights = paddle.to_tensor(data=weights, dtype=
                'float64')
        self.remove_existing_mel()
        self.audio2mel = Audio2Mel(n_fft=hp.audio.filter_length, hop_length
            =hp.audio.hop_length, win_length=hp.audio.win_length,
            sampling_rate=hp.audio.sampling_rate, n_mel_channels=hp.audio.
            n_mel_channels, mel_fmin=hp.audio.mel_fmin, mel_fmax=hp.audio.
            mel_fmax)
        if hp.data.lang == 'cmu':
            self.cmudict = CMUDict(hp.data.cmudict_path)
            self.cmu_pattern = re.compile(
                "^(?P<word>[^!\\'(),-.:~?]+)(?P<punc>[!\\'(),-.:~?]+)$")
        else:
            self.cmudict = None

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if self.train:
            idx = paddle.multinomial(x=self.mapping_weights, num_samples=1
                ).item()
        audiopath, text, spk_id = self.meta[idx]
        audiopath = os.path.join(self.data_dir, audiopath)
        mel = self.get_mel(audiopath)
        text_norm = self.get_text(text)
        spk_id = self.speaker_dict[spk_id]
        return text_norm, mel, spk_id

    def get_mel(self, audiopath):
        melpath = os.path.join(self.data_dir, '{}.pt'.format(audiopath))
        try:
            mel = paddle.load(path=melpath)
            assert mel.shape[0
                ] == self.hp.audio.n_mel_channels, 'Mel dimension mismatch: expected %d, got %d' % (
                self.hp.audio.n_mel_channels, mel.shape[0])
        except (FileNotFoundError, RuntimeError, TypeError, ValueError):
            wav, sr = librosa.load(audiopath, sr=None, mono=True)
            assert sr == self.hp.audio.sampling_rate, 'sample mismatch: expected %d, got %d at %s' % (
                self.hp.audio.sampling_rate, sr, audiopath)
            wav = paddle.to_tensor(data=wav).reshape([1, 1, -1])
            if self.norm:
                wav = wav * (0.99 / (paddle.max(x=paddle.abs(x=wav)) + 1e-07))
            mel = self.audio2mel(wav).squeeze(axis=0)
            paddle.save(obj=mel, path=melpath)
        return mel

    def get_text(self, text):
        if self.cmudict is not None:
            text = ' '.join([self.get_arpabet(word) for word in text.split(
                ' ')])
        text_norm = paddle.to_tensor(data=self.lang.text_to_sequence(text,
            self.hp.data.text_cleaners), dtype='int32')
        return text_norm

    def get_arpabet(self, word):
        arpabet = self.cmudict.lookup(word)
        if arpabet is None:
            match = self.cmu_pattern.search(word)
            if match is None:
                return word
            subword = match.group('word')
            arpabet = self.cmudict.lookup(subword)
            if arpabet is None:
                return word
            punc = match.group('punc')
            arpabet = '{%s}%s' % (arpabet[0], punc)
        else:
            arpabet = '{%s}' % arpabet[0]
        if random.random() < 0.5:
            return word
        else:
            return arpabet

    def load_metadata(self, path, split='|'):
        with open(path, 'r', encoding='utf-8') as f:
            metadata = [line.strip().split(split) for line in f]
        return metadata

    def remove_existing_mel(self):
        for meta in self.meta:
            audiopath = meta[0]
            melpath = os.path.join(self.data_dir, '{}.pt'.format(audiopath))
            if os.path.exists(melpath):
                try:
                    os.remove(melpath)
                except FileNotFoundError:
                    pass


def text_mel_collate(batch):
    input_lengths, ids_sorted_decreasing = paddle.sort(descending=True, x=
        paddle.to_tensor(data=[len(x[0]) for x in batch], dtype='int64'),
        axis=0), paddle.argsort(descending=True, x=paddle.to_tensor(data=[
        len(x[0]) for x in batch], dtype='int64'), axis=0)
    max_input_len = paddle.empty(shape=[len(batch)], dtype='int64')
    max_input_len = paddle.full_like(max_input_len, input_lengths[0].item(), dtype='int64')
    text_padded = paddle.zeros(shape=(len(batch), max_input_len[0]), dtype=
        'int64')
    n_mel_channels = batch[0][1].shape[0]
    max_target_len = max([x[1].shape[1] for x in batch])
    mel_padded = paddle.zeros(shape=[len(batch), n_mel_channels,
        max_target_len])
    output_lengths = paddle.empty(shape=[len(batch)], dtype='int64')
    speakers = paddle.empty(shape=[len(batch)], dtype='int64')
    for idx, key in enumerate(ids_sorted_decreasing):
        text = batch[key][0]
        text_padded[(idx), :text.shape[0]] = text
        mel = batch[key][1]
        mel_padded[(idx), :, :mel.shape[1]] = mel
        output_lengths[idx] = mel.shape[1]
        speakers[idx] = batch[key][2]
    return (text_padded, mel_padded, speakers, input_lengths,
        output_lengths, max_input_len)
