import paddle
import random
import numpy as np
import commons
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cmudict
from text.symbols import symbols


class TextMelLoader(paddle.io.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """

    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.add_noise = hparams.add_noise
        self.add_blank = getattr(hparams, 'add_blank', False)
        if getattr(hparams, 'cmudict_path', None) is not None:
            self.cmudict = cmudict.CMUDict(hparams.cmudict_path)
        self.stft = commons.TacotronSTFT(hparams.filter_length, hparams.
            hop_length, hparams.win_length, hparams.n_mel_channels, hparams
            .sampling_rate, hparams.mel_fmin, hparams.mel_fmax)
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)

    def get_mel_text_pair(self, audiopath_and_text):
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return text, mel

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".
                    format(sampling_rate, self.stft.sampling_rate))
            if self.add_noise:
                audio = audio + paddle.rand(shape=audio.shape, dtype=audio.
                    dtype)
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(axis=0)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = paddle.squeeze(x=melspec, axis=0)
        else:
            melspec = paddle.to_tensor(data=np.load(filename))
            assert melspec.shape[0
                ] == self.stft.n_mel_channels, 'Mel dimension mismatch: given {}, expected {}'.format(
                melspec.shape[0], self.stft.n_mel_channels)
        return melspec

    def get_text(self, text):
        text_norm = text_to_sequence(text, self.text_cleaners, getattr(self,
            'cmudict', None))
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, len(symbols))
        text_norm = paddle.to_tensor(data=text_norm, dtype='int32')
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate:
    """ Zero-pads model inputs and targets based on number of frames per step
    """

    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        input_lengths, ids_sorted_decreasing = paddle.sort(descending=True,
            x=paddle.to_tensor(data=[len(x[0]) for x in batch], dtype=
            'int64'), axis=0), paddle.argsort(descending=True, x=paddle.
            to_tensor(data=[len(x[0]) for x in batch], dtype='int64'), axis=0)
        max_input_len = input_lengths[0]
        text_padded = paddle.empty(shape=[len(batch), max_input_len], dtype
            ='int64')
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[(i), :text.shape[0]] = text
        num_mels = batch[0][1].shape[0]
        max_target_len = max([x[1].shape[1] for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += (self.n_frames_per_step - max_target_len %
                self.n_frames_per_step)
            assert max_target_len % self.n_frames_per_step == 0
        mel_padded = paddle.empty(shape=[len(batch), num_mels,
            max_target_len], dtype='float32')
        mel_padded.zero_()
        output_lengths = paddle.to_tensor([0] * len(batch), dtype='int64')
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[(i), :, :mel.shape[1]] = mel
            output_lengths[i] = mel.shape[1]
        return text_padded, input_lengths, mel_padded, output_lengths


"""Multi speaker version"""


class TextMelSpeakerLoader(paddle.io.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """

    def __init__(self, audiopaths_sid_text, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.add_noise = hparams.add_noise
        self.add_blank = getattr(hparams, 'add_blank', False)
        self.min_text_len = getattr(hparams, 'min_text_len', 1)
        self.max_text_len = getattr(hparams, 'max_text_len', 190)
        if getattr(hparams, 'cmudict_path', None) is not None:
            self.cmudict = cmudict.CMUDict(hparams.cmudict_path)
        self.stft = commons.TacotronSTFT(hparams.filter_length, hparams.
            hop_length, hparams.win_length, hparams.n_mel_channels, hparams
            .sampling_rate, hparams.mel_fmin, hparams.mel_fmax)
        self._filter_text_len()
        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)

    def _filter_text_len(self):
        audiopaths_sid_text_new = []
        for audiopath, sid, text in self.audiopaths_sid_text:
            if self.min_text_len <= len(text) and len(text
                ) <= self.max_text_len:
                audiopaths_sid_text_new.append([audiopath, sid, text])
        self.audiopaths_sid_text = audiopaths_sid_text_new

    def get_mel_text_speaker_pair(self, audiopath_sid_text):
        audiopath, sid, text = audiopath_sid_text[0], audiopath_sid_text[1
            ], audiopath_sid_text[2]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        sid = self.get_sid(sid)
        return text, mel, sid

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".
                    format(sampling_rate, self.stft.sampling_rate))
            if self.add_noise:
                audio = audio + paddle.rand(shape=audio.shape, dtype=audio.
                    dtype)
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(axis=0)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = paddle.squeeze(x=melspec, axis=0)
        else:
            melspec = paddle.to_tensor(data=np.load(filename))
            assert melspec.shape[0
                ] == self.stft.n_mel_channels, 'Mel dimension mismatch: given {}, expected {}'.format(
                melspec.shape[0], self.stft.n_mel_channels)
        return melspec

    def get_text(self, text):
        text_norm = text_to_sequence(text, self.text_cleaners, getattr(self,
            'cmudict', None))
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, len(symbols))
        text_norm = paddle.to_tensor(data=text_norm, dtype='int32')
        return text_norm

    def get_sid(self, sid):
        sid = paddle.to_tensor(data=[int(sid)], dtype='int32')
        return sid

    def __getitem__(self, index):
        return self.get_mel_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextMelSpeakerCollate:
    """ Zero-pads model inputs and targets based on number of frames per step
    """

    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        input_lengths, ids_sorted_decreasing = paddle.sort(descending=True,
            x=paddle.to_tensor(data=[len(x[0]) for x in batch], dtype=
            'int64'), axis=0), paddle.argsort(descending=True, x=paddle.
            to_tensor(data=[len(x[0]) for x in batch], dtype='int64'), axis=0)
        max_input_len = input_lengths[0]
        text_padded = paddle.empty(shape=[len(batch), max_input_len], dtype
            ='int64')
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[(i), :text.shape[0]] = text
        num_mels = batch[0][1].shape[0]
        max_target_len = max([x[1].shape[1] for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += (self.n_frames_per_step - max_target_len %
                self.n_frames_per_step)
            assert max_target_len % self.n_frames_per_step == 0
        mel_padded = paddle.empty(shape=[len(batch), num_mels,
            max_target_len], dtype='float32')
        mel_padded.zero_()
        output_lengths = paddle.to_tensor(data=len(batch), dtype='int64')
        sid = paddle.to_tensor(data=len(batch), dtype='int64')
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[(i), :, :mel.shape[1]] = mel
            output_lengths[i] = mel.shape[1]
            sid[i] = batch[ids_sorted_decreasing[i]][2]
        return text_padded, input_lengths, mel_padded, output_lengths, sid
