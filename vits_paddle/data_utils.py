import paddle
import time
import os
import random
import numpy as np
import commons
from mel_processing import spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cleaned_text_to_sequence


class TextAudioLoader(paddle.io.Dataset):
    """
        1) loads audio, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.cleaned_text = getattr(hparams, 'cleaned_text', False)
        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, 'min_text_len', 1)
        self.max_text_len = getattr(hparams, 'max_text_len', 190)
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        audiopaths_and_text_new = []
        lengths = []
        for audiopath, text in self.audiopaths_and_text:
            if self.min_text_len <= len(text) and len(text
                ) <= self.max_text_len:
                audiopaths_and_text_new.append([audiopath, text])
                lengths.append(os.path.getsize(audiopath) // (2 * self.
                    hop_length))
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_audio_text_pair(self, audiopath_and_text):
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        return text, spec, wav

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(axis=0)
        spec_filename = filename.replace('.wav', '.spec.pt')
        # spec_filename = os.path.join('/home1/zhaoyh/paddlemodel/vits_paddle', spec_filename)
        # if os.path.exists(spec_filename):
        #     spec = paddle.load(path=spec_filename)
        # else:

        spec = spectrogram_torch(audio_norm, self.filter_length, self.
            sampling_rate, self.hop_length, self.win_length, center=False)
        spec = paddle.squeeze(x=spec, axis=0)
        paddle.save(obj=spec, path=spec_filename)
        return spec, audio_norm

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = paddle.to_tensor(data=text_norm, dtype='int64')
        return text_norm

    def __getitem__(self, index):
        self.get_audio_text_pair(self.audiopaths_and_text[index])
        # import pdb
        # pdb.set_trace()
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextAudioCollate:
    """ Zero-pads model inputs and targets
    """

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """


        _, ids_sorted_decreasing = paddle.sort(descending=True, x=paddle.to_tensor(data=[x[1].shape[1] for x in batch], dtype='int64'),axis=0), paddle.argsort(descending=True, x=paddle.to_tensor(data=[x[1].shape[1] for x in batch], dtype='int64'), axis=0)
        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].shape[1] for x in batch])
        max_wav_len = max([x[2].shape[1] for x in batch])
        # text_lengths = paddle.to_tensor(data=len(batch), dtype='int64')

        text_lengths = paddle.zeros(shape=[len(batch)], dtype='int64')
        # spec_lengths = paddle.to_tensor(data=len(batch), dtype='int64')
        # wav_lengths = paddle.to_tensor(data=len(batch), dtype='int64')
        spec_lengths = paddle.zeros(shape=[len(batch)], dtype='int64')
        wav_lengths = paddle.zeros(shape=[len(batch)], dtype='int64')
        text_padded = paddle.empty(shape=[len(batch), max_text_len], dtype=
            'int64')
        spec_padded = paddle.empty(shape=[len(batch), batch[0][1].shape[0],
            max_spec_len], dtype='float32')
        wav_padded = paddle.empty(shape=[len(batch), 1, max_wav_len], dtype
            ='float32')
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            text = row[0]
            text_padded[(i), :text.shape[0]] = text
            text_lengths[i] = text.shape[0]
            spec = row[1]
            spec_padded[(i), :, :spec.shape[1]] = spec
            spec_lengths[i] = spec.shape[1]
            wav = row[2]
            wav_padded[(i), :, :wav.shape[1]] = wav
            wav_lengths[i] = wav.shape[1]
        if self.return_ids:
            return (text_padded, text_lengths, spec_padded, spec_lengths,
                wav_padded, wav_lengths, ids_sorted_decreasing)
        return (text_padded, text_lengths, spec_padded, spec_lengths,
            wav_padded, wav_lengths)


"""Multi speaker version"""


class TextAudioSpeakerLoader(paddle.io.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_sid_text, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.cleaned_text = getattr(hparams, 'cleaned_text', False)
        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, 'min_text_len', 1)
        self.max_text_len = getattr(hparams, 'max_text_len', 190)
        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        audiopaths_sid_text_new = []
        lengths = []
        for audiopath, sid, text in self.audiopaths_sid_text:
            if self.min_text_len <= len(text) and len(text
                ) <= self.max_text_len:
                audiopaths_sid_text_new.append([audiopath, sid, text])
                lengths.append(os.path.getsize(audiopath) // (2 * self.
                    hop_length))
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        audiopath, sid, text = audiopath_sid_text[0], audiopath_sid_text[1
            ], audiopath_sid_text[2]
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        sid = self.get_sid(sid)
        return text, spec, wav, sid

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(axis=0)
        spec_filename = filename.replace('.wav', '.spec.pt')
        if os.path.exists(spec_filename):
            spec = paddle.load(path=spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length, self.
                sampling_rate, self.hop_length, self.win_length, center=False)
            spec = paddle.squeeze(x=spec, axis=0)
            paddle.save(obj=spec, path=spec_filename)
        return spec, audio_norm

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = paddle.to_tensor(data=text_norm, dtype='int64')
        return text_norm

    def get_sid(self, sid):
        sid = paddle.to_tensor(data=[int(sid)], dtype='int64')
        return sid

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index]
            )

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate:
    """ Zero-pads model inputs and targets
    """

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        _, ids_sorted_decreasing = paddle.sort(descending=True, x=paddle.
            to_tensor(data=[x[1].shape[1] for x in batch], dtype='int64'),
            axis=0), paddle.argsort(descending=True, x=paddle.to_tensor(
            data=[x[1].shape[1] for x in batch], dtype='int64'), axis=0)
        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].shape[1] for x in batch])
        max_wav_len = max([x[2].shape[1] for x in batch])
        text_lengths = paddle.to_tensor(data=len(batch), dtype='int64')
        spec_lengths = paddle.to_tensor(data=len(batch), dtype='int64')
        wav_lengths = paddle.to_tensor(data=len(batch), dtype='int64')
        sid = paddle.to_tensor(data=len(batch), dtype='int64')
        text_padded = paddle.empty(shape=[len(batch), max_text_len], dtype=
            'int64')
        spec_padded = paddle.empty(shape=[len(batch), batch[0][1].shape[0],
            max_spec_len], dtype='float32')
        wav_padded = paddle.empty(shape=[len(batch), 1, max_wav_len], dtype
            ='float32')
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            text = row[0]
            text_padded[(i), :text.shape[0]] = text
            text_lengths[i] = text.shape[0]
            spec = row[1]
            spec_padded[(i), :, :spec.shape[1]] = spec
            spec_lengths[i] = spec.shape[1]
            wav = row[2]
            wav_padded[(i), :, :wav.shape[1]] = wav
            wav_lengths[i] = wav.shape[1]
            sid[i] = row[3]
        if self.return_ids:
            return (text_padded, text_lengths, spec_padded, spec_lengths,
                wav_padded, wav_lengths, sid, ids_sorted_decreasing)
        return (text_padded, text_lengths, spec_padded, spec_lengths,
            wav_padded, wav_lengths, sid)


class DistributedBucketSampler(paddle.io.DistributedBatchSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(self, dataset, batch_size, boundaries, num_replicas=None,
        rank=None, shuffle=True):
        super().__init__(dataset, batch_size, num_replicas=num_replicas, rank=rank,
            shuffle=shuffle)
        self.rank = rank
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
        self.num_replicas = num_replicas    
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - len_bucket % total_batch_size
                ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        g = paddle.framework.core.default_cpu_generator()
        g.manual_seed(self.epoch)
        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(paddle.randperm(n=len(bucket)).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))
        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]
            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket
                ) + ids_bucket[:rem % len_bucket]
            ids_bucket = ids_bucket[self.rank::self.num_replicas]
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j * self.
                    batch_size:(j + 1) * self.batch_size]]
                batches.append(batch)
        if self.shuffle:
            batch_ids = paddle.randperm(n=len(batches)).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples

        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1
        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
