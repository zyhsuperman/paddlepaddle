import paddle
import os
import pickle
import json
import numpy as np


class CollateFn(object):

    def __init__(self, frame_size):
        self.frame_size = frame_size

    def make_frames(self, tensor):

        out = tensor.reshape([tensor.shape[0], tensor.shape[1] // self.frame_size, self.frame_size * tensor.shape[2]])
        out = out.transpose([0, 2, 1])
        return out

    def __call__(self, l):
        data_tensor = paddle.to_tensor(data=np.array(l))
        segment = self.make_frames(data_tensor)
        return segment


def get_data_loader(dataset, batch_size, frame_size, shuffle=True,
    num_workers=4, drop_last=False):
    _collate_fn = CollateFn(frame_size=frame_size)
    dataloader = paddle.io.DataLoader(dataset, batch_size=batch_size,
        shuffle=shuffle, num_workers=num_workers, collate_fn=_collate_fn)
    return dataloader


class SequenceDataset(paddle.io.Dataset):

    def __init__(self, data):
        self.data = data
        self.utt_ids = list(self.data.keys())

    def __getitem__(self, ind):
        utt_id = self.utt_ids[ind]
        ret = self.data[utt_id].transpose()
        return ret

    def __len__(self):
        return len(self.utt_ids)


class PickleDataset(paddle.io.Dataset):

    def __init__(self, pickle_path, sample_index_path, segment_size):
        with open(pickle_path, 'rb') as f:
            self.data = pickle.load(f)
        with open(sample_index_path, 'r') as f:
            self.indexes = json.load(f)
        self.segment_size = segment_size

    def __getitem__(self, ind):
        utt_id, t = self.indexes[ind]
        segment = self.data[utt_id][t:t + self.segment_size]
        return segment

    def __len__(self):
        return len(self.indexes)
