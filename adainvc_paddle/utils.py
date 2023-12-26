import paddle
import numpy as np

from visualdl import LogWriter
import editdistance


def cc(net):
    device = str('cuda' if paddle.device.cuda.device_count() >= 1 else 'cpu'
        ).replace('cuda', 'gpu')
    paddle.set_device(device)
    return net


class Logger(object):

    def __init__(self, logdir='./log'):
        self.writer =  LogWriter(logdir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def scalars_summary(self, tags, dictionary, step):
        for tag, value in dictionary.items():
            self.writer.add_scalar(tags+tag, value, step)

    def text_summary(self, tag, value, step):
        self.writer.add_text(tag, value, step)

    def audio_summary(self, tag, value, step, sr):
        self.writer.add_audio(tag, value, step, sample_rate=sr)


def infinite_iter(iterable):
    it = iter(iterable)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(iterable)
