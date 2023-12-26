import sys
import paddle
import os
import pickle
import numpy as np
from math import ceil
from model_vc import Generator


def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0]) / base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0, len_pad), (0, 0)), 'constant'), len_pad

device = 'gpu:0'
paddle.set_device(device)

G = Generator(32, 256, 512, 32)
G.eval()
g_checkpoint = paddle.load(path='checkpoints/autovc_1.pdparams')
G.set_state_dict(state_dict=g_checkpoint['model'])
metadata = pickle.load(open('mymetadata.pkl', 'rb'))
spect_vc = []
for sbmt_i in metadata:
    x_org = sbmt_i[2]
    x_org, len_pad = pad_seq(x_org)
    uttr_org = paddle.to_tensor(data=x_org[(np.newaxis), :, :])
    emb_org = paddle.to_tensor(data=sbmt_i[1][(np.newaxis), :])
    for sbmt_j in metadata:
        emb_trg = paddle.to_tensor(data=sbmt_j[1][(np.newaxis), :])
        with paddle.no_grad():
            _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)
        if len_pad == 0:
            uttr_trg = x_identic_psnt[(0), (0), :, :].cpu().numpy()
        else:
            uttr_trg = x_identic_psnt[(0), (0), :-len_pad, :].cpu().numpy()
        spect_vc.append(('{}x{}'.format(sbmt_i[0], sbmt_j[0]), uttr_trg))
with open('myresults.pkl', 'wb') as handle:
    pickle.dump(spect_vc, handle)
