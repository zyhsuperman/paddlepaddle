import sys
sys.path.append('/home/zyhao/paddlepaddle/Wav2lip_paadle/utils')
import paddle
import os
import sys
import cv2
import random
import datetime
import math
import argparse
import numpy as np
import scipy.io as sio
import zipfile
from .net_s3fd import s3fd
from .bbox import *

paddle.device.set_device('gpu')
def detect(net, img):
    img = img - np.array([104, 117, 123])
    img = img.transpose(2, 0, 1)
    img = img.reshape((1,) + img.shape)
    img = paddle.to_tensor(data=img).astype(dtype='float32')
    BB, CC, HH, WW = img.shape
    with paddle.no_grad():
        olist = net(img)
    bboxlist = []
    for i in range(len(olist) // 2):
        olist[i * 2] = paddle.nn.functional.softmax(x=olist[i * 2], axis=1)
    olist = [oelem.data.cpu() for oelem in olist]
    for i in range(len(olist) // 2):
        ocls, oreg = olist[i * 2], olist[i * 2 + 1]
        FB, FC, FH, FW = ocls.shape
        stride = 2 ** (i + 2)
        anchor = stride * 4
        poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
        for Iindex, hindex, windex in poss:
            axc, ayc = (stride / 2 + windex * stride, stride / 2 + hindex *
                stride)
            score = ocls[0, 1, hindex, windex]
            """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            loc = oreg[0, :, hindex, windex].reshape((1, 4))
            priors = paddle.to_tensor(data=[[axc / 1.0, ayc / 1.0, stride *
                4 / 1.0, stride * 4 / 1.0]], dtype='float32')
            variances = [0.1, 0.2]
            box = decode(loc, priors, variances)
            x1, y1, x2, y2 = box[0] * 1.0
            bboxlist.append([x1, y1, x2, y2, score])
    bboxlist = np.array(bboxlist)
    if 0 == len(bboxlist):
        bboxlist = np.zeros((1, 5))
    return bboxlist


def batch_detect(net, imgs):
    imgs = imgs - np.array([104, 117, 123])
    imgs = imgs.transpose(0, 3, 1, 2)
    imgs = paddle.to_tensor(data=imgs).astype(dtype='float32')
    BB, CC, HH, WW = imgs.shape
    with paddle.no_grad():
        olist = net(imgs)
    bboxlist = []
    for i in range(len(olist) // 2):
        olist[i * 2] = paddle.nn.functional.softmax(x=olist[i * 2], axis=1)
    olist = [oelem.cpu() for oelem in olist]
    for i in range(len(olist) // 2):
        ocls, oreg = olist[i * 2], olist[i * 2 + 1]
        FB, FC, FH, FW = ocls.shape
        stride = 2 ** (i + 2)
        anchor = stride * 4
        poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
        for Iindex, hindex, windex in poss:
            axc, ayc = (stride / 2 + windex * stride, stride / 2 + hindex *
                stride)
            score = ocls[:, 1, hindex, windex]
            """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            loc = oreg[:, :, hindex, windex].reshape((BB, 1, 4))
            """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            priors = paddle.to_tensor(data=[[axc / 1.0, ayc / 1.0, stride *
                4 / 1.0, stride * 4 / 1.0]], dtype='float32').reshape((1, 1, 4))
            variances = [0.1, 0.2]
            box = batch_decode(loc, priors, variances)
            box = box[:, 0] * 1.0
            bboxlist.append(paddle.concat(x=[box, score.unsqueeze(axis=1)],
                axis=1).cpu().numpy())
    bboxlist = np.array(bboxlist)
    if 0 == len(bboxlist):
        bboxlist = np.zeros((1, BB, 5))
    return bboxlist


def flip_detect(net, img):
    img = cv2.flip(img, 1)
    b = detect(net, img)
    bboxlist = np.zeros(b.shape)
    bboxlist[:, 0] = img.shape[1] - b[:, 2]
    bboxlist[:, 1] = b[:, 1]
    bboxlist[:, 2] = img.shape[1] - b[:, 0]
    bboxlist[:, 3] = b[:, 3]
    bboxlist[:, 4] = b[:, 4]
    return bboxlist


def pts_to_bb(pts):
    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)
    return np.array([min_x, min_y, max_x, max_y])
