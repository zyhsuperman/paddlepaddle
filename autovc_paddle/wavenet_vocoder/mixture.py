from __future__ import with_statement, print_function, absolute_import
import paddle

import math
import numpy as np


def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.shape) - 1
    m, _ = paddle.max(x=x, axis=axis), paddle.argmax(x=x, axis=axis)
    m2, _ = paddle.max(x=x, axis=axis, keepdim=True), paddle.argmax(x=x,
        axis=axis, keepdim=True)
    return m + paddle.log(x=paddle.sum(x=paddle.exp(x=x - m2), axis=axis))


def discretized_mix_logistic_loss(y_hat, y, num_classes=256, log_scale_min=
    -7.0, reduce=True):
    """Discretized mixture of logistic distributions loss

    Note that it is assumed that input is scaled to [-1, 1].

    Args:
        y_hat (Tensor): Predicted output (B x C x T)
        y (Tensor): Target (B x T x 1).
        num_classes (int): Number of classes
        log_scale_min (float): Log scale minimum value
        reduce (bool): If True, the losses are averaged or summed for each
          minibatch.

    Returns
        Tensor: loss
    """
    assert y_hat.dim() == 3
    assert y_hat.shape[1] % 3 == 0
    nr_mix = y_hat.shape[1] // 3

    y_hat = paddle.transpose(y_hat, perm=[0, 2, 1])

    logit_probs = y_hat[:, :, :nr_mix]
    means = y_hat[:, :, nr_mix:2 * nr_mix]
    log_scales = paddle.clip(x=y_hat[:, :, 2 * nr_mix:3 * nr_mix], min=
        log_scale_min)
    y = y.expand_as(y=means)
    centered_y = y - means
    inv_stdv = paddle.exp(x=-log_scales)
    plus_in = inv_stdv * (centered_y + 1.0 / (num_classes - 1))
    cdf_plus = paddle.nn.functional.sigmoid(x=plus_in)
    min_in = inv_stdv * (centered_y - 1.0 / (num_classes - 1))
    cdf_min = paddle.nn.functional.sigmoid(x=min_in)
    log_cdf_plus = plus_in - paddle.nn.functional.softplus(x=plus_in)
    log_one_minus_cdf_min = -paddle.nn.functional.softplus(x=min_in)
    cdf_delta = cdf_plus - cdf_min
    mid_in = inv_stdv * centered_y
    log_pdf_mid = mid_in - log_scales - 2.0 * paddle.nn.functional.softplus(x
        =mid_in)
    """
    log_probs = tf.where(x < -0.999, log_cdf_plus,
                         tf.where(x > 0.999, log_one_minus_cdf_min,
                                  tf.where(cdf_delta > 1e-5,
                                           tf.log(tf.maximum(cdf_delta, 1e-12)),
                                           log_pdf_mid - np.log(127.5))))
    """
    inner_inner_cond = (cdf_delta > 1e-05).astype(dtype='float32')
    inner_inner_out = inner_inner_cond * paddle.log(x=paddle.clip(x=
        cdf_delta, min=1e-12)) + (1.0 - inner_inner_cond) * (log_pdf_mid -
        np.log((num_classes - 1) / 2))
    inner_cond = (y > 0.999).astype(dtype='float32')
    inner_out = inner_cond * log_one_minus_cdf_min + (1.0 - inner_cond
        ) * inner_inner_out
    cond = (y < -0.999).astype(dtype='float32')
    log_probs = cond * log_cdf_plus + (1.0 - cond) * inner_out
    log_probs = log_probs + paddle.nn.functional.log_softmax(x=logit_probs,
        axis=-1)
    if reduce:
        return -paddle.sum(x=log_sum_exp(log_probs))
    else:
        return -log_sum_exp(log_probs).unsqueeze(axis=-1)


def to_one_hot(tensor, n, fill_with=1.0):
    one_hot = paddle.zeros(tensor.shape+[n], dtype=paddle.float32)
    if 'gpu' in str(tensor.place):
        one_hot = one_hot
    one_hot.put_along_axis_(axis=len(tensor.shape), indices=tensor.
        unsqueeze(axis=-1), values=fill_with)
    return one_hot


def sample_from_discretized_mix_logistic(y, log_scale_min=-7.0):
    """
    Sample from discretized mixture of logistic distributions

    Args:
        y (Tensor): B x C x T
        log_scale_min (float): Log scale minimum value

    Returns:
        Tensor: sample in range of [-1, 1].
    """
    assert y.shape[1] % 3 == 0
    nr_mix = y.shape[1] // 3
    y = paddle.transpose(y, perm=[0, 2, 1])

    logit_probs = y[:, :, :nr_mix]
    temp = paddle.uniform(shape=logit_probs.shape, min=1e-05, max=1.0 - 1e-05, dtype=logit_probs.dtype)
    temp = logit_probs.data - paddle.log(x=-paddle.log(x=temp))
        # 使用paddle.argmax获取最大值的索引
    argmax = paddle.argmax(temp, axis=-1)

    # 使用paddle.max获取最大值
    max_values = paddle.max(temp, axis=-1)
    one_hot = to_one_hot(argmax, nr_mix)
    means = paddle.sum(x=y[:, :, nr_mix:2 * nr_mix] * one_hot, axis=-1)
    log_scales = paddle.clip(x=paddle.sum(x=y[:, :, 2 * nr_mix:3 * nr_mix] *
        one_hot, axis=-1), min=log_scale_min)
    # u = means.data.new(means.shape).uniform_(min=1e-05, max=1.0 - 1e-05)
    u = paddle.uniform(shape=means.shape, min=1e-05, max=1.0 - 1e-05, dtype=
        means.dtype)
    x = means + paddle.exp(x=log_scales) * (paddle.log(x=u) - paddle.log(x=
        1.0 - u))
    x = paddle.clip(x=paddle.clip(x=x, min=-1.0), max=1.0)
    return x
