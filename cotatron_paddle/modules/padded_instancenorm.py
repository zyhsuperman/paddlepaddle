import paddle


class PaddedInstanceNorm1d(paddle.nn.Layer):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=False,
        track_running_stats=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        if affine is True:
            raise NotImplementedError
        if track_running_stats is True:
            raise NotImplementedError

    def forward(self, x, lengths):
        lengths = lengths.reshape([-1, 1, 1]).astype(dtype='float32')
        sum_ = paddle.sum(x=x, axis=2, keepdim=True)
        mean = sum_ / lengths
        sqsum = paddle.sum(x=paddle.pow(x=x, y=2.0), axis=2, keepdim=True)
        sqmean = sqsum / lengths
        var = sqmean - paddle.pow(x=mean, y=2.0)
        return (x - mean) / paddle.pow(x=var + self.eps, y=0.5)


if __name__ == '__main__':
    instnorm = paddle.nn.InstanceNorm1D(num_features=1, momentum=1 - 0.1)
    p_instnorm = PaddedInstanceNorm1d(1)
    
    x = paddle.to_tensor(data=[-2.0, 1.0, 0.0, 3.0, 4.0]).reshape([1, 1, -1])
    lengths = paddle.to_tensor(data=[5], dtype='int64')
    print('-' * 100)
    print('Check InstanceNorm1d == PaddedInstanceNorm1d')
    print('Input x: %s' % x)
    print('Input lengths: %s' % lengths)
    print('%s - nn.InstanceNorm1d(1)(x)' % instnorm(x))
    print('%s - PaddedInstanceNorm1d(1)(x, lengths)' % p_instnorm(x, lengths))
    print('-' * 100)
    padded = paddle.to_tensor(data=[[-2.0, 1.0, 0.0, 3.0, 4.0, 0.0, 0.0], [
        -2.0, 1.0, 0.0, 3.0, 4.0, 0.0, 0.0]]).unsqueeze(axis=1)
    padded_lengths = paddle.to_tensor(data=[5, 5], dtype='int64')
    print('Input padded: %s, %s' % (padded, padded.shape))
    print('Input padded_lengths: %s' % padded_lengths)
    y = p_instnorm(padded, padded_lengths)
    print('%s - PaddedInstanceNorm1d(1)(x, lengths), %s' % (y, y.shape))
    print('-' * 100)
    instnorm = paddle.nn.InstanceNorm1D(num_features=7, epsilon=1e-06,
        momentum=1 - 0.1)
    p_instnorm = PaddedInstanceNorm1d(7, eps=1e-06)
    x = paddle.randn(shape=[3, 7, 11])
    lengths = paddle.to_tensor(data=[3, 9, 11], dtype='int64')
    x[(0), :, 3:] = 0.0
    x[(1), :, 9:] = 0.0
    y0 = instnorm(x[(0), :, :3].unsqueeze(axis=0))
    y1 = instnorm(x[(1), :, :9].unsqueeze(axis=0))
    y2 = instnorm(x[2].unsqueeze(axis=0))
    p_y = p_instnorm(x, lengths)
    print(y0 - p_y[0][:, :3] < 1e-06)
    print(y1 - p_y[1][:, :9] < 1e-06)
    print(y2 - p_y[2] < 1e-06)
    print(y.shape)
    print(p_y.shape)
