import paddle


class ConditionalBatchNorm1d(paddle.nn.Layer):

    def __init__(self, num_features, condition_dim):
        super().__init__()
        self.num_features = num_features
        self.bn = paddle.nn.BatchNorm1D(num_features=num_features,
            weight_attr=None if False else False, bias_attr=None if False else
            False)
        self.projection = paddle.nn.Linear(in_features=condition_dim,
            out_features=2 * num_features)

    def forward(self, x, cond):
        x = self.bn(x)
        gamma, beta = self.projection(cond).chunk(chunks=2, axis=1)
        x = gamma.unsqueeze(axis=-1) * x + beta.unsqueeze(axis=-1)
        return x
