import paddle


class SpkClassifier(paddle.nn.Layer):

    def __init__(self, hp):
        super().__init__()
        self.mlp = paddle.nn.Sequential(paddle.nn.ReLU(), paddle.nn.Dropout
            (p=0.5), paddle.nn.Linear(in_features=hp.chn.speaker.token,
            out_features=hp.chn.speaker.token), paddle.nn.ReLU(), paddle.nn
            .Dropout(p=0.5), paddle.nn.Linear(in_features=hp.chn.speaker.
            token, out_features=len(hp.data.speakers)))

    def forward(self, x):
        x = self.mlp(x)
        x = paddle.nn.functional.log_softmax(x=x, axis=1)
        return x
