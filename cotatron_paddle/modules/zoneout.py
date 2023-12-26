import paddle


class ZoneoutLSTMCell(paddle.nn.Layer):

    def __init__(self, input_size, hidden_size, bias=True, zoneout_prob=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.zoneout_prob = zoneout_prob
        self.lstm = paddle.nn.LSTMCell(input_size=input_size, hidden_size=
            hidden_size, bias_ih_attr=bias, bias_hh_attr=bias)
        self.dropout = paddle.nn.Dropout(p=zoneout_prob)
        self.lstm.bias_ih[hidden_size:2 * hidden_size].data.fill_(1.0)
        self.lstm.bias_hh[hidden_size:2 * hidden_size].data.fill_(1.0)

    def forward(self, x, prev_hc=None):
        _, (h, c) = self.lstm(x, prev_hc)
        if prev_hc is None:
            prev_h = paddle.zeros(shape=[x.shape[0], self.hidden_size])
            prev_c = paddle.zeros(shape=[x.shape[0], self.hidden_size])
        else:
            prev_h, prev_c = prev_hc
        if self.training:
            
            h = (1.0 - self.zoneout_prob) * self.dropout(h - prev_h) + prev_h
            c = (1.0 - self.zoneout_prob) * self.dropout(c - prev_c) + prev_c
            
        else:
            h = (1.0 - self.zoneout_prob) * h + self.zoneout_prob * prev_h
            c = (1.0 - self.zoneout_prob) * c + self.zoneout_prob * prev_c
        return h, c
