import sys
import paddle


class D_VECTOR(paddle.nn.Layer):
    """d vector speaker embedding."""

    def __init__(self, num_layers=3, dim_input=40, dim_cell=256, dim_emb=64):
        super(D_VECTOR, self).__init__()
        # 定义 LSTM
        self.lstm = paddle.nn.LSTM(input_size=dim_input, hidden_size=dim_cell, 
                            num_layers=num_layers, time_major=False)

        # 定义线性层（全连接层）
        self.embedding = paddle.nn.Linear(in_features=dim_cell, out_features=dim_emb)

    def forward(self, x):
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)
        embeds = self.embedding(lstm_out[:, (-1), :])
        norm = embeds.norm(p=2, axis=-1, keepdim=True)
        embeds_normalized = paddle.divide(embeds, norm)
        return embeds_normalized
