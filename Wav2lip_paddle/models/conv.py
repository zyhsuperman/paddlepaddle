import paddle


class Conv2d(paddle.nn.Layer):

    def __init__(self, cin, cout, kernel_size, stride, padding, residual=
        False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels
            =cin, out_channels=cout, kernel_size=kernel_size, stride=stride,
            padding=padding), paddle.nn.BatchNorm2D(num_features=cout))
        self.act = paddle.nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class nonorm_Conv2d(paddle.nn.Layer):

    def __init__(self, cin, cout, kernel_size, stride, padding, residual=
        False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels
            =cin, out_channels=cout, kernel_size=kernel_size, stride=stride,
            padding=padding))
        self.act = paddle.nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)


class Conv2dTranspose(paddle.nn.Layer):

    def __init__(self, cin, cout, kernel_size, stride, padding,
        output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = paddle.nn.Sequential(paddle.nn.Conv2DTranspose(
            in_channels=cin, out_channels=cout, kernel_size=kernel_size,
            stride=stride, padding=padding, output_padding=output_padding),
            paddle.nn.BatchNorm2D(num_features=cout))
        self.act = paddle.nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)
