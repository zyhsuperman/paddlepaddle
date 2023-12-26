import paddle
from .res_stack import ResStack
MAX_WAV_VALUE = 32768.0


class Generator(paddle.nn.Layer):

    def __init__(self, mel_channel):
        super(Generator, self).__init__()
        self.mel_channel = mel_channel
        self.generator = paddle.nn.Sequential(paddle.nn.Pad1D(padding=3,
            mode='reflect'), paddle.nn.utils.weight_norm(layer=paddle.nn.
            Conv1D(in_channels=mel_channel, out_channels=512, kernel_size=7,
            stride=1)), paddle.nn.LeakyReLU(negative_slope=0.2), paddle.nn.
            utils.weight_norm(layer=paddle.nn.Conv1DTranspose(in_channels=
            512, out_channels=256, kernel_size=16, stride=8, padding=4)),
            ResStack(256), paddle.nn.LeakyReLU(negative_slope=0.2), paddle.
            nn.utils.weight_norm(layer=paddle.nn.Conv1DTranspose(
            in_channels=256, out_channels=128, kernel_size=16, stride=8,
            padding=4)), ResStack(128), paddle.nn.LeakyReLU(negative_slope=
            0.2), paddle.nn.utils.weight_norm(layer=paddle.nn.
            Conv1DTranspose(in_channels=128, out_channels=64, kernel_size=4,
            stride=2, padding=1)), ResStack(64), paddle.nn.LeakyReLU(
            negative_slope=0.2), paddle.nn.utils.weight_norm(layer=paddle.
            nn.Conv1DTranspose(in_channels=64, out_channels=32, kernel_size
            =4, stride=2, padding=1)), ResStack(32), paddle.nn.LeakyReLU(
            negative_slope=0.2), paddle.nn.Pad1D(padding=3, mode='reflect'),
            paddle.nn.utils.weight_norm(layer=paddle.nn.Conv1D(in_channels=
            32, out_channels=1, kernel_size=7, stride=1)), paddle.nn.Tanh())

    def forward(self, mel):
        return self.generator(mel)

    def eval(self, inference=False):
        super(Generator, self).eval()
        if inference:
            self.remove_weight_norm()

    def remove_weight_norm(self):
        for idx, layer in enumerate(self.generator):
            if len(layer.state_dict()) != 0:
                try:
                    paddle.nn.utils.remove_weight_norm(layer=layer)
                except:
                    layer.remove_weight_norm()

    def inference(self, mel):
        hop_length = 256
        audio = self.forward(mel)
        audio = audio.squeeze()
        audio = MAX_WAV_VALUE * audio
        audio = audio.clip(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE - 1)
        audio = audio.astype(dtype='int16')
        return audio


"""
    to run this, fix 
    from . import ResStack
    into
    from res_stack import ResStack
"""
if __name__ == '__main__':
    model = Generator(80)
    x = paddle.randn(shape=[3, 80, 10])
    print(x.shape)
    y = model(x)
    print(y.shape)
    assert y.shape == list([3, 1, 2560])
    total_params = sum(p.size for p in model.parameters() if not p.
        stop_gradient)
    print(total_params)
