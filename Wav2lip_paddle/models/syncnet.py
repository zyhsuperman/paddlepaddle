import paddle
from .conv import Conv2d


class SyncNet_color(paddle.nn.Layer):

    def __init__(self):
        super(SyncNet_color, self).__init__()
        self.face_encoder = paddle.nn.Sequential(Conv2d(15, 32, kernel_size
            =(7, 7), stride=1, padding=3), Conv2d(32, 64, kernel_size=5,
            stride=(1, 2), padding=1), Conv2d(64, 64, kernel_size=3, stride
            =1, padding=1, residual=True), Conv2d(64, 64, kernel_size=3,
            stride=1, padding=1, residual=True), Conv2d(64, 128,
            kernel_size=3, stride=2, padding=1), Conv2d(128, 128,
            kernel_size=3, stride=1, padding=1, residual=True), Conv2d(128,
            128, kernel_size=3, stride=1, padding=1, residual=True), Conv2d
            (128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 256, kernel_size=3, stride=2, padding=1), Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=
            True), Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=
            True), Conv2d(512, 512, kernel_size=3, stride=1, padding=1,
            residual=True), Conv2d(512, 512, kernel_size=3, stride=2,
            padding=1), Conv2d(512, 512, kernel_size=3, stride=1, padding=0
            ), Conv2d(512, 512, kernel_size=1, stride=1, padding=0))
        self.audio_encoder = paddle.nn.Sequential(Conv2d(1, 32, kernel_size
            =3, stride=1, padding=1), Conv2d(32, 32, kernel_size=3, stride=
            1, padding=1, residual=True), Conv2d(32, 32, kernel_size=3,
            stride=1, padding=1, residual=True), Conv2d(32, 64, kernel_size
            =3, stride=(3, 1), padding=1), Conv2d(64, 64, kernel_size=3,
            stride=1, padding=1, residual=True), Conv2d(64, 64, kernel_size
            =3, stride=1, padding=1, residual=True), Conv2d(64, 128,
            kernel_size=3, stride=3, padding=1), Conv2d(128, 128,
            kernel_size=3, stride=1, padding=1, residual=True), Conv2d(128,
            128, kernel_size=3, stride=1, padding=1, residual=True), Conv2d
            (128, 256, kernel_size=3, stride=(3, 2), padding=1), Conv2d(256,
            256, kernel_size=3, stride=1, padding=1, residual=True), Conv2d
            (256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 512, kernel_size=3, stride=1, padding=0), Conv2d(
            512, 512, kernel_size=1, stride=1, padding=0))

    def forward(self, audio_sequences, face_sequences):
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)
        """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
        audio_embedding = audio_embedding.reshape((audio_embedding.shape[0], -1))
        """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
        face_embedding = face_embedding.reshape((face_embedding.shape[0], -1))
        audio_embedding = paddle.nn.functional.normalize(x=audio_embedding,
            p=2, axis=1)
        face_embedding = paddle.nn.functional.normalize(x=face_embedding, p
            =2, axis=1)
        return audio_embedding, face_embedding
