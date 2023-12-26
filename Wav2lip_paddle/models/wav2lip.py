import sys
sys.path.append('/home/zyhao/paddlepaddle/Wav2lip_paadle/utils')
import paddle
import math
from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d


class Wav2Lip(paddle.nn.Layer):

    def __init__(self):
        super(Wav2Lip, self).__init__()
        self.face_encoder_blocks = paddle.nn.LayerList(sublayers=[paddle.nn
            .Sequential(Conv2d(6, 16, kernel_size=7, stride=1, padding=3)),
            paddle.nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2,
            padding=1), Conv2d(32, 32, kernel_size=3, stride=1, padding=1,
            residual=True), Conv2d(32, 32, kernel_size=3, stride=1, padding
            =1, residual=True)), paddle.nn.Sequential(Conv2d(32, 64,
            kernel_size=3, stride=2, padding=1), Conv2d(64, 64, kernel_size
            =3, stride=1, padding=1, residual=True), Conv2d(64, 64,
            kernel_size=3, stride=1, padding=1, residual=True), Conv2d(64, 
            64, kernel_size=3, stride=1, padding=1, residual=True)), paddle
            .nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding
            =1), Conv2d(128, 128, kernel_size=3, stride=1, padding=1,
            residual=True), Conv2d(128, 128, kernel_size=3, stride=1,
            padding=1, residual=True)), paddle.nn.Sequential(Conv2d(128, 
            256, kernel_size=3, stride=2, padding=1), Conv2d(256, 256,
            kernel_size=3, stride=1, padding=1, residual=True), Conv2d(256,
            256, kernel_size=3, stride=1, padding=1, residual=True)),
            paddle.nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2,
            padding=1), Conv2d(512, 512, kernel_size=3, stride=1, padding=1,
            residual=True)), paddle.nn.Sequential(Conv2d(512, 512,
            kernel_size=3, stride=1, padding=0), Conv2d(512, 512,
            kernel_size=1, stride=1, padding=0))])
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
            (256, 512, kernel_size=3, stride=1, padding=0), Conv2d(512, 512,
            kernel_size=1, stride=1, padding=0))
        self.face_decoder_blocks = paddle.nn.LayerList(sublayers=[paddle.nn
            .Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0
            )), paddle.nn.Sequential(Conv2dTranspose(1024, 512, kernel_size
            =3, stride=1, padding=0), Conv2d(512, 512, kernel_size=3,
            stride=1, padding=1, residual=True)), paddle.nn.Sequential(
            Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1,
            output_padding=1), Conv2d(512, 512, kernel_size=3, stride=1,
            padding=1, residual=True), Conv2d(512, 512, kernel_size=3,
            stride=1, padding=1, residual=True)), paddle.nn.Sequential(
            Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1,
            output_padding=1), Conv2d(384, 384, kernel_size=3, stride=1,
            padding=1, residual=True), Conv2d(384, 384, kernel_size=3,
            stride=1, padding=1, residual=True)), paddle.nn.Sequential(
            Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1,
            output_padding=1), Conv2d(256, 256, kernel_size=3, stride=1,
            padding=1, residual=True), Conv2d(256, 256, kernel_size=3,
            stride=1, padding=1, residual=True)), paddle.nn.Sequential(
            Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1,
            output_padding=1), Conv2d(128, 128, kernel_size=3, stride=1,
            padding=1, residual=True), Conv2d(128, 128, kernel_size=3,
            stride=1, padding=1, residual=True)), paddle.nn.Sequential(
            Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1,
            output_padding=1), Conv2d(64, 64, kernel_size=3, stride=1,
            padding=1, residual=True), Conv2d(64, 64, kernel_size=3, stride
            =1, padding=1, residual=True))])
        self.output_block = paddle.nn.Sequential(Conv2d(80, 32, kernel_size
            =3, stride=1, padding=1), paddle.nn.Conv2D(in_channels=32,
            out_channels=3, kernel_size=1, stride=1, padding=0), paddle.nn.
            Sigmoid())

    def forward(self, audio_sequences, face_sequences):
        B = audio_sequences.shape[0]
        # print("+++++++++")
        # print(B)
        # print("---------")
        input_dim_size = len(face_sequences.shape)
        # print(input_dim_size)
        if input_dim_size > 4:
            audio_sequences = paddle.concat(x=[audio_sequences[:, i] for i in
                range(audio_sequences.shape[1])], axis=0)
            face_sequences = paddle.concat(x=[face_sequences[:, :, i] for i in
                range(face_sequences.shape[2])], axis=0)
        audio_embedding = self.audio_encoder(audio_sequences)
        # print(audio_embedding.shape)
        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)
        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = paddle.concat(x=(x, feats[-1]), axis=1)
            except Exception as e:
                print(x.shape)
                print(feats[-1].shape)
                raise e
            feats.pop()
        x = self.output_block(x)
        if input_dim_size > 4:
            num_or_sections = [B] * (face_sequences.shape[0] // B) 
            x = paddle.split(x, num_or_sections=num_or_sections, axis=0)
            # shapes = [split.shape for split in x]
            # print("Shapes of all tensors in list:", shapes)
            outputs = paddle.stack(x, axis=2) # (B, C, T, H, W)
        else:
            outputs = x
        return outputs


class Wav2Lip_disc_qual(paddle.nn.Layer):

    def __init__(self):
        super(Wav2Lip_disc_qual, self).__init__()
        self.face_encoder_blocks = paddle.nn.LayerList(sublayers=[paddle.nn
            .Sequential(nonorm_Conv2d(3, 32, kernel_size=7, stride=1,
            padding=3)), paddle.nn.Sequential(nonorm_Conv2d(32, 64,
            kernel_size=5, stride=(1, 2), padding=2), nonorm_Conv2d(64, 64,
            kernel_size=5, stride=1, padding=2)), paddle.nn.Sequential(
            nonorm_Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2)),
            paddle.nn.Sequential(nonorm_Conv2d(128, 256, kernel_size=5,
            stride=2, padding=2), nonorm_Conv2d(256, 256, kernel_size=5,
            stride=1, padding=2)), paddle.nn.Sequential(nonorm_Conv2d(256, 
            512, kernel_size=3, stride=2, padding=1), nonorm_Conv2d(512, 
            512, kernel_size=3, stride=1, padding=1)), paddle.nn.Sequential
            (nonorm_Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
            paddle.nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3,
            stride=1, padding=0), nonorm_Conv2d(512, 512, kernel_size=1,
            stride=1, padding=0))])
        self.binary_pred = paddle.nn.Sequential(paddle.nn.Conv2D(
            in_channels=512, out_channels=1, kernel_size=1, stride=1,
            padding=0), paddle.nn.Sigmoid())
        self.label_noise = 0.0

    def get_lower_half(self, face_sequences):
        return face_sequences[:, :, face_sequences.shape[2] // 2:]

    def to_2d(self, face_sequences):
        B = face_sequences.shape[0]
        face_sequences = paddle.concat(x=[face_sequences[:, :, i] for i in
            range(face_sequences.shape[2])], axis=0)
        return face_sequences

    def perceptual_forward(self, false_face_sequences):
        false_face_sequences = self.to_2d(false_face_sequences)
        false_face_sequences = self.get_lower_half(false_face_sequences)
        false_feats = false_face_sequences
        for f in self.face_encoder_blocks:
            false_feats = f(false_feats)
        """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
# >>>>>>        false_pred_loss = paddle.nn.functional.binary_cross_entropy(input=
#             self.binary_pred(false_feats).view(len(false_feats), -1), label
#             =paddle.ones(shape=(len(false_feats), 1)))
        false_pred_loss = paddle.nn.functional.binary_cross_entropy(self.binary_pred(false_feats).reshape([len(false_feats), -1]), 
                                         paddle.ones([len(false_feats), 1]).cuda())
        return false_pred_loss

    def forward(self, face_sequences):
        face_sequences = self.to_2d(face_sequences)
        face_sequences = self.get_lower_half(face_sequences)
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
        """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
        return self.binary_pred(x).reshape((len(x), -1))
