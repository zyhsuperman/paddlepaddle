import sys
sys.path.append('/home/zyhao/paddlepaddle/Wav2lip_paadle/utils')
import paddle


# class L2Norm(paddle.nn.Layer):

#     def __init__(self, n_channels, scale=1.0):
#         super(L2Norm, self).__init__()
#         self.n_channels = n_channels
#         self.scale = scale
#         self.eps = 1e-10
#         out_0 = paddle.create_parameter(shape=paddle.to_tensor(data=self.
#             n_channels, dtype='float32').shape, dtype=paddle.to_tensor(data
#             =self.n_channels, dtype='float32').numpy().dtype,
#             default_initializer=paddle.nn.initializer.Assign(paddle.
#             to_tensor(data=self.n_channels, dtype='float32')))
#         out_0.stop_gradient = not True
#         self.weight = out_0
#         self.weight.data *= 0.0
#         self.weight.data += self.scale

#     def forward(self, x):
#         norm = x.pow(y=2).sum(axis=1, keepdim=True).sqrt() + self.eps
#         """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
#         x = x / norm * self.weight.reshape((1, -1, 1, 1))
#         return x

class L2Norm(paddle.nn.Layer):
    def __init__(self, n_channels, scale=1.0):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.scale = scale
        self.eps = 1e-10
        self.weight = self.create_parameter(shape=[self.n_channels], 
                                            default_initializer=paddle.nn.initializer.Assign(paddle.full([self.n_channels], self.scale)))

    def forward(self, x):
        norm = paddle.sqrt(x.pow(2).sum(axis=1, keepdim=True) + self.eps)
        x = x / norm * self.weight.reshape([1, -1, 1, 1])
        return x


class s3fd(paddle.nn.Layer):

    def __init__(self):
        super(s3fd, self).__init__()
        self.conv1_1 = paddle.nn.Conv2D(in_channels=3, out_channels=64,
            kernel_size=3, stride=1, padding=1)
        self.conv1_2 = paddle.nn.Conv2D(in_channels=64, out_channels=64,
            kernel_size=3, stride=1, padding=1)
        self.conv2_1 = paddle.nn.Conv2D(in_channels=64, out_channels=128,
            kernel_size=3, stride=1, padding=1)
        self.conv2_2 = paddle.nn.Conv2D(in_channels=128, out_channels=128,
            kernel_size=3, stride=1, padding=1)
        self.conv3_1 = paddle.nn.Conv2D(in_channels=128, out_channels=256,
            kernel_size=3, stride=1, padding=1)
        self.conv3_2 = paddle.nn.Conv2D(in_channels=256, out_channels=256,
            kernel_size=3, stride=1, padding=1)
        self.conv3_3 = paddle.nn.Conv2D(in_channels=256, out_channels=256,
            kernel_size=3, stride=1, padding=1)
        self.conv4_1 = paddle.nn.Conv2D(in_channels=256, out_channels=512,
            kernel_size=3, stride=1, padding=1)
        self.conv4_2 = paddle.nn.Conv2D(in_channels=512, out_channels=512,
            kernel_size=3, stride=1, padding=1)
        self.conv4_3 = paddle.nn.Conv2D(in_channels=512, out_channels=512,
            kernel_size=3, stride=1, padding=1)
        self.conv5_1 = paddle.nn.Conv2D(in_channels=512, out_channels=512,
            kernel_size=3, stride=1, padding=1)
        self.conv5_2 = paddle.nn.Conv2D(in_channels=512, out_channels=512,
            kernel_size=3, stride=1, padding=1)
        self.conv5_3 = paddle.nn.Conv2D(in_channels=512, out_channels=512,
            kernel_size=3, stride=1, padding=1)
        self.fc6 = paddle.nn.Conv2D(in_channels=512, out_channels=1024,
            kernel_size=3, stride=1, padding=3)
        self.fc7 = paddle.nn.Conv2D(in_channels=1024, out_channels=1024,
            kernel_size=1, stride=1, padding=0)
        self.conv6_1 = paddle.nn.Conv2D(in_channels=1024, out_channels=256,
            kernel_size=1, stride=1, padding=0)
        self.conv6_2 = paddle.nn.Conv2D(in_channels=256, out_channels=512,
            kernel_size=3, stride=2, padding=1)
        self.conv7_1 = paddle.nn.Conv2D(in_channels=512, out_channels=128,
            kernel_size=1, stride=1, padding=0)
        self.conv7_2 = paddle.nn.Conv2D(in_channels=128, out_channels=256,
            kernel_size=3, stride=2, padding=1)
        self.conv3_3_norm = L2Norm(256, scale=10)
        self.conv4_3_norm = L2Norm(512, scale=8)
        self.conv5_3_norm = L2Norm(512, scale=5)
        self.conv3_3_norm_mbox_conf = paddle.nn.Conv2D(in_channels=256,
            out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv3_3_norm_mbox_loc = paddle.nn.Conv2D(in_channels=256,
            out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv4_3_norm_mbox_conf = paddle.nn.Conv2D(in_channels=512,
            out_channels=2, kernel_size=3, stride=1, padding=1)
        self.conv4_3_norm_mbox_loc = paddle.nn.Conv2D(in_channels=512,
            out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv5_3_norm_mbox_conf = paddle.nn.Conv2D(in_channels=512,
            out_channels=2, kernel_size=3, stride=1, padding=1)
        self.conv5_3_norm_mbox_loc = paddle.nn.Conv2D(in_channels=512,
            out_channels=4, kernel_size=3, stride=1, padding=1)
        self.fc7_mbox_conf = paddle.nn.Conv2D(in_channels=1024,
            out_channels=2, kernel_size=3, stride=1, padding=1)
        self.fc7_mbox_loc = paddle.nn.Conv2D(in_channels=1024, out_channels
            =4, kernel_size=3, stride=1, padding=1)
        self.conv6_2_mbox_conf = paddle.nn.Conv2D(in_channels=512,
            out_channels=2, kernel_size=3, stride=1, padding=1)
        self.conv6_2_mbox_loc = paddle.nn.Conv2D(in_channels=512,
            out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv7_2_mbox_conf = paddle.nn.Conv2D(in_channels=256,
            out_channels=2, kernel_size=3, stride=1, padding=1)
        self.conv7_2_mbox_loc = paddle.nn.Conv2D(in_channels=256,
            out_channels=4, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = paddle.nn.functional.relu(x=self.conv1_1(x))
        h = paddle.nn.functional.relu(x=self.conv1_2(h))
        h = paddle.nn.functional.max_pool2d(x=h, kernel_size=2, stride=2)
        h = paddle.nn.functional.relu(x=self.conv2_1(h))
        h = paddle.nn.functional.relu(x=self.conv2_2(h))
        h = paddle.nn.functional.max_pool2d(x=h, kernel_size=2, stride=2)
        h = paddle.nn.functional.relu(x=self.conv3_1(h))
        h = paddle.nn.functional.relu(x=self.conv3_2(h))
        h = paddle.nn.functional.relu(x=self.conv3_3(h))
        f3_3 = h
        h = paddle.nn.functional.max_pool2d(x=h, kernel_size=2, stride=2)
        h = paddle.nn.functional.relu(x=self.conv4_1(h))
        h = paddle.nn.functional.relu(x=self.conv4_2(h))
        h = paddle.nn.functional.relu(x=self.conv4_3(h))
        f4_3 = h
        h = paddle.nn.functional.max_pool2d(x=h, kernel_size=2, stride=2)
        h = paddle.nn.functional.relu(x=self.conv5_1(h))
        h = paddle.nn.functional.relu(x=self.conv5_2(h))
        h = paddle.nn.functional.relu(x=self.conv5_3(h))
        f5_3 = h
        h = paddle.nn.functional.max_pool2d(x=h, kernel_size=2, stride=2)
        h = paddle.nn.functional.relu(x=self.fc6(h))
        h = paddle.nn.functional.relu(x=self.fc7(h))
        ffc7 = h
        h = paddle.nn.functional.relu(x=self.conv6_1(h))
        h = paddle.nn.functional.relu(x=self.conv6_2(h))
        f6_2 = h
        h = paddle.nn.functional.relu(x=self.conv7_1(h))
        h = paddle.nn.functional.relu(x=self.conv7_2(h))
        f7_2 = h
        f3_3 = self.conv3_3_norm(f3_3)
        f4_3 = self.conv4_3_norm(f4_3)
        f5_3 = self.conv5_3_norm(f5_3)
        cls1 = self.conv3_3_norm_mbox_conf(f3_3)
        reg1 = self.conv3_3_norm_mbox_loc(f3_3)
        cls2 = self.conv4_3_norm_mbox_conf(f4_3)
        reg2 = self.conv4_3_norm_mbox_loc(f4_3)
        cls3 = self.conv5_3_norm_mbox_conf(f5_3)
        reg3 = self.conv5_3_norm_mbox_loc(f5_3)
        cls4 = self.fc7_mbox_conf(ffc7)
        reg4 = self.fc7_mbox_loc(ffc7)
        cls5 = self.conv6_2_mbox_conf(f6_2)
        reg5 = self.conv6_2_mbox_loc(f6_2)
        cls6 = self.conv7_2_mbox_conf(f7_2)
        reg6 = self.conv7_2_mbox_loc(f7_2)
        chunk = paddle.chunk(x=cls1, chunks=4, axis=1)
        bmax = paddle.maximum(paddle.maximum(chunk[0], chunk[1]), chunk[2])
        cls1 = paddle.concat(x=[bmax, chunk[3]], axis=1)
        return [cls1, reg1, cls2, reg2, cls3, reg3, cls4, reg4, cls5, reg5,
            cls6, reg6]
