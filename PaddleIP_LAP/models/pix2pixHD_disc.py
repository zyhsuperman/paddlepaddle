import paddle
import functools
import numpy as np


# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find('BatchNorm2d') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         # 使用paddle.nn.initializer.Normal进行权重初始化
#         m.weight.set_value(paddle.nn.initializer.Normal(mean=0.0, std=0.02)(m.weight.shape))
#     elif classname.find('BatchNorm2d') != -1:
#         # 对于BatchNorm层，初始化gamma（权重）和beta（偏置）
#         m.weight.set_value(paddle.nn.initializer.Normal(mean=1.0, std=0.02)(m.weight.shape))
#         m.bias.set_value(paddle.zeros_like(m.bias))

def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and isinstance(m.weight, paddle.Tensor):
        if classname.find('Conv') != -1:
            m.weight.set_value(paddle.normal(mean=0.0, std=0.02, shape=m.weight.shape))
        elif classname.find('BatchNorm2D') != -1:
            m.weight.set_value(paddle.normal(mean=1.0, std=0.02, shape=m.weight.shape))
            m.bias.set_value(paddle.zeros(shape=m.bias.shape))


def define_D(input_nc=3, ndf=64, n_layers_D=3, norm='instance', use_sigmoid
    =False, num_D=2, getIntermFeat=True):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer,
        use_sigmoid, num_D, getIntermFeat)
    netD.apply(weights_init)
    return netD


class NLayerDiscriminator(paddle.nn.Layer):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=paddle.nn.
        BatchNorm2D, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[paddle.nn.Conv2D(in_channels=input_nc, out_channels=
            ndf, kernel_size=kw, stride=2, padding=padw), paddle.nn.
            LeakyReLU(negative_slope=0.2)]]
        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[paddle.nn.Conv2D(in_channels=nf_prev,
                out_channels=nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), paddle.nn.LeakyReLU(negative_slope=0.2)]]
        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[paddle.nn.Conv2D(in_channels=nf_prev, out_channels=nf,
            kernel_size=kw, stride=1, padding=padw), norm_layer(nf), paddle
            .nn.LeakyReLU(negative_slope=0.2)]]
        sequence += [[paddle.nn.Conv2D(in_channels=nf, out_channels=1,
            kernel_size=kw, stride=1, padding=padw)]]
        if use_sigmoid:
            sequence += [[paddle.nn.Sigmoid()]]
        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), paddle.nn.Sequential(*
                    sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = paddle.nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


# def get_norm_layer(norm_type='instance'):
#     if norm_type == 'batch':
#         norm_layer = functools.partial(paddle.nn.BatchNorm2D, affine=True)
#     elif norm_type == 'instance':
#         norm_layer = functools.partial(paddle.nn.InstanceNorm2D, affine=False)
#     else:
#         raise NotImplementedError('normalization layer [%s] is not found' %
#             norm_type)
#     return norm_layer

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(paddle.nn.BatchNorm2D)
    elif norm_type == 'instance':
        norm_layer = functools.partial(paddle.nn.InstanceNorm2D)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class MultiscaleDiscriminator(paddle.nn.Layer):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=paddle.nn.
        BatchNorm2D, use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer,
                use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j),
                        getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)
        self.downsample = paddle.nn.AvgPool2D(kernel_size=3, stride=2,
            padding=[1, 1], exclusive=not False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) +
                    '_layer' + str(j)) for j in range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != num_D - 1:
                input_downsampled = self.downsample(input_downsampled)
        return result
