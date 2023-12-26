import paddle
import math


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    """3x3 convolution with padding"""
    return paddle.nn.Conv2D(in_channels=in_planes, out_channels=out_planes,
        kernel_size=3, stride=strd, padding=padding, bias_attr=bias)


class ConvBlock(paddle.nn.Layer):

    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = paddle.nn.BatchNorm2D(num_features=in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = paddle.nn.BatchNorm2D(num_features=int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = paddle.nn.BatchNorm2D(num_features=int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))
        if in_planes != out_planes:
            self.downsample = paddle.nn.Sequential(paddle.nn.BatchNorm2D(
                num_features=in_planes), paddle.nn.ReLU(), paddle.nn.Conv2D
                (in_channels=in_planes, out_channels=out_planes,
                kernel_size=1, stride=1, bias_attr=False))
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out1 = self.bn1(x)
        out1 = paddle.nn.functional.relu(x=out1)
        out1 = self.conv1(out1)
        out2 = self.bn2(out1)
        out2 = paddle.nn.functional.relu(x=out2)
        out2 = self.conv2(out2)
        out3 = self.bn3(out2)
        out3 = paddle.nn.functional.relu(x=out3)
        out3 = self.conv3(out3)
        out3 = paddle.concat(x=(out1, out2, out3), axis=1)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out3 += residual
        return out3


class Bottleneck(paddle.nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=inplanes, out_channels=
            planes, kernel_size=1, bias_attr=False)
        self.bn1 = paddle.nn.BatchNorm2D(num_features=planes)
        self.conv2 = paddle.nn.Conv2D(in_channels=planes, out_channels=
            planes, kernel_size=3, stride=stride, padding=1, bias_attr=False)
        self.bn2 = paddle.nn.BatchNorm2D(num_features=planes)
        self.conv3 = paddle.nn.Conv2D(in_channels=planes, out_channels=
            planes * 4, kernel_size=1, bias_attr=False)
        self.bn3 = paddle.nn.BatchNorm2D(num_features=planes * 4)
        self.relu = paddle.nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class HourGlass(paddle.nn.Layer):

    def __init__(self, num_modules, depth, num_features):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_sublayer(name='b1_' + str(level), sublayer=ConvBlock(self.
            features, self.features))
        self.add_sublayer(name='b2_' + str(level), sublayer=ConvBlock(self.
            features, self.features))
        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_sublayer(name='b2_plus_' + str(level), sublayer=
                ConvBlock(self.features, self.features))
        self.add_sublayer(name='b3_' + str(level), sublayer=ConvBlock(self.
            features, self.features))

    def _forward(self, level, inp):
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)
        low1 = paddle.nn.functional.avg_pool2d(kernel_size=2, stride=2, x=
            inp, exclusive=False)
        low1 = self._modules['b2_' + str(level)](low1)
        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)
        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)
        up2 = paddle.nn.functional.interpolate(x=low3, scale_factor=2, mode
            ='nearest')
        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)


class FAN(paddle.nn.Layer):

    def __init__(self, num_modules=1):
        super(FAN, self).__init__()
        self.num_modules = num_modules
        self.conv1 = paddle.nn.Conv2D(in_channels=3, out_channels=64,
            kernel_size=7, stride=2, padding=3)
        self.bn1 = paddle.nn.BatchNorm2D(num_features=64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)
        for hg_module in range(self.num_modules):
            self.add_sublayer(name='m' + str(hg_module), sublayer=HourGlass
                (1, 4, 256))
            self.add_sublayer(name='top_m_' + str(hg_module), sublayer=
                ConvBlock(256, 256))
            self.add_sublayer(name='conv_last' + str(hg_module), sublayer=
                paddle.nn.Conv2D(in_channels=256, out_channels=256,
                kernel_size=1, stride=1, padding=0))
            self.add_sublayer(name='bn_end' + str(hg_module), sublayer=
                paddle.nn.BatchNorm2D(num_features=256))
            self.add_sublayer(name='l' + str(hg_module), sublayer=paddle.nn
                .Conv2D(in_channels=256, out_channels=68, kernel_size=1,
                stride=1, padding=0))
            if hg_module < self.num_modules - 1:
                self.add_sublayer(name='bl' + str(hg_module), sublayer=
                    paddle.nn.Conv2D(in_channels=256, out_channels=256,
                    kernel_size=1, stride=1, padding=0))
                self.add_sublayer(name='al' + str(hg_module), sublayer=
                    paddle.nn.Conv2D(in_channels=68, out_channels=256,
                    kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x = paddle.nn.functional.relu(x=self.bn1(self.conv1(x)))
        x = paddle.nn.functional.avg_pool2d(kernel_size=2, stride=2, x=self
            .conv2(x), exclusive=False)
        x = self.conv3(x)
        x = self.conv4(x)
        previous = x
        outputs = []
        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)](previous)
            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)
            ll = paddle.nn.functional.relu(x=self._modules['bn_end' + str(i
                )](self._modules['conv_last' + str(i)](ll)))
            tmp_out = self._modules['l' + str(i)](ll)
            outputs.append(tmp_out)
            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_
        return outputs


class ResNetDepth(paddle.nn.Layer):

    def __init__(self, block=Bottleneck, layers=[3, 8, 36, 3], num_classes=68):
        self.inplanes = 64
        super(ResNetDepth, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=3 + 68, out_channels=64,
            kernel_size=7, stride=2, padding=3, bias_attr=False)
        self.bn1 = paddle.nn.BatchNorm2D(num_features=64)
        self.relu = paddle.nn.ReLU()
        self.maxpool = paddle.nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = paddle.nn.AvgPool2D(kernel_size=7, exclusive=False)
        self.fc = paddle.nn.Linear(in_features=512 * block.expansion,
            out_features=num_classes)
        for m in self.sublayers():
            if isinstance(m, paddle.nn.Conv2D):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, paddle.nn.BatchNorm2D):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=
                self.inplanes, out_channels=planes * block.expansion,
                kernel_size=1, stride=stride, bias_attr=False), paddle.nn.
                BatchNorm2D(num_features=planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return paddle.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
        x = x.reshape((x.shape[0], -1))
        x = self.fc(x)
        return x
