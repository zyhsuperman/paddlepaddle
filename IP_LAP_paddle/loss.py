import paddle
import math
import paddle.vision.models as models

def apply_imagenet_normalization(input):
    """Normalize using ImageNet mean and std.

    Args:
        input (4D tensor NxCxHxW): The input images, assuming to be [-1, 1].

    Returns:
        Normalized inputs using the ImageNet normalization.
    """
    normalized_input = (input + 1) / 2
    mean = paddle.to_tensor(data=[0.485, 0.456, 0.406], dtype=
        normalized_input.dtype).reshape([1, 3, 1, 1])
    std = paddle.to_tensor(data=[0.229, 0.224, 0.225], dtype=
        normalized_input.dtype).reshape([1, 3, 1, 1])
    output = (normalized_input - mean) / std
    return output


class PerceptualLoss(paddle.nn.Layer):
    """Perceptual loss initialization.

    Args:
        network (str) : The name of the loss network: 'vgg16' | 'vgg19'.
        layers (str or list of str) : The layers used to compute the loss.
        weights (float or list of float : The loss weights of each layer.
        criterion (str): The type of distance function: 'l1' | 'l2'.
        resize (bool) : If ``True``, resize the input images to 224x224.
        resize_mode (str): Algorithm used for resizing.
        instance_normalized (bool): If ``True``, applies instance normalization
            to the feature maps before computing the distance.
        num_scales (int): The loss will be evaluated at original size and
            this many times downsampled sizes.
    """

    def __init__(self, network='vgg19', layers='relu_4_1', weights=None,
        criterion='l1', resize=False, resize_mode='bilinear',
        instance_normalized=False, num_scales=1):
        super().__init__()
        if isinstance(layers, str):
            layers = [layers]
        if weights is None:
            weights = [1.0] * len(layers)
        elif isinstance(layers, float) or isinstance(layers, int):
            weights = [weights]
        assert len(layers) == len(weights
            ), 'The number of layers (%s) must be equal to the number of weights (%s).' % (
            len(layers), len(weights))
        if network == 'vgg19':
            self.model = _vgg19(layers)
        elif network == 'vgg16':
            self.model = _vgg16(layers)
        # elif network == 'alexnet':
        #     self.model = _alexnet(layers)
        # elif network == 'inception_v3':
        #     self.model = _inception_v3(layers)
        # elif network == 'resnet50':
        #     self.model = _resnet50(layers)
        # elif network == 'robust_resnet50':
        #     self.model = _robust_resnet50(layers)
        # elif network == 'vgg_face_dag':
        #     self.model = _vgg_face_dag(layers)
        else:
            raise ValueError('Network %s is not recognized' % network)
        self.num_scales = num_scales
        self.layers = layers
        self.weights = weights
        if criterion == 'l1':
            self.criterion = paddle.nn.L1Loss()
        elif criterion == 'l2' or criterion == 'mse':
            self.criterion = paddle.nn.MSELoss()
        else:
            raise ValueError('Criterion %s is not recognized' % criterion)
        self.resize = resize
        self.resize_mode = resize_mode
        self.instance_normalized = instance_normalized
        print('Perceptual loss:')
        print('\tMode: {}'.format(network))

    def forward(self, inp_source, target_source, mask=None, use_style_loss=
        False, weight_style_to_perceptual=0.0, warp=False):
        inp = inp_source.clone()
        target = target_source.clone()
        if warp:
            inp[:, :, 0:96 // 2, :] = 0
            target[:, :, 0:96 // 2, :] = 0
        """Perceptual loss forward.

        Args:
           inp (4D tensor) : Input tensor.
           target (4D tensor) : Ground truth tensor, same shape as the input.

        Returns:
           (scalar tensor) : The perceptual loss.
        """
        self.model.eval()
        inp, target = apply_imagenet_normalization(inp
            ), apply_imagenet_normalization(target)
        if self.resize:
            inp = paddle.nn.functional.interpolate(x=inp, mode=self.
                resize_mode, size=(256, 256), align_corners=False)
            target = paddle.nn.functional.interpolate(x=target, mode=self.
                resize_mode, size=(256, 256), align_corners=False)
        loss = 0
        style_loss = 0
        for scale in range(self.num_scales):
            input_features, target_features = self.model(inp), self.model(
                target)
            for layer, weight in zip(self.layers, self.weights):
                input_feature = input_features[layer]
                target_feature = target_features[layer].detach()
                if self.instance_normalized:
                    input_feature = paddle.nn.functional.instance_norm(x=
                        input_feature)
                    target_feature = paddle.nn.functional.instance_norm(x=
                        target_feature)
                if mask is not None:
                    mask_ = paddle.nn.functional.interpolate(x=mask, size=
                        input_feature.shape[2:], mode='bilinear',
                        align_corners=False)
                    input_feature = input_feature * mask_
                    target_feature = target_feature * mask_
                loss += weight * self.criterion(input_feature, target_feature)
                if use_style_loss and scale == 0:
                    style_loss += self.criterion(self.compute_gram(
                        input_feature), self.compute_gram(target_feature))
            if scale != self.num_scales - 1:
                inp = paddle.nn.functional.interpolate(
                    inp, mode=self.resize_mode, scale_factor=0.5, align_corners=False)
                target = paddle.nn.functional.interpolate(
                    target, mode=self.resize_mode, scale_factor=0.5, align_corners=False)
        if use_style_loss:
            return loss + style_loss * weight_style_to_perceptual
        else:
            return loss

    def compute_gram(self, x):
        b, ch, h, w = x.shape
        f = x.reshape([b, ch, w * h])
        x = f
        perm_1 = list(range(x.ndim))
        perm_1[1] = 2
        perm_1[2] = 1
        f_T = x.transpose(perm=perm_1)
        G = f.bmm(y=f_T) / (h * w * ch)
        return G


class _PerceptualNetwork(paddle.nn.Layer):
    """The network that extracts features to compute the perceptual loss.

    Args:
        network (nn.Sequential) : The network that extracts features.
        layer_name_mapping (dict) : The dictionary that
            maps a layer's index to its name.
        layers (list of str): The list of layer names that we are using.
    """

    def __init__(self, network, layer_name_mapping, layers):
        super().__init__()
        assert isinstance(network, paddle.nn.Sequential
            ), 'The network needs to be of type "nn.Sequential".'
        self.network = network
        self.layer_name_mapping = layer_name_mapping
        self.layers = layers
        for param in self.parameters():
            param.stop_gradient = not False

    def forward(self, x):
        """Extract perceptual features."""
        output = {}
        for i, layer in enumerate(self.network):
            x = layer(x)
            layer_name = self.layer_name_mapping.get(i, None)
            if layer_name in self.layers:
                output[layer_name] = x
        return output


# def _vgg19(layers):
#     """Get vgg19 layers"""
# >>>>>>    network = torchvision.models.vgg19(pretrained=True).features
#     layer_name_mapping = {(1): 'relu_1_1', (3): 'relu_1_2', (6): 'relu_2_1',
#         (8): 'relu_2_2', (11): 'relu_3_1', (13): 'relu_3_2', (15):
#         'relu_3_3', (17): 'relu_3_4', (20): 'relu_4_1', (22): 'relu_4_2', (
#         24): 'relu_4_3', (26): 'relu_4_4', (29): 'relu_5_1'}
#     return _PerceptualNetwork(network, layer_name_mapping, layers)

def _vgg19(layers):
    """获取 VGG19 网络的层"""
    network = paddle.vision.models.vgg19(pretrained=True).features
    layer_name_mapping = {
        1: 'relu_1_1', 3: 'relu_1_2', 6: 'relu_2_1',
        8: 'relu_2_2', 11: 'relu_3_1', 13: 'relu_3_2',
        15: 'relu_3_3', 17: 'relu_3_4', 20: 'relu_4_1',
        22: 'relu_4_2', 24: 'relu_4_3', 26: 'relu_4_4', 29: 'relu_5_1'
    }
    return _PerceptualNetwork(network, layer_name_mapping, layers)


# def _vgg16(layers):
#     """Get vgg16 layers"""
# >>>>>>    network = torchvision.models.vgg16(pretrained=True).features
#     layer_name_mapping = {(1): 'relu_1_1', (3): 'relu_1_2', (6): 'relu_2_1',
#         (8): 'relu_2_2', (11): 'relu_3_1', (13): 'relu_3_2', (15):
#         'relu_3_3', (18): 'relu_4_1', (20): 'relu_4_2', (22): 'relu_4_3', (
#         25): 'relu_5_1'}
#     return _PerceptualNetwork(network, layer_name_mapping, layers)

# def _vgg16(layers):
#     """获取 VGG19 网络的层"""
#     network = paddle.vision.models.vgg16(pretrained=True).features
#     layer_name_mapping = {
#         1: 'relu_1_1', 3: 'relu_1_2', 6: 'relu_2_1',
#         8: 'relu_2_2', 11: 'relu_3_1', 13: 'relu_3_2',
#         15: 'relu_3_3', 18: 'relu_4_1', 20: 'relu_4_2',
#         22: 'relu_4_3', 25: 'relu_5_1'
#     }
#     return _PerceptualNetwork(network, layer_name_mapping, layers)

def _vgg16(layers):
    network = models.vgg16(pretrained=True).features
    layer_name_mapping = {
        1: 'relu_1_1',
        3: 'relu_1_2',
        6: 'relu_2_1',
        8: 'relu_2_2',
        11: 'relu_3_1',
        13: 'relu_3_2',
        15: 'relu_3_3',
        18: 'relu_4_1',
        20: 'relu_4_2',
        22: 'relu_4_3',
        25: 'relu_5_1'
    }
    return _PerceptualNetwork(network, layer_name_mapping, layers)


# def _alexnet(layers):
#     """Get alexnet layers"""
# >>>>>>    network = torchvision.models.alexnet(pretrained=True).features
#     layer_name_mapping = {(0): 'conv_1', (1): 'relu_1', (3): 'conv_2', (4):
#         'relu_2', (6): 'conv_3', (7): 'relu_3', (8): 'conv_4', (9):
#         'relu_4', (10): 'conv_5', (11): 'relu_5'}
#     return _PerceptualNetwork(network, layer_name_mapping, layers)


# def _inception_v3(layers):
#     """Get inception v3 layers"""
# >>>>>>    inception = torchvision.models.inception_v3(pretrained=True)
#     network = paddle.nn.Sequential(inception.Conv2d_1a_3x3, inception.
#         Conv2d_2a_3x3, inception.Conv2d_2b_3x3, paddle.nn.MaxPool2D(
#         kernel_size=3, stride=2), inception.Conv2d_3b_1x1, inception.
#         Conv2d_4a_3x3, paddle.nn.MaxPool2D(kernel_size=3, stride=2),
#         inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d,
#         inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c,
#         inception.Mixed_6d, inception.Mixed_6e, inception.Mixed_7a,
#         inception.Mixed_7b, inception.Mixed_7c, paddle.nn.AdaptiveAvgPool2D
#         (output_size=(1, 1)))
#     layer_name_mapping = {(3): 'pool_1', (6): 'pool_2', (14): 'mixed_6e', (
#         18): 'pool_3'}
#     return _PerceptualNetwork(network, layer_name_mapping, layers)


# def _resnet50(layers):
#     """Get resnet50 layers"""
# >>>>>>    resnet50 = torchvision.models.resnet50(pretrained=True)
#     network = paddle.nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.
#         relu, resnet50.maxpool, resnet50.layer1, resnet50.layer2, resnet50.
#         layer3, resnet50.layer4, resnet50.avgpool)
#     layer_name_mapping = {(4): 'layer_1', (5): 'layer_2', (6): 'layer_3', (
#         7): 'layer_4'}
#     return _PerceptualNetwork(network, layer_name_mapping, layers)


# def _robust_resnet50(layers):
#     """Get robust resnet50 layers"""
# >>>>>>    resnet50 = torchvision.models.resnet50(pretrained=False)
# >>>>>>    state_dict = torch.utils.model_zoo.load_url(
#         'http://andrewilyas.com/ImageNet.pt')
#     new_state_dict = {}
#     for k, v in state_dict['model'].items():
#         if k.startswith('module.model.'):
#             new_state_dict[k[13:]] = v
#     resnet50.set_state_dict(state_dict=new_state_dict)
#     network = paddle.nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.
#         relu, resnet50.maxpool, resnet50.layer1, resnet50.layer2, resnet50.
#         layer3, resnet50.layer4, resnet50.avgpool)
#     layer_name_mapping = {(4): 'layer_1', (5): 'layer_2', (6): 'layer_3', (
#         7): 'layer_4'}
#     return _PerceptualNetwork(network, layer_name_mapping, layers)


# def _vgg_face_dag(layers):
#     """Get vgg face layers"""
# >>>>>>    network = torchvision.models.vgg16(num_classes=2622)
# >>>>>>    state_dict = torch.utils.model_zoo.load_url(
#         'http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.pth'
#         )
#     feature_layer_name_mapping = {(0): 'conv1_1', (2): 'conv1_2', (5):
#         'conv2_1', (7): 'conv2_2', (10): 'conv3_1', (12): 'conv3_2', (14):
#         'conv3_3', (17): 'conv4_1', (19): 'conv4_2', (21): 'conv4_3', (24):
#         'conv5_1', (26): 'conv5_2', (28): 'conv5_3'}
#     new_state_dict = {}
#     for k, v in feature_layer_name_mapping.items():
#         new_state_dict['features.' + str(k) + '.weight'] = state_dict[v +
#             '.weight']
#         new_state_dict['features.' + str(k) + '.bias'] = state_dict[v + '.bias'
#             ]
#     classifier_layer_name_mapping = {(0): 'fc6', (3): 'fc7', (6): 'fc8'}
#     for k, v in classifier_layer_name_mapping.items():
#         new_state_dict['classifier.' + str(k) + '.weight'] = state_dict[v +
#             '.weight']
#         new_state_dict['classifier.' + str(k) + '.bias'] = state_dict[v +
#             '.bias']
#     network.set_state_dict(state_dict=new_state_dict)


#     class Flatten(paddle.nn.Layer):
#         """Flatten the tensor"""

#         def forward(self, x):
#             """Flatten it"""
#             return x.reshape([x.shape[0], -1])
#     layer_name_mapping = {(1): 'avgpool', (3): 'fc6', (4): 'relu_6', (6):
#         'fc7', (7): 'relu_7', (9): 'fc8'}
#     seq_layers = [network.features, network.avgpool, Flatten()]
#     for i in range(7):
#         seq_layers += [network.classifier[i]]
#     network = paddle.nn.Sequential(*seq_layers)
#     return _PerceptualNetwork(network, layer_name_mapping, layers)


class GANLoss(paddle.nn.Layer):

    def __init__(self, use_lsgan=True, target_real_label=1.0,
        target_fake_label=0.0, tensor=paddle.Tensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = paddle.nn.MSELoss()
        else:
            self.loss = paddle.nn.BCELoss()

    # def get_target_tensor(self, input, target_is_real):
    #     target_tensor = None
    #     if target_is_real:
    #         create_label = (self.real_label_var is None or self.
    #             real_label_var.size != input.size)
    #         if create_label:
    #             real_tensor = self.Tensor(input.shape).fill_(value=self.
    #                 real_label)
    #             self.real_label_var = paddle.to_tensor(real_tensor, stop_gradient=True)
    #         target_tensor = self.real_label_var
    #     else:
    #         create_label = (self.fake_label_var is None or self.
    #             fake_label_var.size != input.size)
    #         if create_label:
    #             fake_tensor = self.Tensor(input.shape).fill_(value=self.
    #                 fake_label)
    #             self.fake_label_var = paddle.to_tensor(fake_tensor, stop_gradient=True)
    #         target_tensor = self.fake_label_var
    #     return target_tensor
    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            # 检查是否需要创建新的标签张量
            create_label = (self.real_label_var is None or self.real_label_var.shape != input.shape)
            if create_label:
                # 创建一个与输入形状相同、填充为real_label的张量
                real_tensor = paddle.full(input.shape, self.real_label, dtype=input.dtype)
                self.real_label_var = real_tensor.detach()  # 使用detach()而不是stop_gradient=True
                self.real_label_var.stop_gradient = True   # 确保不计算梯度
            target_tensor = self.real_label_var
        else:
            # 检查是否需要创建新的标签张量
            create_label = (self.fake_label_var is None or self.fake_label_var.shape != input.shape)
            if create_label:
                # 创建一个与输入形状相同、填充为fake_label的张量
                fake_tensor = paddle.full(input.shape, self.fake_label, dtype=input.dtype)
                self.fake_label_var = fake_tensor.detach()
                self.fake_label_var.stop_gradient = True
            target_tensor = self.fake_label_var
        return target_tensor


    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)
