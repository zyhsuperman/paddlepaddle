import paddle


class AdaINLayer(paddle.nn.Layer):

    def __init__(self, input_nc, modulation_nc):
        super().__init__()
        self.InstanceNorm2d = paddle.nn.InstanceNorm2D(num_features=
            input_nc, weight_attr=False, bias_attr=False, momentum=1 - 0.1)
        nhidden = 128
        use_bias = True
        self.mlp_shared = paddle.nn.Sequential(paddle.nn.Linear(in_features
            =modulation_nc, out_features=nhidden, bias_attr=use_bias),
            paddle.nn.ReLU())
        self.mlp_gamma = paddle.nn.Linear(in_features=nhidden, out_features
            =input_nc, bias_attr=use_bias)
        self.mlp_beta = paddle.nn.Linear(in_features=nhidden, out_features=
            input_nc, bias_attr=use_bias)

    def forward(self, input, modulation_input):
        normalized = self.InstanceNorm2d(input)
        modulation_input = modulation_input.reshape([modulation_input.shape[0], -1])
        actv = self.mlp_shared(modulation_input)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        gamma = gamma.reshape([*gamma.shape[:2], 1, 1])
        beta = beta.reshape([*beta.shape[:2], 1, 1])
        out = normalized * (1 + gamma) + beta
        return out


class AdaIN(paddle.nn.Layer):

    def __init__(self, input_channel, modulation_channel, kernel_size=3,
        stride=1, padding=1):
        super(AdaIN, self).__init__()
        self.conv_1 = paddle.nn.Conv2D(in_channels=input_channel,
            out_channels=input_channel, kernel_size=kernel_size, stride=
            stride, padding=padding)
        self.conv_2 = paddle.nn.Conv2D(in_channels=input_channel,
            out_channels=input_channel, kernel_size=kernel_size, stride=
            stride, padding=padding)
        self.leaky_relu = paddle.nn.LeakyReLU(negative_slope=0.2)
        self.adain_layer_1 = AdaINLayer(input_channel, modulation_channel)
        self.adain_layer_2 = AdaINLayer(input_channel, modulation_channel)

    def forward(self, x, modulation):
        x = self.adain_layer_1(x, modulation)
        x = self.leaky_relu(x)
        x = self.conv_1(x)
        x = self.adain_layer_2(x, modulation)
        x = self.leaky_relu(x)
        x = self.conv_2(x)
        return x


class SPADELayer(paddle.nn.Layer):

    def __init__(self, input_channel, modulation_channel, hidden_size=256,
        kernel_size=3, stride=1, padding=1):
        super(SPADELayer, self).__init__()
        self.instance_norm = paddle.nn.InstanceNorm2D(num_features=
            input_channel, momentum=1 - 0.1)
        
        self.instance_norm.scale.stop_gradient = True
        self.instance_norm.bias.stop_gradient = True
        self.conv1 = paddle.nn.Conv2D(in_channels=modulation_channel,
            out_channels=hidden_size, kernel_size=kernel_size, stride=
            stride, padding=padding)
        self.gamma = paddle.nn.Conv2D(in_channels=hidden_size, out_channels
            =input_channel, kernel_size=kernel_size, stride=stride, padding
            =padding)
        self.beta = paddle.nn.Conv2D(in_channels=hidden_size, out_channels=
            input_channel, kernel_size=kernel_size, stride=stride, padding=
            padding)

    def forward(self, input, modulation):
        norm = self.instance_norm(input)
        conv_out = self.conv1(modulation)
        gamma = self.gamma(conv_out)
        beta = self.beta(conv_out)
        return norm + norm * gamma + beta


class SPADE(paddle.nn.Layer):

    def __init__(self, num_channel, num_channel_modulation, hidden_size=256,
        kernel_size=3, stride=1, padding=1):
        super(SPADE, self).__init__()
        self.conv_1 = paddle.nn.Conv2D(in_channels=num_channel,
            out_channels=num_channel, kernel_size=kernel_size, stride=
            stride, padding=padding)
        self.conv_2 = paddle.nn.Conv2D(in_channels=num_channel,
            out_channels=num_channel, kernel_size=kernel_size, stride=
            stride, padding=padding)
        self.leaky_relu = paddle.nn.LeakyReLU(negative_slope=0.2)
        self.spade_layer_1 = SPADELayer(num_channel, num_channel_modulation,
            hidden_size, kernel_size=kernel_size, stride=stride, padding=
            padding)
        self.spade_layer_2 = SPADELayer(num_channel, num_channel_modulation,
            hidden_size, kernel_size=kernel_size, stride=stride, padding=
            padding)

    def forward(self, input, modulations):
        input = self.spade_layer_1(input, modulations)
        input = self.leaky_relu(input)
        input = self.conv_1(input)
        input = self.spade_layer_2(input, modulations)
        input = self.leaky_relu(input)
        input = self.conv_2(input)
        return input


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


def downsample(x, size):
    if len(x.shape) == 5:
        size = x.shape[2], size[0], size[1]
        return paddle.nn.functional.interpolate(x=x, size=size, mode='nearest')
    return paddle.nn.functional.interpolate(x=x, size=size, mode='nearest')


# def convert_flow_to_deformation(flow):
#     """convert flow fields to deformations.
#     Args:
#         flow (tensor): Flow field obtained by the model
#     Returns:
#         deformation (tensor): The deformation used for warpping
#     """
#     b, c, h, w = flow.shape
#     flow_norm = 2 * paddle.concat(x=[flow[:, :1, (...)] / (w - 1), flow[:, 
#         1:, (...)] / (h - 1)], axis=1)
#     grid = make_coordinate_grid(flow)
#     deformation = grid + flow_norm.transpose(perm=[0, 2, 3, 1])
#     return deformation


# # def make_coordinate_grid(flow):
# #     """obtain coordinate grid with the same size as the flow filed.
# #     Args:
# #         flow (tensor): Flow field obtained by the model
# #     Returns:
# #         grid (tensor): The grid with the same size as the input flow
# #     """
# #     b, c, h, w = flow.shape
# #     x = paddle.arange(end=w).to(flow)
# #     y = paddle.arange(end=h).to(flow)
# #     x = 2 * (x / (w - 1)) - 1
# #     y = 2 * (y / (h - 1)) - 1
# #     yy = y.reshape([-1, 1]).repeat(1, w)
# #     xx = x.reshape(1, -1).repeat(h, 1)
# #     meshed = paddle.concat(x=[xx.unsqueeze_(axis=2), yy.unsqueeze_(axis=2)],
# #         axis=2)
# #     meshed = meshed.expand(shape=[b, -1, -1, -1])
# #     return meshed


# def make_coordinate_grid(flow):
#     b, c, h, w = flow.shape

#     x = paddle.linspace(-1, 1, w)
#     y = paddle.linspace(-1, 1, h)

#     xx = x.reshape((1, -1)).tile((h, 1))
#     yy = y.reshape((-1, 1)).tile((1, w))

#     xx = xx.reshape((h, w, 1)).tile((b, 1, 1))
#     yy = yy.reshape((h, w, 1)).tile((b, 1, 1))
    
#     meshed = paddle.concat([xx, yy], axis=-1)

#     return meshed



def convert_flow_to_deformation(flow):
    b, c, h, w = flow.shape
    flow_norm = paddle.concat([
        flow[:, :1, ...] * 2 / (w - 1), 
        flow[:, 1:, ...] * 2 / (h - 1)], 1)
    grid = make_coordinate_grid(flow)
    deformation = grid + flow_norm.transpose([0, 2, 3, 1])
    return deformation

def make_coordinate_grid(flow):
    b, c, h, w = flow.shape

    x = paddle.linspace(-1, 1, w, dtype=flow.dtype).reshape([1, 1, -1, 1]).tile([b, h, 1, 1])
    y = paddle.linspace(-1, 1, h, dtype=flow.dtype).reshape([1, -1, 1, 1]).tile([b, 1, w, 1])

    grid = paddle.concat([x, y], -1)
    return grid

# Usage example
# Assuming `flow` is a PaddlePaddle tensor with shape [batch_size, 2, height, width]
# deformation = convert_flow_to_deformation(flow)



def warping(source_image, deformation):
    """warp the input image according to the deformation
    Args:
        source_image (tensor): source images to be warpped
        deformation (tensor): deformations used to warp the images; value in range (-1, 1)
    Returns:
        output (tensor): the warpped images
    """
    _, h_old, w_old, _ = deformation.shape
    _, _, h, w = source_image.shape
    if h_old != h or w_old != w:
        deformation = deformation.transpose(perm=[0, 3, 1, 2])
        deformation = paddle.nn.functional.interpolate(x=deformation, size=
            (h, w), mode='bilinear')
        deformation = deformation.transpose(perm=[0, 2, 3, 1])
    return paddle.nn.functional.grid_sample(x=source_image, grid=deformation)


class DenseFlowNetwork(paddle.nn.Layer):

    def __init__(self, num_channel=6, num_channel_modulation=3 * 5,
        hidden_size=256):
        super(DenseFlowNetwork, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=num_channel, out_channels
            =32, kernel_size=7, stride=1, padding=3)
        self.conv1_bn = paddle.nn.BatchNorm2D(num_features=32, weight_attr=
            None if True else False, bias_attr=None if True else False)
        self.conv1_relu = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv2D(in_channels=32, out_channels=256,
            kernel_size=3, stride=2, padding=1)
        self.conv2_bn = paddle.nn.BatchNorm2D(num_features=256, weight_attr
            =None if True else False, bias_attr=None if True else False)
        self.conv2_relu = paddle.nn.ReLU()
        self.spade_layer_1 = SPADE(256, num_channel_modulation, hidden_size)
        self.spade_layer_2 = SPADE(256, num_channel_modulation, hidden_size)
        self.pixel_shuffle_1 = paddle.nn.PixelShuffle(upscale_factor=2)
        self.spade_layer_4 = SPADE(64, num_channel_modulation, hidden_size)
        self.conv_4 = paddle.nn.Conv2D(in_channels=64, out_channels=2,
            kernel_size=7, stride=1, padding=3)
        self.conv_5 = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=64,
            out_channels=32, kernel_size=7, stride=1, padding=3), paddle.nn
            .ReLU(), paddle.nn.Conv2D(in_channels=32, out_channels=1,
            kernel_size=7, stride=1, padding=3), paddle.nn.Sigmoid())

    def forward(self, ref_N_frame_img, ref_N_frame_sketch, T_driving_sketch):
        ref_N = ref_N_frame_img.shape[1]
        driving_sketch = paddle.concat(x=[T_driving_sketch[:, (i)] for i in
            range(T_driving_sketch.shape[1])], axis=1)
        wrapped_h1_sum, wrapped_h2_sum, wrapped_ref_sum = 0.0, 0.0, 0.0
        softmax_denominator = 0.0
        T = 1
        for ref_idx in range(ref_N):
            ref_img = ref_N_frame_img[:, (ref_idx)]
            ref_img = ref_img.unsqueeze(axis=1).expand(shape=[-1, T, -1, -1,
                -1])
            ref_img = paddle.concat(x=[ref_img[i] for i in range(ref_img.
                shape[0])], axis=0)
            ref_sketch = ref_N_frame_sketch[:, (ref_idx)]
            ref_sketch = ref_sketch.unsqueeze(axis=1).expand(shape=[-1, T, 
                -1, -1, -1])
            ref_sketch = paddle.concat(x=[ref_sketch[i] for i in range(
                ref_sketch.shape[0])], axis=0)
            flow_module_input = paddle.concat(x=(ref_img, ref_sketch), axis=1)
            h1 = self.conv1_relu(self.conv1_bn(self.conv1(flow_module_input)))
            h2 = self.conv2_relu(self.conv2_bn(self.conv2(h1)))
            downsample_64 = downsample(driving_sketch, (64, 64))
            spade_layer = self.spade_layer_1(h2, downsample_64)
            spade_layer = self.spade_layer_2(spade_layer, downsample_64)
            spade_layer = self.pixel_shuffle_1(spade_layer)
            spade_layer = self.spade_layer_4(spade_layer, driving_sketch)
            output_flow = self.conv_4(spade_layer)
            output_weight = self.conv_5(spade_layer)
            deformation = convert_flow_to_deformation(output_flow)
            wrapped_h1 = warping(h1, deformation)
            wrapped_h2 = warping(h2, deformation)
            wrapped_ref = warping(ref_img, deformation)
            softmax_denominator += output_weight
            wrapped_h1_sum += wrapped_h1 * output_weight
            wrapped_h2_sum += wrapped_h2 * downsample(output_weight, (64, 64))
            wrapped_ref_sum += wrapped_ref * output_weight
        softmax_denominator += 1e-05
        wrapped_h1_sum = wrapped_h1_sum / softmax_denominator
        wrapped_h2_sum = wrapped_h2_sum / downsample(softmax_denominator, (
            64, 64))
        wrapped_ref_sum = wrapped_ref_sum / softmax_denominator
        return wrapped_h1_sum, wrapped_h2_sum, wrapped_ref_sum


class TranslationNetwork(paddle.nn.Layer):

    def __init__(self):
        super(TranslationNetwork, self).__init__()
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
        self.conv1 = paddle.nn.Conv2D(in_channels=3 + 3 * 5, out_channels=
            32, kernel_size=7, stride=1, padding=3, bias_attr=False)
        self.conv1_bn = paddle.nn.BatchNorm2D(num_features=32, weight_attr=
            None if True else False, bias_attr=None if True else False)
        self.conv1_relu = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv2D(in_channels=32, out_channels=256,
            kernel_size=3, stride=2, padding=1, bias_attr=False)
        self.conv2_bn = paddle.nn.BatchNorm2D(num_features=256, weight_attr
            =None if True else False, bias_attr=None if True else False)
        self.conv2_relu = paddle.nn.ReLU()
        self.spade_1 = SPADE(num_channel=256, num_channel_modulation=256)
        self.adain_1 = AdaIN(256, 512)
        self.pixel_suffle_1 = paddle.nn.PixelShuffle(upscale_factor=2)
        self.spade_2 = SPADE(num_channel=64, num_channel_modulation=32)
        self.adain_2 = AdaIN(input_channel=64, modulation_channel=512)
        self.spade_4 = SPADE(num_channel=64, num_channel_modulation=3)
        self.leaky_relu = paddle.nn.LeakyReLU()
        self.conv_last = paddle.nn.Conv2D(in_channels=64, out_channels=3,
            kernel_size=7, stride=1, padding=3, bias_attr=False)
        self.Sigmoid = paddle.nn.Sigmoid()

    def forward(self, translation_input, wrapped_ref, wrapped_h1,
        wrapped_h2, T_mels):
        T_mels = paddle.concat(x=[T_mels[i] for i in range(T_mels.shape[0])
            ], axis=0)
        x = self.conv1_relu(self.conv1_bn(self.conv1(translation_input)))
        x = self.conv2_relu(self.conv2_bn(self.conv2(x)))
        audio_feature = self.audio_encoder(T_mels).squeeze(axis=-1).transpose(
            perm=[0, 2, 1])
        x = self.spade_1(x, wrapped_h2)
        x = self.adain_1(x, audio_feature)
        x = self.pixel_suffle_1(x)
        x = self.spade_2(x, wrapped_h1)
        x = self.adain_2(x, audio_feature)
        x = self.spade_4(x, wrapped_ref)
        x = self.leaky_relu(x)
        x = self.conv_last(x)
        x = self.Sigmoid(x)
        return x


class Renderer(paddle.nn.Layer):

    def __init__(self):
        super(Renderer, self).__init__()
        self.flow_module = DenseFlowNetwork()
        self.translation = TranslationNetwork()
        self.perceptual = PerceptualLoss(network='vgg19', layers=[
            'relu_1_1', 'relu_2_1', 'relu_3_1', 'relu_4_1', 'relu_5_1'],
            num_scales=2)

    def forward(self, face_frame_img, target_sketches, ref_N_frame_img,
        ref_N_frame_sketch, audio_mels):
        wrapped_h1, wrapped_h2, wrapped_ref = self.flow_module(ref_N_frame_img,
            ref_N_frame_sketch, target_sketches)
        target_sketches = paddle.concat(x=[target_sketches[:, (i)] for i in
            range(target_sketches.shape[1])], axis=1)
        gt_face = paddle.concat(x=[face_frame_img[i] for i in range(
            face_frame_img.shape[0])], axis=0)
        gt_mask_face = gt_face.clone()
        gt_mask_face[:, :, gt_mask_face.shape[2] // 2:, :] = 0
        translation_input = paddle.concat(x=[gt_mask_face, target_sketches],
            axis=1)
        generated_face = self.translation(translation_input, wrapped_ref,
            wrapped_h1, wrapped_h2, audio_mels)
        perceptual_gen_loss = self.perceptual(generated_face, gt_face,
            use_style_loss=True, weight_style_to_perceptual=250).mean()
        perceptual_warp_loss = self.perceptual(wrapped_ref, gt_face,
            use_style_loss=False, weight_style_to_perceptual=0.0).mean()
        return generated_face, wrapped_ref, paddle.unsqueeze(x=
            perceptual_warp_loss, axis=0), paddle.unsqueeze(x=
            perceptual_gen_loss, axis=0)


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

    def forward(self, inp, target, mask=None, use_style_loss=False,
        weight_style_to_perceptual=0.0):
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
        perm_0 = list(range(x.ndim))
        perm_0[1] = 2
        perm_0[2] = 1
        f_T = x.transpose(perm=perm_0)
        G = f.bmm(y=f_T) / (h * w * ch)
        return G
