import paddle


class GuidedAttentionLoss(paddle.nn.Layer):
    """Wrapper around all loss functions including the loss of Tacotron 2.

    Details:
        - L2 of the prediction before and after the postnet.
        - Cross entropy of the stop tokens
        - Guided attention loss:
            prompt the attention matrix to be nearly diagonal, this is how people usualy read text
            introduced by 'Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention'
    Arguments:
        guided_att_steps -- number of training steps for which the guided attention is enabled
        guided_att_variance -- initial allowed variance of the guided attention (strictness of diagonal) 
        guided_att_gamma -- multiplier which is applied to guided_att_variance at every update_states call
    """

    def __init__(self, guided_att_steps, guided_att_variance, guided_att_gamma
        ):
        super(GuidedAttentionLoss, self).__init__()
        self._g = guided_att_variance
        self._gamma = guided_att_gamma
        self._g_steps = guided_att_steps

    def forward(self, alignments, input_lengths, target_lengths, global_step):
        if self._g_steps < global_step:
            return 0
        self._g = self._gamma ** global_step
        weights = paddle.zeros_like(x=alignments)
        for i, (f, l) in enumerate(zip(target_lengths, input_lengths)):
            grid_f, grid_l = paddle.meshgrid(paddle.arange(dtype='float32',
                end=f), paddle.arange(dtype='float32', end=l))
            weights[(i), :f, :l] = 1 - paddle.exp(x=-(grid_l / l - grid_f /
                f) ** 2 / (2 * self._g ** 2))
        loss = paddle.sum(x=weights * alignments, axis=(1, 2))
        loss = paddle.mean(x=loss / target_lengths.astype(dtype='float32'))
        return loss
