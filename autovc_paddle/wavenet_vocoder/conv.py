import paddle


class Conv1d(paddle.nn.Conv1D):
    """Extended nn.Conv1d for incremental dilated convolutions
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clear_buffer()
        self._linearized_weight = None
#         """Class Method: *.register_backward_hook, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
# >>>>>>        self.register_backward_hook(self._clear_linearized_weight)

    def incremental_forward(self, input):
        if self.training:
            raise RuntimeError('incremental_forward only supports eval mode')
        for hook in self._forward_pre_hooks.values():
            hook(self, input)
        weight = self._get_linearized_weight()
        kw = self._kernel_size[0]
        dilation = self._dilation[0]
        bsz = input.shape[0]
        if kw > 1:
            input = input.data
            if self.input_buffer is None:
                self.input_buffer = paddle.empty(shape=[bsz, kw + (kw - 1) * (dilation - 1), input.shape[2]], 
                                     dtype=input.dtype)
                self.input_buffer.zero_()
            else:
                self.input_buffer[:, :-1, :] = self.input_buffer[:, 1:, :
                    ].clone()
            self.input_buffer[:, (-1), :] = input[:, (-1), :]
            input = self.input_buffer
            if dilation > 1:
                input = input[:, 0::dilation, :]
        
        output = paddle.nn.functional.linear(weight=weight.T, bias=self.
            bias, x=input.reshape([bsz, -1]))
        return output.reshape([bsz, 1, -1])

    def clear_buffer(self):
        self.input_buffer = None

    def _get_linearized_weight(self):
        if self._linearized_weight is None:
            kw = self._kernel_size[0]
            if self.weight.shape == [self._out_channels, self._in_channels, kw]:
                weight = paddle.transpose(self.weight, perm=[0, 2, 1])
            else:
                weight = paddle.transpose(self.weight, perm=[2, 1, 0])
            assert weight.shape == [self._out_channels, kw, self._in_channels]
            """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            self._linearized_weight = weight.reshape([self._out_channels, -1])
        return self._linearized_weight

    def _clear_linearized_weight(self, *args):
        self._linearized_weight = None
