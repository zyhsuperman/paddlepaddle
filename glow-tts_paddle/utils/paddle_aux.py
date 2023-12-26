
# This file is generated by PaConvert ToolKit, please Don't edit it!
import paddle

def to(self, *args, **kwargs):
    args_list = ["x", "y", "non_blocking", "copy", "memory_format"]
    new_kwargs = {}
    for i, node in enumerate(args):
        k = args_list[i]
        new_kwargs[k] = node
    for node in kwargs:
        v = kwargs[node]
        new_kwargs[node] = v
    kwargs = new_kwargs
    if not kwargs:
        return self
    elif "tensor" in kwargs:
        return paddle.cast(self, "{}.dtype".format(kwargs["tensor"]))
    elif "dtype" in kwargs:
        return paddle.cast(self, "{}".format(kwargs["dtype"]))
    elif "device" in kwargs and "dtype" not in kwargs:
        return self
    elif kwargs:
        if "y" not in kwargs and "x" in kwargs:
            if isinstance(kwargs["x"], paddle.dtype):
                dtype = kwargs["x"]
            elif isinstance(kwargs["x"], str) and kwargs["x"] not in ['cpu', 'cuda', 'ipu', 'xpu']:
                dtype = kwargs["x"]
            elif isinstance(kwargs["x"], paddle.Tensor):
                dtype = kwargs["x"].dtype
            else:
                dtype = self.dtype
            return paddle.cast(self, dtype)

        elif "y" in kwargs and "x" in kwargs:
            if isinstance(kwargs["x"], paddle.dtype):
                dtype = kwargs["x"]
            elif isinstance(kwargs["x"], str):
                if x not in ['cpu', 'cuda', 'ipu', 'xpu']:
                    dtype = kwargs["x"]
                else:
                    dtype = kwargs["y"] if isinstance(kwargs["y"], str) else self.dtype
            else:
                dtype = kwargs["x"]
            return paddle.cast(self, dtype)
        else:
            return self

setattr(paddle.Tensor, 'to', to)

def reshape(self, *args, **kwargs):
    if args:
        if len(args)==1 and isinstance(args[0], (tuple, list)):
            return paddle.reshape(self, args[0])
        else:
            return paddle.reshape(self, list(args))
    elif kwargs:
        assert 'shape' in kwargs
        return paddle.reshape(self, shape=kwargs['shape'])

setattr(paddle.Tensor, 'reshape', reshape)

def split_tensor_func(self, split_size, dim=0):
    if isinstance(split_size, int):
        return paddle.split(self, self.shape[dim]//split_size, dim)
    else:
        return paddle.split(self, split_size, dim)

setattr(paddle.Tensor, 'split', split_tensor_func)

def min_class_func(self, *args, **kwargs):
    if 'other' in kwargs:
        kwargs['y'] = kwargs.pop('other')
        ret = paddle.minimum(self, *args, **kwargs)
    elif len(args)==1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.minimum(self, *args, **kwargs)
    else:
        if 'dim' in kwargs:
            kwargs['axis'] = kwargs.pop('dim')

        if 'axis' in kwargs or len(args) >= 1:
            ret = paddle.min(self, *args, **kwargs), paddle.argmin(self, *args, **kwargs)
        else:
            ret = paddle.min(self, *args, **kwargs)

    return ret

def max_class_func(self, *args, **kwargs):
    if 'other' in kwargs:
        kwargs['y'] = kwargs.pop('other')
        ret = paddle.maximum(self, *args, **kwargs)
    elif len(args)==1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.maximum(self, *args, **kwargs)
    else:
        if 'dim' in kwargs:
            kwargs['axis'] = kwargs.pop('dim')

        if 'axis' in kwargs or len(args) >= 1:
            ret = paddle.max(self, *args, **kwargs), paddle.argmax(self, *args, **kwargs)
        else:
            ret = paddle.max(self, *args, **kwargs)

    return ret

setattr(paddle.Tensor, "min", min_class_func)
setattr(paddle.Tensor, "max", max_class_func)

def repeat(self, *args, **kwargs):
    if args:
        if len(args)==1 and isinstance(args[0], (tuple, list)):
            return paddle.tile(self, args[0])
        else:
            return paddle.tile(self, list(args))
    elif kwargs:
        assert 'repeats' in kwargs
        return paddle.tile(self, repeat_times=kwargs['repeats'])

setattr(paddle.Tensor, 'repeat', repeat)
