import paddle
import numpy as np
DEFAULT_MIN_BIN_WIDTH = 0.001
DEFAULT_MIN_BIN_HEIGHT = 0.001
DEFAULT_MIN_DERIVATIVE = 0.001


def piecewise_rational_quadratic_transform(inputs, unnormalized_widths,
    unnormalized_heights, unnormalized_derivatives, inverse=False, tails=
    None, tail_bound=1.0, min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT, min_derivative=
    DEFAULT_MIN_DERIVATIVE):
    if tails is None:
        spline_fn = rational_quadratic_spline
        spline_kwargs = {}
    else:
        spline_fn = unconstrained_rational_quadratic_spline
        spline_kwargs = {'tails': tails, 'tail_bound': tail_bound}
    outputs, logabsdet = spline_fn(inputs=inputs, unnormalized_widths=
        unnormalized_widths, unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives, inverse=inverse,
        min_bin_width=min_bin_width, min_bin_height=min_bin_height,
        min_derivative=min_derivative, **spline_kwargs)
    return outputs, logabsdet


def searchsorted(bin_locations, inputs, eps=1e-06):
    bin_locations[..., -1] += eps
    return paddle.sum(x=inputs[..., None] >= bin_locations, axis=-1) - 1


def unconstrained_rational_quadratic_spline(inputs, unnormalized_widths,
    unnormalized_heights, unnormalized_derivatives, inverse=False, tails=
    'linear', tail_bound=1.0, min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT, min_derivative=
    DEFAULT_MIN_DERIVATIVE):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask
    outputs = paddle.zeros_like(x=inputs)
    logabsdet = paddle.zeros_like(x=inputs)
    if tails == 'linear':
        unnormalized_derivatives = paddle.nn.functional.pad(unnormalized_derivatives, pad=[0, 0,0, 0, 0, 0, 1, 1])
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant
        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError('{} tails are not implemented.'.format(tails))
    outputs[inside_interval_mask], logabsdet[inside_interval_mask
        ] = rational_quadratic_spline(inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[(inside_interval_mask), :],
        unnormalized_heights=unnormalized_heights[(inside_interval_mask), :
        ], unnormalized_derivatives=unnormalized_derivatives[(
        inside_interval_mask), :], inverse=inverse, left=-tail_bound, right
        =tail_bound, bottom=-tail_bound, top=tail_bound, min_bin_width=
        min_bin_width, min_bin_height=min_bin_height, min_derivative=
        min_derivative)
    return outputs, logabsdet


def rational_quadratic_spline(inputs, unnormalized_widths,
    unnormalized_heights, unnormalized_derivatives, inverse=False, left=0.0,
    right=1.0, bottom=0.0, top=1.0, min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT, min_derivative=
    DEFAULT_MIN_DERIVATIVE):
    if paddle.min(x=inputs) < left or paddle.max(x=inputs) > right:
        raise ValueError('Input to a transform is not within its domain')
    num_bins = unnormalized_widths.shape[-1]
    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')
    widths = paddle.nn.functional.softmax(x=unnormalized_widths, axis=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = paddle.cumsum(x=widths, axis=-1)
    
    cumwidths = paddle.nn.functional.pad(cumwidths, pad=[0, 0, 1, 0], mode=
        'constant', value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[(...), 1:] - cumwidths[(...), :-1]
    derivatives = min_derivative + paddle.nn.functional.softplus(x=
        unnormalized_derivatives)
    heights = paddle.nn.functional.softmax(x=unnormalized_heights, axis=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = paddle.cumsum(x=heights, axis=-1)
    cumheights = paddle.nn.functional.pad(cumheights, pad=[0, 0, 1, 0], mode=
        'constant', value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[(...), 1:] - cumheights[(...), :-1]
    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]
    input_cumwidths = cumwidths.take_along_axis(axis=-1, indices=bin_idx)[
        ..., 0]
    input_bin_widths = widths.take_along_axis(axis=-1, indices=bin_idx)[..., 0]
    input_cumheights = cumheights.take_along_axis(axis=-1, indices=bin_idx)[
        ..., 0]
    delta = heights / widths
    input_delta = delta.take_along_axis(axis=-1, indices=bin_idx)[..., 0]
    input_derivatives = derivatives.take_along_axis(axis=-1, indices=bin_idx)[
        ..., 0]
    input_derivatives_plus_one = derivatives[(...), 1:].take_along_axis(axis
        =-1, indices=bin_idx)[..., 0]
    input_heights = heights.take_along_axis(axis=-1, indices=bin_idx)[..., 0]
    if inverse:
        a = (inputs - input_cumheights) * (input_derivatives +
            input_derivatives_plus_one - 2 * input_delta) + input_heights * (
            input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta)
        c = -input_delta * (inputs - input_cumheights)
        discriminant = b.pow(y=2) - 4 * a * c
        assert (discriminant >= 0).astype('bool').all()
        root = 2 * c / (-b - paddle.sqrt(x=discriminant))
        outputs = root * input_bin_widths + input_cumwidths
        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (input_derivatives +
            input_derivatives_plus_one - 2 * input_delta
            ) * theta_one_minus_theta
        derivative_numerator = input_delta.pow(y=2) * (
            input_derivatives_plus_one * root.pow(y=2) + 2 * input_delta *
            theta_one_minus_theta + input_derivatives * (1 - root).pow(y=2))
        logabsdet = paddle.log(x=derivative_numerator) - 2 * paddle.log(x=
            denominator)
        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)
        numerator = input_heights * (input_delta * theta.pow(y=2) + 
            input_derivatives * theta_one_minus_theta)
        denominator = input_delta + (input_derivatives +
            input_derivatives_plus_one - 2 * input_delta
            ) * theta_one_minus_theta
        outputs = input_cumheights + numerator / denominator
        derivative_numerator = input_delta.pow(y=2) * (
            input_derivatives_plus_one * theta.pow(y=2) + 2 * input_delta *
            theta_one_minus_theta + input_derivatives * (1 - theta).pow(y=2))
        logabsdet = paddle.log(x=derivative_numerator) - 2 * paddle.log(x=
            denominator)
        return outputs, logabsdet
