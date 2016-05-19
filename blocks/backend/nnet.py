from __future__ import division, absolute_import

import numpy as np

import theano
from theano import tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv3d2d
try:
    from theano.tensor.nnet.nnet import softsign as T_softsign
except ImportError:
    from theano.sandbox.softsign import softsign as T_softsign

from blocks.backend import tensor
from blocks import autoconfig
from blocks.utils import as_tuple


FLOATX = autoconfig.floatX
EPSILON = autoconfig.epsilon


# ===========================================================================
# Helper
# ===========================================================================
def pool_output_length(input_length, pool_size, stride, pad, ignore_border):
    """ Copyright (c) 2014-2015 Lasagne contributors
    All rights reserved.
    LICENSE: https://github.com/Lasagne/Lasagne/blob/master/LICENSE

    Compute the output length of a pooling operator
    along a single dimension.

    Parameters
    ----------
    input_length : integer
        The length of the input in the pooling dimension
    pool_size : integer
        The length of the pooling region
    stride : integer
        The stride between successive pooling regions
    pad : integer
        The number of elements to be added to the input on each side.
    ignore_border: bool
        If ``True``, partial pooling regions will be ignored.
        Must be ``True`` if ``pad != 0``.

    Returns
    -------
    output_length
        * None if either input is None.
        * Computed length of the pooling operator otherwise.

    Notes
    -----
    When ``ignore_border == True``, this is given by the number of full
    pooling regions that fit in the padded input length,
    divided by the stride (rounding down).

    If ``ignore_border == False``, a single partial pooling region is
    appended if at least one input element would be left uncovered otherwise.
    """
    if input_length is None or pool_size is None:
        return None

    if ignore_border:
        output_length = input_length + 2 * pad - pool_size + 1
        output_length = (output_length + stride - 1) // stride

    # output length calculation taken from:
    # https://github.com/Theano/Theano/blob/master/theano/tensor/signal/downsample.py
    else:
        assert pad == 0

        if stride >= pool_size:
            output_length = (input_length + stride - 1) // stride
        else:
            output_length = max(
                0, (input_length - pool_size + stride - 1) // stride) + 1

    return output_length

# ===========================================================================
# NN OPERATIONS
# ===========================================================================


def relu(x, alpha=0., max_value=None):
    assert hasattr(T.nnet, 'relu'), ('It looks like like your version of '
                                     'Theano is out of date. '
                                     'Install the latest version with:\n'
                                     'pip install git+git://github.com/Theano/Theano.git --upgrade --no-deps')
    x = T.nnet.relu(x, alpha)
    if max_value is not None:
        x = T.minimum(x, max_value)
    return x


def softmax(x):
    return T.nnet.softmax(x)


def softplus(x):
    return T.nnet.softplus(x)


def softsign(x):
    return T_softsign(x)


def linear(x):
    return x


def categorical_crossentropy(output, target, from_logits=False):
    if from_logits:
        output = T.nnet.softmax(output)
    # avoid numerical instability with EPSILON clipping
    output = T.clip(output, EPSILON, 1.0 - EPSILON)
    return T.nnet.categorical_crossentropy(output, target)


def binary_crossentropy(output, target, from_logits=False):
    if from_logits:
        output = T.nnet.sigmoid(output)
    # avoid numerical instability with EPSILON clipping
    output = T.clip(output, EPSILON, 1.0 - EPSILON)
    return T.nnet.binary_crossentropy(output, target)


def sigmoid(x):
    return T.nnet.sigmoid(x)


def hard_sigmoid(x):
    return T.nnet.hard_sigmoid(x)


def tanh(x):
    return T.tanh(x)


# ==================== Regularizations ==================== #
def l2_normalize(x, axis):
    norm = T.sqrt(T.sum(T.square(x), axis=axis, keepdims=True))
    return x / norm


def l2_regularize(x):
    return T.sum(T.square(x))


def l1_regularize(x):
    return T.sum(T.abs_(x))


def jacobian_regularize(hidden, params):
    """ Computes the jacobian of the hidden layer with respect to
    the input, reshapes are necessary for broadcasting the
    element-wise product on the right axis
    """
    hidden = hidden * (1 - hidden)
    L = tensor.expand_dims(hidden, 1) * tensor.expand_dims(params, 0)
    # Compute the jacobian and average over the number of samples/minibatch
    L = T.sum(T.pow(L, 2)) / hidden.shape[0]
    return T.mean(L)


def correntropy_regularize(x, sigma=1.):
    """
    Note
    ----
    origin implementation from seya:
    https://github.com/EderSantana/seya/blob/master/seya/regularizers.py
    Copyright (c) EderSantana
    """
    return -T.sum(T.mean(T.exp(x**2 / sigma), axis=0)) / T.sqrt(2 * np.pi * sigma)


# ===========================================================================
# CONVOLUTIONS
# ===========================================================================
def conv2d(x, kernel, strides=(1, 1),
           border_mode='valid', image_shape=None, filter_shape=None):
    """
    border_mode: string, "same" or "valid".
    dim_ordering : th (defaults)
        TH input shape: (samples, input_depth, rows, cols)
        TH kernel shape: (depth, input_depth, rows, cols)
    """
    if border_mode == 'same':
        th_border_mode = 'half'
        np_kernel = kernel.eval()
    elif border_mode == 'valid':
        th_border_mode = 'valid'
    elif border_mode == 'full':
        th_border_mode = 'full'
    elif isinstance(border_mode, (tuple, list)):
        th_border_mode = border_mode
    else:
        raise Exception('Border mode not supported: ' + str(border_mode))

    # Theano might not accept long type
    def int_or_none(value):
        try:
            return int(value)
        except TypeError:
            return None

    if image_shape is not None:
        image_shape = tuple(int_or_none(v) for v in image_shape)

    if filter_shape is not None:
        filter_shape = tuple(int_or_none(v) for v in filter_shape)

    conv_out = T.nnet.conv2d(x, kernel,
                             border_mode=th_border_mode,
                             subsample=strides,
                             input_shape=image_shape,
                             filter_shape=filter_shape)

    if border_mode == 'same':
        if np_kernel.shape[2] % 2 == 0:
            conv_out = conv_out[:, :, :(x.shape[2] + strides[0] - 1) // strides[0], :]
        if np_kernel.shape[3] % 2 == 0:
            conv_out = conv_out[:, :, :, :(x.shape[3] + strides[1] - 1) // strides[1]]

    return conv_out


def deconv2d(x, kernel, image_shape, filter_shape=None,
    strides=(1, 1), border_mode='valid', flip_filters=True):
    """
    Run on cuDNN if available.
    border_mode: string, "same" or "valid".
    img_shape: (n, channels, width, height) of original image
    filter_shape: (n_filter, channels, w, h) of original filters
    """
    if len(image_shape) != 4:
        raise ValueError('img_shape for deconvolution operator must be 4-D')
    border_mode = 'half' if border_mode == 'same' else border_mode
    op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
        imshp=tuple([int(i) if isinstance(i, (long, float, int)) else None
                     for i in image_shape]),
        kshp=filter_shape,
        subsample=strides, border_mode=border_mode,
        filter_flip=flip_filters)
    return op(kernel, x, image_shape[2:])


def conv3d(x, kernel, strides=(1, 1, 1), border_mode='valid',
           image_shape=None, filter_shape=None):
    """
    Run on cuDNN if available.
    border_mode: string, "same" or "valid".
    dim_ordering : th (defaults)
        TH input shape: (samples, input_depth, conv_dim1, conv_dim2, conv_dim3)
        TH kernel shape: (out_depth, input_depth, kernel_dim1, kernel_dim2, kernel_dim3)
    """
    if tensor.on_gpu(): # Using DNN on GPU
        from theano.sandbox.cuda import dnn
        if border_mode == 'same':
            border_mode = 'half'
        conv_out = dnn.dnn_conv3d(img=x,
                                kerns=kernel,
                                subsample=strides,
                                border_mode=border_mode,
                                conv_mode='conv')
    else: # Using default implementation of Theano
        if border_mode not in {'same', 'valid', 'full'} and not isinstance(border_mode, (tuple, list)):
            raise Exception('Invalid border mode: ' + str(border_mode))

        if border_mode == 'same':
            assert(strides == (1, 1, 1))
            pad_dim1 = (kernel.shape[2] - 1)
            pad_dim2 = (kernel.shape[3] - 1)
            pad_dim3 = (kernel.shape[4] - 1)
            output_shape = (x.shape[0], x.shape[1],
                            x.shape[2] + pad_dim1,
                            x.shape[3] + pad_dim2,
                            x.shape[4] + pad_dim3)
            output = T.zeros(output_shape)
            indices = (slice(None), slice(None),
                       slice(pad_dim1 // 2, x.shape[2] + pad_dim1 // 2),
                       slice(pad_dim2 // 2, x.shape[3] + pad_dim2 // 2),
                       slice(pad_dim3 // 2, x.shape[4] + pad_dim3 // 2))
            x = T.set_subtensor(output[indices], x)
            border_mode = 'valid'

        border_mode_3d = (border_mode, border_mode, border_mode)
        conv_out = conv3d2d.conv3d(signals=x.dimshuffle(0, 2, 1, 3, 4),
                                   filters=kernel.dimshuffle(0, 2, 1, 3, 4),
                                   border_mode=border_mode_3d,
                                   signals_shape=None,
                                   filter_shape=None)
        conv_out = conv_out.dimshuffle(0, 2, 1, 3, 4)

        # support strides by manually slicing the output
        if strides != (1, 1, 1):
            conv_out = conv_out[:, :, ::strides[0], ::strides[1], ::strides[2]]
    return conv_out


def pool2d(x, pool_size, ignore_border=True,
           strides=(1, 1), pad=(0, 0), mode='max'):
    """
    Parameters
    ----------
    x : N-D theano tensor of input images
        Input images. Max pooling will be done over the 2 last dimensions.
    pool_size : tuple of length 2
        Factor by which to downscale (vertical ds, horizontal ds).
        (2,2) will halve the image in each dimension.
    strides : tuple of two ints
        Stride size, which is the number of shifts over rows/cols to get the
        next pool region. If st is None, it is considered equal to ds
        (no overlap on pooling regions).
    ignore_border : bool (default None, will print a warning and set to False)
        When True, (5,5) input with ds=(2,2) will generate a (2,2) output.
        (3,3) otherwise.
    padding : tuple of two ints
        (pad_h, pad_w), pad zeros to extend beyond four borders of the
        images, pad_h is the size of the top and bottom margins, and
        pad_w is the size of the left and right margins.
    mode : {'max', 'sum', 'average_inc_pad', 'average_exc_pad'}
        Operation executed on each window. `max` and `sum` always exclude
        the padding in the computation. `average` gives you the choice to
        include or exclude it.
    """
    pool_size = as_tuple(pool_size, 2, int)
    strides = as_tuple(strides, 2, int)
    pad = as_tuple(pad, 2, int)
    # ====== On GPU: use CuDNN ====== #
    if mode != 'sum' and tensor.on_gpu():
        from theano.sandbox.cuda import dnn
        if not ignore_border:
            raise ValueError('CuDNN does not support ignore_border = False.')
        pool_out = dnn.dnn_pool(x, ws=pool_size, stride=strides, mode=mode, pad=pad)
    # ====== Use default Theano implementation ====== #
    else:
        pool_out = pool.pool_2d(x, ds=pool_size, st=strides,
                                ignore_border=ignore_border,
                                padding=pad,
                                mode=mode)
    # ====== Estimate output shape ====== #
    input_shape = tensor.shape(x)
    output_shape = list(input_shape)  # copy / convert to mutable list
    output_shape[2] = pool_output_length(input_shape[2],
                                         pool_size=pool_size[0],
                                         stride=strides[0],
                                         pad=pad[0],
                                         ignore_border=ignore_border)
    output_shape[3] = pool_output_length(input_shape[3],
                                         pool_size=pool_size[1],
                                         stride=strides[1],
                                         pad=pad[1],
                                         ignore_border=ignore_border)
    tensor.add_shape(pool_out, tuple(output_shape))
    return pool_out


def pool3d(x, pool_size, ignore_border=True,
           strides=(1, 1, 1), pad=(0, 0, 0), mode='max'):
    """
    Parameters
    ----------
    x : N-D theano tensor of input images
        Input images. Max pooling will be done over the 2 last dimensions.
    pool_size : tuple of length 3
        Factor by which to downscale (vertical ds, horizontal ds).
        (2,2,2) will halve the image in each dimension.
    strides : tuple of 3 ints
        Stride size, which is the number of shifts over rows/cols to get the
        next pool region. If st is None, it is considered equal to ds
        (no overlap on pooling regions).
    ignore_border : bool (default None, will print a warning and set to False)
        When True, (5,5,5) input with ds=(2,2,2) will generate a (2,2,2) output.
        (3,3,3) otherwise.
    padding : tuple of 3 ints
        (pad_h, pad_w, pad_l), pad zeros to extend beyond four borders of the
        images, pad_h is the size of the top and bottom margins, and
        pad_w is the size of the left and right margins.
    mode : {'max', 'sum', 'average_inc_pad', 'average_exc_pad'}
        Operation executed on each window. `max` and `sum` always exclude
        the padding in the computation. `average` gives you the choice to
        include or exclude it.
    """
    pool_size = as_tuple(pool_size, 3, int)
    strides = as_tuple(strides, 3, int)
    pad = as_tuple(pad, 3, int)
    # ====== On GPU: use CuDNN ====== #
    if mode != 'sum' and tensor.on_gpu():
        from theano.sandbox.cuda import dnn
        if not ignore_border:
            raise ValueError('CuDNN does not support ignore_border = False.')
        pool_out = dnn.dnn_pool(x, ws=pool_size, stride=strides, mode=mode, pad=pad)
    # ====== Use default Theano implementation ====== #
    else:
        if len(set(pad)) > 1:
            raise ValueError('Only support same padding on CPU.')
        padding = (pad[0], pad[0])
        output = pool.pool_2d(input=tensor.dimshuffle(x, (0, 1, 4, 3, 2)),
                              ds=(pool_size[1], pool_size[0]),
                              st=(strides[1], strides[0]),
                              ignore_border=ignore_border,
                              padding=padding,
                              mode=mode)
        # pooling over conv_dim3
        pool_out = pool.pool_2d(input=tensor.dimshuffle(output, (0, 1, 4, 3, 2)),
                                ds=(1, pool_size[2]),
                                st=(1, strides[2]),
                                ignore_border=ignore_border,
                                padding=padding,
                                mode=mode)
    # ====== Estimate output shape ====== #
    input_shape = tensor.shape(x)
    output_shape = list(input_shape)  # copy / convert to mutable list
    output_shape[2] = pool_output_length(input_shape[2],
                                         pool_size=pool_size[0],
                                         stride=strides[0],
                                         pad=pad[0],
                                         ignore_border=ignore_border)
    output_shape[3] = pool_output_length(input_shape[3],
                                         pool_size=pool_size[1],
                                         stride=strides[1],
                                         pad=pad[1],
                                         ignore_border=ignore_border)
    output_shape[4] = pool_output_length(input_shape[4],
                                         pool_size=pool_size[2],
                                         stride=strides[2],
                                         pad=pad[2],
                                         ignore_border=ignore_border)
    tensor.add_shape(pool_out, tuple(output_shape))
    return pool_out


def poolWTA(x, pool_size, axis=1):
    """ This function is adpated from Lasagne
    Original work Copyright (c) 2014-2015 lasagne contributors
    All rights reserved.
    LICENSE: https://github.com/Lasagne/Lasagne/blob/master/LICENSE

    'Winner Take All' layer

    This layer performs 'Winner Take All' (WTA) across feature maps: zero out
    all but the maximal activation value within a region.

    Parameters
    ----------
    pool_size : integer
        the number of feature maps per region.

    axis : integer
        the axis along which the regions are formed.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    Notes
    -----
    This layer requires that the size of the axis along which it groups units
    is a multiple of the pool size.
    """
    input_shape = tensor.shape(x)
    num_feature_maps = input_shape[axis]
    num_pools = num_feature_maps // pool_size

    if input_shape[axis] % pool_size != 0:
        raise ValueError("Number of input feature maps (%d) is not a "
                         "multiple of the region size (pool_size=%d)" %
                         (num_feature_maps, pool_size))

    pool_shape = ()
    arange_shuffle_pattern = ()
    for k in range(axis):
        pool_shape += (input_shape[k],)
        arange_shuffle_pattern += ('x',)

    pool_shape += (num_pools, pool_size)
    arange_shuffle_pattern += ('x', 0)

    for k in range(axis + 1, x.ndim):
        pool_shape += (input_shape[k],)
        arange_shuffle_pattern += ('x',)

    input_reshaped = tensor.reshape(x, pool_shape)
    max_indices = tensor.argmax(input_reshaped, axis=axis + 1, keepdims=True)

    arange = tensor.arange(pool_size).dimshuffle(*arange_shuffle_pattern)
    mask = tensor.reshape(tensor.eq(max_indices, arange),
                          input_shape)
    output = x * mask
    tensor.add_shape(output, input_shape)
    return output


def poolGlobal(x, pool_function=tensor.mean):
    """ Global pooling

    This layer pools globally across all trailing dimensions beyond the 2nd.

    Parameters
    ----------
    pool_function : callable
        the pooling function to use. This defaults to `theano.tensor.mean`
        (i.e. mean-pooling) and can be replaced by any other aggregation
        function.

    Note
    ----
    output_shape = input_shape[:2]
    """
    input_shape = tensor.shape(x)
    x = pool_function(tensor.flatten(x, 3), axis=2)
    tensor.add_shape(x, input_shape[:2])
    return x
