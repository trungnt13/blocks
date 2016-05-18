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


FLOATX = autoconfig.floatX
EPSILON = autoconfig.epsilon


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
    ''' Computes the jacobian of the hidden layer with respect to
    the input, reshapes are necessary for broadcasting the
    element-wise product on the right axis
    '''
    hidden = hidden * (1 - hidden)
    L = tensor.expand_dims(hidden, 1) * tensor.expand_dims(params, 0)
    # Compute the jacobian and average over the number of samples/minibatch
    L = T.sum(T.pow(L, 2)) / hidden.shape[0]
    return T.mean(L)


def correntropy_regularize(x, sigma=1.):
    '''
    Note
    ----
    origin implementation from seya:
    https://github.com/EderSantana/seya/blob/master/seya/regularizers.py
    Copyright (c) EderSantana
    '''
    return -T.sum(T.mean(T.exp(x**2 / sigma), axis=0)) / T.sqrt(2 * np.pi * sigma)


# ===========================================================================
# CONVOLUTIONS
# ===========================================================================
def conv2d(x, kernel, strides=(1, 1),
           border_mode='valid', image_shape=None, filter_shape=None):
    '''
    border_mode: string, "same" or "valid".
    dim_ordering : th (defaults)
        TH input shape: (samples, input_depth, rows, cols)
        TH kernel shape: (depth, input_depth, rows, cols)
    '''
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
    '''
    Run on cuDNN if available.
    border_mode: string, "same" or "valid".
    img_shape: (n, channels, width, height) of original image
    filter_shape: (n_filter, channels, w, h) of original filters
    '''
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


def conv3d(x, kernel, strides=(1, 1, 1), border_mode='valid'):
    '''
    Run on cuDNN if available.
    border_mode: string, "same" or "valid".
    dim_ordering : th (defaults)
        TH input shape: (samples, input_depth, conv_dim1, conv_dim2, conv_dim3)
        TH kernel shape: (out_depth, input_depth, kernel_dim1, kernel_dim2, kernel_dim3)
    '''
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
                                   border_mode=border_mode_3d)
        conv_out = conv_out.dimshuffle(0, 2, 1, 3, 4)

        # support strides by manually slicing the output
        if strides != (1, 1, 1):
            conv_out = conv_out[:, :, ::strides[0], ::strides[1], ::strides[2]]
    return conv_out


def pool2d(x, pool_size, strides=(1, 1), border_mode='valid',
           dim_ordering='th', pool_mode='max'):
    # ====== dim ordering ====== #
    if dim_ordering not in {'th', 'tf'}:
        raise Exception('Unknown dim_ordering ' + str(dim_ordering))
    if dim_ordering == 'tf':
        x = x.dimshuffle((0, 3, 1, 2))
    # ====== border mode ====== #
    if border_mode == 'same':
        w_pad = pool_size[0] - 2 if pool_size[0] % 2 == 1 else pool_size[0] - 1
        h_pad = pool_size[1] - 2 if pool_size[1] % 2 == 1 else pool_size[1] - 1
        padding = (w_pad, h_pad)
    elif border_mode == 'valid':
        padding = (0, 0)
    elif isinstance(border_mode, (tuple, list)):
        padding = tuple(border_mode)
    else:
        raise Exception('Invalid border mode: ' + str(border_mode))

    # ====== pooling ====== #
    if tensor.on_gpu() and dnn.dnn_available():
        pool_out = dnn.dnn_pool(x, pool_size,
                                stride=strides,
                                mode=pool_mode,
                                pad=padding)
    else: # CPU veresion support by theano
        pool_out = pool.pool_2d(x, ds=pool_size, st=strides,
                                ignore_border=True,
                                padding=padding,
                                mode=pool_mode)

    if dim_ordering == 'tf':
        pool_out = pool_out.dimshuffle((0, 2, 3, 1))
    return pool_out


def pool3d(x, pool_size, strides=(1, 1, 1), border_mode='valid',
           dim_ordering='th', pool_mode='max'):
    # ====== dim ordering ====== #
    if dim_ordering not in {'th', 'tf'}:
        raise Exception('Unknown dim_ordering ' + str(dim_ordering))
    if dim_ordering == 'tf':
        x = x.dimshuffle((0, 4, 1, 2, 3))
    # ====== border mode ====== #
    if border_mode == 'same':
        w_pad = pool_size[0] - 2 if pool_size[0] % 2 == 1 else pool_size[0] - 1
        h_pad = pool_size[1] - 2 if pool_size[1] % 2 == 1 else pool_size[1] - 1
        d_pad = pool_size[2] - 2 if pool_size[2] % 2 == 1 else pool_size[2] - 1
        padding = (w_pad, h_pad, d_pad)
    elif border_mode == 'valid':
        padding = (0, 0, 0)
    elif isinstance(border_mode, (tuple, list)):
        padding = tuple(border_mode)
    else:
        raise Exception('Invalid border mode: ' + str(border_mode))
    # ====== pooling ====== #
    if tensor.on_gpu() and dnn.dnn_available():
        pool_out = dnn.dnn_pool(x, pool_size,
                                stride=strides,
                                mode=pool_mode,
                                pad=padding)
    else:
        padding = padding[:2]
        # pooling over conv_dim2, conv_dim1 (last two channels)
        output = pool.pool_2d(input=x.dimshuffle(0, 1, 4, 3, 2),
                              ds=(pool_size[1], pool_size[0]),
                              st=(strides[1], strides[0]),
                              ignore_border=True,
                              padding=padding,
                              mode=pool_mode)
        # pooling over conv_dim3
        pool_out = pool.pool_2d(input=output.dimshuffle(0, 1, 4, 3, 2),
                                ds=(1, pool_size[2]),
                                st=(1, strides[2]),
                                ignore_border=True,
                                padding=padding,
                                mode=pool_mode)
    # ====== output ====== #
    if dim_ordering == 'tf':
        pool_out = pool_out.dimshuffle((0, 2, 3, 4, 1))
    return pool_out
