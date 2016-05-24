from __future__ import division, absolute_import

from blocks import backend as K
from blocks import autoconfig
from blocks.roles import add_role, PARAMETER
from .base import NNOps, NNConfig


def antirectify(x):
    """
    This is the combination of a sample-wise L2 normalization with the
    concatenation of:
        - the positive part of the input
        - the negative part of the input
    The result is a tensor of samples that are twice as large as
    the input samples.
    It can be used in place of a ReLU.
        - Input shape: 2D tensor of shape (samples, n)
        - Output shape: 2D tensor of shape (samples, 2*n)

    Notes
    -----
    When applying ReLU, assuming that the distribution of the previous
    output is approximately centered around 0., you are discarding half of
    your input. This is inefficient.
    Antirectifier allows to return all-positive outputs like ReLU, without
    discarding any data.
    Tests on MNIST show that Antirectifier allows to train networks with
    twice less parameters yet with comparable classification accuracy
    as an equivalent ReLU-based network.

    """
    if x.ndim != 2:
        raise Exception('This Ops only support 2D input.')
    input_shape = K.shape(x)
    x -= K.mean(x, axis=1, keepdims=True)
    # l2 normalization
    x /= K.sqrt(K.sum(K.square(x), axis=1, keepdims=True))
    x = K.concatenate([K.relu(x, 0), K.relu(-x, 0)], axis=1)
    if isinstance(input_shape, (tuple, list)):
        K.add_shape(x, (input_shape[0], input_shape[1] * 2))
    return x


def randrectify(x, lower=0.3, upper=0.8, shared_axes='auto', seed=None):
    """ This function is adpated from Lasagne
    Original work Copyright (c) 2014-2015 lasagne contributors
    All rights reserved.
    LICENSE: https://github.com/Lasagne/Lasagne/blob/master/LICENSE

    Applies a randomized leaky rectify activation to x.

    The randomized leaky rectifier was first proposed and used in the Kaggle
    NDSB Competition, and later evaluated in [1]_. Compared to the standard
    leaky rectifier :func:`leaky_rectify`, it has a randomly sampled slope
    for negative input during training, and a fixed slope during evaluation.

    Equation for the randomized rectifier linear unit during training:
    :math:`\\varphi(x) = \\max((\\sim U(lower, upper)) \\cdot x, x)`

    During evaluation, the factor is fixed to the arithmetic mean of `lower`
    and `upper`.

    Parameters
    ----------
    lower : Theano shared variable, expression, or constant
        The lower bound for the randomly chosen slopes.

    upper : Theano shared variable, expression, or constant
        The upper bound for the randomly chosen slopes.

    shared_axes : 'auto', 'all', int or tuple of int
        The axes along which the random slopes of the rectifier units are
        going to be shared. If ``'auto'`` (the default), share over all axes
        except for the second - this will share the random slope over the
        minibatch dimension for dense layers, and additionally over all
        spatial dimensions for convolutional layers. If ``'all'``, share over
        all axes, thus using a single random slope.

     References
    ----------
    .. [1] Bing Xu, Naiyan Wang et al. (2015):
       Empirical Evaluation of Rectified Activations in Convolutional Network,
       http://arxiv.org/abs/1505.00853
    """
    input_shape = K.shape(x)
    # ====== check lower and upper ====== #
    if K.is_shared_variable(lower):
        add_role(lower, PARAMETER)
        lower.name = 'lower'
    if K.is_shared_variable(upper):
        add_role(upper, PARAMETER)
        upper.name = 'upper'
    if not K.is_variable(lower > upper) and lower > upper:
        raise ValueError("Upper bound for Randomized Rectifier needs "
                         "to be higher than lower bound.")
    # ====== check shared_axes ====== #
    if shared_axes == 'auto':
        shared_axes = (0,) + tuple(range(2, len(input_shape)))
    elif shared_axes == 'all':
        shared_axes = tuple(range(len(input_shape)))
    elif isinstance(shared_axes, int):
        shared_axes = (shared_axes,)
    else:
        shared_axes = shared_axes
    # ====== main logic ====== #
    if not K.is_training(x) or upper == lower:
        x = K.relu(x, (upper + lower) / 2.0)
    else: # Training mode
        shape = list(input_shape)
        if any(s is None for s in shape):
            shape = list(x.shape)
        for ax in shared_axes:
            shape[ax] = 1

        rnd = K.random_uniform(tuple(shape),
                               low=lower,
                               high=upper,
                               dtype=autoconfig.floatX,
                               seed=seed)
        rnd = K.addbroadcast(rnd, *shared_axes)
        x = K.relu(x, rnd)
    if isinstance(input_shape, (tuple, list)):
        K.add_shape(x, input_shape)
    return x


def rectify(x, alpha=0.):
    return K.relu(x, alpha)


def softmax(x):
    return K.softmax(x)


def softplus(x):
    return K.softplus(x)


def softsign(x):
    return K.softsign(x)


def linear(x):
    return K.linear(x)


def sigmoid(x):
    return K.sigmoid(x)


def hard_sigmoid(x):
    return K.hard_sigmoid(x)


def tanh(x):
    return K.tanh(x)
