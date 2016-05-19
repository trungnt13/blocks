# ===========================================================================
# This module is adpated from: https://github.com/fchollet/keras
# Revision: @80927fa
# Original work Copyright (c) 2014-2015 keras contributors
# Some idea are also borrowed from Lasagne library
# Original work Copyright (c) 2014-2015 Lasagne contributors
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import division, absolute_import

from collections import OrderedDict
import math

import numpy as np

import theano
from theano import tensor as T
from theano import Variable
from theano.gof.graph import Constant
from theano.tensor.shared_randomstreams import RandomStateSharedVariable
from theano.tensor.sharedvar import SharedVariable

from blocks import autoconfig
from blocks.utils import as_tuple
from blocks.graph import ComputationGraph
from blocks.roles import add_role, has_roles, INPUT, TRAINING

FLOATX = autoconfig.floatX
EPSILON = autoconfig.epsilon


# remember original min and max
_min = min
_max = max


# ===========================================================================
# INTERNAL UTILS
# ===========================================================================
def on_gpu():
    """Return whether the session is set to
    run on GPU or not (i.e. on CPU).
    """
    import theano.sandbox.cuda

    return 'gpu' in theano.config.device or \
    'cuda' in theano.config.device or \
    'gpu' in theano.config.contexts or \
    'cuda' in theano.config.contexts or \
    theano.sandbox.cuda.cuda_enabled

if on_gpu():
    """Import cuDNN only if running on GPU:
    not having Cuda installed should not
    prevent from running the present code.
    """
    # dummy initialization to remove the overhead of running libgpuarray backend
    T.zeros(0, dtype='int').eval()
    _ = theano.shared(value=np.asarray(1., dtype='float32'),
                     name='temporary_var')
    T.grad(2 * _, _).eval()
    _.set_value(None)
    del _


def add_shape(var, shape):
    if not isinstance(shape, (tuple, list)):
        shape = (shape,)
    if len(shape) != var.ndim:
        raise ValueError('Variable has ndim={} but given shape has ndim={}'
                         '.'.format(var.ndim, len(shape)))
    old_shape = getattr(var.tag, 'shape', var.shape)
    new_shape = []
    for i in shape:
        if isinstance(i, (tuple, list)):
            new_shape.append(old_shape[int(i[0])])
        elif i is not None:
            new_shape.append(int(i))
        else:
            new_shape.append(i)
    var.tag.shape = tuple(new_shape)


# ===========================================================================
# VARIABLE MANIPULATION
# ===========================================================================
def variable(value, dtype=FLOATX, name=None, broadcastable=None, target=None):
    """Instantiate a tensor variable.
    """
    value = np.asarray(value, dtype=dtype)
    if not on_gpu():
        target = None

    kwargs = {}
    if broadcastable is not None:
        kwargs['broadcastable'] = broadcastable
    if target is not None:
        kwargs['target'] = target

    variable = theano.shared(value=value, name=name, strict=False, **kwargs)
    add_shape(variable, value.shape)
    return variable


def zeros_var(shape, dtype=FLOATX, name=None):
    """Instantiate an all-zeros variable.
    """
    return variable(np.zeros(shape), dtype, name)


def ones_var(shape, dtype=FLOATX, name=None):
    """Instantiate an all-ones variable.
    """
    return variable(np.ones(shape), dtype, name)


def is_shared_variable(variable):
    """Check if a variable is a Theano shared variable.

    Notes
    -----
    This function excludes shared variables that store the state of Theano
    random number generators.

    """
    return (isinstance(variable, SharedVariable) and
            not isinstance(variable, RandomStateSharedVariable) and
            not hasattr(variable.tag, 'is_rng'))


def is_variable(v):
    return isinstance(v, Variable)


def is_training(v):
    """ A variable is in TRAINING mode if at least one of its descendant
    has TRAINING roles

    Note
    ----
    TRAINING role can be override by: ``add_role(x, DEPLOYING)``

    """
    if hasattr(v, 'tag') and hasattr(v.tag, 'roles'):
        temp = ComputationGraph(v)
        for i in temp.variables:
            if has_roles(i, TRAINING, exact=True):
                return True
    return False

_PLACEHOLDER_ID = 0


def placeholder(shape=None, ndim=None, dtype=FLOATX, name=None, for_training=False):
    """Instantiate an input data placeholder variable.
    """
    if shape is None and ndim is None:
        raise Exception('Specify either a shape or ndim value.')
    if shape is not None:
        ndim = len(shape)
    broadcast = (False,) * ndim
    # ====== Modify add name prefix ====== #
    global _PLACEHOLDER_ID
    name_prefix = '[ID%02d]' % _PLACEHOLDER_ID
    _PLACEHOLDER_ID += 1
    name = [name_prefix] if name is None else [name_prefix, name]
    name = ''.join(name)
    placeholder = T.TensorType(dtype, broadcast)(name)
    add_role(placeholder, INPUT)
    if for_training:
        add_role(placeholder, TRAINING)
    # store the predefined shape of placeholder
    if shape is not None:
        add_shape(placeholder, shape)
    return placeholder


def is_placeholder(variable):
    """Check if variable is a user-provided graph input.

    To be considered an input the variable must have no owner, and not
    be a constant or shared variable.

    Parameters
    ----------
    variable : :class:`~tensor.TensorVariable`

    Returns
    -------
    bool
        ``True`` If the variable is a user-provided input to the graph.

    """
    return (not variable.owner and
            not isinstance(variable, SharedVariable) and
            not isinstance(variable, Constant))


def eval(x):
    """Run a graph.
    """
    # just a hack to return placeholder shape when eval
    return x.eval()


# ===========================================================================
# Shape operator
# ===========================================================================
def shape(x, none=True):
    """Return the shape of a tensor, this function search for predefined shape
    of `x` first, otherwise, return the theano shape

    Warning: type returned will be different for
    Theano backend (Theano tensor type) and TF backend (TF TensorShape).

    Parameters
    ----------
    none : bool
        allow None value, otherwise, all None (and -1) dimensions are converted to
        intermediate shape variable
    """
    shape = x.shape
    if hasattr(x, 'tag') and hasattr(x.tag, 'shape') and x.tag.shape is not None:
        shape = x.tag.shape
    # remove None value
    if not none:
        shape = tuple([x.shape[i] if j is None or j < 0 else j for i, j in enumerate(shape)])
    return shape


def int_shape(x):
    s = shape(x)
    if hasattr(s, 'eval'):
        return s.eval()
    return tuple([i.eval() if hasattr(i, 'eval') else i for i in s])


def ndim(x):
    return x.ndim


def broadcastable(x):
    return x.broadcastable


def addbroadcast(x, *axes):
    return T.addbroadcast(x, *axes)


# ===========================================================================
# Predefined data
# ===========================================================================
def zeros(shape, dtype=FLOATX, name=None):
    """Instantiate an all-zeros variable.
    """
    return T.zeros(shape=shape, dtype=dtype)


def ones(shape, dtype=FLOATX, name=None):
    """Instantiate an all-ones variable.
    """
    return T.ones(shape=shape, dtype=dtype)


def ones_like(x):
    return T.ones_like(x)


def zeros_like(x):
    return T.zeros_like(x)


def count_params(x):
    """Return number of scalars in a tensor.

    Return: numpy integer.
    """
    return np.prod(x.shape.eval())


def cast(x, dtype):
    if 'theano.' in str(x.__class__):
        return T.cast(x, dtype)
    return np.cast[dtype](x)


def castX(x):
    return cast(x, FLOATX)


# ===========================================================================
# LINEAR ALGEBRA
# Assumed overridden:
# +, -, /, *, +=, -=, *=, /=
# ===========================================================================
def dot(x, y):
    # TODO: float16 overflow
    output = T.dot(x, y)
    add_shape(output, shape(x)[:-1] + shape(y)[1:])
    return output


def transpose(x, axes=None):
    output_shape = shape(x)
    if axes is None:
        output_shape = output_shape[::-1]
    else:
        output_shape = [output_shape[i] for i in axes]
    x = T.transpose(x, axes=axes)
    add_shape(x, tuple(output_shape))
    return x


def gather(reference, indices):
    """reference: a tensor.
    indices: an int tensor of indices.

    Return: a tensor of same type as reference.
    """
    return reference[indices]


def diag(x):
    input_shape = shape(x)
    x = T.diag(x)
    add_shape(x, (_min(input_shape),))
    return x


def eye(n, dtype=FLOATX):
    x = T.eye(n, dtype=dtype)
    add_shape(x, (n, n))
    return x


# ===========================================================================
# ELEMENT-WISE OPERATIONS
# ===========================================================================
def var(x, axis=None, keepdims=False):
    return T.var(x, axis=axis, keepdims=keepdims)


def max(x, axis=None, keepdims=False):
    return T.max(x, axis=axis, keepdims=keepdims)


def min(x, axis=None, keepdims=False):
    return T.min(x, axis=axis, keepdims=keepdims)


def sum(x, axis=None, keepdims=False):
    """Sum of the values in a tensor, alongside the specified axis.
    """
    return T.sum(x, axis=axis, keepdims=keepdims)


def mul(x, y):
    return T.mul(x, y)


def prod(x, axis=None, keepdims=False):
    """Multiply the values in a tensor, alongside the specified axis.
    """
    return T.prod(x, axis=axis, keepdims=keepdims)


def mean(x, axis=None, keepdims=False):
    dtype = x.dtype
    if 'int' in dtype:
        dtype = FLOATX
    return T.mean(x, axis=axis, keepdims=keepdims, dtype=dtype)


def std(x, axis=None, keepdims=False):
    return T.std(x, axis=axis, keepdims=keepdims)


def any(x, axis=None, keepdims=False):
    """Bitwise reduction (logical OR).
    """
    return T.any(x, axis=axis, keepdims=keepdims)


def argmax(x, axis=-1, keepdims=False):
    return T.argmax(x, axis=axis, keepdims=keepdims)


def arange(start, stop=None, step=1, dtype=None):
    return T.arange(start=start, stop=stop, step=step, dtype=dtype)


def argsort(x, axis=-1):
    return T.argsort(x, axis)


def argtop_k(x, k=1):
    # top-k accuracy
    top = T.argsort(x, axis=-1)
    # (Theano cannot index with [..., -top_k:], we need to simulate that)
    top = top[[slice(None) for _ in range(top.ndim - 1)] +
              [slice(-k, None)]]
    top = top[(slice(None),) * (top.ndim - 1) + (slice(None, None, -1),)]
    return top


def argmin(x, axis=-1):
    return T.argmin(x, axis=axis, keepdims=False)


def square(x):
    input_shape = shape(x)
    x = T.sqr(x)
    add_shape(x, input_shape)
    return x


def abs(x):
    input_shape = shape(x)
    x = T.abs_(x)
    add_shape(x, input_shape)
    return x


def inv(x):
    input_shape = shape(x)
    x = T.inv(x)
    add_shape(x, input_shape)
    return x


def sqrt(x):
    x = T.clip(x, 0., np.inf)
    return T.sqrt(x)


def exp(x):
    input_shape = shape(x)
    x = T.exp(x)
    add_shape(x, input_shape)
    return x


def log(x):
    input_shape = shape(x)
    x = T.log(x)
    add_shape(x, input_shape)
    return x


def round(x):
    input_shape = shape(x)
    x = T.round(x)
    add_shape(x, input_shape)
    return x


def pow(x, a):
    input_shape = shape(x)
    x = T.pow(x, a)
    add_shape(x, input_shape)
    return x


def clip(x, min_value, max_value):
    if max_value < min_value:
        max_value = min_value
    input_shape = shape(x)
    x = T.clip(x, min_value, max_value)
    add_shape(x, input_shape)
    return x


def maximum(x, y):
    return T.maximum(x, y)


def minimum(x, y):
    return T.minimum(x, y)


# ===========================================================================
# SHAPE OPERATIONS
# ===========================================================================
def reverse(x, axis=-1):
    """Apply [::-1] to appropriate axis"""
    if axis < 0:
        axis += x.ndim
    input_shape = shape(x)
    x = x[(slice(None),) * axis + (slice(None, None, -1),)]
    add_shape(x, input_shape)
    return x


def concatenate(tensors, axis=-1):
    return T.concatenate(tensors, axis=axis)


def reshape(x, shape_):
    """ x.shape = [25, 08, 12]
    reshape(shape=([1], [2], [0]))
    => x.shape = (08, 12, 25)
    """
    input_shape = shape(x)
    new_shape = []
    for i in shape_:
        if i is None:
            new_shape.append(-1)
        elif isinstance(i, (list, tuple)):
            new_shape.append(input_shape[i[0]])
        else:
            new_shape.append(i)
    x = T.reshape(x, tuple(new_shape))
    add_shape(x, new_shape)
    return x


def dimshuffle(x, pattern):
    """Transpose dimensions.

    pattern should be a tuple or list of
    dimension indices, e.g. [0, 2, 1].
    """
    pattern = tuple(pattern)
    input_shape = shape(x)
    new_shape = tuple([1 if i == 'x' else input_shape[i] for i in pattern])
    x = x.dimshuffle(pattern)
    add_shape(x, new_shape)
    return x


def repeat_elements(x, rep, axis):
    """Repeat the elements of a tensor along an axis, like np.repeat.

    If x has shape (s1, s2, s3) and axis=1, the output
    will have shape (s1, s2 * rep, s3).
    """
    return T.repeat(x, rep, axis=axis)


def resize_images(X, height_factor, width_factor, dim_ordering):
    """Resize the images contained in a 4D tensor of shape
    - [batch, channels, height, width] (for 'th' dim_ordering)
    - [batch, height, width, channels] (for 'tf' dim_ordering)
    by a factor of (height_factor, width_factor). Both factors should be
    positive integers.
    """
    if dim_ordering == 'th':
        output = repeat_elements(X, height_factor, axis=2)
        output = repeat_elements(output, width_factor, axis=3)
        return output
    elif dim_ordering == 'tf':
        output = repeat_elements(X, height_factor, axis=1)
        output = repeat_elements(output, width_factor, axis=2)
        return output
    else:
        raise Exception('Invalid dim_ordering: ' + dim_ordering)


def repeat(x, n):
    """Repeat a 2D tensor.

    If x has shape (samples, dim) and n=2,
    the output will have shape (samples, 2, dim).
    """
    assert x.ndim == 2
    x = x.dimshuffle((0, 'x', 1))
    return T.extra_ops.repeat(x, n, axis=1)


def tile(x, n):
    return T.tile(x, n)


def flatten(x, outdim=2):
    input_shape = shape(x)
    x = T.flatten(x, outdim)
    if outdim <= 1:
        input_shape = (None,) if None in input_shape else (int(np.prod(input_shape)),)
    else:
        input_shape = input_shape[:outdim - 1] + (int(np.prod(input_shape[outdim - 1:])),)
    add_shape(x, input_shape)
    return x


def expand_dims(x, dim=-1):
    """Add a 1-sized dimension at index "dim".
    """
    pattern = [i for i in range(x.type.ndim)]
    if dim < 0:
        if x.type.ndim == 0:
            dim = 0
        else:
            dim = dim % x.type.ndim + 1
    pattern.insert(dim, 'x')
    return dimshuffle(x, pattern)


def squeeze(x, axis):
    """Remove a 1-dimension from the tensor at index "axis".
    """
    input_shape = shape(x)
    x = T.addbroadcast(x, axis)
    x = T.squeeze(x)
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, tuple([j for i, j in enumerate(input_shape) if i != axis]))
    return x


def pad(x, axes=1, padding=1):
    """Pad the all dimension given in `axes` of a N-D tensor
    with "padding" zeros left and right.

    Example
    -------
    >>> X = [[1, 1, 1],
             [1, 1, 1]]
    >>> Y1 = pad(X, axes=1, padding=1)
    >>> Y1 = [[0, 1, 1, 1, 0],
              [0, 1, 1, 1, 0]]
    >>> Y2 = pad(X, axes=(0, 1), padding=1)
    >>> Y2 = [[0, 0, 0, 0, 0],
              [0, 1, 1, 1, 0],
              [0, 1, 1, 1, 0],
              [0, 0, 0, 0, 0]]
    """
    if not isinstance(axes, (tuple, list)):
        axes = (axes,)
    axes = tuple([i % x.ndim for i in axes])
    padding = as_tuple(padding, len(axes), int)

    input_shape = x.shape
    output_shape = tuple([input_shape[i] if i not in axes
                         else input_shape[i] + 2 * padding[axes.index(i)]
                         for i in range(x.ndim)])
    output = T.zeros(output_shape)
    indices = tuple([slice(None) if i not in axes
                    else slice(padding[axes.index(i)], input_shape[i] + padding[axes.index(i)])
                    for i in range(x.ndim)])
    input_shape = shape(x)
    x = T.set_subtensor(output[indices], x)
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, tuple([input_shape[i] if i not in axes or input_shape[i] is None
                            else input_shape[i] + 2 * padding[axes.index(i)]
                            for i in range(x.ndim)]))
    return x


def stack(*x):
    return T.stack(*x)


# ===========================================================================
# VALUE MANIPULATION
# ===========================================================================
def get_value(x, borrow=False):
    if not hasattr(x, 'get_value'):
        raise Exception("'get_value() can only be called on a variable. " +
                        "If you have an expression instead, use eval().")
    return x.get_value(borrow=borrow)


def set_value(x, value):
    x.set_value(np.asarray(value, dtype=x.dtype))


def set_subtensor(x, y):
    return T.set_subtensor(x, y)


# ===========================================================================
# GRAPH MANIPULATION
# ===========================================================================
class Function(object):

    def __init__(self, inputs, outputs, updates=[], **kwargs):
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
        if isinstance(updates, OrderedDict):
            updates = updates.items()
        # ====== add and reset global update ====== #
        self.function = theano.function(
            inputs, outputs,
            updates=updates,
            on_unused_input='raise', # TODO: remove this when stop testing
            allow_input_downcast=True, **kwargs)

    def __call__(self, *inputs):
        return self.function(*inputs)


def function(inputs, outputs, updates=[]):
    return Function(inputs, outputs, updates=updates)


def grad_clip(x, clip):
    """
    This clip the gradient of expression, used on forward pass but clip the
    gradient on backward pass

    This is an elemwise operation.

    Parameters
    ----------
    x: expression
        the variable we want its gradient inputs clipped
    lower_bound: float
        The lower bound of the gradient value
    upper_bound: float
        The upper bound of the gradient value.

    Example
    -------
    >>> x = theano.tensor.scalar()
    >>>
    >>> z = theano.tensor.grad(grad_clip(x, -1, 1)**2, x)
    >>> z2 = theano.tensor.grad(x**2, x)
    >>>
    >>> f = theano.function([x], outputs = [z, z2])
    >>>
    >>> print(f(2.0))  # output (1.0, 4.0)

    Note
    ----
    We register an opt in tensor/opt.py that remove the GradClip.
    So it have 0 cost in the forward and only do work in the grad.

    """
    return theano.gradient.grad_clip(x, -clip, clip)


def gradients(loss, variables, consider_constant=None, known_grads=None):
    """
    Return symbolic gradients for one or more variables with respect to some
    cost.

    For more information about how automatic differentiation works in Theano,
    see :mod:`gradient`. For information on how to implement the gradient of
    a certain Op, see :func:`grad`.

    Parameters
    ----------
    cost : scalar (0-dimensional) tensor variable or None
        Value with respect to which we are differentiating.  May be
        `None` if known_grads is provided.
    wrt : variable or list of variables
        term[s] for which we want gradients
    consider_constant : list of expressions(variables)
        expressions not to backpropagate through
    known_grads : dict, optional
        A dictionary mapping variables to their gradients. This is
        useful in the case where you know the gradient on some
        variables but do not know the original cost.
    Returns
    -------
    variable or list/tuple of variables (matches `wrt`)
        symbolic expression of gradient of `cost` with respect to each
        of the `wrt` terms.  If an element of `wrt` is not
        differentiable with respect to the output, then a zero
        variable is returned.

    Example
    -------
    >>> # For consider_constant:
    >>> a = T.variable(1.2)
    >>> b = T.variable(1.3)
    >>> x = a * b
    >>>
    >>> y = T.variable(2.)
    >>> z = T.variable(1.)
    >>>
    >>> z_pred = x * y
    >>> loss = T.pow((z - z_pred), 2)
    >>>
    >>> G = T.gradients(loss, [a, b, y], consider_constant=[x])
    >>>
    >>> for g in G:
    >>>     print(g.eval())
    >>> # a_grad=0. b_grad=0. y_grad=6.614
    """
    # TODO: float16 overflow, unsupport DeepCopyOps
    return T.grad(loss, wrt=variables,
        consider_constant=consider_constant, known_grads=known_grads,
        disconnected_inputs='raise')


def jacobian(loss, variables):
    return theano.gradient.jacobian(loss, variables, disconnected_inputs='warn')


def hessian(loss, variables):
    return theano.gradient.hessian(loss, variables, disconnected_inputs='warn')

# ===========================================================================
# CONTROL FLOW
# ===========================================================================


def scan(step_fn, sequences=None, outputs_info=None, non_sequences=None,
    n_steps=None, truncate_gradient=-1, go_backwards=False):
    return theano.scan(step_fn,
        sequences=sequences,
        outputs_info=outputs_info,
        non_sequences=non_sequences,
        n_steps=n_steps, truncate_gradient=truncate_gradient,
        go_backwards=go_backwards,
        strict=False)


def loop(step_fn, n_steps,
    sequences=None, outputs_info=None, non_sequences=None,
    go_backwards=False):
    """
    Helper function to unroll for loops. Can be used to unroll theano.scan.
    The parameter names are identical to theano.scan, please refer to here
    for more information.

    Note that this function does not support the truncate_gradient
    setting from theano.scan.

    Parameters
    ----------
    step_fn : function
        Function that defines calculations at each step.

    sequences : TensorVariable or list of TensorVariables
        List of TensorVariable with sequence data. The function iterates
        over the first dimension of each TensorVariable.

    outputs_info : list of TensorVariables
        List of tensors specifying the initial values for each recurrent
        value. Specify output_info to None for non-arguments to
        the step_function

    non_sequences: list of TensorVariables
        List of theano.shared variables that are used in the step function.

    n_steps: int
        Number of steps to unroll.

    go_backwards: bool
        If true the recursion starts at sequences[-1] and iterates
        backwards.

    Returns
    -------
    List of TensorVariables. Each element in the list gives the recurrent
    values at each time step.

    """
    if not isinstance(sequences, (list, tuple)):
        sequences = [] if sequences is None else [sequences]

    # When backwards reverse the recursion direction
    counter = range(n_steps)
    if go_backwards:
        counter = counter[::-1]

    output = []
    # ====== check if outputs_info is None ====== #
    if outputs_info is not None:
        prev_vals = outputs_info
    else:
        prev_vals = []
    output_idx = [i for i in range(len(prev_vals)) if prev_vals[i] is not None]
    # ====== check if non_sequences is None ====== #
    if non_sequences is None:
        non_sequences = []
    # ====== Main loop ====== #
    for i in counter:
        step_input = [s[i] for s in sequences] + \
                     [prev_vals[idx] for idx in output_idx] + \
            non_sequences
        out_ = step_fn(*step_input)
        # The returned values from step can be either a TensorVariable,
        # a list, or a tuple.  Below, we force it to always be a list.
        if isinstance(out_, T.TensorVariable):
            out_ = [out_]
        if isinstance(out_, tuple):
            out_ = list(out_)
        output.append(out_)
        prev_vals = output[-1]

    # iterate over each scan output and convert it to same format as scan:
    # [[output11, output12,...output1n],
    # [output21, output22,...output2n],...]
    output_scan = []
    for i in range(len(output[0])):
        l = map(lambda x: x[i], output)
        output_scan.append(T.stack(*l))

    return output_scan


def rnn(step_function, inputs, initial_states,
        go_backwards=False, mask=None, constants=None):
    """Iterates over the time dimension of a tensor.
    Parameters
    ----------
    inputs: tensor of temporal data of shape (samples, time, ...)
        (at least 3D).
    step_function:
        Parameters:
            input: tensor with shape (samples, ...) (no time dimension),
                representing input for the batch of samples at a certain
                time step.
            states: list of tensors.
        Returns:
            output: tensor with shape (samples, ...) (no time dimension),
            new_states: list of tensors, same length and shapes
                as 'states'.
    initial_states: tensor with shape (samples, ...) (no time dimension),
        containing the initial values for the states used in
        the step function.
    go_backwards: boolean. If True, do the iteration over
        the time dimension in reverse order.
    mask: binary tensor with shape (samples, time),
        with a zero for every element that is masked.
    constants: a list of constant values passed at each step.
    Returns
    -------
    A tuple (last_output, outputs, new_states).
        last_output: the latest output of the rnn, of shape (samples, ...)
        outputs: tensor with shape (samples, time, ...) where each
            entry outputs[s, t] is the output of the step function
            at time t for sample s.
        new_states: list of tensors, latest states returned by
            the step function, of shape (samples, ...).
    """
    ndim = inputs.ndim
    assert ndim >= 3, 'Input should be at least 3D.'

    axes = [1, 0] + list(range(2, ndim))
    inputs = inputs.dimshuffle(axes)

    if mask is not None:
        if mask.ndim == ndim - 1:
            mask = expand_dims(mask)
        assert mask.ndim == ndim
        mask = mask.dimshuffle(axes)

        if constants is None:
            constants = []
        # build an all-zero tensor of shape (samples, output_dim)
        initial_output = step_function(inputs[0], initial_states + constants)[0] * 0
        # Theano gets confused by broadcasting patterns in the scan op
        initial_output = T.unbroadcast(initial_output, 0, 1)

        def _step(input, mask, output_tm1, *states):
            output, new_states = step_function(input, states)
            # output previous output if masked.
            output = T.switch(mask, output, output_tm1)
            return_states = []
            for state, new_state in zip(states, new_states):
                return_states.append(T.switch(mask, new_state, state))
            return [output] + return_states

        results, _ = theano.scan(
            _step,
            sequences=[inputs, mask],
            outputs_info=[initial_output] + initial_states,
            non_sequences=constants,
            go_backwards=go_backwards)
    else:
        def _step(input, *states):
            output, new_states = step_function(input, states)
            return [output] + new_states

        results, _ = theano.scan(
            _step,
            sequences=inputs,
            outputs_info=[None] + initial_states,
            non_sequences=constants,
            go_backwards=go_backwards)

    # deal with Theano API inconsistency
    if type(results) is list:
        outputs = results[0]
        states = results[1:]
    else:
        outputs = results
        states = []

    outputs = T.squeeze(outputs)
    last_output = outputs[-1]

    axes = [1, 0] + list(range(2, outputs.ndim))
    outputs = outputs.dimshuffle(axes)
    states = [T.squeeze(state[-1]) for state in states]
    return last_output, outputs, states


def switch(condition, then_expression, else_expression):
    """condition: scalar tensor.
    """
    return T.switch(condition, then_expression, else_expression)


# ===========================================================================
# Comparator
# ===========================================================================
def neq(a, b):
    """a != b"""
    return T.neq(a, b)


def eq(a, b):
    """a == b"""
    return T.eq(a, b)


def gt(a, b):
    """a > b"""
    return T.gt(a, b)


def ge(a, b):
    """a >= b"""
    return T.ge(a, b)


def lt(a, b):
    """a < b"""
    return T.lt(a, b)


def le(a, b):
    """a <= b"""
    return T.le(a, b)


def one_hot(x, nb_class):
    """ x: 1D-integer vector """
    ret = T.zeros((x.shape[0], nb_class), dtype=FLOATX)
    ret = T.set_subtensor(ret[T.arange(x.shape[0]), x], 1)
    return ret


def confusion_matrix(y_pred, y_true, labels=None):
    """
    Computes the confusion matrix of given vectors containing
    actual observations and predicted observations.
    Parameters
    ----------
    pred : 1-d or 2-d tensor variable
    actual : 1-d or 2-d tensor variable
    labels : array, shape = [n_classes], optional
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If none is given, those that appear at least once
        in ``y_true`` or ``y_pred`` are used in sorted order.

    Returns
    -------
    conf_mat : Confusion matrix of actual and predictions observations as shown below.
               | Predicted
    ___________|___________
       Actual  |
               |
    Examples
    --------
    >>> import theano
    >>> from theano.tensor.nnet import confusion_matrix
    >>> x = theano.tensor.vector()
    >>> y = theano.tensor.vector()
    >>> f = theano.function([x, y], confusion_matrix(x, y))
    >>> a = [0, 1, 2, 1, 0]
    >>> b = [0, 0, 2, 1, 2]
    >>> print(f(a, b))
    [array([[0, 0, 1],
            [2, 1, 0],
            [0, 0, 1]]), array([ 0.,  1.,  2.])]
    """
    if y_true.ndim == 2:
        y_true = T.argmax(y_true, axis=-1)
    elif y_true.ndim != 1:
        raise ValueError('actual must be 1-d or 2-d tensor variable')
    if y_pred.ndim == 2:
        y_pred = T.argmax(y_pred, axis=-1)
    elif y_pred.ndim != 1:
        raise ValueError('pred must be 1-d or 2-d tensor variable')

    if labels is None:
        labels = T.extra_ops.Unique(False, False, False)(T.concatenate([y_true, y_pred]))

    colA = y_true.dimshuffle(0, 'x')
    colP = y_pred.dimshuffle(0, 'x')

    oneHotA = T.eq(colA, labels).astype('int64')
    oneHotP = T.eq(colP, labels).astype('int64')

    conf_mat = T.dot(oneHotA.T, oneHotP)
    return conf_mat


def one_hot_max(x, axis=-1):
    """
    Example
    -------
    >>> Input: [[0.0, 0.0, 0.5],
    >>>         [0.0, 0.3, 0.1],
    >>>         [0.6, 0.0, 0.2]]
    >>> Output: [[0.0, 0.0, 1.0],
    >>>         [0.0, 1.0, 0.0],
    >>>         [1.0, 0.0, 0.0]]
    """
    return T.cast(
        T.eq(T.arange(x.shape[axis])[None, :],
             T.argmax(x, axis=axis, keepdims=True)),
        FLOATX
    )


def apply_mask(x, mask):
    """
    x : 3D tensor
    mask : 2D tensor

    Example
    -------
    >>> Input: [128, 500, 120]
    >>> Mask:  [1, 1, 0]
    >>> Output: [128, 500, 0]
    """
    return T.mul(x, expand_dims(mask, -1))
