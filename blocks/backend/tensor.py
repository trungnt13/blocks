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
from theano.tensor.signal import pool
from theano.tensor.nnet import conv3d2d
from theano.gof import graph
from theano.gof.graph import Constant
from theano.tensor.shared_randomstreams import RandomStateSharedVariable
from theano.tensor.sharedvar import SharedVariable
try:
    from theano.tensor.nnet.nnet import softsign as T_softsign
except ImportError:
    from theano.sandbox.softsign import softsign as T_softsign

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


def add_shape(var, shape, override=True):
    if not isinstance(shape, (tuple, list)):
        shape = (shape,)
    if len(shape) != var.ndim:
        raise ValueError('Variable has ndim={} but given shape has ndim={}'
                         '.'.format(var.ndim, len(shape)))
    # ====== NO override ====== #
    if (not override and hasattr(var.tag, 'shape') and
            isinstance(var.tag.shape, (tuple, list))):
        return
    # ====== override or assign ====== #
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


def is_graph_input(variable):
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
    """ A variable is in TRAINING mode if at least one of its inputs has
    training role.

    Note
    ----
    TRAINING role can be override by: ``add_role(x, DEPLOYING)``

    """
    if not isinstance(v, (tuple, list)):
        v = [v]
    inputs = graph.inputs(v)
    for i in inputs:
        if is_graph_input(i) and has_roles(i, TRAINING, exact=True):
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
    # global _PLACEHOLDER_ID
    # name_prefix = '[ID%02d]' % _PLACEHOLDER_ID
    # _PLACEHOLDER_ID += 1
    # name = [name_prefix] if name is None else [name_prefix, name]
    # name = ''.join(name)
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


def int_shape(x, none=True):
    s = shape(x, none=True)
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
    shapeX = shape(x, none=True)
    shapeY = shape(y)
    if isinstance(shapeX, (tuple, list)) and isinstance(shapeY, (tuple, list)):
        add_shape(output, shapeX[:-1] + shapeY[1:])
    return output


def transpose(x, axes=None):
    output_shape = shape(x, none=True)
    x = T.transpose(x, axes=axes)
    if isinstance(output_shape, (tuple, list)):
        if axes is None:
            output_shape = output_shape[::-1]
        else:
            output_shape = [output_shape[i] for i in axes]
        add_shape(x, tuple(output_shape))
    return x


def gather(reference, indices):
    """reference: a tensor.
    indices: an int tensor of indices.

    Return: a tensor of same type as reference.
    """
    return reference[indices]


def diag(x):
    input_shape = shape(x, none=True)
    x = T.diag(x)
    if isinstance(input_shape, (tuple, list)):
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


def add(x, y):
    return T.add(x, y)


def sub(x, y):
    return T.sub(x, y)


def mul(x, y):
    return T.mul(x, y)


def div(x, y):
    return T.true_div(x, y)


def mod(x, y):
    return T.mod(x, y)


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
    input_shape = shape(x, none=True)
    x = T.sqr(x)
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, input_shape)
    return x


def abs(x):
    input_shape = shape(x, none=True)
    x = T.abs_(x)
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, input_shape)
    return x


def inv(x):
    input_shape = shape(x, none=True)
    x = T.inv(x)
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, input_shape)
    return x


def sqrt(x):
    input_shape = shape(x, none=True)
    x = T.clip(x, 0., np.inf)
    x = T.sqrt(x)
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, input_shape)
    return x


def exp(x):
    input_shape = shape(x, none=True)
    x = T.exp(x)
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, input_shape)
    return x


def log(x):
    input_shape = shape(x, none=True)
    x = T.log(x)
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, input_shape)
    return x


def round(x):
    input_shape = shape(x, none=True)
    x = T.round(x)
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, input_shape)
    return x


def pow(x, a):
    input_shape = shape(x, none=True)
    x = T.pow(x, a)
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, input_shape)
    return x


def clip(x, min_value, max_value):
    if max_value < min_value:
        max_value = min_value
    input_shape = shape(x, none=True)
    x = T.clip(x, min_value, max_value)
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, input_shape)
    return x


def maximum(x, y):
    x = T.maximum(x, y)
    output_shape = shape(y)
    if isinstance(output_shape, (tuple, list)):
        add_shape(x, output_shape)
    return x


def minimum(x, y):
    x = T.minimum(x, y)
    output_shape = shape(y)
    if isinstance(output_shape, (tuple, list)):
        add_shape(x, output_shape)
    return x


# ===========================================================================
# SHAPE OPERATIONS
# ===========================================================================
def reverse(x, axis=-1):
    """Apply [::-1] to appropriate axis"""
    if axis < 0:
        axis += x.ndim
    input_shape = shape(x, none=True)
    x = x[(slice(None),) * axis + (slice(None, None, -1),)]
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, input_shape, override=True)
    return x


def concatenate(tensors, axis=-1):
    return T.concatenate(tensors, axis=axis)


def reshape(x, shape_):
    """ x.shape = [25, 08, 12]
    reshape(shape=([1], [2], [0]))
    => x.shape = (08, 12, 25)
    """
    input_shape = shape(x, none=True)
    new_shape = []
    for i in shape_:
        if i is None:
            new_shape.append(-1)
        elif isinstance(i, (list, tuple)):
            new_shape.append(input_shape[i[0]])
        else:
            new_shape.append(i)
    x = T.reshape(x, tuple(new_shape))
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, new_shape, override=True)
    return x


def dimshuffle(x, pattern):
    """Transpose dimensions.

    pattern should be a tuple or list of
    dimension indices, e.g. [0, 2, 1].
    """
    pattern = tuple(pattern)
    input_shape = shape(x, none=True)
    new_shape = tuple([1 if i == 'x' else input_shape[i] for i in pattern])
    x = x.dimshuffle(pattern)
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, new_shape, override=True)
    return x


def repeat(x, n, axes=None):
    """Repeat a N-D tensor.

    If x has shape (s1, s2, s3) and axis=(1, -1), the output
    will have shape (s1, s2 * n[0], s3 * n[1]).
    """
    input_shape = shape(x, none=True)
    if axes is not None:
        if not isinstance(axes, (tuple, list)):
            axes = (axes,)
        axes = tuple([i % x.ndim for i in axes])
        n = as_tuple(n, len(axes), int)
        for i, j in zip(n, axes):
            x = T.extra_ops.repeat(x, repeats=i, axis=j)
    else:
        x = T.extra_ops.repeat(x, n, None)
    if isinstance(input_shape, (tuple, list)):
        if axes is None and None not in input_shape:
            add_shape(x, int(np.prod(input_shape) * n), override=True)
        else:
            add_shape(x, tuple([j if i not in axes or j is None
                                else j * n[axes.index(i)]
                                for i, j in enumerate(input_shape)]), override=True)
    return x


def tile(x, n):
    return T.tile(x, n)


def flatten(x, outdim=2):
    input_shape = shape(x, none=True)
    x = T.flatten(x, outdim)
    if outdim <= 1:
        input_shape = (None,) if None in input_shape else (int(np.prod(input_shape)),)
    else:
        input_shape = input_shape[:outdim - 1] + (int(np.prod(input_shape[outdim - 1:])),)
    add_shape(x, input_shape, override=True)
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
    input_shape = shape(x, none=True)
    x = T.addbroadcast(x, axis)
    x = T.squeeze(x)
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, tuple([j for i, j in enumerate(input_shape) if i != axis]), override=True)
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
    input_shape = shape(x, none=True)
    x = T.set_subtensor(output[indices], x)
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, tuple([input_shape[i] if i not in axes or input_shape[i] is None
                            else input_shape[i] + 2 * padding[axes.index(i)]
                            for i in range(x.ndim)]), override=True)
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


def recurrent(*args, **kwargs):
    """Wraps an apply method to allow its iterative application.

    This decorator allows you to implement only one step of a recurrent
    network and enjoy applying it to sequences for free. The idea behind is
    that its most general form information flow of an RNN can be described
    as follows: depending on the context and driven by input sequences the
    RNN updates its states and produces output sequences.

    Given a method describing one step of an RNN and a specification
    which of its inputs are the elements of the input sequence,
    which are the states and which are the contexts, this decorator
    returns an application method which implements the whole RNN loop.
    The returned application method also has additional parameters,
    see documentation of the `recurrent_apply` inner function below.

    Parameters
    ----------
    sequences : list of strs
        Specifies which of the arguments are elements of input sequences.
    states : list of strs
        Specifies which of the arguments are the states.
    contexts : list of strs
        Specifies which of the arguments are the contexts.
    outputs : list of strs
        Names of the outputs. The outputs whose names match with those
        in the `state` parameter are interpreted as next step states.

    Returns
    -------
    recurrent_apply : :class:`~blocks.bricks.base.Application`
        The new application method that applies the RNN to sequences.

    See Also
    --------
    :doc:`The tutorial on RNNs </rnn>`

    """
    def recurrent_wrapper(application_function):
        arg_spec = inspect.getargspec(application_function)
        arg_names = arg_spec.args[1:]

        @wraps(application_function)
        def recurrent_apply(brick, application, application_call,
                            *args, **kwargs):
            """Iterates a transition function.

            Parameters
            ----------
            iterate : bool
                If ``True`` iteration is made. By default ``True``.
            reverse : bool
                If ``True``, the sequences are processed in backward
                direction. ``False`` by default.
            return_initial_states : bool
                If ``True``, initial states are included in the returned
                state tensors. ``False`` by default.

            """
            # Extract arguments related to iteration and immediately relay the
            # call to the wrapped function if `iterate=False`
            iterate = kwargs.pop('iterate', True)
            if not iterate:
                return application_function(brick, *args, **kwargs)
            reverse = kwargs.pop('reverse', False)
            scan_kwargs = kwargs.pop('scan_kwargs', {})
            return_initial_states = kwargs.pop('return_initial_states', False)

            # Push everything to kwargs
            for arg, arg_name in zip(args, arg_names):
                kwargs[arg_name] = arg

            # Make sure that all arguments for scan are tensor variables
            scan_arguments = (application.sequences + application.states +
                              application.contexts)
            for arg in scan_arguments:
                if arg in kwargs:
                    if kwargs[arg] is None:
                        del kwargs[arg]
                    else:
                        kwargs[arg] = tensor.as_tensor_variable(kwargs[arg])

            # Check which sequence and contexts were provided
            sequences_given = dict_subset(kwargs, application.sequences,
                                          must_have=False)
            contexts_given = dict_subset(kwargs, application.contexts,
                                         must_have=False)

            # Determine number of steps and batch size.
            if len(sequences_given):
                # TODO Assumes 1 time dim!
                shape = list(sequences_given.values())[0].shape
                n_steps = shape[0]
                batch_size = shape[1]
            else:
                # TODO Raise error if n_steps and batch_size not found?
                n_steps = kwargs.pop('n_steps')
                batch_size = kwargs.pop('batch_size')

            # Handle the rest kwargs
            rest_kwargs = {key: value for key, value in kwargs.items()
                           if key not in scan_arguments}
            for value in rest_kwargs.values():
                if (isinstance(value, Variable) and not
                        is_shared_variable(value)):
                    logger.warning("unknown input {}".format(value) +
                                   unknown_scan_input)

            # Ensure that all initial states are available.
            initial_states = brick.initial_states(batch_size, as_dict=True,
                                                  *args, **kwargs)
            for state_name in application.states:
                dim = brick.get_dim(state_name)
                if state_name in kwargs:
                    if isinstance(kwargs[state_name], NdarrayInitialization):
                        kwargs[state_name] = tensor.alloc(
                            kwargs[state_name].generate(brick.rng, (1, dim)),
                            batch_size, dim)
                    elif isinstance(kwargs[state_name], Application):
                        kwargs[state_name] = (
                            kwargs[state_name](state_name, batch_size,
                                               *args, **kwargs))
                else:
                    try:
                        kwargs[state_name] = initial_states[state_name]
                    except KeyError:
                        raise KeyError(
                            "no initial state for '{}' of the brick {}".format(
                                state_name, brick.name))
            states_given = dict_subset(kwargs, application.states)

            # Theano issue 1772
            for name, state in states_given.items():
                states_given[name] = tensor.unbroadcast(state,
                                                        *range(state.ndim))

            def scan_function(*args):
                args = list(args)
                arg_names = (list(sequences_given) +
                             [output for output in application.outputs
                              if output in application.states] +
                             list(contexts_given))
                kwargs = dict(equizip(arg_names, args))
                kwargs.update(rest_kwargs)
                outputs = application(iterate=False, **kwargs)
                # We want to save the computation graph returned by the
                # `application_function` when it is called inside the
                # `theano.scan`.
                application_call.inner_inputs = args
                application_call.inner_outputs = pack(outputs)
                return outputs
            outputs_info = [
                states_given[name] if name in application.states
                else None
                for name in application.outputs]
            result, updates = theano.scan(
                scan_function, sequences=list(sequences_given.values()),
                outputs_info=outputs_info,
                non_sequences=list(contexts_given.values()),
                n_steps=n_steps,
                go_backwards=reverse,
                name='{}_{}_scan'.format(
                    brick.name, application.application_name),
                **scan_kwargs)
            result = pack(result)
            if return_initial_states:
                # Undo Subtensor
                for i in range(len(states_given)):
                    assert isinstance(result[i].owner.op,
                                      tensor.subtensor.Subtensor)
                    result[i] = result[i].owner.inputs[0]
            if updates:
                application_call.updates = dict_union(application_call.updates,
                                                      updates)

            return result

        return recurrent_apply

    # Decorator can be used with or without arguments
    assert (args and not kwargs) or (not args and kwargs)
    if args:
        application_function, = args
        return application(recurrent_wrapper(application_function))
    else:
        def wrap_application(application_function):
            return application(**kwargs)(
                recurrent_wrapper(application_function))
        return wrap_application


# ===========================================================================
# Comparator
# ===========================================================================
def switch(condition, then_expression, else_expression):
    """condition: scalar tensor.
    """
    return T.switch(condition, then_expression, else_expression)


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


# ===========================================================================
# Ops
# ===========================================================================
def relu(x, alpha=0.):
    input_shape = shape(x, none=True)
    x = T.nnet.relu(x, alpha)
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, input_shape, override=True)
    return x


def softmax(x):
    input_shape = shape(x, none=True)
    x = T.nnet.softmax(x)
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, input_shape, override=True)
    return x


def softplus(x):
    input_shape = shape(x, none=True)
    x = T.nnet.softplus(x)
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, input_shape, override=True)
    return x


def softsign(x):
    input_shape = shape(x, none=True)
    x = T_softsign(x)
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, input_shape, override=True)
    return x


def linear(x):
    return x


def sigmoid(x):
    input_shape = shape(x, none=True)
    x = T.nnet.sigmoid(x)
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, input_shape, override=True)
    return x


def hard_sigmoid(x):
    input_shape = shape(x, none=True)
    x = T.nnet.hard_sigmoid(x)
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, input_shape, override=True)
    return x


def tanh(x):
    input_shape = shape(x, none=True)
    x = T.tanh(x)
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, input_shape, override=True)
    return x


def categorical_crossentropy(output, target):
    input_shape = shape(output)
    x = T.nnet.categorical_crossentropy(output, target)
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, (input_shape[0],), override=True)
    return x


def binary_crossentropy(output, target):
    input_shape = shape(output)
    x = T.nnet.binary_crossentropy(output, target)
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, (input_shape[0],), override=True)
    return x


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
    if on_gpu(): # Using DNN on GPU
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


def pool2d(x, pool_size=(2, 2), ignore_border=True,
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
    if mode != 'sum' and on_gpu():
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
    input_shape = shape(x, none=True)
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
    add_shape(pool_out, tuple(output_shape), override=True)
    return pool_out


def pool3d(x, pool_size=(2, 2, 2), ignore_border=True,
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
    if mode != 'sum' and on_gpu():
        from theano.sandbox.cuda import dnn
        if not ignore_border:
            raise ValueError('CuDNN does not support ignore_border = False.')
        pool_out = dnn.dnn_pool(x, ws=pool_size, stride=strides, mode=mode, pad=pad)
    # ====== Use default Theano implementation ====== #
    else:
        if len(set(pad)) > 1:
            raise ValueError('Only support same padding on CPU.')
        padding = (pad[0], pad[0])
        output = pool.pool_2d(input=dimshuffle(x, (0, 1, 4, 3, 2)),
                              ds=(pool_size[1], pool_size[0]),
                              st=(strides[1], strides[0]),
                              ignore_border=ignore_border,
                              padding=padding,
                              mode=mode)
        # pooling over conv_dim3
        pool_out = pool.pool_2d(input=dimshuffle(output, (0, 1, 4, 3, 2)),
                                ds=(1, pool_size[2]),
                                st=(1, strides[2]),
                                ignore_border=ignore_border,
                                padding=padding,
                                mode=mode)
    # ====== Estimate output shape ====== #
    input_shape = shape(x, none=True)
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
    add_shape(pool_out, tuple(output_shape), override=True)
    return pool_out


def poolWTA(x, pool_size=(2, 2), axis=1):
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
    input_shape = shape(x, none=True)
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

    input_reshaped = reshape(x, pool_shape)
    max_indices = argmax(input_reshaped, axis=axis + 1, keepdims=True)

    arange = T.arange(pool_size).dimshuffle(*arange_shuffle_pattern)
    mask = reshape(T.eq(max_indices, arange), input_shape)
    output = x * mask
    add_shape(output, input_shape, override=True)
    return output


def poolGlobal(x, pool_function=mean):
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
    input_shape = shape(x, none=True)
    x = pool_function(T.flatten(x, 3), axis=2)
    add_shape(x, input_shape[:2], override=True)
    return x
