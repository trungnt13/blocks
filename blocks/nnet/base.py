from __future__ import division, absolute_import

import inspect
from abc import ABCMeta, abstractmethod, abstractproperty
from six import add_metaclass
from six.moves import zip, range

import numpy as np

from blocks import backend as K
from blocks.graph import add_annotation, add_shape, Annotation
from blocks.roles import (add_role, has_roles, PARAMETER, VariableRole,
                          WEIGHT, BIAS, OUTPUT)
from blocks.utils.decorators import autoinit
from blocks.utils import np_utils


class NNConfig(object):

    @autoinit
    def __init__(self, **kwargs):
        super(NNConfig, self).__init__()
        self._paramters = []

    @property
    def parameters(self):
        return self._paramters

    def __getattr__(self, name):
        if name in self._arguments:
            return self._arguments[name]
        for i in self._paramters:
            if name == i.name:
                return i
        raise AttributeError('Cannot find attribute={} in arguments and parameters'
                             '.'.format(name))

    def create_params(self, spec, shape, name, roles=[], annotations=[]):
        if not isinstance(roles, (tuple, list)):
            roles = [roles]
        if not isinstance(annotations, (tuple, list)):
            annotations = [annotations]

        shape = tuple(shape)  # convert to tuple if needed
        if any(d <= 0 for d in shape):
            raise ValueError((
                "Cannot create param with a non-positive shape dimension. "
                "Tried to create param with shape=%r, name=%r") %
                (shape, name))

        #####################################
        # 1. Shared variable, just check the shape.
        if K.is_shared_variable(spec):
            spec_shape = K.eval(K.shape(spec))
            if shape is None:
                shape = spec_shape
            elif tuple(shape) != tuple(spec_shape):
                self.raise_arguments('Given variable has different shape '
                                     'from requirement, %s != %s' %
                                     (str(spec_shape), str(shape)))
        #####################################
        # 2. expression, we can only check number of dimension.
        elif K.is_variable(spec):
            # We cannot check the shape here, Theano expressions (even shared
            # variables) do not have a fixed compile-time shape. We can check the
            # dimensionality though.
            # Note that we cannot assign a name here. We could assign to the
            # `name` attribute of the variable, but the user may have already
            # named the variable and we don't want to override this.
            if shape is not None and K.ndim(spec) != len(shape):
                self.raise_arguments("parameter variable has %d dimensions, "
                                   "should be %d" % (spec.ndim, len(shape)))
        #####################################
        # 3. numpy ndarray, create shared variable wraper for it.
        elif isinstance(spec, np.ndarray):
            if shape is not None and spec.shape != shape:
                raise RuntimeError("parameter array has shape %s, should be "
                                   "%s" % (spec.shape, shape))
            spec = K.variable(spec, name=name)
        #####################################
        # 4. initializing function.
        elif hasattr(spec, '__call__'):
            arr = spec(shape)
            if K.is_shared_variable(arr):
                spec = arr
            elif K.is_variable(arr) and K.ndim(arr) == len(shape):
                spec = arr
            elif isinstance(arr, np.ndarray):
                spec = K.variable(arr, name=name)
        #####################################
        # 5. Exception.
        else:
            raise RuntimeError("cannot initialize parameters: 'spec' is not "
                               "a numpy array, a Theano expression, or a "
                               "callable")
        # ====== create and return params ====== #
        for i in roles:
            if isinstance(i, VariableRole):
                add_role(spec, i)
        for i in annotations:
            if isinstance(i, Annotation):
                add_annotation(spec, i)
        spec.name = name
        # return actual variable or expression
        for i, j in enumerate(self._paramters): # override other parameters with same name
            if j.name == name:
                self._paramters[i] = spec
        if spec not in self._paramters:
            self._paramters.append(spec)
        return spec

    def inflate(self, obj):
        """ Infate configuration into given object  """
        for i, j in self._arguments.iteritems():
            setattr(obj, i, j)
        for i in self._paramters:
            setattr(obj, i.name, i)

    def reset(self, obj):
        """  """
        for i in self._arguments.keys():
            setattr(obj, i, None)
        for i in self._paramters:
            setattr(obj, i.name, None)

    def __eq__(self, other):
        if hasattr(other, '_arguments'):
            other = other._arguments
        return self._arguments.__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def copy(self):
        config = NNConfig(**self._arguments)
        config._paramters = self._paramters
        return config

    def __str__(self):
        s = 'Arguments:\n'
        for i, j in self._arguments.iteritems():
            s += str(i) + ':' + str(j) + '\n'
        s += 'Parameters: ' + ', '.join([str(i) for i in self._paramters])
        return s


@add_metaclass(ABCMeta)
class NNOps(Annotation):

    ID = 0

    def __init__(self, name=None):
        super(NNOps, self).__init__()
        self._id = NNOps.ID
        self._name = str(name)
        self._configuration = None
        self._transpose_ops = None
        NNOps.ID += 1

    def _check_configuration(self):
        if self._configuration is None:
            raise Exception('Configuration have not been initilized.')

    # ==================== properties ==================== #
    @property
    def name(self):
        return '[' + str(self._id) + ']' + self.__class__.__name__ + '/' + self._name

    @property
    def T(self):
        """ Return new ops which is transpose of this ops """
        self._check_configuration()
        if self._transpose_ops is None:
            self._transpose_ops = self._transpose()
        return self._transpose_ops

    @property
    def parameters(self):
        self._check_configuration()
        return [i for i in self._configuration.parameters if has_roles(i, PARAMETER)]

    def config(self, *args, **kwargs):
        """
        Note
        ----
        New configuration will be created based on kwargs
        args only for setting the NNConfig directly
        """
        for i in args:
            if isinstance(i, NNConfig):
                self._configuration = i

        # initialized but mismatch configuration
        if self._configuration is not None:
            if len(kwargs) != 0 and self._configuration != kwargs:
                raise ValueError('Initialized configuration: {} is mismatch with new configuration, '
                                 'no support for kwargs={}'.format(self._configuration._arguments, kwargs))
        # not initialized but no information
        elif len(kwargs) == 0:
            raise ValueError('Configuration have not initialized.')

        # still None, initialize configuration
        if self._configuration is None:
            self._configuration = self._initialize(**kwargs)
        self._configuration.inflate(self)
        return self._configuration

    # ==================== abstract method ==================== #
    @abstractmethod
    def _initialize(self, *args, **kwargs):
        """ This function return NNConfig for given configuration from arg
        and kwargs
        """
        raise NotImplementedError

    @abstractmethod
    def _apply(self, *args, **kwargs):
        raise NotImplementedError

    def _transpose(self):
        raise NotImplementedError

    def apply(self, *args, **kwargs):
        out = self._apply(*args, **kwargs)
        # ====== add roles ====== #
        tmp = out
        if not isinstance(tmp, (tuple, list)):
            tmp = [out]
        for o in tmp:
            add_role(o, OUTPUT)
            add_annotation(o, self)
        # return outputs
        return out

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def __str__(self):
        return self.name


# ===========================================================================
# Simple ops
# ===========================================================================
class Dense(NNOps):

    @autoinit
    def __init__(self, num_units,
                 W_init=K.init.symmetric_uniform,
                 b_init=K.init.constant,
                 nonlinearity=K.relu,
                 **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.nonlinearity = (K.linear if nonlinearity is None else nonlinearity)
        # hack to prevent infinite useless loop of transpose
        self._original_dense = None

    # ==================== abstract methods ==================== #
    def _transpose(self):
        if self._original_dense is not None:
            return self._original_dense

        # flip the input and hidden
        num_inputs = self.num_units
        num_units = self.num_inputs
        # create the new dense
        transpose = Dense(num_units=num_units,
                          W_init=self.W_init, b_init=self.b_init,
                          nonlinearity=self.nonlinearity,
                          name=self._name + '_transpose')
        transpose._original_dense = self
        #create the config
        config = NNConfig(num_inputs=num_inputs)
        config.create_params(self.W.T, shape=(num_inputs, num_units), name='W')
        if self.b_init is not None:
            config.create_params(self.b_init, shape=(num_units,), name='b',
                                 roles=BIAS, annotations=transpose)
        # modify the config
        transpose.config(config)
        return transpose

    def _initialize(self, num_inputs):
        config = NNConfig(num_inputs=num_inputs)
        shape = (num_inputs, self.num_units)
        config.create_params(self.W_init, shape, 'W', WEIGHT, self)
        if self.b_init is not None:
            config.create_params(self.b_init, (self.num_units,), 'b', BIAS, self)
        return config

    def _apply(self, x):
        input_shape = K.shape(x)
        self.config(num_inputs=input_shape[-1])
        # calculate projection
        activation = K.dot(x, self.W)
        if hasattr(self, 'b') and self.b is not None:
            activation = activation + self.b
        activation = self.nonlinearity(activation)
        # set shape for output
        add_shape(activation, input_shape[:-1] + (self.num_units,))
        return activation
