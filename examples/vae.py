from __future__ import print_function, absolute_import, division

import os
os.environ['ODIN'] = 'cpu,float32,theano'

import numpy as np

from blocks import backend as K
from blocks import nnet as N
from blocks import fuel
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks import roles

import theano

ds = fuel.load_mnist()
print(ds)


X = K.placeholder(shape=(None,) + ds['X_train'].shape[1:], name='X',
                  for_training=True)
y = K.placeholder(shape=(None,), name='y')

# ==================== Building model ==================== #
y = K.flatten(X)
h1 = N.Dense(512, activation=N.activations.sigmoid)
y = h1(y)

# variational
h2 = N.VariationalDense(128, activation=K.linear)
y = h2(y)

h3 = N.Dense(512, activation=N.activations.sigmoid)
y = h3(y)

y = h1.T(y)

graph = ComputationGraph(y)
mean = VariableFilter(roles=roles.VARIATIONAL_MEAN)(graph.variables)[0]
logsigma = VariableFilter(roles=roles.VARIATIONAL_LOGSIGMA)(graph.variables)[0]
kl = K.mean(K.kl_gaussian(mean, logsigma, prior_mu=0., prior_logsigma=0.))

f = K.function(X, kl)
print(f(np.random.uniform(size=(16, 28, 28))))
print(f(np.random.rand(16, 28, 28)))
print(f(np.random.randn(16, 28, 28)))
