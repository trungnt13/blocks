from __future__ import print_function, division, absolute_import

import os
os.environ['ODIN'] = 'cpu,theano,float32'

import numpy as np

from blocks import backend as K
from blocks import nnet as N
from blocks import fuel

ds = fuel.load_mnist()
print(ds)
Xshape = (None,) + ds['X_train'].shape[1:]
yshape = (None,)


# ===========================================================================
# Variables
# ===========================================================================
X = K.placeholder(shape=Xshape, name='X', for_training=True)
y = K.placeholder(shape=yshape, name='y')

x = K.dimshuffle(X, (0, 'x', 1, 2))

# ===========================================================================
# Model 1
# ===========================================================================
encoder = N.Sequence([
    N.Conv2D(32, (3, 3), stride=(2, 2), pad='same', activation=N.activations.linear),
    N.BatchNorm(),
    N.Conv2D(64, (3, 3), stride=(2, 2), pad='same', activation=N.activations.linear),
    N.Flatten(outdim=2),
    N.Dense(10, activation=N.activations.linear)
])

latent = encoder(x)
y_train = N.activations.softmax(latent)

decoder = encoder.T
X_reconstruct = decoder(latent)

# ===========================================================================
# Model 2
# ===========================================================================
