# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
os.environ['ODIN'] = 'cpu,theano,float32'
import cPickle
from six.moves import zip, zip_longest

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
try:
    import seaborn # for pretty plot
except:
    pass

from blocks import backend as K
from blocks import nnet as N
from blocks.utils import Progbar
from blocks.utils.decorators import functionable
from blocks import fuel, graph, algorithms, visual

# ===========================================================================
# Const
# ===========================================================================
input_shape = (None, 1)
mu = -2.
std = 0.3


# ===========================================================================
# Helper
# ===========================================================================
def draw_training_samples(N=256):
    # x ∼ normal(-1,1)
    N = N * input_shape[-1]
    return sp.stats.norm.rvs(loc=mu, scale=std, size=N).reshape(-1, input_shape[-1])


def draw_latent_samples(N=256):
    # z ∼ uniform(0,1)
    N = N * input_shape[-1]
    return (np.float32(np.linspace(-5.0, 5.0, N) +
            np.random.random(N))).reshape(-1, input_shape[-1])


def visual_training(f_gen, f_dis):
    # p_data
    N = 2000
    xs = np.linspace(-5, 5, N)

    # true distribution
    plt.plot(xs, sp.stats.norm.pdf(xs, loc=mu, scale=std))
    plt.vlines(x=mu, ymin=0., ymax=2., color='r')
    # discriminator decision boundary
    if f_dis is not None:
        DX = f_dis(xs.reshape(-1, input_shape[-1]))
        plt.plot(xs, DX, label='decision boundary')
    # generator distribution
    if f_gen is not None:
        GZ = f_gen(draw_latent_samples(N))
        plt.hist(GZ, bins=12, normed=True, alpha=0.8)
    plt.xlim(-6, 6)
    plt.show(block=True)

# ===========================================================================
# Create networks
# ===========================================================================
Z = K.placeholder(shape=input_shape, name='Z')
X = K.placeholder(shape=input_shape, name='X')

# generator
G_net = N.Sequence([
    N.Dense(10, activation=N.activations.rectify),
    N.Dense(10, activation=N.activations.rectify),
    N.Dense(input_shape[-1], activation=N.activations.linear)
])
# discriminators
D_net = N.Sequence([
    N.Dense(10, activation=N.activations.tanh),
    N.Dense(10, activation=N.activations.tanh),
    N.Dense(1, activation=N.activations.sigmoid)
])

# ===========================================================================
# Output Variables
# ===========================================================================
Gz = G_net(Z)
Dx = D_net(X)
DGz = D_net(Gz)

Gz.name = 'G(z)'
Dx.name = 'D(x)'
DGz.name = 'D(G(z))'
# output
f_gen = K.function(Z, Gz)
f_dis = K.function(X, Dx)

# objectives
G_obj = 1 - K.mean(K.log(DGz))
D_obj = 1 - K.mean(K.log(Dx) + K.log(1 - DGz))

visual_training(f_gen, f_dis)

# ===========================================================================
# TRaining
# ===========================================================================
# !Remember: G_train is trained on parameters of Generative Network,
# vice versa, D_train is trained on parameters of Discriminative Network only.
G_train = algorithms.GradientDescent(cost=G_obj,
                                     parameters=G_net.parameters,
                                     step_rule=algorithms.Adam(learning_rate=0.002))
D_train = algorithms.GradientDescent(cost=D_obj,
                                     consider_constant=Gz,
                                     # parameters=D_net.parameters,
                                     step_rule=algorithms.Adam(learning_rate=0.002))

# training loop
epochs = 400
k = 20
M = 200  # mini-batch size
input_shape = tuple([-1 if i is None else i for i in input_shape])

p = Progbar(epochs * (20 + 1), title='Adversarial Training:')
discriminator_cost = []
generator_cost = []

plt.ion()
for i in range(epochs):
    for j in range(k):
        x = np.float32(np.random.normal(mu, std, M))  # sampled orginal batch
        z = draw_latent_samples(M)
        discriminator_cost.append(
            D_train.process_batch({
                'X': x.reshape(input_shape),
                'Z': z.reshape(input_shape)})
        )
        p.add(1)
    z = draw_latent_samples(M)
    p.add(1)
    generator_cost.append(
        G_train.process_batch({'Z': z.reshape(input_shape)})
    )
    if False: # doing interactive visualization
        plt.clf()
        visual_training(f_gen, f_dis)
        plt.draw()
plt.ioff()

print('Discriminator Cost:')
print(visual.print_bar(discriminator_cost, bincount=39))
print('Generator Cost:')
print(visual.print_bar(generator_cost, bincount=39))

plt.clf()
visual_training(f_gen, f_dis)
