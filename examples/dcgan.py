# -*- coding: utf-8 -*-
# Original implementation and credits:
# https://github.com/soumith/dcgan.torch
# Alec Radford, Luke Metz, Soumith Chintala
# ===========================================================================
# Some useful note from author:
# * Replace any pooling layers with strided convolutions (discriminator) and
#   fractional-strided convolutions (generator).
# * Use batchnorm in both the generator and the discriminator
# * Remove fully connected hidden layers, just use average pooling at the end.
# * ReLU in generator, and Tanh for the output
# * Use LeakyReLU activation in the discriminator for all layers.
# This network can be used for:
# * Use the discriminator as a pre-trained net for CIFAR-10 classification.
# * show the interpolated latent space, where transitions are really smooth and
#   every image in the latent space is a bedroom.
# * figure out a way to identify and remove filters that draw windows in generation.
# * Control the generator to not output certain objects.
# e.g. Smiling woman - neutral woman + neutral man = Smiling man. Whuttttt!
# (learnt a latent space in a completely unsupervised fashion where
#  ROTATIONS ARE LINEAR in this latent space.)
# ===========================================================================
from __future__ import print_function, division, absolute_import

import os
os.environ['ODIN'] = 'cpu,theano,float32'

import numpy as np

from blocks import backend as K
from blocks import nnet as N


# ===========================================================================
# Generator is an Autoencoder
# ===========================================================================
N_gen = N.Sequence([
    N.Conv2D(64, filter_size=(4, 4)),
    N.BatchNorm(activation=N.activations.rectify),
    N.Conv2D(32, filter_size=(4, 4), stride=(2, 2), pad='same'),
    N.BatchNorm(activation=N.activations.rectify),
    N.Conv2D(16, filter_size=(4, 4), stride=(2, 2), pad='same'),
    N.BatchNorm(activation=N.activations.rectify),
    N.Conv2D(8, filter_size=(4, 4), stride=(2, 2), pad='same', activation=N.activations.tanh),
])
N_gen += N_gen.T

# ===========================================================================
# Header
# ===========================================================================
N_dis = N.Sequence([
    N_gen[-1].T,
    lambda x:N.activations.rectify(x, alpha=0.2),
    N_gen[-3].T,
    N.BatchNorm(activation=lambda x:N.activations.rectify(x, alpha=0.2)),
    N_gen[-5].T,
    N.BatchNorm(activation=lambda x:N.activations.rectify(x, alpha=0.2)),
    N_gen[-7].T,
    N.activations.sigmoid
])
# ===========================================================================
# TRaining objectives:
# * Generator: MSE + maximum D(G(z))
# * Discriminator: binary_entropy(fake, target)
# ===========================================================================
