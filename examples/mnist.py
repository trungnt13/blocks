#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import types

import numpy as np

import os
os.environ['ODIN'] = 'float32,gpu,theano'

from blocks import algorithms
from blocks import backend as K
from blocks import nnet as N
from blocks import fuel
from blocks import training
from blocks.roles import add_role, DEPLOYING, has_roles, TRAINING
from blocks.graph import ComputationGraph
from blocks import visual
import cPickle

ds = fuel.load_mnist()
print(ds)

X = K.placeholder(shape=(None,) + ds['X_train'].shape[1:], name='X', for_training=True)
y = K.placeholder(shape=ds['y_train'].shape, name='y', dtype='int32')


ops = N.Sequence([
    lambda x: K.dimshuffle(x, (0, 'x', 1, 2)),
    N.Conv2D(16, (3, 3), stride=(1, 1), pad='same', activation=N.activations.rectify),
    K.pool2d,
    N.Dropout(level=0.3),
    N.Conv2D(32, (3, 3), stride=(1, 1), pad='same', activation=N.activations.rectify),
    K.pool2d,
    N.Dropout(level=0.3),
    K.flatten,
    N.Dense(64, activation=N.activations.rectify),
    N.Dense(10, activation=N.activations.softmax)
])
ops = cPickle.loads(cPickle.dumps(ops)) # test if the ops is pickle-able

y_pred_train = ops(X)
add_role(X, DEPLOYING)
y_pred_test = ops(X)

cost_train = N.cost.categorical_crossentropy(y_pred_train, y)
cost_test = N.cost.categorical_accuracy(y_pred_test, y)

graph = ComputationGraph(cost_train)
alg = algorithms.GradientDescent(cost=cost_train, step_rule=algorithms.RMSProp(learning_rate=0.01)).initialize()
f_train = alg.function
f_test = K.function([X, y], cost_test)

task = training.MainLoop(dataset=ds, batch_size=128)
task.add_callback(training.ProgressMonitor(title='Results: %.2f'), training.History())
task.set_task(f_train, ('X_train', 'y_train'), epoch=1, name='train')
task.add_subtask(f_test, ('X_valid', 'y_valid'), freq=0.3, name='valid')
task.add_subtask(f_test, ('X_test', 'y_test'), epoch=1, when=-1, name='test')
task.run()

valid = task.callback[1].get(task='valid', event='epoch_end')
valid = [np.mean(i) for i in valid]
print(valid)
print(visual.print_bar(valid, bincount=len(valid)))

test = np.mean(task.callback[1].get(task='test', event='epoch_end')[0])
print('Test accuracy:', test)
