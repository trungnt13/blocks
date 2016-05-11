from __future__ import print_function, division, absolute_import

import logging
import inspect
import signal

from six.moves import range, zip

import numpy as np

from blocks import RNG_GENERATOR
from blocks.fuel.dataset import Dataset
from blocks.fuel.data import Data
from blocks.utils import struct, queue

from .callbacks import Callback, CallbackList

__all__ = [
    'TaskDescriptor',
    'Task',
]


# ===========================================================================
# Tasks
# ===========================================================================
def _parse_data(data):
    if not isinstance(data, (list, tuple)):
        data = [data]
    if any(not isinstance(i, Data) for i in data):
        raise ValueError('only instances of Data are accepted')
    return data


class TaskDescriptor(object):

    def __init__(self, func, data, epoch, p, batch_size, seed, preprocess=None, name=None):
        super(TaskDescriptor, self).__init__()
        if not hasattr(func, '__call__'):
            raise ValueError('func must be instance of theano.Function or '
                             'python function, method, or hasattr __call__.')
        data = _parse_data(data)

        self._func = func
        self._data = data
        self._epoch = epoch
        self._p = np.clip(p, 0., 1.)

        self._batch_size = batch_size
        self.set_seed(seed)

        self._preprocess = preprocess if hasattr(preprocess, '__call__') else lambda x: x

        self._iter_per_epoch = int(np.ceil(
            min([len(i) for i in data]) / self._batch_size
        ))
        self._name = name

    @property
    def name(self):
        return str(self._name)

    def set_seed(self, seed):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        else:
            self._rng = struct()
            self._rng.randint = lambda x: None
            self._rng.rand = RNG_GENERATOR.rand
        return self

    @property
    def iter_per_epoch(self):
        ''' Estimated number of iteration for each epoch '''
        return self._iter_per_epoch

    def __iter__(self):
        '''
        Return
        ------
        'start_epoch' : beginning of epoch
        'end_epoch' : epoch ended
        'end_task' : task ended
        (results, n_iter, n_epoch) : results of execute function on data

        Note
        ----
        'end_task' also end of final epoch
        '''
        n_iter = 0
        p = self._p
        _ = 0
        yield 'start_task'
        while _ < self._epoch:
            _ += 1
            seed = self._rng.randint(10e8)
            data = zip(*[iter(i.set_batch(batch_size=self._batch_size, seed=seed))
                         for i in self._data])
            yield 'start_epoch'
            for i, x in enumerate(data):
                x = self._preprocess(x)
                if not isinstance(x, (tuple, list)):
                    x = [x]
                n_iter += 1
                if p >= 1. or (p < 1 and self._rng.rand() < p):
                    results = self._func(*x)
                else:
                    results = None
                yield (results, n_iter, _)
            # end_epoch or task
            if _ >= self._epoch:
                yield 'end_task'
            else:
                yield 'end_epoch'
        # keep ending so no Exception
        while True:
            yield 'end_task'


class Task(object):

    """Task"""

    def __init__(self, batch_size=256, dataset=None, shuffle=True, name=None):
        super(Task, self).__init__()
        self._batch_size = batch_size
        self._name = name
        self._task = None
        self._subtask = {} # run 1 epoch after given frequence
        self._crosstask = {} # randomly run 1 iter given probability

        if shuffle:
            self._rng = np.random.RandomState(RNG_GENERATOR.randint(10e8))
        else:
            self._rng = struct()
            self._rng.randint = lambda *args, **kwargs: None

        if isinstance(dataset, str):
            dataset = Dataset(dataset)
        elif not isinstance(dataset, Dataset):
            raise Exception('input dataset can be path (string) or Dataset instance.')
        self._dataset = dataset
        self._callback = Callback()

    # ==================== properties ==================== #
    @property
    def name(self):
        return self._name

    def __str__(self):
        return 'Task'

    def set_callback(self, callback):
        if isinstance(callback, Callback):
            self._callback = callback
        return self

    # ==================== main ==================== #
    def set_task(self, func, data, epoch=1, p=1., preprocess=None, name=None):
        '''
        '''
        self._task = TaskDescriptor(func, data, epoch, 1.,
                                    batch_size=self._batch_size,
                                    seed=self._rng.randint(10e8),
                                    preprocess=preprocess,
                                    name=name)
        return self

    def set_subtask(self, func, data, epoch=float('inf'), p=1., freq=0.,
                    when=0, preprocess=None, name=None):
        '''
        Parameters
        ----------
        when : float or int
            int => number of main task's iteration before this task is executed
            float => percentage of epoch of main task before this task is executed
        '''
        self._subtask[TaskDescriptor(func, data, epoch, p,
                                     batch_size=self._batch_size,
                                     seed=self._rng.randint(10e8),
                                     preprocess=preprocess,
                                     name=name)] = (freq, when)
        return self

    def set_crosstask(self, func, data, epoch=float('inf'), p=0.5,
                      when=0, preprocess=None, name=None):
        self._crosstask[TaskDescriptor(func, data, epoch, p,
                                       batch_size=self._batch_size,
                                       seed=self._rng.randint(10e8),
                                       preprocess=preprocess,
                                       name=name)] = when
        return self

    # ==================== logic ==================== #
    def run(self):
        if self._task is None:
            raise ValueError('You must call set_task and set the main task first.')
        callback = self._callback
        epoch_results = []
        task_results = []
        # ====== prepare subtask ====== #
        # iterator, task_results, is_ended=False
        subtask_map = {i: [iter(i), [], False] for i in self._subtask}
        # iterator, epoch_results, task_results, is_ended=False
        crosstask_map = {i: [iter(i), [], [], False] for i in self._crosstask}

        # ====== main logics ====== #
        for i in self._task:
            callback.mode = 'task' # dirty hack
            callback.reset(); callback.task = self._task
            if isinstance(i, str): # start_epoch, end_epoch or end_task
                if i == 'start_task':
                    callback.task_start()
                elif i == 'start_epoch':
                    callback.epoch_start()
                elif i == 'end_epoch' or i == 'end_task':
                    callback.results = epoch_results
                    callback.epoch_end()
                    epoch_results = []
                    if i == 'end_task':
                        callback.results = task_results
                        callback.task_end()
                        break # end everything
            else:
                results, niter, nepoch = i
                epoch_results.append(results)
                task_results.append(results)
                callback.results = results
                callback.iter = niter
                callback.epoch = nepoch
                callback.batch_end()
                # ====== run subtask ====== #
                callback.mode = 'subtask'
                for subtask, (freq, when) in self._subtask.iteritems():
                    subtask_iter, subtask_results, is_end = subtask_map[subtask]
                    if is_end: continue # already ended
                    if isinstance(when, float): when = int(when * self._task.iter_per_epoch)
                    if isinstance(freq, float): freq = int(freq * self._task.iter_per_epoch)
                    if niter >= when and (niter - when) % freq == 0: # OK to run
                        callback.reset(); callback.task = subtask
                        x = subtask_iter.next()
                        if x == 'start_task':
                            callback.task_start()
                            x = subtask_iter.next()
                        if x == 'start_epoch':
                            callback.epoch_start()
                            subepoch_results = []
                            while x != 'end_epoch' and x != 'end_task':
                                x = subtask_iter.next()
                                if isinstance(x, tuple):
                                    subepoch_results.append(x[0])
                                    subtask_results.append(x[0])
                                    callback.results = x[0]
                                    callback.iter = x[1]
                                    callback.epoch = x[2]
                                    callback.batch_end()
                            callback.results = subepoch_results
                            callback.epoch_end()
                        if x == 'end_task':
                            callback.results = subtask_results
                            callback.task_end()
                            subtask_map[subtask][-1] = True
                # ====== run crosstask ====== #
                callback.mode = 'crosstask'
                for crosstask, when in self._crosstask.iteritems():
                    if isinstance(when, float): when = int(when * self._task.iter_per_epoch)
                    crosstask_iter, crosstask_epoch, crosstask_results, is_end = crosstask_map[crosstask]
                    if niter >= when and not is_end: # OK to run
                        callback.reset(); callback.task = crosstask
                        x = crosstask_iter.next()
                        if x == 'start_task':
                            callback.task_start()
                            x = crosstask_iter.next()
                        if x == 'start_epoch':
                            callback.epoch_start()
                            x = crosstask_iter.next()
                        if isinstance(x, tuple):
                            crosstask_epoch.append(x[0])
                            crosstask_results.append(x[0])
                            callback.results = x[0]
                            callback.iter = x[1]
                            callback.epoch = x[2]
                            callback.batch_end()
                        elif x == 'end_epoch' or x == 'end_task':
                            callback.results = crosstask_epoch
                            crosstask_map[crosstask][1] = [] # reset epoch results
                            callback.epoch_end()
                            if x == 'end_task':
                                callback.results = crosstask_results
                                callback.task_end()
                                crosstask_map[crosstask][-1] = True


# ======================================================================
# Trainer
# ======================================================================
class MainLoop(CallbackList):

    def __init__(self):
        super(MainLoop, self).__init__()
        self._tasks = queue()

    def add_task(self):
        pass

    def run(self):
        pass
