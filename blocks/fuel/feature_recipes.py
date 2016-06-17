# ===========================================================================
# Collection of features extraction recipes
# Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import print_function, division, absolute_import

import os
import re
import warnings
from collections import defaultdict
from six.moves import zip, range

import numpy as np
import sidekit

from blocks.utils import get_all_files, as_tuple, play_audio
from blocks.utils.decorators import autoinit
from blocks.preprocessing.textgrid import TextGrid
from blocks.preprocessing import speech
from blocks.preprocessing.preprocess import normalize, segment_axis

from .data import Hdf5Data, MmapData
from .dataset import Dataset
from .features import FeatureRecipe

try: # this library may not available
    from scikits.samplerate import resample
except:
    pass


# ===========================================================================
# General global normalization
# ===========================================================================
class Normalization(FeatureRecipe):
    """ There are 2 schemes for organizing the data we support:
    1. dataset = [indices.csv, data, data_mean, data_std, ...]
    2. dataset = [data1, data2, data3, ..., mean, std]
    """
    @autoinit
    def __init__(self, dataset, global_normalize=True, local_normalize=False):
        super(Normalization, self).__init__('Normalization')

    def initialize(self, mr):
        dataset = self.dataset
        if isinstance(dataset, str) and os.path.isdir(dataset):
            dataset = Dataset(dataset)
        elif isinstance(dataset, Dataset):
            pass
        else:
            raise Exception('Only a fuel.Dataset object or path to Dataset '
                            'are supported.')

        # ====== scheme 2 ====== #
        if 'indices.csv' not in os.listdir(dataset.path):
            indices = None
            names = [n for n, t, s in dataset.keys()
                     if 'mean' not in n and 'std' not in n]
            mean = dataset['mean'][:]
            std = dataset['std'][:]
            self.jobs = [(i, mean, std) for i in names]
        # ====== scheme 1 ====== #
        else:
            indices = os.path.join(dataset.path, 'indices.csv')
            names = [n.replace('_mean', '')
                     for n, t, s in dataset.keys()
                     if '_mean' in n]
            self.jobs = [(i, i + '_mean', i + '_std') for i in names]

        # ====== inititalize function ====== #
        self.wrap_map(dataset=dataset, indices=indices,
            global_normalize=self.global_normalize,
            local_normalize=self.local_normalize)
        self.wrap_reduce()

    @staticmethod
    def _map(f, dataset, indices, global_normalize, local_normalize):
        # ====== scheme 1 ====== #
        if indices is not None:
            indices = np.genfromtxt(indices, dtype=str, delimiter=' ')
            data, mean, std = f
            data = dataset[data]
            mean = dataset[mean][:]
            std = dataset[std][:]
            for name, start, end in indices:
                start = int(start); end = int(end)
                x = data[start:end]
                if local_normalize:
                    data[start:end] = (x - x.mean(axis=0)) / x.std(axis=0)
                if global_normalize:
                    data[start:end] = (x - mean) / std
            data.flush()
        # ====== scheme 2 ====== #
        else:
            raise Exception('Waiting for testing')
            name, mean, std = f
            data = dataset[name]
            if global_normalize:
                data[:] = (data[:] - mean) / std
            if local_normalize:
                data[:] = (data[:] - data.mean(axis=0)) / data.std(axis=0)
            data.flush()

    @staticmethod
    def _reduce(results):
        return None


# ===========================================================================
# General Sequencing
# ===========================================================================
class FrameSequence(FeatureRecipe):

    '''
    Parameters
    ----------
    jobs : list
        can be list of string tuple or just a string represent name of data
        int dataset_in. Any object cannot be found will be duplicated
        according to the number of samples.
    shuffle : False, 1, 2
        False: no shuffle jobs is performed, 1: only shuffle the jobs list,
        2: shuffle the jobs list and block of examples in reducing phase.
    data_name : str or list(str)
        accorded name for each job in job list

    Example
    -------
    >>> train_jobs = \
    ... [('Kautokeino_03_05', 'Kautokeino_03_05_vad', 0),
    ...  ('Utsjoki_03_13', 'Utsjoki_03_13_vad', 3),
    ...  ('Ivalo_06_05', 'Ivalo_06_05_vad', 2)]

    >>> train = FrameSequence(train_jobs, dataset_path, dataset_out,
    ...     data_name=('Xtrain', 'Xtrain_vad', 'ytrain'),
    ...     dtype=('float32', 'int8', 'int8'),
    ...     win_length=win_length, hop_length=hop_length,
    ...     end=end, endvalue=0, shuffle=True, seed=12)

    '''

    @autoinit
    def __init__(self, jobs, dataset_in, dataset_out,
        data_name, dtype,
        win_length=256, hop_length=256, end='cut', endvalue=0,
        shuffle=True, seed=12, name='Sequencing'):
        super(FrameSequence, self).__init__(name)
        self.jobs = jobs

    def initialize(self, mr):
        # save dataset
        dataset_in = self.dataset_in
        # save dataset_out
        dataset_out = self.dataset_out

        # load dataset_in
        if not os.path.exists(dataset_in):
            dataset_in = mr[dataset_in]
            if not isinstance(dataset_in, Dataset):
                dataset_in = Dataset(dataset_in)
        else:
            dataset_in = Dataset(dataset_in)
        # load dataset_out
        if os.path.isfile(dataset_out):
            raise ValueError('path to dataset must be a folder.')
        dataset_out = Dataset(dataset_out)

        # ====== parse jobs information ====== #
        jobs = self.jobs
        if jobs is None or not isinstance(jobs, (tuple, list)):
            raise ValueError('jobs cannot be None or list of data name.')
        rng = np.random.RandomState(self.seed)
        if self.shuffle:
            rng.shuffle(jobs)
        self.jobs = jobs

        # ====== inititalize function ====== #
        self.wrap_map(dataset_in=dataset_in,
            win_length=self.win_length, hop_length=self.hop_length,
            end=self.end, endvalue=self.endvalue)
        self.wrap_reduce(dataset_out=dataset_out,
            data_name=self.data_name, dtype=self.dtype,
            shuffle=self.shuffle, rng=rng)
        self.wrap_finalize(dataset_out=dataset_out)

    @staticmethod
    def _map(f, dataset_in, win_length, hop_length, end, endvalue):
        if not isinstance(f, (tuple, list)):
            f = (f,)
        results = []
        for i in f:
            if isinstance(i, str):
                x = dataset_in.get_data(i)[:]
                x = segment_axis(x, frame_length=win_length, hop_length=hop_length,
                                 axis=0, end=end, endvalue=endvalue)
                results.append(x)
                n = x.shape[0]
            else:
                results.append([i] * n)
        return results

    @staticmethod
    def _reduce(results, dataset_out, data_name, dtype, shuffle, rng):
        if len(results) > 0 and (len(data_name) != len(results[0]) or
                                 len(dtype) != len(results[0])):
            raise ValueError('Returned [{}] results but only given [{}] name and'
                             ' [{}] dtype'.format(
                                 len(results[0]), len(data_name), len(dtype)))

        final = [[] for i in range(len(results[0]))]
        for res in results:
            for i, j in zip(res, final):
                j.append(i)
        final = [np.vstack(i)
                 if isinstance(i[0], np.ndarray)
                 else np.asarray(reduce(lambda x, y: x + y, i))
                 for i in final]
        # shufle features
        if shuffle > 2:
            permutation = rng.permutation(final[0].shape[0])
            final = [i[permutation] for i in final]
        # save to dataset
        for i, name, dt in zip(final, data_name, dtype):
            shape = i.shape
            dt = np.dtype(dt)
            x = dataset_out.get_data(name, dtype=dt, shape=shape, value=i)
            x.flush()
        return None

    @staticmethod
    def _finalize(results, dataset_out):
        dataset_out.flush()
        dataset_out.close()
