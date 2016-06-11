# ===========================================================================
# Parallel features processing using multi-core CPU and multiprocessing
# Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import print_function, division, absolute_import

import sys
import os
from multiprocessing import Pool, Manager
from six import add_metaclass
from abc import ABCMeta

from collections import defaultdict
import numpy as np

from blocks.utils import queue, Progbar
from blocks.utils.decorators import functionable, abstractstatic


__all__ = [
    'FeatureRecipe',
    'MapReduce'
]


def segment_list(l, size=None, n_seg=None):
    '''
    Example
    -------
    >>> segment_list([1,2,3,4,5],2)
    >>> [[1, 2, 3], [4, 5]]
    >>> segment_list([1,2,3,4,5],4)
    >>> [[1], [2], [3], [4, 5]]
    '''
    # by floor, make sure and process has it own job
    if size is None:
        size = int(np.ceil(len(l) / float(n_seg)))
    else:
        n_seg = int(np.ceil(len(l) / float(size)))
    if size * n_seg - len(l) > size:
        size = int(np.floor(len(l) / float(n_seg)))
    # start segmenting
    segments = []
    for i in range(n_seg):
        start = i * size
        if i < n_seg - 1:
            end = start + size
        else:
            end = max(start + size, len(l))
        segments.append(l[start:end])
    return segments


# ===========================================================================
# Predefined tasks
# ===========================================================================
@add_metaclass(ABCMeta)
class FeatureRecipe(object):

    ''' Pickle-able recipe for extracting object, that can be used with
    MapReduce

    '''

    def __init__(self, name=None):
        self.name = name
        self._map_func = None
        self._reduce_func = None
        self._finalize_func = None
        self.jobs = []
        self.seq_jobs = []

    # ==================== helper function ==================== #
    def update(self, key, value):
        '''Update all argument with given name to given value'''
        for i in [self._map_func, self._reduce_func, self._finalize_func]:
            if isinstance(i, functionable):
                i[key] = value

    def wrap_map(self, *args, **kwargs):
        self._map_func = functionable(self._map, *args, **kwargs)
        return self

    def wrap_reduce(self, *args, **kwargs):
        self._reduce_func = functionable(self._reduce, *args, **kwargs)
        return self

    def wrap_finalize(self, *args, **kwargs):
        self._finalize_func = functionable(self._finalize, *args, **kwargs)
        return self

    def initialize(self, mr):
        ''' This function will be called before the recipe is executed '''
        pass

    # ==================== non-touchable properties ==================== #
    @property
    def map_func(self):
        if not isinstance(self._map_func, functionable):
            raise ValueError('map_func must be instance of functionable')
        return self._map_func

    @property
    def reduce_func(self):
        if not isinstance(self._reduce_func, functionable):
            raise ValueError('reduce_func must be instance of functionable')
        return self._reduce_func

    @property
    def finalize_func(self):
        if not isinstance(self._finalize_func, functionable) and \
           self._finalize_func is not None:
            raise ValueError('finalize_func only can be None or functionable')
        return self._finalize_func

    # ==================== main function ==================== #
    @abstractstatic
    def _map(*args, **kwargs):
        raise NotImplementedError

    @abstractstatic
    def _reduce(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _finalize(*args, **kwargs):
        raise NotImplementedError

    # ==================== load from yaml ==================== #
    @classmethod
    def load(cls, path):
        if isinstance(path, str):
            if os.path.isfile(path):
                data = open(path, 'r').read()
            else:
                data = path
            import yaml
            from StringIO import StringIO
            data = yaml.load(StringIO(data))
            if isinstance(data, dict):
                if cls.__name__ in data:
                    data = data[cls.__name__]
                return cls(**data)
        raise Exception('Cannot load yaml recipe from path:%s' % path)

    def dump(self, path=None):
        """ Return yaml string represent this class """
        if not hasattr(self, '_arguments'):
            raise Exception('This method only support @autoinit class, which '
                            'store all its parameters in _arguments.')
        import yaml
        data = {self.__class__.__name__: self._arguments}
        styles = {'default_flow_style': False, 'encoding': 'utf-8'}
        if path is not None:
            yaml.dump(data, open(path, 'w'), **styles)
            return path
        return yaml.dump(data, **styles)


# ===========================================================================
# MPI MapReduce
# ===========================================================================
class MapReduce(object):

    """ This class manage all MapReduce task by callback function:

    map_function : argmuents(static_data, job)
        static_data: dictionary, which initialized right after you set the
        init_funciotn
        job: is a single job that automatically scheduled to each MPI process

    reduce_function : arguments(static_data, results, finnished)
        static_data: dictionary, which initialized right after you set the
        init_funciotn
        results: list, of returned result from each map_function (None returned
        will be ignored)
        finnished: bool, whether this function is called at the end of MapReduce
        task

    Example
    -------
    >>> def function(a):
    ...     x, y = a
    ...     return x + y

    >>> def function1(x):
    ...     return x - 1

    >>> mr = MapReduce(2, 1)
    >>> mr.cache = 12
    >>> mr.push(zip(range(26), reversed(range(26))),
    ...         function, lambda x: x, name='hello')
    >>> mr.push('hello', function1, lambda x: x, name='hi')
    >>> mr()
    >>> print(mr['hello']) # [25, ...]
    >>> print(mr['hi']) # [24, ...]

    """

    def __init__(self, processes=8, verbose=1):
        super(MapReduce, self).__init__()
        # variables
        self._cache = 5
        self._tasks = queue()
        self._processes = processes
        self._pool = Pool(processes)
        self._results = defaultdict(list)

    # ==================== Get & set ==================== #
    @property
    def cache(self):
        return self._cache

    def set_cache(self, value):
        if value > 0:
            self._cache = int(value)

    @property
    def processes(self):
        return self._processes

    # ==================== Task manager ==================== #
    def add_recipe(self, recipe):
        if isinstance(recipe, str): # path to pickled file or pickled string
            import cPickle
            if os.path.exists(recipe):
                recipe = cPickle.load(open(recipe, 'r'))
            else:
                recipe = cPickle.loads(recipe)

        if not isinstance(recipe, (tuple, list)):
            recipe = (recipe,)
        if not all(isinstance(i, FeatureRecipe) for i in recipe):
            raise ValueError('Given recipe is not instance of FeatureRecipe, '
                             'but has type={}'.format(map(type, recipe)))
        for i in recipe:
            self._tasks.append(i) # in this case, tasks contain recipe

    def add(self, jobs, map_func, reduce_func=None, finalize_func=None,
            init_func=None, name=None):
        ''' Wrapped preprocessing procedure in multiprocessing.
                ....root
                / / / | \ \ \ ,
                .mapping_func
                \ \ \ | / / /
                .reduce_func
                ......|
                .finalize_func

        Parameters
        ----------
        jobs : list
            [data_concern_job_1, job_2, ....]

        map_func : function(dict, job_i)
            function object to extract feature from each job, the dictionary
            will contain all static data initilized from set_init function

        reduce_func : function(dict, [job_i,...], finnished)
            transfer all data to process 0 as a list for saving to disk, the
            dictionary will contain all static data initilized from set_init
            function

        Notes
        -----
        Any None return by features_func will be ignored

        '''
        if not hasattr(map_func, '__call__') or \
            (reduce_func is not None and not hasattr(reduce_func, '__call__')) or \
            (finalize_func is not None and not hasattr(finalize_func, '__call__')) or \
                (init_func is not None and not hasattr(init_func, '__call__')):
            raise ValueError('map, reduce, finalize and init function must be callable'
                             ' object, but map_func={}, reduce_func={}, '
                             'finalize_func={} and init_func={}'
                             ''.format(type(map_func), type(reduce_func),
                                type(finalize_func), type(init_func)))
        self._tasks.append([jobs, map_func, reduce_func, finalize_func, init_func, name])
        return self

    # ==================== internal helper methods ==================== #
    def _flexible_init(self, init_func):
        import inspect
        # flexible init_func, accept 1 arg or None
        if inspect.ismethod(init_func) or \
            len(inspect.getargspec(init_func).args) == 1:
            init_func(self)
        else:
            init_func()

    def _run_mpi(self, task):
        #####################################
        # 0. parse task information.
        if isinstance(task, (tuple, list)):
            jobs_list, map_func, reduce_func, finalize_func, init_func, name = task
            if init_func is not None:
                self._flexible_init(init_func)
            seq_jobs = []
        elif isinstance(task, FeatureRecipe):
            self._flexible_init(task.initialize) # init first
            jobs_list, map_func, reduce_func, finalize_func, name = \
            task.jobs, task.map_func, task.reduce_func, task.finalize_func, task.name
            seq_jobs = task.seq_jobs
        else:
            raise ValueError('No support for type(task)={}.'.format(type(task)))

        #####################################
        # 1. Scatter jobs for all process.
        try:
            # str => the name of previous jobs
            if isinstance(jobs_list, str):
                jobs_list = len(self._results[jobs_list])
                if len(jobs_list) == 1: # only 1 result in result list
                    jobs_list = jobs_list[0]
            # if still no jobs
            if not isinstance(jobs_list, (tuple, list)) or \
            len(jobs_list) + len(seq_jobs) == 0:
                raise ValueError('no job for running task!')
            # create progbar
            progbar = Progbar(target=len(jobs_list) + len(seq_jobs),
                              title='Task:' + str(name))
            progbar.add(0) # update progress-bar
            # ====== start segment and process jobs ====== #
            jobs = segment_list(jobs_list, size=self._cache * self.processes)
            jobs.append(seq_jobs) # append seq jobs
            final_results = []
            for count, j in enumerate(jobs):
                if len(j) == 0: continue
                elif len(j) > self.processes and count < len(jobs) - 1:
                    results = self._pool.map(map_func, j, chunksize=self._cache)
                else: # execute sequently
                    results = [map_func(i) for i in j]
                # reduce all the results
                results = (reduce_func(results)
                           if reduce_func is not None else None)
                progbar.add(len(j)) # update progress-bar
                if results is not None:
                    final_results.append(results)
            # finalize all reduced results
            if finalize_func is not None:
                final_results = finalize_func(final_results)
            # store results
            if isinstance(final_results, dict):
                self._results.update(final_results)
            else:
                self._results[name].append(final_results)
        except Exception, e:
            sys.stderr.write("\nError! Ignored given task: name={}, error='{}'\n"
                             ''.format(name, e))
            import traceback; traceback.print_exc()

    def __getitem__(self, key):
        x = self._results.__getitem__(key)
        if isinstance(x, (tuple, list)) and len(x) == 1:
            return x[0]
        return x

    def get(self, key):
        return self.__getitem__(key)

    def run(self):
        while not self._tasks.empty():
            self._run_mpi(self._tasks.get())

    def __del__(self):
        try:
            self._pool.close()
            self._pool.join()
            del self._pool
        except:
            pass # already closed
