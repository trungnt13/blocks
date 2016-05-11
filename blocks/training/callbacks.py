from __future__ import division, absolute_import, print_function

import time
import timeit
from datetime import datetime
from collections import defaultdict

import numpy as np

from blocks.utils import Progbar

__all__ = [
    'Callback',
    'CallbackList',
    'History',
    'ProgressMonitor',
    'Debug'
]


# ===========================================================================
# Helpers
# ===========================================================================
def _parse_result(result):
    if isinstance(result, (tuple, list)) and len(str(result)) > 20:
        type_str = ''
        if len(result) > 0:
            type_str = type(result[0]).__name__
        return 'list;%d;%s' % (len(result), type_str)
    s = str(result)
    return s[:20]


# ===========================================================================
# Callbacks
# ===========================================================================
class Callback(object):

    """Callback
    Properties
    ----------
    task

    Note
    ----
    This object can be used for many different task, just call reset before
    switching to other task

    """

    def __init__(self):
        super(Callback, self).__init__()
        self._task = None
        self._results = None
        self._iter = defaultdict(int)
        self._epoch = defaultdict(int)

        self._mode = None # 'task', 'subtask', 'crosstask'

    # ==================== helpers ==================== #
    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, value):
        if hasattr(value, 'name'):
            self._task = value

    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, value):
        self._results = value

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value in ('task', 'subtask', 'crosstask'):
            self._mode = value

    @property
    def iter(self):
        return self._iter[self.task]

    @iter.setter
    def iter(self, value):
        self._iter[self.task] = value

    @property
    def epoch(self):
        return self._epoch[self.task]

    @epoch.setter
    def epoch(self, value):
        self._epoch[self.task] = value

    def reset(self):
        self._task = None
        self._results = None

    # ==================== main callback methods ==================== #
    def task_start(self):
        pass

    def task_end(self):
        pass

    def epoch_start(self):
        pass

    def epoch_end(self):
        pass

    def batch_end(self):
        pass


class CallbackList(Callback):

    ''' Broadcast signal to all its children'''

    def __init__(self, *args):
        super(CallbackList, self).__init__()
        self._callbacks = []
        for i in args:
            self.add_callback(i)

    def add_callback(self, callback):
        if isinstance(callback, Callback) and callback not in self._callbacks:
            self._callbacks.append(callback)

    def __getitem__(self, key):
        return self._callbacks[key]

    # ==================== helpers ==================== #
    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, value):
        if hasattr(value, 'name'):
            self._task = value
            for i in self._callbacks:
                i.task = value

    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, value):
        self._results = value
        for i in self._callbacks:
            i.results = value

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value in ('task', 'subtask', 'crosstask'):
            self._mode = value
            for i in self._callbacks:
                i.mode = value

    @property
    def iter(self):
        return self._iter[self.task]

    @iter.setter
    def iter(self, value):
        self._iter[self.task] = value
        for i in self._callbacks:
            i.iter = value

    @property
    def epoch(self):
        return self._epoch[self.task]

    @epoch.setter
    def epoch(self, value):
        self._epoch[self.task] = value
        for i in self._callbacks:
            i.epoch = value

    def reset(self):
        self.task = None
        self.results = None
        for i in self._callbacks:
            i.reset()

    # ==================== main callback methods ==================== #
    def task_start(self):
        for i in self._callbacks:
            i.task_start()

    def task_end(self):
        for i in self._callbacks:
            i.task_end()

    def epoch_start(self):
        for i in self._callbacks:
            i.epoch_start()

    def epoch_end(self):
        for i in self._callbacks:
            i.epoch_end()

    def batch_end(self):
        for i in self._callbacks:
            i.batch_end()


# ===========================================================================
# Extension
# ===========================================================================
class ProgressMonitor(Callback):

    '''
    Example
    -------
    >>> t = training.Task(dataset=ds, batch_size=512)
    >>> t.set_callback(training.ProgressMonitor(title='Result: %.2f'))
    >>> t.run()
        # Result: 52751.29 98/98 [=======================================] - 0s
    '''

    def __init__(self, title=''):
        super(ProgressMonitor, self).__init__()
        self._format_results = False
        if len(list(title._formatter_parser())) > 0:
            self._format_results = True
        self._prog = Progbar(100, title='')
        self._title = title

    def batch_end(self):
        # do nothing for crosstask
        if self._mode == 'crosstask':
            return

        if self._format_results:
            title = self._title % self.results
        else:
            title = self._title
        self._prog.title = '%-8s:%2d:' % (self.task.name[:8], self.epoch) + title
        iter_per_epoch = self.task.iter_per_epoch
        n = round(((self.iter % iter_per_epoch) / iter_per_epoch) * 100)
        self._prog.update(int(n))

    def task_end(self):
        if self._mode == 'task': # main task ended
            self._prog.update(100)


class History(Callback):

    ''' Record the executing history in following format
    |Datatime; event_type; task; result; iter; epoch|
    '''
    @staticmethod
    def time2date(timestamp):
        return datetime.fromtimestamp(timestamp).strftime('%y-%m-%d %H:%M:%S')

    @staticmethod
    def date2time(date):
        return time.mktime(datetime.datetime.strptime(date, '%y-%m-%d %H:%M:%S').timetuple())

    def __init__(self):
        super(History, self).__init__()
        self._history = []

    def task_start(self):
        t = timeit.default_timer()
        self._history.append((t, 'task_start', self.task.name,
                              self.results, self.iter, self.epoch))

    def task_end(self):
        t = timeit.default_timer()
        self._history.append((t, 'task_end', self.task.name,
                              self.results, self.iter, self.epoch))

    def epoch_start(self):
        t = timeit.default_timer()
        self._history.append((t, 'epoch_start', self.task.name,
                              self.results, self.iter, self.epoch))

    def epoch_end(self):
        t = timeit.default_timer()
        self._history.append((t, 'epoch_end', self.task.name,
                              self.results, self.iter, self.epoch))

    def batch_end(self):
        t = timeit.default_timer()
        self._history.append((t, 'batch_end', self.task.name,
                              self.results, self.iter, self.epoch))

    def benchmark(self, task, event):
        '''
        Parameters
        ----------
        task : str
            name of given task want to benchmark
        event : 'batch', 'epoch', 'task'
            kind of event (e.g benchmark for each epoch, or batch)

        Return
        ------
        time : in second

        '''
        # ====== prepare ====== #
        if 'task' in event:
            event = 'task'
        elif 'epoch' in event:
            event = 'epoch'
        else:
            event = 'batch'
        history = [(i[0], i[1]) for i in self._history if task == i[2] and event in i[1]]
        # ====== benchmark ====== #
        if len(history) >= 2:
            if event == 'batch':
                history = [i[0] for i in history]
                return np.mean([j - i for i, j in zip(history, history[1:])])
            start = [i[0] for i in history if 'start' in i[1]]
            end = [i[0] for i in history if 'end' in i[1]]
            return np.mean([j - i for i, j in zip(start, end)])
        return None

    def __str__(self):
        format_str = "%s | %-12s | %-8s | %-20s | %-4s | %-4s"
        s = "=" * 24 + " N: %d " % len(self._history) + "=" * 24 + '\n'
        for i in self._history:
            i = (History.time2date(i[0]),) + i[1:]
            s += format_str % tuple([_parse_result(j) for j in i]) + '\n'
        return s

    # ==================== pickle interface ==================== #
    def __getstate__(self):
        return self._history

    def __setstate__(self, value):
        self._history = value


class Debug(Callback):

    def task_start(self):
        print()
        print('%-12s' % 'task_start', '%-12s' % self.task.name,
              None, '%4d' % self.iter, '%4d' % self.epoch)

    def task_end(self):
        print('%-12s' % 'task_end', '%-12s' % self.task.name,
              '%4d' % len(self.results), '%4d' % self.iter, '%4d' % self.epoch)
        print()

    def epoch_start(self):
        print('%-12s' % 'epoch_start', '%-12s' % self.task.name,
              None, '%4d' % self.iter, '%4d' % self.epoch)

    def epoch_end(self):
        print('%-12s' % 'epoch_end', '%-12s' % self.task.name,
              '%4d' % len(self.results), '%4d' % self.iter, '%4d' % self.epoch)

    def batch_end(self):
        # print('batch end', self.results)
        pass
