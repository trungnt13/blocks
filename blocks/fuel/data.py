# ===========================================================================
# Class handle extreme large numpy ndarray
# ===========================================================================
from __future__ import print_function, division, absolute_import

import os
import re
from math import ceil
from abc import ABCMeta, abstractmethod
from six import add_metaclass
from six.moves import range, zip, zip_longest

import numpy as np

from blocks import RNG_GENERATOR
from blocks.utils.decorators import autoattr, cache
from blocks.utils import queue, struct


__all__ = [
    'open_hdf5',
    'close_all_hdf5',
    'resize_memmap',
    'get_all_hdf_dataset',

    'Data',
    'MmapData',
    'Hdf5Data',
    'DataIterator',
    'DataMerge'
]


# ===========================================================================
# Helper function
# ===========================================================================
def resize_memmap(memmap, shape):
    if not isinstance(memmap, np.core.memmap):
        raise ValueError('Object must be instance of numpy memmap')

    if not isinstance(shape, (tuple, list)):
        shape = (shape,)
    if any(i != j for i, j in zip(shape[1:], memmap.shape[1:])):
        raise ValueError('Resize only support the first dimension, but '
                         '{} != {}'.format(shape[1:], memmap.shape[1:]))
    if shape[0] < memmap.shape[0]:
        raise ValueError('Only support extend memmap, and do not shrink the memory')
    elif shape[0] == memmap.shape[0]:
        return memmap
    memmap.flush()
    memmap = np.memmap(memmap.filename,
                       dtype=memmap.dtype,
                       mode='r+',
                       shape=(shape[0],) + tuple(memmap.shape[1:]))
    return memmap


def _estimate_shape(shape, func):
    ''' This method cannot estimate the shape accurately if you use slice '''
    shape0 = (12 + 8) // 10 * 13
    # func on 1 array
    if not isinstance(shape[0], (list, tuple)):
        if len(shape) > 0:
            n = int(min(shape0, shape[0])) # lucky number :D
            tmp = np.empty((n,) + shape[1:])
        else:
            tmp = np.empty(shape)
        old_shape = tmp.shape
        new_shape = func(tmp).shape
    else: # func on multiple-array
        tmp = []
        for s in shape:
            if len(s) > 0:
                n = int(min(shape0, s[0])) # lucky number :D
                tmp.append(np.empty((n,) + s[1:]))
            else:
                tmp.append(np.empty(s))
        old_shape = tmp[0].shape
        new_shape = func(tmp).shape
        shape = shape[0] # list of shape to a shape

    # ====== omitted the some dimensions ====== #
    if len(new_shape) == 0:
        return new_shape
    elif len(new_shape) < len(old_shape):
        if old_shape[0] != new_shape[0]: # first dimension omitted
            old_shape = old_shape[1:]
            shape = shape[1:]
        else: # other dimension omitted (no problem because we already know all of them)
            pass

    zip_func = zip if len(new_shape) <= len(old_shape) else zip_longest
    new_shape_ratio = [i / j if j is not None else i
                       for i, j in zip_func(new_shape, old_shape)]
    return tuple([int(round(i * j)) if i is not None else j
                  for i, j in zip_func(shape, new_shape_ratio)])


def _get_chunk_size(shape, size):
    if isinstance(size, int):
        return (2**int(np.ceil(np.log2(size))),) + shape[1:]
    elif size is None:
        return False
    return True


def _validate_operate_axis(axis):
    ''' as we iterate over first dimension, it is prerequisite to
    have 0 in the axis of operator
    '''
    if not isinstance(axis, (tuple, list)):
        axis = [axis]
    axis = tuple(int(i) for i in axis)
    if 0 not in axis:
        raise ValueError('Expect 0 in the operating axis because we always'
                         ' iterate data over the first dimension.')
    return axis


# x can be percantage or number of samples
_apply_approx = lambda n, x: int(round(n * x)) if x < 1. + 1e-12 else int(x)


# ===========================================================================
# Data
# ===========================================================================
@add_metaclass(ABCMeta)
class Data(object):

    def __init__(self):
        # batch information
        self._batch_size = 256
        self._start = 0.
        self._end = 1.
        self._seed = None
        self._status = 0 # flag show that array valued changed
        # main data object that have shape, dtype ...
        self._data = None

        self._transformer = lambda x: x

    # ====== transformer ====== #
    def transform(self, transformer):
        if hasattr(transformer, '__call__'):
            self._transformer = transformer
        return self

    # ==================== internal utilities ==================== #
    ''' BigData instance store large dataset that need to be iterate over to
    perform any operators.
    '''
    @cache('_status')
    def _iterating_operator(self, ops, axis, merge_func=sum, init_val=0.):
        '''Execute a list of ops on X given the axis or axes'''
        if axis is not None:
            axis = _validate_operate_axis(axis)
        if not isinstance(ops, (tuple, list)):
            ops = [ops]

        # init values all zeros
        s = None
        old_seed = self._seed
        old_start = self._start
        old_end = self._end
        self.set_batch(start=0., end=1., seed=None)
        # less than million data points, not a big deal
        for X in iter(self):
            if s is None:
                s = [o(X, axis) for o in ops]
            else:
                s = [merge_func((i, o(X, axis))) for i, o in zip(s, ops)]
        self.set_batch(start=old_start, end=old_end, seed=old_seed)
        return s

    def _iterate_update(self, y, ops):
        shape = self._data.shape
        # custom batch_size
        idx = list(range(0, shape[0], 1024))
        if idx[-1] < shape[0]:
            idx.append(shape[0])
        idx = list(zip(idx, idx[1:]))
        Y = lambda start, end: (y[start:end] if hasattr(y, 'shape') and
                                y.shape[0] == shape[0]
                                else y)
        for i in idx:
            start, end = i
            if 'add' == ops:
                self._data[start:end] += Y
            elif 'mul' == ops:
                self._data[start:end] *= Y
            elif 'div' == ops:
                self._data[start:end] /= Y
            elif 'sub' == ops:
                self._data[start:end] -= Y
            elif 'floordiv' == ops:
                self._data[start:end] //= Y
            elif 'pow' == ops:
                self._data[start:end] **= Y

    # ==================== properties ==================== #
    @property
    def shape(self):
        # auto infer new shape
        return _estimate_shape(self._data.shape, self._transformer)

    @property
    def T(self):
        return self.array.T

    @property
    def dtype(self):
        if not hasattr(self._data, 'dtype'):
            return self._data[0].dtype
        return self._data.dtype

    @property
    def array(self):
        return self._transformer(self._data[:])

    def tolist(self):
        return self.array.tolist()

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def data(self):
        return self._data

    def set_batch(self, batch_size=None, seed=None, start=None, end=None):
        if isinstance(batch_size, int) and batch_size > 0:
            self._batch_size = batch_size
        self._seed = seed
        if start is not None and start > 0. - 1e-12:
            self._start = start
        if end is not None and end > 0. - 1e-12:
            self._end = end
        return self

    # ==================== Slicing methods ==================== #
    def __getitem__(self, y):
        return self._transformer(self._data.__getitem__(y))

    @autoattr(_status=lambda x: x + 1)
    def __setitem__(self, x, y):
        return self._data.__setitem__(x, y)

    # ==================== iteration ==================== #
    def __iter__(self):
        batch_size = self._batch_size
        seed = self._seed
        shape = self._data.shape

        # custom batch_size
        start = _apply_approx(shape[0], self._start)
        end = _apply_approx(shape[0], self._end)
        if start > shape[0] or end > shape[0]:
            raise ValueError('start={} or end={} excess data_size={}'
                             ''.format(start, end, shape[0]))

        idx = list(range(start, end, batch_size))
        if idx[-1] < end:
            idx.append(end)
        idx = list(zip(idx, idx[1:]))
        if seed is not None:
            np.random.seed(seed)
            np.random.shuffle(idx)
            self._seed = None

        for i in idx:
            start, end = i
            yield self._transformer(self._data[start:end])

    # ==================== Strings ==================== #
    def __len__(self):
        return self.shape[0]

    def __str__(self):
        return self._data.__str__()

    def __repr__(self):
        return self._data.__repr__()

    # ==================== manipulation ==================== #
    @autoattr(_status=lambda x: x + 1)
    def append(self, *arrays):
        accepted_arrays = []
        new_size = 0
        shape = self._data.shape

        for a in arrays:
            if hasattr(a, 'shape'):
                if a.shape[1:] == shape[1:]:
                    accepted_arrays.append(a)
                    new_size += a.shape[0]
        old_size = shape[0]
        self.resize(old_size + new_size) # resize only once will be faster
        for a in accepted_arrays:
            self._data[old_size:old_size + a.shape[0]] = a
            old_size = old_size + a.shape[0]
        return self

    @autoattr(_status=lambda x: x + 1)
    def prepend(self, *arrays):
        accepted_arrays = []
        new_size = 0
        shape = self._data.shape

        for a in arrays:
            if hasattr(a, 'shape'):
                if a.shape[1:] == shape[1:]:
                    accepted_arrays.append(a)
                    new_size += a.shape[0]
        if new_size > shape[0]:
            self.resize(new_size) # resize only once will be faster
        size = 0
        for a in accepted_arrays:
            self._data[size:size + a.shape[0]] = a
            size = size + a.shape[0]
        return self

    # ==================== abstract ==================== #
    @abstractmethod
    def resize(self, shape):
        raise NotImplementedError

    @abstractmethod
    def flush(self):
        raise NotImplementedError

    # ==================== high-level operators ==================== #
    @abstractmethod
    def sum(self, axis=0):
        raise NotImplementedError

    @abstractmethod
    def cumsum(self, axis=None):
        raise NotImplementedError

    @abstractmethod
    def sum2(self, axis=0):
        raise NotImplementedError

    @abstractmethod
    def pow(self, y):
        raise NotImplementedError

    @abstractmethod
    def min(self, axis=None):
        raise NotImplementedError

    @abstractmethod
    def argmin(self, axis=None):
        raise NotImplementedError

    @abstractmethod
    def max(self, axis=None):
        raise NotImplementedError

    @abstractmethod
    def argmax(self, axis=None):
        raise NotImplementedError

    @abstractmethod
    def mean(self, axis=0):
        raise NotImplementedError

    @abstractmethod
    def var(self, axis=0):
        raise NotImplementedError

    @abstractmethod
    def std(self, axis=0):
        raise NotImplementedError

    @abstractmethod
    def normalize(self, axis, mean=None, std=None):
        raise NotImplementedError


class MutableData(Data):

    ''' Can only read, NO write or modify the values '''

    def __setitem__(self, x, y):
        raise NotImplementedError

    # ==================== manipulation ==================== #
    def append(self, *arrays):
        raise NotImplementedError

    def prepend(self, *arrays):
        raise NotImplementedError

    # ==================== abstract ==================== #
    def resize(self, shape):
        raise NotImplementedError

    def flush(self):
        pass

    # ==================== high-level operators ==================== #
    def sum(self, axis=0):
        ops = lambda x, axis: np.sum(x, axis=axis)
        return self._iterating_operator(ops, axis)[0]

    def cumsum(self, axis=None):
        return self.array.cumsum(axis)

    def sum2(self, axis=0):
        ops = lambda x, axis: np.sum(np.power(x, 2), axis=axis)
        return self._iterating_operator(ops, axis)[0]

    def pow(self, y):
        return self.array.__pow__(y)

    def min(self, axis=None):
        ops = lambda x, axis: np.min(x, axis=axis)
        return self._iterating_operator(ops, axis,
            merge_func=lambda x: np.where(x[0] < x[1], x[0], x[1]),
            init_val=float('inf'))[0]

    def argmin(self, axis=None):
        return self.array.argmin(axis)

    def max(self, axis=None):
        ops = lambda x, axis: np.max(x, axis=axis)
        return self._iterating_operator(ops, axis,
            merge_func=lambda x: np.where(x[0] > x[1], x[0], x[1]),
            init_val=float('-inf'))[0]

    def argmax(self, axis=None):
        return self.array.argmax(axis)

    def mean(self, axis=0):
        sum1 = self.sum(axis)

        axis = _validate_operate_axis(axis)
        n = np.prod([self.shape[i] for i in axis])
        return sum1 / n

    def var(self, axis=0):
        sum1 = self.sum(axis)
        sum2 = self.sum2(axis)

        axis = _validate_operate_axis(axis)
        n = np.prod([self.shape[i] for i in axis])
        return (sum2 - np.power(sum1, 2) / n) / n

    def std(self, axis=0):
        return np.sqrt(self.var(axis))

    def normalize(self, axis, mean=None, std=None):
        raise NotImplementedError

    # ==================== low-level operator ==================== #
    def __add__(self, y):
        return self.array.__add__(y)

    def __sub__(self, y):
        return self.array.__sub__(y)

    def __mul__(self, y):
        return self.array.__mul__(y)

    def __div__(self, y):
        return self.array.__div__(y)

    def __floordiv__(self, y):
        return self.array.__floordiv__(y)

    def __pow__(self, y):
        return self.array.__pow__(y)

    def __neg__(self):
        return self.array.__neg__()

    def __pos__(self):
        return self.array.__pos__()

    def __iadd__(self, y):
        raise NotImplementedError

    def __isub__(self, y):
        raise NotImplementedError

    def __imul__(self, y):
        raise NotImplementedError

    def __idiv__(self, y):
        raise NotImplementedError

    def __ifloordiv__(self, y):
        raise NotImplementedError

    def __ipow__(self, y):
        raise NotImplementedError

# ===========================================================================
# Memmap Data object
# ===========================================================================
MAX_OPEN_MMAP = 120


class MmapData(Data):

    """Create a memory-map to an array stored in a *binary* file on disk.

    Memory-mapped files are used for accessing small segments of large files
    on disk, without reading the entire file into memory.  Numpy's
    memmap's are array-like objects.  This differs from Python's ``mmap``
    module, which uses file-like objects.

    This subclass of ndarray has some unpleasant interactions with
    some operations, because it doesn't quite fit properly as a subclass.
    An alternative to using this subclass is to create the ``mmap``
    object yourself, then create an ndarray with ndarray.__new__ directly,
    passing the object created in its 'buffer=' parameter.

    This class may at some point be turned into a factory function
    which returns a view into an mmap buffer.

    Delete the memmap instance to close.

    Parameters
    ----------
    filename : str or file-like object
        The file name or file object to be used as the array data buffer.
    dtype : data-type, optional
        The data-type used to interpret the file contents.
        Default is `uint8`.
    mode : {'r+', 'r', 'w+', 'c'}, optional
        The file is opened in this mode:

        +------+-------------------------------------------------------------+
        | 'r'  | Open existing file for reading only.                        |
        +------+-------------------------------------------------------------+
        | 'r+' | Open existing file for reading and writing.                 |
        +------+-------------------------------------------------------------+
        | 'w+' | Create or overwrite existing file for reading and writing.  |
        +------+-------------------------------------------------------------+
        | 'c'  | Copy-on-write: assignments affect data in memory, but       |
        |      | changes are not saved to disk.  The file on disk is         |
        |      | read-only.                                                  |
        +------+-------------------------------------------------------------+

        Default is 'r+'.
    shape : tuple, optional
        The desired shape of the array. If ``mode == 'r'`` and the number
        of remaining bytes after `offset` is not a multiple of the byte-size
        of `dtype`, you must specify `shape`. By default, the returned array
        will be 1-D with the number of elements determined by file size
        and data-type.
    """

    # name.float32.(8,12)
    PATTERN = re.compile('[a-zA-Z0-9_-]*\.[<a-zA-Z]*\d{1,2}.\(\d+(,\d*)*\)')
    NAME_PATTERN = re.compile('[a-zA-Z0-9_-]*')
    SHAPE_PATTERN = re.compile('\.\(\d+(,\d*)*\)')
    DTYPE_PATTERN = re.compile('\.[<a-zA-Z]*\d{1,2}')
    COUNT = 0

    SUPPORT_EXT = ['.mmap', '.memmap', '.mem']

    def __init__(self, path, dtype='float32', shape=None, mode='r+',
        override=True):
        super(MmapData, self).__init__()
        if MmapData.COUNT > MAX_OPEN_MMAP:
            raise ValueError('Only allowed to open maximum of {} memmap file'.format(MAX_OPEN_MMAP))
        MmapData.COUNT += 1

        # validate path
        name = os.path.basename(path)
        path = path.replace(name, '')
        path = path if len(path) > 0 else '.'
        if name[0] == '.':
            name = name[1:]

        name_match = self.NAME_PATTERN.search(name)
        shape_match = self.SHAPE_PATTERN.search(name)
        dtype_match = self.DTYPE_PATTERN.search(name)
        if shape_match is not None:
            shape = name[shape_match.start():shape_match.end()][1:]
            shape = eval(shape)
        if dtype_match is not None:
            dtype = name[dtype_match.start():dtype_match.end()][1:]
            dtype = np.dtype(dtype)
        if name_match is not None:
            name = name[name_match.start():name_match.end()]
        else:
            raise ValueError('Cannot find name of memmap from given path')

        if dtype is None:
            raise ValueError('Must specify dtype at the beginning, either in '
                             'the path or dtype argument.')
        if shape is not None:
            shape = tuple([1 if i is None else i for i in shape])

        # check mode
        if 'r' in mode or 'c' in mode:
            files = os.listdir(path)
            for f in files:
                match = self.PATTERN.match(f)
                if match is not None:
                    _name, _dtype, _shape = f.split('.')
                    _shape = eval(_shape)
                    if name == _name and dtype == _dtype:
                        if shape is None or shape[1:] == _shape[1:]:
                            shape = _shape
            mmap_path = os.path.join(path, MmapData.info_to_name(name, shape, dtype))
            if not os.path.exists(mmap_path):
                raise ValueError('File not exist, given path:' + mmap_path)
        elif 'w' in mode:
            if shape is None:
                raise ValueError('dtype and shape must be specified in write '
                                 'mode, but shape={} and dtype={}'
                                 ''.format(shape, dtype))
            # the main issue is 2 files with the same name can have different
            # shape and dtype (this cannot happen)
            files = os.listdir(path)
            for f in files:
                if self.PATTERN.match(f) is not None:
                    _name, _dtype, _shape = f.split('.')
                    _shape = eval(_shape)
                    if _name == name and _dtype == dtype and shape[1:] == _shape[1:]:
                        if not override:
                            raise ValueError("memmap with the same name={} and dtype={} "
                                            "but with different shape already existed "
                                            "with name={}".format(name, dtype, f))
                        else: # remove old file to override
                            try:
                                os.remove(f)
                            except:
                                pass
            mmap_path = os.path.join(path, MmapData.info_to_name(name, shape, dtype))
        else:
            raise ValueError('No support for given mode:' + str(mode))

        # store variables
        self._data = np.memmap(mmap_path, dtype=dtype, shape=shape, mode=mode)
        self._name = name.split('.')[0]
        self._path = path
        self._mode = mode

    def __del__(self):
        MmapData.COUNT -= 1
        if hasattr(self, '_data') and self._data is not None:
            self._data._mmap.close()
            del self._data

    @staticmethod
    def name_to_info(name):
        shape = eval(name.split('.')[1])
        dtype = name.split('.')[2]
        return shape, dtype

    @staticmethod
    def info_to_name(name, shape, dtype):
        shape = [str(i) for i in shape]
        #(1000) will be understand as int (error)
        # should be (1000,)
        if len(shape) == 1:
            shape.append('')
        return '.'.join([name,
                        str(dtype),
                        '(' + ','.join([str(i) for i in shape]) + ')'])

    # ==================== properties ==================== #
    @property
    def path(self):
        return self._data.filename

    @property
    def name(self):
        return self._name

    # ==================== High-level operator ==================== #
    @cache('_status')
    def sum(self, axis=0):
        return self._data.sum(axis)

    @cache('_status')
    def cumsum(self, axis=None):
        return self._data.cumsum(axis)

    @cache('_status')
    def sum2(self, axis=0):
        return self._data.__pow__(2).sum(axis)

    @cache('_status')
    def pow(self, y):
        return self._data.__pow__(y)

    @cache('_status')
    def min(self, axis=None):
        return self._data.min(axis)

    @cache('_status')
    def argmin(self, axis=None):
        return self._data.argmin(axis)

    @cache('_status')
    def max(self, axis=None):
        return self._data.max(axis)

    @cache('_status')
    def argmax(self, axis=None):
        return self._data.argmax(axis)

    @cache('_status')
    def mean(self, axis=0):
        sum1 = self.sum(axis)
        if not isinstance(axis, (tuple, list)):
            axis = (axis,)
        n = np.prod([self._data.shape[i] for i in axis])
        return sum1 / n

    @cache('_status')
    def var(self, axis=0):
        sum1 = self.sum(axis)
        sum2 = self.sum2(axis)
        if not isinstance(axis, (tuple, list)):
            axis = (axis,)
        n = np.prod([self._data.shape[i] for i in axis])
        return (sum2 - np.power(sum1, 2) / n) / n

    @cache('_status')
    def std(self, axis=0):
        return np.sqrt(self.var(axis))

    @autoattr(_status=lambda x: x + 1)
    def normalize(self, axis, mean=None, std=None):
        mean = mean if mean is not None else self.mean(axis)
        std = std if std is not None else self.std(axis)
        self._data -= mean
        self._data /= std
        return self

    # ==================== Special operators ==================== #
    def __add__(self, y):
        return self._data.__add__(y)

    def __sub__(self, y):
        return self._data.__sub__(y)

    def __mul__(self, y):
        return self._data.__mul__(y)

    def __div__(self, y):
        return self._data.__div__(y)

    def __floordiv__(self, y):
        return self._data.__floordiv__(y)

    def __pow__(self, y):
        return self._data.__pow__(y)

    @autoattr(_status=lambda x: x + 1)
    def __iadd__(self, y):
        self._data.__iadd__(y)
        return self

    @autoattr(_status=lambda x: x + 1)
    def __isub__(self, y):
        self._data.__isub__(y)
        return self

    @autoattr(_status=lambda x: x + 1)
    def __imul__(self, y):
        self._data.__imul__(y)
        return self

    @autoattr(_status=lambda x: x + 1)
    def __idiv__(self, y):
        self._data.__idiv__(y)
        return self

    @autoattr(_status=lambda x: x + 1)
    def __ifloordiv__(self, y):
        self._data.__ifloordiv__(y)
        return self

    @autoattr(_status=lambda x: x + 1)
    def __ipow__(self, y):
        return self._data.__ipow__(y)

    def __neg__(self):
        self._data.__neg__()
        return self

    def __pos__(self):
        self._data.__pos__()
        return self

    # ==================== Save ==================== #
    def resize(self, shape):
        mode = self._mode
        mmap = self._data
        if mode == 'r' or mode == 'c':
            return

        if not isinstance(shape, (tuple, list)):
            shape = (shape,)
        if any(i != j for i, j in zip(shape[1:], mmap.shape[1:])):
            raise ValueError('Resize only support the first dimension, but '
                             '{} != {}'.format(shape[1:], mmap.shape[1:]))
        if shape[0] < mmap.shape[0]:
            raise ValueError('Only support extend memmap, and do not shrink the memory')
        elif shape[0] == self._data.shape[0]:
            return self
        mmap.flush()
        # resize by create new memmap and also rename old file
        shape = (shape[0],) + tuple(mmap.shape[1:])
        new_name = os.path.join(os.path.dirname(self.path),
                                MmapData.info_to_name(self.name, shape, mmap.dtype))
        os.rename(mmap.filename, new_name)
        self._data = np.memmap(new_name,
                               dtype=mmap.dtype,
                               mode='r+',
                               shape=shape)
        return self

    def flush(self):
        mode = self._mode
        if mode == 'r' or mode == 'c':
            return

        old_path = self._data.filename
        new_path = os.path.join(os.path.dirname(self.path),
                    MmapData.info_to_name(self.name, self._data.shape, self.dtype))
        self._data.flush()
        if old_path != new_path:
            del self._data
            os.rename(old_path, new_path)
            if 'w' in mode:
                mode = 'r+'
            self._data = np.memmap(new_path, mode=mode)


# ===========================================================================
# Hdf5 Data object
# ===========================================================================
try:
    import h5py
except:
    pass


def get_all_hdf_dataset(hdf, fileter_func=None, path='/'):
    res = []
    # init queue
    q = queue()
    for i in hdf[path].keys():
        q.put(i)
    # get list of all file
    while not q.empty():
        p = q.pop()
        if 'Dataset' in str(type(hdf[p])):
            if fileter_func is not None and not fileter_func(p):
                continue
            res.append(p)
        elif 'Group' in str(type(hdf[p])):
            for i in hdf[p].keys():
                q.put(p + '/' + i)
    return res


_HDF5 = {}


def open_hdf5(path):
    '''
    Parameters
    ----------
    mode : one of the following options
        +------------------------------------------------------------+
        |r        | Readonly, file must exist                        |
        +------------------------------------------------------------+
        |r+       | Read/write, file must exist                      |
        +------------------------------------------------------------+
        |w        | Create file, truncate if exists                  |
        +------------------------------------------------------------+
        |w- or x  | Create file, fail if exists                      |
        +------------------------------------------------------------+
        |a        | Read/write if exists, create otherwise (default) |
        +------------------------------------------------------------+

    check : bool
        if enable, only return openned files, otherwise, None

    Note
    ----
    If given file already open in read mode, mode = 'w' will cause error
    (this is good error and you should avoid this situation)

    '''
    key = os.path.abspath(path)
    if key in _HDF5:
        f = _HDF5[key]
        if 'Closed' in str(f):
            f = h5py.File(path, mode='a')
    else:
        # h5py._errors.silence_errors()
        f = h5py.File(path, mode='a')
        _HDF5[key] = f
    return f


def close_all_hdf5():
    import gc
    for obj in gc.get_objects():   # Browse through ALL objects
        if isinstance(obj, h5py.File):   # Just HDF5 files
            try:
                obj.close()
            except:
                pass # Was already closed


class Hdf5Data(Data):

    SUPPORT_EXT = ['.h5', '.hdf', '.hdf5']

    def __init__(self, dataset, hdf=None, dtype='float32', shape=None,
        chunk_size=32):
        super(Hdf5Data, self).__init__()

        self._chunk_size = chunk_size
        if isinstance(hdf, str):
            hdf = open_hdf5(hdf, mode='a')
        if hdf is None and not isinstance(dataset, h5py.Dataset):
            raise ValueError('Cannot initialize dataset without hdf file')

        if isinstance(dataset, h5py.Dataset):
            self._data = dataset
            self._hdf = dataset.file
        else:
            if dataset not in hdf: # not created dataset
                if dtype is None or shape is None:
                    raise ValueError('dtype and shape must be specified if '
                                     'dataset has not created in hdf5 file.')
                shape = tuple([0 if i is None else i for i in shape])
                hdf.create_dataset(dataset, dtype=dtype,
                    chunks=_get_chunk_size(shape, chunk_size),
                    shape=shape, maxshape=(None, ) + shape[1:])

            self._data = hdf[dataset]
            if shape is not None and self._data.shape[1:] != shape[1:]:
                raise ValueError('Shape mismatch between predefined dataset '
                                 'and given shape, {} != {}'
                                 ''.format(shape, self._data.shape))
            self._hdf = hdf

    # ==================== properties ==================== #
    @property
    def path(self):
        return self._hdf.filename

    @property
    def name(self):
        _ = self._data.name
        if _[0] == '/':
            _ = _[1:]
        return _

    @property
    def hdf5(self):
        return self._hdf

    # ==================== High-level operator ==================== #
    @cache('_status')
    def sum(self, axis=0):
        ops = lambda x, axis: np.sum(x, axis=axis)
        return self._iterating_operator(ops, axis)[0]

    @cache('_status')
    def cumsum(self, axis=None):
        return self._data[:].cumsum(axis)

    @cache('_status')
    def sum2(self, axis=0):
        ops = lambda x, axis: np.sum(np.power(x, 2), axis=axis)
        return self._iterating_operator(ops, axis)[0]

    @cache('_status')
    def pow(self, y):
        return self._data[:].__pow__(y)

    @cache('_status')
    def min(self, axis=None):
        ops = lambda x, axis: np.min(x, axis=axis)
        return self._iterating_operator(ops, axis,
            merge_func=lambda x: np.where(x[0] < x[1], x[0], x[1]),
            init_val=float('inf'))[0]

    @cache('_status')
    def argmin(self, axis=None):
        return self._data[:].argmin(axis)

    @cache('_status')
    def max(self, axis=None):
        ops = lambda x, axis: np.max(x, axis=axis)
        return self._iterating_operator(ops, axis,
            merge_func=lambda x: np.where(x[0] > x[1], x[0], x[1]),
            init_val=float('-inf'))[0]

    @cache('_status')
    def argmax(self, axis=None):
        return self._data[:].argmax(axis)

    @cache('_status')
    def mean(self, axis=0):
        sum1 = self.sum(axis)

        axis = _validate_operate_axis(axis)
        n = np.prod([self._data.shape[i] for i in axis])
        return sum1 / n

    @cache('_status')
    def var(self, axis=0):
        sum1 = self.sum(axis)
        sum2 = self.sum2(axis)

        axis = _validate_operate_axis(axis)
        n = np.prod([self._data.shape[i] for i in axis])
        return (sum2 - np.power(sum1, 2) / n) / n

    @cache('_status')
    def std(self, axis=0):
        return np.sqrt(self.var(axis))

    @autoattr(_status=lambda x: x + 1)
    def normalize(self, axis, mean=None, std=None):
        mean = mean if mean is not None else self.mean(axis)
        std = std if std is not None else self.std(axis)
        self._iterate_update(mean, 'sub')
        self._iterate_update(std, 'div')
        return self

    # ==================== low-level operator ==================== #
    def __add__(self, y):
        return self._data.__add__(y)

    def __sub__(self, y):
        return self._data.__sub__(y)

    def __mul__(self, y):
        return self._data.__mul__(y)

    def __div__(self, y):
        return self._data.__div__(y)

    def __floordiv__(self, y):
        return self._data.__floordiv__(y)

    def __pow__(self, y):
        return self._data.__pow__(y)

    @autoattr(_status=lambda x: x + 1)
    def __iadd__(self, y):
        self._iterate_update(y, 'add')
        return self

    @autoattr(_status=lambda x: x + 1)
    def __isub__(self, y):
        self._iterate_update(y, 'sub')
        return self

    @autoattr(_status=lambda x: x + 1)
    def __imul__(self, y):
        self._iterate_update(y, 'mul')
        return self

    @autoattr(_status=lambda x: x + 1)
    def __idiv__(self, y):
        self._iterate_update(y, 'div')
        return self

    @autoattr(_status=lambda x: x + 1)
    def __ifloordiv__(self, y):
        self._iterate_update(y, 'floordiv')
        return self

    @autoattr(_status=lambda x: x + 1)
    def __ipow__(self, y):
        self._iterate_update(y, 'pow')
        return self

    def __neg__(self):
        self._data.__neg__()
        return self

    def __pos__(self):
        self._data.__pos__()
        return self

    # ==================== Save ==================== #
    def resize(self, shape):
        if self._hdf.mode == 'r':
            return

        if not isinstance(shape, (tuple, list)):
            shape = (shape,)
        if any(i != j for i, j in zip(shape[1:], self._data.shape[1:])):
            raise ValueError('Resize only support the first dimension, but '
                             '{} != {}'.format(shape[1:], self._data.shape[1:]))
        if shape[0] < self._data.shape[0]:
            raise ValueError('Only support extend memmap, and do not shrink the memory')
        elif shape[0] == self._data.shape[0]:
            return self

        self._data.resize(self._data.shape[0] + shape[0], axis=0)
        return self

    def flush(self):
        try:
            if self._hdf.mode == 'r':
                return
            self._hdf.flush()
        except:
            pass


# ===========================================================================
# data iterator
# ===========================================================================
def _approximate_continuos_by_discrete(distribution):
    '''original distribution: [ 0.47619048  0.38095238  0.14285714]
       best approximated: [ 5.  4.  2.]
    '''
    if len(distribution) == 1:
        return distribution

    inv_distribution = 1 - distribution
    x = np.round(1 / inv_distribution)
    x = np.where(distribution == 0, 0, x)
    return x.astype(int)


class DataIterator(MutableData):

    ''' Vertically merge several data object for iteration
    '''

    def __init__(self, data):
        super(DataIterator, self).__init__()
        if not isinstance(data, (tuple, list)):
            data = (data,)
        # ====== validate args ====== #
        if any(not isinstance(i, Data) for i in data):
            raise ValueError('data must be instance of MmapData or Hdf5Data, '
                             'but given data have types: {}'
                             ''.format(map(lambda x: str(type(x)).split("'")[1],
                                          data)))
        shape = data[0].shape[1:]
        if any(i.shape[1:] != shape for i in data):
            raise ValueError('all data must have the same trial dimension, but'
                             'given shape of all data as following: {}'
                             ''.format([i.shape for i in data]))
        # ====== defaults parameters ====== #
        self._data = data
        self._batch_size = 256

        self._start = 0.
        self._end = 1.

        self._sequential = False
        self._distribution = [1.] * len(data)
        self._seed = None

    # ==================== properties ==================== #
    @property
    def shape(self):
        orig_shape = (len(self),) + self._data[0].shape[1:]
        return _estimate_shape(orig_shape, self._transformer)

    @property
    def array(self):
        start = self._start
        end = self._end
        idx = [(_apply_approx(i.shape[0], start), _apply_approx(i.shape[0], end))
               for i in self._data]
        idx = [(j[0], int(round(j[0] + i * (j[1] - j[0]))))
               for i, j in zip(self._distribution, idx)]
        return self._transformer(
            np.vstack([i[j[0]:j[1]] for i, j in zip(self._data, idx)]))

    def __len__(self):
        start = self._start
        end = self._end
        return sum(round(i * (_apply_approx(j.shape[0], end) - _apply_approx(j.shape[0], start)))
                   for i, j in zip(self._distribution, self._data))

    @property
    def distribution(self):
        return self._distribution

    def __str__(self):
        s = ['====== Iterator: ======']
        # ====== Find longest string ====== #
        longest_name = 0
        longest_shape = 0
        for d in self._data:
            name = d.name
            dtype = d.dtype
            shape = d.shape
            longest_name = max(len(name), longest_name)
            longest_shape = max(len(str(shape)), longest_shape)
        # ====== return print string ====== #
        format_str = ('Name:%-' + str(longest_name) + 's  '
                      'dtype:%-7s  '
                      'shape:%-' + str(longest_shape) + 's  ')
        for d in self._data:
            name = d.name
            dtype = d.dtype
            shape = d.shape
            s.append(format_str % (name, dtype, shape))
        # ====== batch configuration ====== #
        s.append('Batch: %d' % self._batch_size)
        s.append('Sequential: %r' % self._sequential)
        s.append('Distibution: %s' % str(self._distribution))
        s.append('Seed: %s' % str(self._seed))
        s.append('Range: [%.2f, %.2f]' % (self._start, self._end))
        return '\n'.join(s)

    def __repr__(self):
        return self.__str__()

    # ==================== batch configuration ==================== #
    def set_mode(self, distribution=None, sequential=None):
        '''
        Parameters
        ----------
        distribution : str, list or float
            'up', 'over': over-sampling all Data
            'down', 'under': under-sampling all Data
            list: percentage of each Data in the iterator will be iterated
            float: the same percentage for all Data
        sequential : bool
            if True, read each Data one-by-one, otherwise, mix all Data

        '''
        if sequential is not None:
            self._sequential = sequential
        if distribution is not None:
            # upsampling or downsampling
            if isinstance(distribution, str):
                distribution = distribution.lower()
                if 'up' in distribution or 'over' in distribution:
                    n = max(i.shape[0] for i in self._data)
                elif 'down' in distribution or 'under' in distribution:
                    n = min(i.shape[0] for i in self._data)
                else:
                    raise ValueError("Only upsampling (keyword: up, over) "
                                     "or undersampling (keyword: down, under) "
                                     "are supported.")
                self._distribution = [n / i.shape[0] for i in self._data]
            # real values distribution
            elif isinstance(distribution, (tuple, list)):
                if len(distribution) != len(self._data):
                    raise ValueError('length of given distribution must equal '
                                     'to number of data in the iterator, but '
                                     'len_data={} != len_distribution={}'
                                     ''.format(len(self._data), len(self._distribution)))
                self._distribution = distribution
            # all the same value
            elif isinstance(distribution, float):
                self._distribution = [distribution] * len(self._data)
        return self

    # ==================== main logic of batch iterator ==================== #
    def __iter__(self):
        if self._seed is not None:
            rng = np.random.RandomState(self._seed)
        else: # deterministic RandomState
            rng = struct()
            rng.randint = lambda x: None
            rng.permutation = lambda x: range(x)
        self._seed = None # remove the seed for next time
        # ====== easy access many private variables ====== #
        sequential = self._sequential
        start, end = self._start, self._end
        batch_size = self._batch_size
        data = np.asarray(self._data)
        distribution = np.asarray(self._distribution)
        # shuffle order of data (good for sequential mode)
        idx = rng.permutation(len(data))
        data = data[idx]
        distribution = distribution[idx]
        shape = [i.shape[0] for i in data]
        # ====== prepare distribution information ====== #
        # number of sample should be traversed
        n = np.asarray([i * (_apply_approx(j, end) - _apply_approx(j, start))
                        for i, j in zip(distribution, shape)])
        n = np.round(n).astype(int)
        # normalize the distribution (base on new sample n of each data)
        distribution = n / n.sum()
        distribution = _approximate_continuos_by_discrete(distribution)
        # somehow heuristic, rescale distribution to get more benifit from cache
        if distribution.sum() <= len(data):
            distribution = distribution * 3
        # distribution now the actual batch size of each data
        distribution = (batch_size * distribution).astype(int)
        assert distribution.sum() % batch_size == 0, 'wrong distribution size!'
        # predefined (start,end) pair of each batch (e.g (0,256), (256,512))
        idx = list(range(0, batch_size + distribution.sum(), batch_size))
        idx = list(zip(idx, idx[1:]))
        #####################################
        # 1. optimized parallel code.
        if not sequential:
            # first iterators
            it = [iter(dat.set_batch(bs, rng.randint(10e8), start, end))
                  for bs, dat in zip(distribution, data)]
            # iterator
            while sum(n) > 0:
                batch = []
                for i, x in enumerate(it):
                    if n[i] <= 0:
                        continue
                    try:
                        x = x.next()[:n[i]]
                        n[i] -= x.shape[0]
                        batch.append(x)
                    except StopIteration: # one iterator stopped
                        it[i] = iter(data[i].set_batch(
                            distribution[i], rng.randint(10e8), start, end))
                        x = it[i].next()[:n[i]]
                        n[i] -= x.shape[0]
                        batch.append(x)
                # got final batch
                batch = np.vstack(batch)
                # no idea why random permutation is much faster than shuffle
                batch = batch[rng.permutation(batch.shape[0])]
                for i, j in idx[:int(ceil(batch.shape[0] / batch_size))]:
                    yield self._transformer(batch[i:j])
        #####################################
        # 2. optimized sequential code.
        else:
            # first iterators
            batch_size = distribution.sum()
            it = [iter(dat.set_batch(batch_size, rng.randint(10e8), start, end))
                  for dat in data]
            current_data = 0
            # iterator
            while sum(n) > 0:
                if n[current_data] <= 0:
                    current_data += 1
                try:
                    x = it[current_data].next()[:n[current_data]]
                    n[current_data] -= x.shape[0]
                except StopIteration: # one iterator stopped
                    it[current_data] = iter(data[current_data].set_batch(
                        batch_size, rng.randint(10e8), start, end))
                    x = it[current_data].next()[:n[current_data]]
                    n[current_data] -= x.shape[0]
                # shuffle x
                x = x[rng.permutation(x.shape[0])]
                for i, j in idx[:int(ceil(x.shape[0] / self._batch_size))]:
                    yield self._transformer(x[i:j])

    # ==================== Slicing methods ==================== #
    def __getitem__(self, y):
        pass

# ===========================================================================
# DataMerge
# ===========================================================================


class DataMerge(MutableData):

    '''
    Parameters
    ----------
    data : list
        list of Data objects
    merge_func : __call__
        function take a list of Data as argument (i.e func([data1, data2]))

    Note
    ----
    First data in the list will be used as root to infer the shape after merge
    '''

    def __init__(self, data, merge_func):
        super(DataMerge, self).__init__()

        if not isinstance(data, (tuple, list)):
            data = (data,)
        self._data = [i for i in data if isinstance(i, Data)]
        if len(self._data) == 0:
            raise ValueError('Cannot find any instance of Data from given argument.')

        if not hasattr(merge_func, '__call__'):
            raise ValueError('Merge operator must be callable and accept at '
                             'least one argument.')
        self._merge_func = merge_func

    # ==================== properties ==================== #
    @property
    def path(self):
        return [d.path for d in self._data]

    @property
    def name(self):
        return [d.name for d in self._data]

    # ==================== properties ==================== #
    @property
    def shape(self):
        shape = [i.shape for i in self._data]
        return _estimate_shape(shape,
                               lambda x: self._transformer(self._merge_func(x)))

    @property
    def dtype(self):
        n = (12 + 8) // 10 # lucky number :D
        tmp = [np.ones((n,) + i.shape[1:]).astype(i.dtype) for i in self._data]
        return self._merge_func(tmp).dtype

    @property
    def array(self):
        return self._transformer(self._merge_func([i[:] for i in self._data]))

    # ==================== Slicing methods ==================== #
    def __getitem__(self, y):
        n = self._data[0].shape[0]
        data = [i.__getitem__(y) if len(i.shape) > 0 and i.shape[0] == n else i
                for i in self._data]
        x = self._merge_func(data)
        return self._transformer(x)

    # ==================== iteration ==================== #
    def __iter__(self):
        batch_size = self._batch_size
        seed = self._seed
        # ====== prepare root first ====== #
        shape = self._data[0].shape
        # custom batch_size
        start = _apply_approx(shape[0], self._start)
        end = _apply_approx(shape[0], self._end)
        if start > shape[0] or end > shape[0]:
            raise ValueError('start={} or end={} excess data_size={}'
                             ''.format(start, end, shape[0]))

        idx = list(range(start, end, batch_size))
        if idx[-1] < end:
            idx.append(end)
        idx = list(zip(idx, idx[1:]))
        if seed is not None:
            np.random.seed(seed)
            np.random.shuffle(idx)
            self._seed = None
        idx = [slice(i[0], i[1]) for i in idx]
        none_idx = [slice(None, None)] * len(idx)
        # ====== check other data ====== #
        batches = [idx]
        for d in self._data[1:]:
            if len(d.shape) > 0 and d.shape[0] == shape[0]:
                batches.append(idx)
            else:
                batches.append(none_idx)
        for b in zip(*batches):
            data = self._merge_func([i[j] for i, j in zip(self._data, b)])
            yield self._transformer(data)
