from __future__ import print_function, division, absolute_import

import os
import numpy as np
from math import ceil

from .data import MmapData, Hdf5Data, open_hdf5, get_all_hdf_dataset, MAX_OPEN_MMAP, Data

from blocks.utils import get_file, Progbar

from six.moves import zip, range
from collections import OrderedDict

__all__ = [
    'DataIterator',
    'Dataset',
    'load_mnist'
]


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

_apply_cut = lambda n, x: n * x if x < 1. + 1e-12 else int(x)


class DataIterator(object):

    ''' Vertically merge several data object for iteration
    '''

    def __init__(self, data, batch_size=256, shuffle=True, seed=None):
        if not isinstance(data, (tuple, list)):
            data = (data,)
        if any(not isinstance(i, (MmapData, Hdf5Data)) for i in data):
            raise ValueError('data must be instance of MmapData or Hdf5Data, '
                             'but given data have types: {}'
                             ''.format(map(lambda x: str(type(x)).split("'")[1],
                                          data)))
        shape = data[0].shape[1:]
        if any(i.shape[1:] != shape for i in data):
            raise ValueError('all data must have the same trial dimension, but'
                             'given shape of all data as following: {}'
                             ''.format([i.shape for i in data]))
        self._data = data
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._rng = np.random.RandomState(seed)

        self._start = 0.
        self._end = 1.

        self._sequential = False
        self._distribution = [1.] * len(data)
        self._seed = seed

    # ==================== properties ==================== #
    def __len__(self):
        start = self._start
        end = self._end
        return sum(round(i * (_apply_cut(j.shape[0], end) - _apply_cut(j.shape[0], start)))
                   for i, j in zip(self._distribution, self._data))

    @property
    def data(self):
        return self._data

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
        s.append('Shuffle: %r' % self._shuffle)
        s.append('Sequential: %r' % self._sequential)
        s.append('Distibution: %s' % str(self._distribution))
        s.append('Seed: %d' % self._seed)
        s.append('Range: [%.2f, %.2f]' % (self._start, self._end))
        return '\n'.join(s)

    # ==================== batch configuration ==================== #
    def set_mode(self, sequential=None, distribution=None):
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

    def set_range(self, start, end):
        if start < 0 or end < 0:
            raise ValueError('start and end must > 0, but start={} and end={}'
                             ''.format(start, end))
        self._start = start
        self._end = end
        return self

    def set_batch(self, batch_size=None, shuffle=None, seed=None):
        if batch_size is not None:
            self._batch_size = batch_size
        if shuffle is not None:
            self._shuffle = shuffle
        if seed is not None:
            self._rng.seed(seed)
            self._seed = seed
        return self

    # ==================== main logic of batch iterator ==================== #
    def _randseed(self):
        if self._shuffle:
            return self._rng.randint(10e8)
        return None

    def __iter__(self):
        # ====== easy access many private variables ====== #
        sequential = self._sequential
        start, end = self._start, self._end
        batch_size = self._batch_size
        data = np.asarray(self._data)
        distribution = np.asarray(self._distribution)
        if self._shuffle: # shuffle order of data (good for sequential mode)
            idx = self._rng.permutation(len(data))
            data = data[idx]
            distribution = distribution[idx]
        shape = [i.shape[0] for i in data]
        # ====== prepare distribution information ====== #
        # number of sample should be traversed
        n = np.asarray([i * (_apply_cut(j, end) - _apply_cut(j, start))
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
        # ==================== optimized parallel code ==================== #
        if not sequential:
            # first iterators
            it = [iter(dat.set_batch(bs, self._randseed(), start, end))
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
                            distribution[i], self._randseed(), start, end))
                        x = it[i].next()[:n[i]]
                        n[i] -= x.shape[0]
                        batch.append(x)
                # got final batch
                batch = np.vstack(batch)
                if self._shuffle:
                    # no idea why random permutation is much faster than shuffle
                    batch = batch[self._rng.permutation(batch.shape[0])]
                    # self._rng.shuffle(data)
                for i, j in idx[:int(ceil(batch.shape[0] / batch_size))]:
                    yield batch[i:j]
        # ==================== optimized sequential code ==================== #
        else:
            # first iterators
            batch_size = distribution.sum()
            it = [iter(dat.set_batch(batch_size, self._randseed(), start, end))
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
                        batch_size, self._randseed(), start, end))
                    x = it[current_data].next()[:n[current_data]]
                    n[current_data] -= x.shape[0]
                if self._shuffle:
                    x = x[self._rng.permutation(x.shape[0])]
                for i, j in idx[:int(ceil(x.shape[0] / self._batch_size))]:
                    yield x[i:j]


# ===========================================================================
# dataset
# ===========================================================================
def _parse_data_descriptor(path, name):
    path = os.path.join(path, name)
    if not os.path.isfile(path):
        return None

    is_mmap = MmapData.PATTERN.search(name)
    # memmap files
    if is_mmap is not None:
        name, dtype, shape = name.split('.')
        dtype = np.dtype(dtype)
        shape = eval(shape)
        # shape[1:], because first dimension can be resize afterward
        return [((name, dtype, shape[1:]), (shape[0], None))]
    # hdf5 files
    elif any(i in name for i in Hdf5Data.SUPPORT_EXT):
        try:
            f = open_hdf5(path)
            ds = get_all_hdf_dataset(f)
            data = [Hdf5Data(i, f) for i in ds]
            return [((i.name, i.dtype, i.shape[1:]), i) for i in data]
        except Exception, e:
            import traceback; traceback.print_exc()
            raise ValueError('Error loading hdf5 data, error:{}, file:{} '
                             ''.format(e, path))
    return None


class Dataset(object):

    '''
    Note
    ----
    for developer: _data_map contains, key=(name, dtype shape); value=Data

    '''

    def __init__(self, path):
        if path is not None:
            if os.path.isfile(path) and '.zip' in os.path.basename(path):
                self._load_archive(path,
                                   extract_path=path.replace(os.path.basename(path), ''))
            else:
                self._set_path(path)

    def _set_path(self, path):
        # all files are opened with default_mode=r+
        path = os.path.abspath(path)
        self._data_map = OrderedDict()

        if not os.path.exists(path):
            os.mkdir(path)
        elif not os.path.isdir(path):
            raise ValueError('Dataset path must be folder.')

        files = os.listdir(path)
        for f in files:
            data = _parse_data_descriptor(path, f)
            if data is None:
                continue
            for key, d in data:
                if key in self._data_map:
                    raise ValueError('Found duplicated data with follow info: '
                                     '{}'.format(key))
                else:
                    self._data_map[key] = d

        self._path = path
        self._name = os.path.basename(path)
        if len(self._name) == 1:
            self._name = os.path.basename(os.path.abspath(path))
        self._default_hdf5 = self.name + '_default.h5'

    # ==================== archive loading ==================== #
    def _load_archive(self, path, extract_path):
        from zipfile import ZipFile, ZIP_DEFLATED
        try:
            zfile = ZipFile(path, mode='r', compression=ZIP_DEFLATED)
            allfile = zfile.namelist()
            # validate extract_path
            if not os.path.isdir(extract_path):
                raise ValueError('Extract path must be path folder, but path'
                                 '={} is a file'.format(extract_path))
            extract_path = os.path.join(extract_path,
                                        os.path.basename(path).replace('.zip', ''))
            if os.path.isdir(extract_path) and \
               set(os.listdir(extract_path)) == set(allfile):
                self._set_path(extract_path)
                return
            # decompress everything
            if not os.path.exists(extract_path):
                os.mkdir(extract_path)
            maxlen = max([len(i) for i in allfile])
            progbar = Progbar(len(allfile))
            for i, f in enumerate(allfile):
                zfile.extract(f, path=extract_path)
                progbar.title = ('Unarchiving: %-' + str(maxlen) + 's') % f
                progbar.update(i + 1)
            self._set_path(extract_path)
        except IOError, e:
            raise IOError('Error loading archived dataset, path:{}, error:{}'
                          '.'.format(path, e))
        return None

    # ==================== properties ==================== #
    @property
    def path(self):
        return self._path

    @property
    def archive_path(self):
        return os.path.join(self._path, '..', self._name + '.zip')

    @property
    def name(self):
        return self._name

    @property
    def size(self):
        ''' return size in MegaByte'''
        size_bytes = 0
        for (name, dtype, shape), value in self._data_map.iteritems():
            size = np.dtype(dtype).itemsize
            if hasattr(value, 'shape'):
                shape = value.shape
            else: # memmap descriptor
                shape = (value[0],) + shape
            n = np.prod(shape)
            size_bytes += size * n
        return size_bytes / 1024. / 1024.

    @property
    def keys(self):
        '''
        Return
        ------
        (name, dtype, shape): tuple
        '''
        return [(name, dtype, value.shape) if hasattr(value, 'shape')
                else (name, dtype, (value[0],) + shape)
                for (name, dtype, shape), value in self._data_map.iteritems()]

    @property
    def info(self):
        '''
        Return
        ------
        (name, dtype, shape): tuple
        '''
        return [(name, dtype, value.shape, type(value))
                if hasattr(value, 'shape')
                else (name, dtype, (value[0],) + shape, type(MmapData))
                for (name, dtype, shape), value in self._data_map.iteritems()]

    # ==================== manipulate data ==================== #
    def get_data(self, name, dtype=None, shape=None, datatype='mmap', value=None):
        datatype = '.' + datatype.lower() if '.' not in datatype else datatype.lower()
        if datatype not in MmapData.SUPPORT_EXT and \
           datatype not in Hdf5Data.SUPPORT_EXT:
            raise ValueError("No support for data type: {}, following formats "
                             " are supported: {} and {}"
                             "".format(
                            datatype, Hdf5Data.SUPPORT_EXT, MmapData.SUPPORT_EXT))
        return_data = None
        return_key = None
        # ====== find defined data ====== #
        for k in self._data_map.keys():
            _name, _dtype, _shape = k
            if name == _name:
                if dtype is not None and np.dtype(_dtype) != np.dtype(dtype):
                    continue
                if shape is not None and shape[1:] != _shape:
                    continue
                return_data = self._data_map[k]
                return_key = k
                # return type is just a descriptor, create MmapData for it
                if not isinstance(return_data, Data):
                    return_data = MmapData(os.path.join(self.path, _name),
                        dtype=_dtype, shape=(return_data[0],) + _shape, mode='r+')
                    self._data_map[return_key] = return_data
                # append value
                if value is not None and value.shape[1:] == _shape:
                    return_data.append(value)
                    return_data.flush()
                break
        # ====== auto create new data, if cannot find any match ====== #
        if return_data is None and dtype is not None and shape is not None:
            if datatype in MmapData.SUPPORT_EXT:
                return_data = MmapData(os.path.join(self.path, name),
                    dtype=dtype, shape=shape, mode='w+', override=True)
            else:
                f = open_hdf5(os.path.join(self.path, self._default_hdf5))
                return_data = Hdf5Data(name, f, dtype=dtype, shape=shape)
            # first time create the dataset, assign init value
            if value is not None and value.shape == return_data.shape:
                return_data[:] = value
                return_data.flush()
            # store new key
            return_key = (return_data.name, return_data.dtype, return_data.shape[1:])
            self._data_map[return_key] = return_data
        # data still None
        if return_data is None:
            raise ValueError('Cannot find or create data with name={}, dtype={} '
                             'shape={}, and datatype={}'
                             ''.format(name, dtype, shape, datatype))
        # ====== check if excess limit, close 1 files ====== #
        if MmapData.COUNT > MAX_OPEN_MMAP:
            for i, j in self._data_map.iteritems():
                if isinstance(j, MmapData) and i != return_key:
                    break
            n = j.shape[0]
            del self._data_map[i]
            self._data_map[i] = (n, None)
        return return_data

    def create_iter(self, names,
        batch_size=256, shuffle=True, seed=None, start=0., end=1., mode=0):
        pass

    def archive(self):
        from zipfile import ZipFile, ZIP_DEFLATED
        path = self.archive_path
        zfile = ZipFile(path, mode='w', compression=ZIP_DEFLATED)

        files = []
        for key, value in self._data_map.iteritems():
            if hasattr(value, 'path'):
                files.append(value.path)
            else: # unloaded data
                name, dtype, shape = key
                n = value[0]
                name = MmapData.info_to_name(name, (n,) + shape, dtype)
                files.append(os.path.join(self.path, name))
        files = set(files)
        progbar = Progbar(len(files), title='Archiving:')

        maxlen = max([len(os.path.basename(i)) for i in files])
        for i, f in enumerate(files):
            zfile.write(f, os.path.basename(f))
            progbar.title = ('Archiving: %-' + str(maxlen) + 's') % os.path.basename(f)
            progbar.update(i + 1)
        zfile.close()
        return path

    def flush(self, name=None, dtype=None, shape=None):
        if name is None: # flush all files
            for v in self._data_map.values():
                if isinstance(v, Data):
                    v.flush()
        else: # flush a particular file
            for (n, d, s), j in self._data_map.items():
                if not isinstance(j, Data): continue
                if name == n:
                    if dtype is not None and np.dtype(dtype) != np.dtype(d):
                        continue
                    if shape is not None and shape[1:] != s:
                        continue
                    self._data_map[(n, d, s)].flush()

    def close(self, name=None, dtype=None, shape=None):
        if name is None: # close all files
            for k in self._data_map.keys():
                del self._data_map[k]
        else: # close a particular file
            for (n, d, s), j in self._data_map.items():
                if name == n:
                    if dtype is not None and np.dtype(dtype) != np.dtype(d):
                        continue
                    if shape is not None and shape[1:] != s:
                        continue
                    del self._data_map[(n, d, s)]

    # ==================== Some info ==================== #
    def __getitem__(self, key):
        if isinstance(key, (tuple, list)):
            return self.get_data(*key)
        return self.get_data(name=key)

    def __str__(self):
        s = ['====== Dataset:%s Total:%d======' %
             (self.path, len(self._data_map))]
        # ====== Find longest string ====== #
        longest_name = 0
        longest_shape = 0
        longest_file = len(str('unloaded'))
        for (name, dtype, _), data in self._data_map.iteritems():
            shape = data.shape if hasattr(data, 'shape') else (data[0],) + _
            longest_name = max(len(name), longest_name)
            longest_shape = max(len(str(shape)), longest_shape)
            if isinstance(data, Data):
                longest_file = max(len(str(data.path)), longest_file)
        # ====== return print string ====== #
        format_str = ('Name:%-' + str(longest_name) + 's  '
                      'dtype:%-7s  '
                      'shape:%-' + str(longest_shape) + 's  '
                      'file:%-' + str(longest_file) + 's')
        for (name, dtype, _), data in self._data_map.iteritems():
            shape = data.shape if hasattr(data, 'shape') else (data[0],) + _
            path = data.path if isinstance(data, Data) else 'unloaded'
            s.append(format_str % (name, dtype, shape, path))
        return '\n'.join(s)

    # ==================== Pickle ==================== #
    def __getstate__(self):
        config = OrderedDict()
        # convert to byte
        config['path'] = self.path
        return config

    def __setstate__(self, config):
        self._set_path(config['path'])


# ===========================================================================
# Predefined dataset
# ===========================================================================
def _load_data_from_path(datapath):
    from zipfile import ZipFile, ZIP_DEFLATED
    if not os.path.isdir(datapath):
        datapath_tmp = datapath + '.tmp'
        os.rename(datapath, datapath_tmp)
        zf = ZipFile(datapath_tmp, mode='r', compression=ZIP_DEFLATED)
        zf.extractall(path=datapath)
        zf.close()
        os.remove(datapath_tmp)
    ds = Dataset(datapath)
    return ds


def load_mnist(path='https://s3.amazonaws.com/ai-datasets/MNIST'):
    '''
    path : str
        local path or url to hdf5 datafile
    '''
    datapath = get_file('MNIST', path)
    return _load_data_from_path(datapath)
