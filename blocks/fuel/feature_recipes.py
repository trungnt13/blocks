# ===========================================================================
# Collection of features extraction recipes
# Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import print_function, division, absolute_import

import os
import re
from collections import defaultdict
from six.moves import zip, range

import numpy as np
from .data import Hdf5Data, MmapData
from .dataset import Dataset
from .features import FeatureRecipe

from blocks.utils import get_all_files, as_tuple, play_audio
from blocks.utils.decorators import autoinit
from blocks.preprocessing.textgrid import TextGrid
from blocks.preprocessing import speech
from blocks.preprocessing.preprocess import normalize, segment_axis

try: # this library may not available
    from scikits.samplerate import resample
    import sidekit
except:
    pass

support_delimiter = re.compile('[\s;,.:]')


# ===========================================================================
# Speech features
# ===========================================================================
def _auto_detect_seperator(f):
    f = open(f, 'r')
    l = f.readline().replace('\n', '')
    f.close()
    search = support_delimiter.search(l)
    if search is None:
        raise ValueError("Only support for following delimiters \s;:,.")
    delimiter = l[search.start():search.end()]
    return delimiter


def _read_transcript_file(file_path):
    transcripter = None
    if '.TextGrid' in file_path:
        tg = TextGrid(file_path)
        transcripter = tg.tiers[0]
    else:
        raise ValueError('No support for file:{}'.format(file_path))
    if not hasattr(transcripter, 'transcript'):
        raise ValueError('Transcripter must has method transcript.')
    return transcripter


def _append_energy_and_deltas(s, energy, delta_order):
    # s.shape = [Time, Dimension]
    if s is None:
        return None
    if energy is not None:
        s = np.hstack((s, energy[:, None]))
    # compute delta
    if delta_order > 0:
        deltas = speech.compute_delta(s.T, order=delta_order)
        # tranpose back to [Time, Dim]
        s = np.hstack([s] + [i.T for i in deltas])
    return s


def speech_features_extraction(s, fs, n_filters, n_ceps, win, shift,
                               delta_order, energy, vad, dtype,
                               get_spec, get_mspec, get_mfcc):
    """ return: spec, mspec, and mfcc """
    if s.ndim >= 2:
        raise Exception('Speech Feature Extraction only accept 1-D signal')
    # speech features, shape: [Time, Dimension]
    mfcc, logEnergy, spec, mspec = sidekit.frontend.mfcc(
        s, fs=fs, lowfreq=64, maxfreq=fs // 2, nlogfilt=n_filters,
        nwin=win, shift=shift, nceps=n_ceps,
        get_spec=get_spec, get_mspec=get_mspec)
    mfcc = mfcc if get_mfcc else None
    # VAD
    vad_idx = None
    if vad:
        distribNb, nbTrainIt = 8, 12
        if isinstance(vad, (tuple, list)):
            distribNb, nbTrainIt = vad
        # vad_idx = sidekit.frontend.vad.vad_snr(s, threshold,
        # fs=fs, shift=shift, nwin=int(fs * win)).astype('int8')
        vad_idx = sidekit.frontend.vad.vad_energy(logEnergy,
            distribNb=distribNb, nbTrainIt=nbTrainIt).astype('int8')
    # Energy
    logEnergy = logEnergy if energy else None

    # everything is (T, D)
    mfcc = (_append_energy_and_deltas(mfcc, logEnergy, delta_order)
            if mfcc is not None else None)
    # we don't calculate deltas for spectrogram features
    spec = (_append_energy_and_deltas(spec, logEnergy, 0)
            if spec is not None else None)
    mspec = (_append_energy_and_deltas(mspec, logEnergy, delta_order)
            if mspec is not None else None)
    # normalization
    mfcc = (mfcc.astype(dtype),
            np.sum(mfcc, axis=0, dtype='float64'),
            np.sum(mfcc**2, axis=0, dtype='float64')) if mfcc is not None else None
    spec = (spec.astype(dtype),
            np.sum(spec, axis=0, dtype='float64'),
            np.sum(spec**2, axis=0, dtype='float64')) if spec is not None else None
    mspec = (mspec.astype(dtype),
             np.sum(mspec, axis=0, dtype='float64'),
             np.sum(mspec**2, axis=0, dtype='float64')) if mspec is not None else None
    return spec, mspec, mfcc, vad_idx


class SpeechFeature(FeatureRecipe):

    ''' Extract speech features from all audio files in given directory or
    file list, then saves them to a `keras.ext.dataset.Dataset`

    Parameters
    ----------
    segments : path, list
        if path, directory of all audio file, or segment csv file in format
            name                     path          start end
        sw02001-A_000098-001156 /path/to/sw02001-A  0.0  -1
        sw02001-A_001980-002131 /path/to/sw02001-A  0.0  -1
    win : float
        frame or window length in second
    shift : float
        frame or window, or hop length in second
    n_filters : int
        number of log-mel filter banks
    n_ceps : int
        number of cepstrum for MFCC
    delta_order : int
        compute deltas featues (e.g 2 means delta1 and delta2)
    energy : bool
        if True, append log energy to features
    vad : bool, tuple or list
        save Voice Activities Detection mask
        if tuple or list provodied, it must represents (distribNb, nbTrainIt)
        where distribNb is number of distribution, nbTrainIt is number of iteration
        (default: distribNb=8, nbTrainIt=12)
    downsample : str
        One of the following algorithms:
        sinc_medium : Band limited sinc interpolation, medium quality, 121dB SNR, 90% BW.
        linear : Linear interpolator, very fast, poor quality.
        sinc_fastest : Band limited sinc interpolation, fastest, 97dB SNR, 80% BW.
        zero_order_hold : Zero order hold interpolator, very fast, poor quality.
        sinc_best : Band limited sinc interpolation, best quality, 145dB SNR, 96% BW.
        (default: best quality algorithm is used)
    get_spec : bool
        return spectrogram
    get_mspec : bool
        return log-mel filterbank
    get_mfcc : bool
        return mfcc features

    Example
    -------
    '''

    @autoinit
    def __init__(self, segments, output, audio_ext=None, fs=8000,
                 win=0.025, shift=0.01, n_filters=40, n_ceps=13,
                 downsample='sinc_best', delta_order=2, energy=True, vad=True,
                 datatype='mmap', dtype='float32',
                 get_spec=False, get_mspec=True, get_mfcc=False):
        super(SpeechFeature, self).__init__('SpeechFeatures')

    def initialize(self, mr):
        if not self.get_spec and not self.get_mspec and not self.get_mfcc:
            raise Exception('You must specify which features you want: spectrogram'
                            'filter-banks, or MFCC.')
        # ====== super function should be called at the beginning ====== #
        segments = self.segments
        output = self.output
        audio_ext = as_tuple('' if self.audio_ext is None else self.audio_ext, 1, str)
        datatype = self.datatype

        # ====== load jobs ====== #
        if isinstance(segments, str):
            if not os.path.exists(segments):
                raise ValueError('Path to segments must exists, however, '
                                 'exist(segments)={}'.format(os.path.exists(segments)))
            if os.path.isdir(segments):
                file_list = get_all_files(segments)
                file_list = [(os.path.basename(i), i, 0.0, -1.0)
                             for i in file_list] # segment, path, start, end
            else: # csv file
                sep = _auto_detect_seperator(segments)
                file_list = np.genfromtxt(segments, dtype='str', delimiter=sep)
                segments = segments.replace(os.path.basename(segments), '')
                file_list = map(lambda x:
                                (x[0], os.path.join(segments, x[1]), float(x[2]), float(x[3]))
                                if not os.path.exists(x[1])
                                else x,
                                file_list)
        elif isinstance(segments, (tuple, list)):
            if isinstance(segments[0], str): # just a list of path to file
                file_list = [(os.path.basename(i), os.path.abspath(i), 0.0, -1.0)
                             for i in segments]
            elif isinstance(segments[0], (tuple, list)):
                if len(segments[0]) != 4:
                    raise Exception('segments must contain information in following for:'
                                    '[name] [path] [start] [end]')
                file_list = segments
        # filter using support audio extension
        file_list = [f for f in file_list if any(ext in f[1] for ext in audio_ext)]
        # convert into audio_path -> segment
        self.jobs = defaultdict(list)
        for segment, file, start, end in file_list:
            self.jobs[file].append((segment, start, end))
        self.jobs = self.jobs.items()

        # ====== check output ====== #
        dataset = Dataset(output)
        # create map_func
        self.wrap_map(n_filters=self.n_filters, n_ceps=self.n_ceps,
                      fs=self.fs, downsample=self.downsample,
                      win=self.win, shift=self.shift,
                      delta_order=self.delta_order, energy=self.energy,
                      vad=self.vad, dtype=self.dtype,
                      get_spec=self.get_spec, get_mspec=self.get_mspec,
                      get_mfcc=self.get_mfcc)
        # create reduce
        self.wrap_reduce(dataset=dataset, datatype=datatype)
        # create finalize
        self.wrap_finalize(dataset=dataset, get_spec=self.get_spec,
                           get_mspec=self.get_mspec, get_mfcc=self.get_mfcc)

    @staticmethod
    def _map(f, n_filters=40, n_ceps=13, fs=8000, downsample='sinc_best',
             win=0.025, shift=0.01, delta_order=2, energy=True, vad=True,
             dtype='float32', get_spec=False, get_mspec=True, get_mfcc=False):
        '''
        Return
        ------
        (name, features, vad, sum1, sum2)

        '''
        audio_path, segments = f
        # load audio data
        s, _ = speech.read(audio_path)
        # check frequency for downsampling (if necessary)
        if _ is not None:
            if fs is not None and fs != _:
                if fs < _: # downsample
                    s = resample(s, fs / _, 'sinc_best')
                else:
                    raise ValueError('Cannot perform upsample from frequency: '
                                     '{}Hz to {}Hz'.format(_, fs))
            else:
                fs = _
        N = len(s)
        features = []
        for name, start, end in segments:
            start = int(float(start) * fs)
            end = int(N if end < 0 else end * fs)
            data = s[start:end, :]
            # ====== 2 channels ====== #
            if len(data.shape) == 2:
                tmp = speech_features_extraction(s[:, 0].ravel(), fs=fs,
                    n_filters=n_filters, n_ceps=n_ceps,
                    win=win, shift=shift, delta_order=delta_order,
                    energy=energy, vad=vad, dtype=dtype,
                    get_spec=get_spec, get_mspec=get_mspec, get_mfcc=get_mfcc)
                features.append((name + '-A',) + tmp)
                tmp = speech_features_extraction(s[:, 1].ravel(), fs=fs,
                    n_filters=n_filters, n_ceps=n_ceps,
                    win=win, shift=shift, delta_order=delta_order,
                    energy=energy, vad=vad, dtype=dtype,
                    get_spec=get_spec, get_mspec=get_mspec, get_mfcc=get_mfcc)
                features.append((name + '-B',) + tmp)
            # ====== Only 1 channel ====== #
            else:
                tmp = speech_features_extraction(s.ravel(), fs=fs,
                    n_filters=n_filters, n_ceps=n_ceps,
                    win=win, shift=shift, delta_order=delta_order,
                    energy=energy, vad=vad, dtype=dtype,
                    get_spec=get_spec, get_mspec=get_mspec, get_mfcc=get_mfcc)
                features.append((name,) + tmp)
        return features

    @staticmethod
    def _reduce(results, dataset, datatype):
        # contains (name, spec, mspec, mfcc, vad)
        index = []
        spec_sum1, spec_sum2 = 0., 0.
        mspec_sum1, mspec_sum2 = 0., 0.
        mfcc_sum1, mfcc_sum2 = 0., 0.
        for r in results:
            for name, spec, mspec, mfcc, vad in r:
                if spec is not None:
                    X, sum1, sum2 = spec
                    dataset.get_data('spec', dtype=X.dtype, shape=X.shape,
                                     datatype=datatype, value=X)
                    spec_sum1 += sum1; spec_sum2 += sum2
                if mspec is not None:
                    X, sum1, sum2 = mspec
                    dataset.get_data('mspec', dtype=X.dtype, shape=X.shape,
                                     datatype=datatype, value=X)
                    mspec_sum1 += sum1; mspec_sum2 += sum2
                if mfcc is not None:
                    X, sum1, sum2 = mfcc
                    dataset.get_data('mfcc', dtype=X.dtype, shape=X.shape,
                                     datatype=datatype, value=X)
                    mfcc_sum1 += sum1; mfcc_sum2 += sum2
                # index
                index.append([name, X.shape[0]])
                # VAD
                if vad is not None:
                    assert vad.shape[0] == X.shape[0],\
                        'VAD mismatch features shape: %d != %d' % (vad.shape[0], X.shape[0])
                    dataset.get_data('vad', dtype=vad.dtype, shape=vad.shape,
                                 datatype=datatype, value=vad)
        return ((spec_sum1, spec_sum2),
                (mspec_sum1, mspec_sum2),
                (mfcc_sum1, mfcc_sum2), index)

    @staticmethod
    def _finalize(results, dataset, get_spec, get_mspec, get_mfcc):
        # contains (sum1, sum2, n)
        path = dataset.path
        spec_sum1, spec_sum2 = 0., 0.
        mspec_sum1, mspec_sum2 = 0., 0.
        mfcc_sum1, mfcc_sum2 = 0., 0.
        n = 0
        indices = []
        for spec, mspec, mfcc, index in results:
            # spec
            spec_sum1 += spec[0]
            spec_sum2 += spec[1]
            # mspec
            mspec_sum1 += mspec[0]
            mspec_sum2 += mspec[1]
            # mfcc
            mfcc_sum1 += mfcc[0]
            mfcc_sum2 += mfcc[1]
            for name, size in index:
                # name, start, end
                indices.append([name, int(n), int(n + size)])
                n += size
        # ====== saving indices ====== #
        with open(os.path.join(path, 'indices.csv'), 'w') as f:
            for name, start, end in indices:
                f.write('%s %d %d\n' % (name, start, end))

        # ====== helper ====== #
        def save_mean_std(sum1, sum2, n, name, dataset):
            mean = sum1 / n
            std = np.sqrt(sum2 / n - mean**2)
            assert not np.any(np.isnan(mean)), 'Mean contains NaN'
            assert not np.any(np.isnan(std)), 'Std contains NaN'
            dataset.get_data(name + '_mean', dtype=mean.dtype,
                             shape=mean.shape, value=mean)
            dataset.get_data(name + '_std', dtype=std.dtype,
                             shape=std.shape, value=std)
        # ====== save mean and std ====== #
        if get_spec:
            save_mean_std(spec_sum1, spec_sum2, n, 'spec', dataset)
        if get_mspec:
            save_mean_std(mspec_sum1, mspec_sum2, n, 'mspec', dataset)
        if get_mfcc:
            save_mean_std(mfcc_sum1, mfcc_sum2, n, 'mfcc', dataset)
        dataset.close()
        return {'dataset': path}


# ===========================================================================
# General global normalization
# ===========================================================================
class Normalization(FeatureRecipe):

    '''
    Parameters
    ----------
    dataset_filter : function
        input: [name,...],if None, select all dataset.
        output: filtered jobs list

    Example
    -------
    >>> norm = Normalization(dataset_path,
    ...                      dataset_filter=lambda x: ('_vad' not in x and
    ...                                                'mean' not in x and
    ...                                                'std' not in x),
    ...                      global_normalize=True, local_normalize=False)

    '''

    @autoinit
    def __init__(self, dataset_path, dataset_filter=None,
        global_normalize=True, local_normalize=False,
        global_mean=None, global_std=None,
        name='Normalization'):
        super(Normalization, self).__init__(name)

    def initialize(self, mr):
        dataset_path = self.dataset_path
        dataset_filter = self.dataset_filter

        # load dataset
        if not os.path.exists(dataset_path):
            dataset = mr[dataset_path]
            if isinstance(dataset, str):
                dataset = Dataset(dataset)
        else:
            dataset = Dataset(dataset_path)
        if not isinstance(dataset, Dataset):
            raise ValueError('Cannot load Dataset from path:{}'.format(dataset_path))
        # try to search for mean and std value from previous task
        if self.global_normalize:
            mean = 'mean' if self.global_mean is None else self.global_mean
            if isinstance(mean, str):
                mean = mr[mean] if len(mr[mean]) > 0 else dataset.get_data(mean)[:]

            std = 'std' if self.global_std is None else self.global_std
            if isinstance(std, str):
                std = mr[std] if len(mr[std]) > 0 else dataset.get_data(std)[:]

        # parse jobs_list
        jobs = dataset.info
        # filter dataset
        if dataset_filter is not None and hasattr(dataset_filter, '__call__'):
            jobs = [i for i in jobs if dataset_filter(i[0])]
        self.seq_jobs = [i[:-1] for i in jobs if i[-1] == Hdf5Data]
        self.jobs = [i[:-1] for i in jobs if i[-1] != Hdf5Data]

        # ====== inititalize function ====== #
        self.wrap_map(dataset=dataset,
            global_normalize=self.global_normalize,
            local_normalize=self.local_normalize,
            mean=mean, std=std)
        self.wrap_reduce()

    @staticmethod
    def _map(f, dataset, global_normalize, local_normalize, mean, std):
        # ====== global normalization ====== #
        name, dtype, shape = f
        x = dataset.get_data(name, dtype=dtype, shape=shape)
        if global_normalize and mean is not None and std is not None:
            x[:] = (x[:] - mean) / std
        if local_normalize:
            x[:] = (x[:] - x.mean(axis=0)) / x.std(axis=0)
        x.flush()
        dataset.close(f)

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
