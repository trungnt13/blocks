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
except:
    pass

support_delimiter = re.compile('[\s;,.:]')


# ===========================================================================
# Speech features
# ===========================================================================
def _compute_delta(s, delta1, delta2):
    _ = [s]
    d1, d2 = None, None
    if delta1:
        d1 = speech.compute_delta(s, order=1)
        if delta2:
            d2 = speech.compute_delta(s, order=2)
    if d1 is not None: _.append(d1)
    if d2 is not None: _.append(d2)
    return np.concatenate(_, 0)


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


def speech_features_extraction(s, fs,
             n_fft, n_filters, n_ceps, win, shift,
             delta1, delta2, energy,
             vad, feature_type, dtype):
    # VAD
    vad_idx = None
    if vad:
        import sidekit
        vad_idx = sidekit.frontend.vad.vad_snr(s, 30,
            fs=fs, shift=shift, nwin=n_fft)
    # speech features
    s = speech.spectrogram(s, n_fft=n_fft, hop_length=int(shift * fs),
        win_length=int(win * fs), center=False)
    if feature_type != 'spec':
        logEnergy = speech.compute_logenergy(s) if energy else None
        s = speech.melspectrogram(S=s, sr=fs, n_mels=n_filters,
            fmin=0., fmax=fs // 2, center=False)
        s = speech.logamplitude(s) # LOG-mel filterbanks
        if feature_type == 'mfcc':
            s = speech.mfcc(S=s, sr=fs, n_mfcc=n_ceps)
        if logEnergy is not None: # append log-energy
            s = np.vstack((s, logEnergy[None, :]))
        s = _compute_delta(s, delta1, delta2)
    # transpose into (T, D)
    s = s.T
    # validate vad shape
    if vad_idx is not None:
        assert vad_idx.shape[0] == s.shape[0], \
        'VAD and features mismatch: {} != {}'
        ''.format(vad_idx.shape[0], s.shape[0])
        vad = vad_idx.astype('int8')
    else:
        vad = None
    # type check
    s = s.astype(dtype)
    # normalization
    sum1 = np.sum(s, axis=0, dtype='float64')
    sum2 = np.sum(s**2, axis=0, dtype='float64')
    return s, vad, sum1, sum2


class SpeechFeature(FeatureRecipe):

    ''' Extract speech features from all audio files in given directory or
    file list, then saves them to a `keras.ext.dataset.Dataset`

    Parameters
    ----------
    downsample : str
        One of the following algorithms:
        sinc_medium : Band limited sinc interpolation, medium quality, 121dB SNR, 90% BW.
        linear : Linear interpolator, very fast, poor quality.
        sinc_fastest : Band limited sinc interpolation, fastest, 97dB SNR, 80% BW.
        zero_order_hold : Zero order hold interpolator, very fast, poor quality.
        sinc_best : Band limited sinc interpolation, best quality, 145dB SNR, 96% BW.
        (default: best quality algorithm is used)


    Example
    -------
    >>> spec = SpeechFeature('/Volumes/backup/data/DigiSami/preprocessed',
    ...                      dataset_path,
    ...                      fs=8000, win=0.025, shift=0.01, audio_ext='.wav',
    ...                      n_filters=40, n_fft=512, feature_type='filt',
    ...                      local_normalize=False, vad=True,
    ...                      delta1=True, delta2=True,
    ...                      datatype='mem')

    '''
    FEATURE_TYPES = ['spec', 'mfcc', 'fbank']

    @autoinit
    def __init__(self, segments, output, audio_ext=None,
                 n_fft=256, n_filters=40, n_ceps=13,
                 fs=8000, downsample='sinc_best', win=0.025, shift=0.01,
                 delta1=True, delta2=True, energy=True,
                 vad=True, feature_type='spec',
                 datatype='mmap', dtype='float32'):
        name = 'Extract %s ' % feature_type
        super(SpeechFeature, self).__init__(name)

    def initialize(self, mr):
        # ====== super function should be called at the beginning ====== #
        if all(i not in self.feature_type for i in SpeechFeature.FEATURE_TYPES):
            raise ValueError("Only accept one of following feature types:"
                             "'spec', 'mfcc', 'fbank'.")

        segments = self.segments
        output = self.output
        audio_ext = as_tuple('' if self.audio_ext is None else self.audio_ext, 1, str)
        datatype = self.datatype

        # ====== load jobs ====== #
        if not os.path.exists(segments):
            raise ValueError('Path to segments must exists, however, '
                             'exist(segments)={}'.format(os.path.exists(segments)))
        if os.path.isdir(segments):
            file_list = get_all_files(segments)
            file_list = [(os.path.basename(i), i, 0.0, -1.0) for i in file_list] # segment, path, start, end
        else: # csv file
            sep = _auto_detect_seperator(segments)
            file_list = np.genfromtxt(segments, dtype='str', delimiter=sep)
            segments = segments.replace(os.path.basename(segments), '')
            file_list = map(lambda x:
                            (x[0], os.path.join(segments, x[1]), float(x[2]), float(x[3]))
                            if not os.path.exists(x[1])
                            else x,
                            file_list)
        # filter using support audio extension
        file_list = [f for f in file_list if any(ext in f[1] for ext in audio_ext)]
        # convert into audio_path -> segment
        self.jobs = defaultdict(list)
        for segment, file, start, end in file_list:
            self.jobs[file].append((segment, start, end))
        self.jobs = self.jobs.items()[:100]

        # ====== check output ====== #
        dataset = Dataset(output)
        # create map_func
        self.wrap_map(n_fft=self.n_fft, n_filters=self.n_filters, n_ceps=self.n_ceps,
                      fs=self.fs, downsample=self.downsample,
                      win=self.win, shift=self.shift,
                      delta1=self.delta1, delta2=self.delta2, energy=self.energy,
                      vad=self.vad, feature_type=self.feature_type, dtype=self.dtype)
        # create reduce
        self.wrap_reduce(dataset=dataset, datatype=datatype, dataname=self.feature_type)
        # create finalize
        self.wrap_finalize(dataset=dataset)

    @staticmethod
    def _map(f, n_fft=256, n_filters=40, n_ceps=13,
             fs=8000, downsample='sinc_best',
             win=0.025, shift=0.01,
             delta1=True, delta2=True, energy=True,
             vad=True, feature_type='spec',
             dtype='float32'):
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
        # clean the signal
        s = speech.pre_emphasis(s, 0.97)
        features = []
        for name, start, end in segments:
            start = int(float(start) * fs)
            end = int(N if end < 0 else end * fs)
            data = s[start:end, :]
            if len(data.shape) == 2: # 2 channel
                x0 = speech_features_extraction(s[:, 0].ravel(), fs=fs,
                    n_fft=n_fft, n_filters=n_filters, n_ceps=n_ceps,
                    win=win, shift=shift, delta1=delta1, delta2=delta2,
                    energy=energy, vad=vad, feature_type=feature_type, dtype=dtype)
                x1 = speech_features_extraction(s[:, 1].ravel(), fs=fs,
                    n_fft=n_fft, n_filters=n_filters, n_ceps=n_ceps,
                    win=win, shift=shift, delta1=delta1, delta2=delta2,
                    energy=energy, vad=vad, feature_type=feature_type, dtype=dtype)
                features.append((name + '-0',) + x0)
                features.append((name + '-1',) + x1)
            else: # only 1 channel
                x = speech_features_extraction(s.ravel(), fs=fs,
                    n_fft=n_fft, n_filters=n_filters, n_ceps=n_ceps,
                    win=win, shift=shift, delta1=delta1, delta2=delta2,
                    energy=energy, vad=vad, feature_type=feature_type, dtype=dtype)
                features.append((name,) + x)
        return features

    @staticmethod
    def _reduce(results, dataset, datatype, dataname):
        # contains (name, features, vad, sum1, sum2)
        index = []
        sum1, sum2 = 0, 0
        for r in results:
            for name, features, vad, s1, s2 in r:
                index.append([name, features.shape[0]])
                # features
                dataset.get_data(dataname, dtype=features.dtype,
                                 shape=features.shape, datatype=datatype,
                                 value=features)
                if vad is not None: # vad
                    dataset.get_data(dataname + '_vad', dtype=vad.dtype,
                                     shape=vad.shape, datatype=datatype, value=vad)
                # just pass sum1, sum2, len(x) for finalize
                sum1 += s1
                sum2 += s2
        return (sum1, sum2, index)

    @staticmethod
    def _finalize(results, dataset):
        # contains (sum1, sum2, n)
        sum1, sum2, n = 0, 0, 0
        indices = []
        for s1, s2, index in results:
            sum1 += s1
            sum2 += s2
            for name, size in index:
                # name, start, end
                indices.append([name, int(n), int(n + size)])
                n += size
        mean = sum1 / n
        std = np.sqrt((sum2 - sum1**2 / n) / n)
        assert not np.any(np.isnan(mean)), 'Mean contains NaN'
        assert not np.any(np.isnan(std)), 'Std contains NaN'
        dataset.get_data('mean', dtype=mean.dtype, shape=mean.shape, value=mean)
        dataset.get_data('std', dtype=std.dtype, shape=std.shape, value=std)
        path = dataset.path
        dataset.close()
        # ====== saving indices ====== #
        with open(os.path.join(path, 'indices.csv'), 'w') as f:
            for name, start, end in indices:
                f.write('%s %d %d \n' % (name, start, end))
        return {'dataset': path, 'mean': mean, 'std': std}


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
