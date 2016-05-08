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

from blocks.utils import get_all_files
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
    FEATURE_TYPES = ['spec', 'mfcc', 'filt']

    @autoinit
    def __init__(self, input_dir, output_dir,
                 audio_ext=None, transcript_ext=None,
                 n_fft=256, n_filters=40, n_ceps=13,
                 fs=8000, downsample='sinc_best', win=0.025, shift=0.01,
                 delta1=True, delta2=True, energy=True,
                 local_normalize=False,
                 vad=True, feature_type='spec',
                 datatype='mmap', dtype='float32', name=None):
        # ====== super function should be called at the beginning ====== #
        if all(i not in feature_type for i in SpeechFeature.FEATURE_TYPES):
            raise ValueError("Only accept one of following feature types:"
                             "'spec', 'mfcc', 'filt'.")
        else:
            feature_type = [i for i in SpeechFeature.FEATURE_TYPES
                            if i in feature_type][0]
        if name is None:
            name = feature_type
        super(SpeechFeature, self).__init__(name)

        if transcript_ext is not None and not isinstance(transcript_ext, (tuple, list)):
            self.transcript_ext = [transcript_ext]

        if audio_ext is None:
            audio_ext = ''
        if not isinstance(audio_ext, (tuple, list)):
            self.audio_ext = [audio_ext]

    def initialize(self, mr):
        input_dir = self.input_dir
        output_dir = self.output_dir
        audio_ext = self.audio_ext
        transcript_ext = self.transcript_ext
        datatype = self.datatype

        # ====== load jobs ====== #
        if not os.path.exists(input_dir):
            raise ValueError('Path to input_dir must exists, however, '
                             'exist(input_dir)={}'.format(os.path.exists(input_dir)))
        if os.path.isdir(input_dir):
            file_list = get_all_files(input_dir)
        else: # csv file
            sep = _auto_detect_seperator(input_dir)
            file_list = np.genfromtxt(input_dir, dtype='str', delimiter=sep)[:, 0]
            input_dir = input_dir.replace(os.path.basename(input_dir), '')
            file_list = map(lambda x: os.path.join(input_dir, x)
                            if not os.path.exists(x)
                            else x,
                            file_list)

        audio_list = sorted([f for f in file_list if any(ext in f for ext in audio_ext)])
        transcript_list = [None] * len(audio_list)
        if transcript_ext is not None:
            transcript_list = sorted(
                [f for f in file_list if any(ext in f for ext in transcript_ext)])
        self.jobs = list(zip(audio_list, transcript_list))

        # ====== check output_dir ====== #
        dataset = Dataset(output_dir)
        if dataset.size > 0:
            raise ValueError('Dataset at path={} already exists with size={}(MB)'
                             ''.format(os.path.abspath(output_dir), dataset.size))
        # create map_func
        self.wrap_map(n_fft=self.n_fft, n_filters=self.n_filters, n_ceps=self.n_ceps,
                      fs=self.fs, downsample=self.downsample,
                      win=self.win, shift=self.shift,
                      delta1=self.delta1, delta2=self.delta2, energy=self.energy,
                      local_normalize=self.local_normalize,
                      vad=self.vad, feature_type=self.feature_type, dtype=self.dtype)
        # create reduce
        self.wrap_reduce(dataset=dataset, datatype=datatype)
        # create finalize
        self.wrap_finalize(dataset=dataset)

    @staticmethod
    def _map(f, n_fft=256, n_filters=40, n_ceps=13,
             fs=8000, downsample='sinc_best',
             win=0.025, shift=0.01,
             delta1=True, delta2=True, energy=True,
             local_normalize=False,
             vad=True, feature_type='spec',
             dtype='float32'):
        '''
        Return
        ------
        (name, features, vad, sum1, sum2)

        '''
        audio_path, transcript_path = f
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
        if local_normalize:
            s = normalize(s, axis=0)
        # name without extension
        name = os.path.basename(audio_path).split('.')
        name = ''.join(name[:-1])
        # reading the transcript
        trans = None
        if transcript_path is not None:
            transcripter = _read_transcript_file(transcript_path)
            xmin, xmax = transcripter.min_max()
            idx = np.linspace(xmin, xmax, num=N, endpoint=True)
            idx = segment_axis(a=idx, frame_length=n_fft, hop_length=int(shift * fs),
                               axis=0, end='cut', endvalue=0)
            # start, end of a segment
            trans = np.asarray([transcripter.transcript((i[0], i[-1])) for i in idx])
            assert trans.shape[0] == s.shape[0], 'Shape of transcript and features'\
                ' are mismatch, {} != {}'.format(trans.shape[0], s.shape[0])
        return (name, s, vad, trans, sum1, sum2)

    @staticmethod
    def _reduce(results, dataset, datatype):
        # contains (name, features, vad, transcript, sum1, sum2)
        new_results = []
        for name, features, vad, trans, sum1, sum2 in results:
            # features
            x = dataset.get_data(name, dtype=features.dtype, shape=features.shape,
                                 datatype=datatype, value=features)
            dataset.close(name)
            if vad is not None: # vad
                x = dataset.get_data(name + '_vad', dtype=vad.dtype, shape=vad.shape,
                                     datatype=datatype, value=vad)
                dataset.close(name + '_vad')
            if trans is not None: # transcript
                x = dataset.get_data(name + '_trans', dtype=trans.dtype, shape=trans.shape,
                                     datatype=datatype, value=trans)
                dataset.close(name + '_trans')
            # just pass sum1, sum2, len(x) for finalize
            new_results.append((sum1, sum2, len(x)))
        return new_results

    @staticmethod
    def _finalize(results, dataset):
        # contains (sum1, sum2, n)
        sum1, sum2, n = 0, 0, 0
        for s1, s2, nb in results:
            sum1 += s1
            sum2 += s2
            n += nb
        mean = sum1 / n
        std = np.sqrt((sum2 - sum1**2 / n) / n)
        assert not np.any(np.isnan(mean)), 'Mean contains NaN'
        assert not np.any(np.isnan(std)), 'Std contains NaN'
        dataset.get_data('mean', dtype=mean.dtype, shape=mean.shape, value=mean)
        dataset.get_data('std', dtype=std.dtype, shape=std.shape, value=std)
        path = dataset.path
        dataset.close()
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
