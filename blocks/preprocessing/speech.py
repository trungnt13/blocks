# -*- coding: utf-8 -*-
# ===========================================================================
# The waveform and spectrogram preprocess utilities is adapted from:
# [librosa](https://github.com/bmcfee/librosa)
# Copyright (c) 2016, librosa development team.
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import print_function, division, absolute_import

import six
from collections import OrderedDict
import warnings

import numpy as np
import scipy.fftpack as fft
import scipy.signal

from blocks.preprocessing import preprocess as pp

MAX_MEM_BLOCK = 2**8 * 2**10
SMALL_FLOAT = 1e-20

# ===========================================================================
# Predefined variables of speech datasets
# ===========================================================================
# ==================== Predefined datasets information ==================== #
nist15_cluster_lang = OrderedDict([
    ['ara', ['ara-arz', 'ara-acm', 'ara-apc', 'ara-ary', 'ara-arb']],
    ['zho', ['zho-yue', 'zho-cmn', 'zho-cdo', 'zho-wuu']],
    ['eng', ['eng-gbr', 'eng-usg', 'eng-sas']],
    ['fre', ['fre-waf', 'fre-hat']],
    ['qsl', ['qsl-pol', 'qsl-rus']],
    ['spa', ['spa-car', 'spa-eur', 'spa-lac', 'por-brz']]
])
nist15_lang_list = np.asarray([
    # Egyptian, Iraqi, Levantine, Maghrebi, Modern Standard
    'ara-arz', 'ara-acm', 'ara-apc', 'ara-ary', 'ara-arb',
    # Cantonese, Mandarin, Min Dong, Wu
    'zho-yue', 'zho-cmn', 'zho-cdo', 'zho-wuu',
    # British, American, South Asian (Indian)
    'eng-gbr', 'eng-usg', 'eng-sas',
    # West african, Haitian
    'fre-waf', 'fre-hat',
    # Polish, Russian
    'qsl-pol', 'qsl-rus',
    # Caribbean, European, Latin American, Brazilian
    'spa-car', 'spa-eur', 'spa-lac', 'por-brz'])


def nist15_label(label):
    '''
    Return
    ------
    lang_id : int
        idx in the list of 20 language, None if not found
    cluster_id : int
        idx in the list of 6 clusters, None if not found
    within_cluster_id : int
        idx in the list of each clusters, None if not found
    '''
    label = label.replace('spa-brz', 'por-brz')
    rval = [None, None, None]
    # lang_id
    if label not in nist15_lang_list:
        raise ValueError('Cannot found label:%s' % label)
    rval[0] = np.argmax(label == nist15_lang_list)
    # cluster_id
    for c, x in enumerate(nist15_cluster_lang.iteritems()):
        j = x[1]
        if label in j:
            rval[1] = c
            rval[2] = j.index(label)
    return rval

# ==================== Timit ==================== #
timit_61 = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay',
    'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en',
    'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih',
    'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow',
    'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th',
    'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
timit_39 = ['aa', 'ae', 'ah', 'aw', 'ay', 'b', 'ch', 'd',
    'dh', 'dx', 'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k',
    'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 'sil', 't',
    'th', 'uh', 'uw', 'v', 'w', 'y', 'z']
timit_map = {'ao': 'aa', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er',
    'hv': 'hh', 'ix': 'ih', 'el': 'l', 'em': 'm',
    'en': 'n', 'nx': 'n',
    'eng': 'ng', 'zh': 'sh', 'ux': 'uw',
    'pcl': 'sil', 'tcl': 'sil', 'kcl': 'sil', 'bcl': 'sil',
    'dcl': 'sil', 'gcl': 'sil', 'h#': 'sil', 'pau': 'sil', 'epi': 'sil'}


def timit_phonemes(phn, map39=False, blank=False):
    ''' Included blank '''
    if type(phn) not in (list, tuple, np.ndarray):
        phn = [phn]
    if map39:
        timit = timit_39
        timit_map = timit_map
        l = 39
    else:
        timit = timit_61
        timit_map = {}
        l = 61
    # ====== return phonemes ====== #
    rphn = []
    for p in phn:
        if p not in timit_map and p not in timit:
            if blank: rphn.append(l)
        else:
            rphn.append(timit.index(timit_map[p]) if p in timit_map else timit.index(p))
    return rphn


# ===========================================================================
# Audio helper
# ===========================================================================
def read(f, pcm=False):
    '''
    Return
    ------
        waveform (ndarray), sample rate (int)
    '''
    if pcm or (isinstance(f, str) and
               any(i in f for i in ['pcm', 'PCM', 'SPH', 'sph'])):
        return (np.memmap(f, dtype=np.int16, mode='r'), None)

    from soundfile import read
    return read(f)


def resample(s, fs, fs_new, algorithm='sinc_best'):
    '''
    sinc_medium : Band limited sinc interpolation, medium quality, 121dB SNR, 90% BW.
    linear : Linear interpolator, very fast, poor quality.
    sinc_fastest : Band limited sinc interpolation, fastest, 97dB SNR, 80% BW.
    zero_order_hold : Zero order hold interpolator, very fast, poor quality.
    sinc_best : Band limited sinc interpolation, best quality, 145dB SNR, 96% BW.
    '''
    from scikits.samplerate import resample
    if fs_new == fs:
        return s
    return resample(s, fs_new / fs, 'sinc_best')


def save(f, s, fs, subtype=None):
    '''
    Return
    ------
        waveform (ndarray), sample rate (int)
    '''
    from soundfile import write
    return write(f, s, fs, subtype=subtype)


# ===========================================================================
# Spectrogram manipulation
# ===========================================================================
def _validate_stft_arguments(n_fft, hop_length, win_length):
    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length / 4)
    return n_fft, hop_length, win_length


def pre_emphasis(input, pre):
    """Pre-emphasis of an audio signal.

    :param pre: value that defines the pre-emphasis filter.
    """
    return scipy.signal.lfilter([1.0, -pre], 1, input)


def compute_delta(data, width=9, order=1, axis=-1, trim=True):
    r'''Compute delta features: local estimate of the derivative
    of the input data along the selected axis.
    Parameters
    ----------
    data      : np.ndarray
        the input data matrix (eg, spectrogram)
    width     : int >= 3, odd [scalar]
        Number of frames over which to compute the delta feature
    order     : int > 0 [scalar]
        the order of the difference operator.
        1 for first derivative, 2 for second, etc.
    axis      : int [scalar]
        the axis along which to compute deltas.
        Default is -1 (columns).
    trim      : bool
        set to `True` to trim the output matrix to the original size.
    Returns
    -------
    delta_data   : np.ndarray [shape=(d, t) or (d, t + window)]
        delta matrix of `data`.
    Examples
    --------
    Compute MFCC deltas, delta-deltas
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> mfcc = librosa.feature.mfcc(y=y, sr=sr)
    >>> mfcc_delta = librosa.feature.delta(mfcc)
    >>> mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    '''

    data = np.atleast_1d(data)

    if width < 3 or np.mod(width, 2) != 1:
        raise ValueError('width must be an odd integer >= 3')

    if order <= 0 or not isinstance(order, int):
        raise ValueError('order must be a positive integer')

    half_length = 1 + int(width // 2)
    window = np.arange(half_length - 1., -half_length, -1.)

    # Normalize the window so we're scale-invariant
    window /= np.sum(np.abs(window)**2)

    # Pad out the data by repeating the border values (delta=0)
    padding = [(0, 0)] * data.ndim
    width = int(width)
    padding[axis] = (width, width)
    delta_x = np.pad(data, padding, mode='edge')

    for _ in range(order):
        delta_x = scipy.signal.lfilter(window, 1, delta_x, axis=axis)

    # Cut back to the original shape of the input data
    if trim:
        idx = [slice(None)] * delta_x.ndim
        idx[axis] = slice(- half_length - data.shape[axis], - half_length)
        delta_x = delta_x[idx]

    return delta_x


def compute_logenergy(S):
    ''' This version is faster and simpler than other version of calculation

    Parameters
    ----------
    S : ndarray
        spectrogram [1 + nfft/2, T]
    '''

    energy = np.sum(S, 0) # this stores the total energy in each frame
    # if energy is zero, we get problems with log
    energy = np.where(energy == 0, np.finfo(float).eps, energy)
    return np.log(energy)


def fft_covert(fs, window=0.025, shift = 0.01):
    ''' Convert conventional information if window_length (in ms) and shift
    (in ms) to parameters for `stft` function

    Return
    ------
    n_fft: int
        number of fft should use
    hop_length: int
        number of sample points should be shifted
    win_length: int
        number of sample points for each windows
    '''
    window_length = int(round(window * fs))
    overlap = window_length - int(shift * fs)
    nfft = 2**int(np.ceil(np.log2(window_length)))
    return nfft, overlap, window_length


def stft(y, n_fft=2048, hop_length=None, win_length=None, window=None,
         center=True, dtype=np.complex64):
    """Short-time Fourier transform (STFT)

    Returns a complex-valued matrix D such that
        `np.abs(D[f, t])` is the magnitude of frequency bin `f`
        at frame `t`

        `np.angle(D[f, t])` is the phase of frequency bin `f`
        at frame `t`

    Parameters
    ----------
    y : np.ndarray [shape=(n,)], real-valued
        the input signal (audio time series)

    n_fft : int > 0 [scalar]
        FFT window size

    hop_length : int > 0 [scalar]
        number audio of frames between STFT columns.
        If unspecified, defaults `win_length / 4`.

    win_length  : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()`.
        The window will be of length `win_length` and then padded
        with zeros to match `n_fft`.

        If unspecified, defaults to ``win_length = n_fft``.

    window : None, function, np.ndarray [shape=(n_fft,)]
        - None (default): use an asymmetric Hann window
        - a window function, such as `scipy.signal.hanning`
        - a vector or array of length `n_fft`

    center      : boolean
        - If `True`, the signal `y` is padded so that frame
          `D[:, t]` is centered at `y[t * hop_length]`.
        - If `False`, then `D[:, t]` begins at `y[t * hop_length]`

    dtype       : numeric type
        Complex numeric type for `D`.  Default is 64-bit complex.


    Returns
    -------
    D : np.ndarray [shape=(1 + n_fft/2, t), dtype=dtype]
        STFT matrix, shape: [number_of_fft, time]


    Raises
    ------
    ValueError
        If `window` is supplied as a vector of length `n_fft`.


    See Also
    --------
    istft : Inverse STFT

    ifgram : Instantaneous frequency spectrogram

    """

    n_fft, hop_length, win_length = _validate_stft_arguments(
        n_fft, hop_length, win_length)

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length / 4)

    if window is None:
        # Default is an asymmetric Hann window
        fft_window = scipy.signal.hamming(win_length, sym=False)

    elif six.callable(window):
        # User supplied a window function
        fft_window = window(win_length)

    else:
        # User supplied a window vector.
        # Make sure it's an array:
        fft_window = np.asarray(window)

        # validate length compatibility
        if fft_window.size != n_fft:
            raise ValueError('Size mismatch between n_fft and len(window)')

    # Pad the window out to n_fft size
    fft_window = pp.pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Pad the time series so that frames are centered
    if center:
        pp.valid_audio(y)
        y = np.pad(y, int(n_fft // 2), mode='reflect')

    # Window the time series.
    y_frames = pp.frame(y, frame_length=n_fft, hop_length=hop_length)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
                           dtype=dtype,
                           order='F')

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = int(MAX_MEM_BLOCK / (stft_matrix.shape[0] * stft_matrix.itemsize))

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        # RFFT and Conjugate here to match phase from DPWE code
        stft_matrix[:, bl_s:bl_t] = fft.fft(fft_window *
                                            y_frames[:, bl_s:bl_t],
                                            axis=0)[:stft_matrix.shape[0]].conj()

    return stft_matrix


def istft(stft_matrix, hop_length=None, win_length=None, window=None,
          center=True, dtype=np.float32):
    """
    Inverse short-time Fourier transform (ISTFT).

    Converts a complex-valued spectrogram `stft_matrix` to time-series `y`
    by minimizing the mean squared error between `stft_matrix` and STFT of
    `y` as described in [1]_.

    In general, window function, hop length and other parameters should be same
    as in stft, which mostly leads to perfect reconstruction of a signal from
    unmodified `stft_matrix`.

    .. [1] D. W. Griffin and J. S. Lim,
        "Signal estimation from modified short-time Fourier transform,"
        IEEE Trans. ASSP, vol.32, no.2, pp.236–243, Apr. 1984.

    Parameters
    ----------
    stft_matrix : np.ndarray [shape=(1 + n_fft/2, t)]
        STFT matrix from `stft`

    hop_length  : int > 0 [scalar]
        Number of frames between STFT columns.
        If unspecified, defaults to `win_length / 4`.

    win_length  : int <= n_fft = 2 * (stft_matrix.shape[0] - 1)
        When reconstructing the time series, each frame is windowed
        and each sample is normalized by the sum of squared window
        according to the `window` function (see below).

        If unspecified, defaults to `n_fft`.

    window      : None, function, np.ndarray [shape=(n_fft,)]
        - None (default): use an asymmetric Hann window
        - a window function, such as `scipy.signal.hanning`
        - a user-specified window vector of length `n_fft`

    center      : boolean
        - If `True`, `D` is assumed to have centered frames.
        - If `False`, `D` is assumed to have left-aligned frames.

    dtype       : numeric type
        Real numeric type for `y`.  Default is 32-bit float.

    Returns
    -------
    y : np.ndarray [shape=(n,)]
        time domain signal reconstructed from `stft_matrix`

    Raises
    ------
    ValueError
        If `window` is supplied as a vector of length `n_fft`

    See Also
    --------
    stft : Short-time Fourier Transform

    """

    n_fft = 2 * (stft_matrix.shape[0] - 1)

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length / 4)

    if window is None:
        # Default is an asymmetric Hann window.
        ifft_window = scipy.signal.hann(win_length, sym=False)

    elif six.callable(window):
        # User supplied a windowing function
        ifft_window = window(win_length)

    else:
        # User supplied a window vector.
        # Make it into an array
        ifft_window = np.asarray(window)

        # Verify that the shape matches
        if ifft_window.size != n_fft:
            raise ValueError('Size mismatch between n_fft and window size')

    # Pad out to match n_fft
    ifft_window = pp.pad_center(ifft_window, n_fft)

    n_frames = stft_matrix.shape[1]
    expected_signal_len = n_fft + hop_length * (n_frames - 1)
    y = np.zeros(expected_signal_len, dtype=dtype)
    ifft_window_sum = np.zeros(expected_signal_len, dtype=dtype)
    ifft_window_square = ifft_window * ifft_window

    for i in range(n_frames):
        sample = i * hop_length
        spec = stft_matrix[:, i].flatten()
        spec = np.concatenate((spec.conj(), spec[-2:0:-1]), 0)
        ytmp = ifft_window * fft.ifft(spec).real

        y[sample:(sample + n_fft)] = y[sample:(sample + n_fft)] + ytmp
        ifft_window_sum[sample:(sample + n_fft)] += ifft_window_square

    # Normalize by sum of squared window
    approx_nonzero_indices = ifft_window_sum > SMALL_FLOAT
    y[approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]

    if center:
        y = y[int(n_fft // 2):-int(n_fft // 2)]

    return y


def ifgram(y, sr=22050, n_fft=2048, hop_length=None, win_length=None,
           norm=False, center=True, ref_power=1e-6, clip=True, dtype=np.complex64):
    '''Compute the instantaneous frequency (as a proportion of the sampling rate)
    obtained as the time-derivative of the phase of the complex spectrum as
    described by [1]_.

    Calculates regular STFT as a side effect.

    .. [1] Abe, Toshihiko, Takao Kobayashi, and Satoshi Imai.
        "Harmonics tracking and pitch extraction based on instantaneous
        frequency."
        International Conference on Acoustics, Speech, and Signal Processing,
        ICASSP-95., Vol. 1. IEEE, 1995.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series

    sr : number > 0 [scalar]
        sampling rate of `y`

    n_fft : int > 0 [scalar]
        FFT window size

    hop_length : int > 0 [scalar]
        hop length, number samples between subsequent frames.
        If not supplied, defaults to `win_length / 4`.

    win_length : int > 0, <= n_fft
        Window length. Defaults to `n_fft`.
        See `stft` for details.

    norm : bool
        Normalize the STFT.

    center      : boolean
        - If `True`, the signal `y` is padded so that frame
            `D[:, t]` (and `if_gram`) is centered at `y[t * hop_length]`.
        - If `False`, then `D[:, t]` at `y[t * hop_length]`

    ref_power : float >= 0 or callable
        Minimum power threshold for estimating instantaneous frequency.
        Any bin with `np.abs(D[f, t])**2 < ref_power` will receive the
        default frequency estimate.

        If callable, the threshold is set to `ref_power(np.abs(D)**2)`.

    clip : boolean
        - If `True`, clip estimated frequencies to the range `[0, 0.5 * sr]`.
        - If `False`, estimated frequencies can be negative or exceed
          `0.5 * sr`.

    dtype : numeric type
        Complex numeric type for `D`.  Default is 64-bit complex.

    Returns
    -------
    if_gram : np.ndarray [shape=(1 + n_fft/2, t), dtype=real]
        Instantaneous frequency spectrogram:
        `if_gram[f, t]` is the frequency at bin `f`, time `t`

    D : np.ndarray [shape=(1 + n_fft/2, t), dtype=complex]
        Short-time Fourier transform

    See Also
    --------
    stft : Short-time Fourier Transform


    '''

    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = int(win_length // 4)

    # Construct a padded hann window
    window = pp.pad_center(scipy.signal.hann(win_length, sym=False), n_fft)

    # Window for discrete differentiation
    freq_angular = np.linspace(0, 2 * np.pi, n_fft, endpoint=False)

    d_window = np.sin(-freq_angular) * np.pi / n_fft

    stft_matrix = stft(y, n_fft=n_fft, hop_length=hop_length,
                       window=window, center=center, dtype=dtype)

    diff_stft = stft(y, n_fft=n_fft, hop_length=hop_length,
                     window=d_window, center=center, dtype=dtype).conj()

    # Compute power normalization. Suppress zeros.
    mag, phase = magphase(stft_matrix)

    if six.callable(ref_power):
        ref_power = ref_power(mag**2)
    elif ref_power < 0:
        raise ValueError('ref_power must be non-negative or callable.')

    # Pylint does not correctly infer the type here, but it's correct.
    # pylint: disable=maybe-no-member
    freq_angular = freq_angular.reshape((-1, 1))
    bin_offset = (phase * diff_stft).imag / mag

    bin_offset[mag < ref_power**0.5] = 0

    if_gram = freq_angular[:n_fft // 2 + 1] + bin_offset

    if norm:
        stft_matrix = stft_matrix * 2.0 / window.sum()

    if clip:
        np.clip(if_gram, 0, np.pi, out=if_gram)

    if_gram *= float(sr) * 0.5 / np.pi

    return if_gram, stft_matrix


def magphase(D):
    """Separate a complex-valued spectrogram D into its magnitude (S)
    and phase (P) components, so that `D = S * P`.


    Parameters
    ----------
    D       : np.ndarray [shape=(d, t), dtype=complex]
        complex-valued spectrogram


    Returns
    -------
    D_mag   : np.ndarray [shape=(d, t), dtype=real]
        magnitude of `D`
    D_phase : np.ndarray [shape=(d, t), dtype=complex]
        `exp(1.j * phi)` where `phi` is the phase of `D`


    """

    mag = np.abs(D)
    phase = np.exp(1.j * np.angle(D))

    return mag, phase


def phase_vocoder(D, rate, hop_length=None):
    """Phase vocoder.  Given an STFT matrix D, speed up by a factor of `rate`

    Based on the implementation provided by [1]_.

    .. [1] Ellis, D. P. W. "A phase vocoder in Matlab."
        Columbia University, 2002.
        http://www.ee.columbia.edu/~dpwe/resources/matlab/pvoc/

    Parameters
    ----------
    D : np.ndarray [shape=(d, t), dtype=complex]
        STFT matrix

    rate :  float > 0 [scalar]
        Speed-up factor: `rate > 1` is faster, `rate < 1` is slower.

    hop_length : int > 0 [scalar] or None
        The number of samples between successive columns of `D`.

        If None, defaults to `n_fft/4 = (D.shape[0]-1)/2`

    Returns
    -------
    D_stretched  : np.ndarray [shape=(d, t / rate), dtype=complex]
        time-stretched STFT
    """

    n_fft = 2 * (D.shape[0] - 1)

    if hop_length is None:
        hop_length = int(n_fft // 4)

    time_steps = np.arange(0, D.shape[1], rate, dtype=np.float)

    # Create an empty output array
    d_stretch = np.zeros((D.shape[0], len(time_steps)), D.dtype, order='F')

    # Expected phase advance in each bin
    phi_advance = np.linspace(0, np.pi * hop_length, D.shape[0])

    # Phase accumulator; initialize to the first sample
    phase_acc = np.angle(D[:, 0])

    # Pad 0 columns to simplify boundary logic
    D = np.pad(D, [(0, 0), (0, 2)], mode='constant')

    for (t, step) in enumerate(time_steps):

        columns = D[:, int(step):int(step + 2)]

        # Weighting for linear magnitude interpolation
        alpha = np.mod(step, 1.0)
        mag = ((1.0 - alpha) * np.abs(columns[:, 0]) + alpha * np.abs(columns[:, 1]))

        # Store to output array
        d_stretch[:, t] = mag * np.exp(1.j * phase_acc)

        # Compute phase advance
        dphase = (np.angle(columns[:, 1]) - np.angle(columns[:, 0]) - phi_advance)

        # Wrap to -pi:pi range
        dphase = dphase - 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))

        # Accumulate phase
        phase_acc += phi_advance + dphase

    return d_stretch


def logamplitude(S, ref_power=1.0, amin=1e-10, top_db=80.0):
    """Log-scale the amplitude of a spectrogram.

    Parameters
    ----------
    S : np.ndarray [shape=(d, t)]
        input spectrogram

    ref_power : scalar or function
        If scalar, `log(abs(S))` is compared to `log(ref_power)`.

        If a function, `log(abs(S))` is compared to `log(ref_power(abs(S)))`.

        This is primarily useful for comparing to the maximum value of `S`.

    amin    : float > 0[scalar]
        minimum amplitude threshold for `abs(S)` and `ref_power`

    top_db  : float >= 0 [scalar]
        threshold log amplitude at top_db below the peak:
        ``max(log(S)) - top_db``

    Returns
    -------
    log_S   : np.ndarray [shape=(d, t)]
        ``log_S ~= 10 * log10(S) - 10 * log10(abs(ref_power))``


    """

    if amin <= 0:
        raise ValueError('amin must be strictly positive')

    magnitude = np.abs(S)

    if six.callable(ref_power):
        # User supplied a function to calculate reference power
        __ref = ref_power(magnitude)
    else:
        __ref = np.abs(ref_power)

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, __ref))

    if top_db is not None:
        if top_db < 0:
            raise ValueError('top_db must be non-negative positive')
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


# ===========================================================================
# Filter
# ===========================================================================
def _fft_frequencies(sr=22050, n_fft=2048):
    '''Alternative implementation of `np.fft.fftfreqs`

    Parameters
    ----------
    sr : number > 0 [scalar]
        Audio sampling rate

    n_fft : int > 0 [scalar]
        FFT window size


    Returns
    -------
    freqs : np.ndarray [shape=(1 + n_fft/2,)]
        Frequencies `(0, sr/n_fft, 2*sr/n_fft, ..., sr/2)`


    Examples
    --------
    >>> librosa.fft_frequencies(sr=22050, n_fft=16)
    array([     0.   ,   1378.125,   2756.25 ,   4134.375,
             5512.5  ,   6890.625,   8268.75 ,   9646.875,  11025.   ])

    '''

    return np.linspace(0,
                       float(sr) / 2,
                       int(1 + n_fft // 2),
                       endpoint=True)


def _mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False):
    """Compute the center frequencies of mel bands.

    Parameters
    ----------
    n_mels    : int > 0 [scalar]
        number of Mel bins

    fmin      : float >= 0 [scalar]
        minimum frequency (Hz)

    fmax      : float >= 0 [scalar]
        maximum frequency (Hz)

    htk       : bool
        use HTK formula instead of Slaney

    Returns
    -------
    bin_frequencies : ndarray [shape=(n_mels,)]
        vector of n_mels frequencies in Hz which are uniformly spaced on the Mel
        axis.

    Examples
    --------
    >>> librosa.mel_frequencies(n_mels=40)
    array([     0.   ,     85.317,    170.635,    255.952,
              341.269,    426.586,    511.904,    597.221,
              682.538,    767.855,    853.173,    938.49 ,
             1024.856,   1119.114,   1222.042,   1334.436,
             1457.167,   1591.187,   1737.532,   1897.337,
             2071.84 ,   2262.393,   2470.47 ,   2697.686,
             2945.799,   3216.731,   3512.582,   3835.643,
             4188.417,   4573.636,   4994.285,   5453.621,
             5955.205,   6502.92 ,   7101.009,   7754.107,
             8467.272,   9246.028,  10096.408,  11025.   ])

    """

    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = pp.hz_to_mel(fmin, htk=htk)
    max_mel = pp.hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return pp.mel_to_hz(mels, htk=htk)


def mel_filter(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False):
    """Create a Filterbank matrix to combine FFT bins into Mel-frequency bins

    Parameters
    ----------
    sr        : number > 0 [scalar]
        sampling rate of the incoming signal

    n_fft     : int > 0 [scalar]
        number of FFT components

    n_mels    : int > 0 [scalar]
        number of Mel bands to generate

    fmin      : float >= 0 [scalar]
        lowest frequency (in Hz)

    fmax      : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use `fmax = sr / 2.0`

    htk       : bool [scalar]
        use HTK formula instead of Slaney

    Returns
    -------
    M         : np.ndarray [shape=(n_mels, 1 + n_fft/2)]
        Mel transform matrix

    """

    if fmax is None:
        fmax = float(sr) / 2

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)))

    # Center freqs of each FFT bin
    fftfreqs = _fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    freqs = _mel_frequencies(n_mels + 2,
                            fmin=fmin,
                            fmax=fmax,
                            htk=htk)

    # Slaney-style mel is scaled to be approx constant energy per channel
    enorm = 2.0 / (freqs[2:n_mels + 2] - freqs[:n_mels])

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = (fftfreqs - freqs[i]) / (freqs[i + 1] - freqs[i])
        upper = (freqs[i + 2] - fftfreqs) / (freqs[i + 2] - freqs[i + 1])

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper)) * enorm[i]

    return weights


def dct(n_filters, n_input):
    """Discrete cosine transform (DCT type-III) basis.

    .. [1] http://en.wikipedia.org/wiki/Discrete_cosine_transform

    Parameters
    ----------
    n_filters : int > 0 [scalar]
        number of output components (DCT filters)

    n_input : int > 0 [scalar]
        number of input components (frequency bins)

    Returns
    -------
    dct_basis: np.ndarray [shape=(n_filters, n_input)]
        DCT (type-III) basis vectors [1]_

    """

    basis = np.empty((n_filters, n_input))
    basis[0, :] = 1.0 / np.sqrt(n_input)

    samples = np.arange(1, 2 * n_input, 2) * np.pi / (2.0 * n_input)

    for i in range(1, n_filters):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / n_input)

    return basis


# ===========================================================================
# Spectrogram
# ===========================================================================
def spectrogram(y, n_fft=2048, hop_length=None, win_length=None, power=1,
    window=None, center=True):
    '''Helper function to retrieve a magnitude spectrogram.

    This is primarily used in feature extraction functions that can operate on
    either audio time-series or spectrogram input.


    Parameters
    ----------
    y : None or np.ndarray [ndim=1]
        If provided, an audio time series

    n_fft : int > 0
        STFT window size

    win_length  : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()`.
        The window will be of length `win_length` and then padded
        with zeros to match `n_fft`.

        If unspecified, defaults to ``win_length = n_fft``.

    hop_length : int > 0
        STFT hop length

    power : float > 0
        Exponent for the magnitude spectrogram,
        e.g., 1 for energy, 2 for power, etc.

    Returns
    -------
    S_out : np.ndarray [dtype=np.float32]
        - If `S` is provided as input, then `S_out == S`
        - Else, `S_out = |stft(y, n_fft=n_fft, hop_length=hop_length)|**power`

    '''
    n_fft, hop_length, win_length = _validate_stft_arguments(
        n_fft, hop_length, win_length)

    # Otherwise, compute a magnitude spectrogram from input
    S = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                    center=center, window=window)) ** power
    return S


def ispectrogram(y, n_fft, hop_length=None, win_length=None, power=1,
    normalize=True, excit=None):
    ''' Attempt to go back from specgram-like powerspec to audio waveform
    by scaling specgram of white noise

    Parameters
    ----------
    y : None or np.ndarray [ndim=1]
        If provided, an audio time series

    n_fft : int > 0
        STFT window size

    win_length  : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()`.
        The window will be of length `win_length` and then padded
        with zeros to match `n_fft`.

        If unspecified, defaults to ``win_length = n_fft``.

    hop_length : int > 0
        STFT hop length

    power : float > 0
        Exponent for the magnitude spectrogram,
        e.g., 1 for energy, 2 for power, etc.

    normalize : bool
        it is suggested to normalize the reconstructed signal or you
        won't hear anything

    '''
    # default values:
    # sr = 8000Hz
    # wintime = 25ms (200 samps)
    # steptime = 10ms (80 samps)
    # which means use 256 point fft
    # hamming window
    #
    # excit is input excitation; white noise is used if not specified

    #  for sr = 8000
    # NFFT = 256;
    # NOVERLAP = 120;
    # SAMPRATE = 8000;
    # WINDOW = hamming(200);
    n_fft, hop_length, win_length = _validate_stft_arguments(
        n_fft, hop_length, win_length)

    nrow, ncol = y.shape

    r = None
    if excit is not None:
        r = excit

    # Values coming out of rasta treat samples as integers,
    # not range -1..1, hence scale up here to match (approx)
    xlen = win_length + hop_length * (ncol - 1)

    # generate random noise, then rescale its spectrogram
    if r is None:
        r = np.random.randn(xlen)
    R = stft(r, n_fft, hop_length=hop_length, win_length=win_length)
    # match shape of new and old spectrogram
    if R.shape[1] < y.shape[1]:
        y = y[:, :R.shape[1]]
    elif R.shape[1] > y.shape[1]:
        R = R[:, :y.shape[1]]
    R = R * np.power(y, 1 / power)
    y_ = istft(R, hop_length=hop_length, win_length=win_length)
    if normalize:
        y_ = (y_ - np.mean(y_)) / np.std(y_)
    return y_


def melspectrogram(y=None, S=None, sr=22050,
                   n_fft=2048, hop_length=None, win_length=None,
                   n_mels=128, fmin=0.0, fmax=None,
                   **kwargs):
    """Compute a Mel-scaled power spectrogram.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        audio time-series

    sr : number > 0 [scalar]
        sampling rate of `y`

    S : np.ndarray [shape=(d, t)]
        power spectrogram

    n_fft : int > 0 [scalar]
        length of the FFT window

    hop_length : int > 0 [scalar]
        number of samples between successive frames.
        See `librosa.core.stft`

    kwargs : additional keyword arguments
      Mel filter bank parameters.
      See `librosa.filters.mel` for details.

    Returns
    -------
    S : np.ndarray [shape=(n_mels, t)]
        Mel power spectrogram

    """

    if S is None:
        S = spectrogram(y=y, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, power=2, **kwargs)
    else:
        n_fft = 2 * (S.shape[0] - 1)

    # Build a Mel filter
    mel_basis = mel_filter(sr, n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    return np.dot(mel_basis, S)


# -- Mel spectrogram and MFCCs -- #
def mfcc(y=None, S=None, sr=22050, n_mfcc=20, **kwargs):
    """Mel-frequency cepstral coefficients

    Parameters
    ----------
    y     : np.ndarray [shape=(n,)] or None
        audio time series

    S : np.ndarray [shape=(d, t)]
        power spectrogram

    sr    : number > 0 [scalar]
        sampling rate of `y`

    n_mfcc: int > 0 [scalar]
        number of MFCCs to return

    kwargs : additional keyword arguments
        Arguments to `melspectrogram`, if operating
        on time series input

    Returns
    -------
    M     : np.ndarray [shape=(n_mfcc, t)]
        MFCC sequence

    """
    if S is None:
        S = melspectrogram(y=y, sr=sr, **kwargs)
    S = logamplitude(S)
    return np.dot(dct(n_mfcc, S.shape[0]), S)


# ===========================================================================
# Pitch tracking
# ===========================================================================
def piptrack(S, sr=22050, fmin=150.0, fmax=4000.0, threshold=0.1):
    '''Pitch tracking on thresholded parabolically-interpolated STFT

    .. [1] https://ccrma.stanford.edu/~jos/sasp/Sinusoidal_Peak_Interpolation.html

    Parameters
    ----------
    S: np.ndarray [shape=(d, t)] or None
        magnitude or power spectrogram

    sr : number > 0 [scalar]
        audio sampling rate of `y`

    threshold : float in `(0, 1)`
        A bin in spectrum X is considered a pitch when it is greater than
        `threshold*X.max()`

    fmin : float > 0 [scalar]
        lower frequency cutoff.

    fmax : float > 0 [scalar]
        upper frequency cutoff.

    .. note::
        One of `S` or `y` must be provided.

        If `S` is not given, it is computed from `y` using
        the default parameters of `librosa.core.stft`.

    Returns
    -------
    pitches : np.ndarray [shape=(d, t)]
    magnitudes : np.ndarray [shape=(d,t)]
        Where `d` is the subset of FFT bins within `fmin` and `fmax`.

        `pitches[f, t]` contains instantaneous frequency at bin
        `f`, time `t`

        `magnitudes[f, t]` contains the corresponding magnitudes.

        Both `pitches` and `magnitudes` take value 0 at bins
        of non-maximal magnitude.

    Examples
    --------
    >>> y, sr = read('example_audio_file')
    >>> s = spectrogram(y=s, n_fft=1024, hop_length=512, win_length=1024, power=2)
    >>> pitches, magnitudes = librosa.piptrack(s, sr=sr)


    '''

    # Check that we received an audio time series or STFT
    n_fft = 2 * (S.shape[0] - 1)

    # Make sure we're dealing with magnitudes
    S = np.abs(S)

    # Truncate to feasible region
    fmin = np.maximum(fmin, 0)
    fmax = np.minimum(fmax, float(sr) / 2)

    fft_freqs = _fft_frequencies(sr=sr, n_fft=n_fft)

    # Do the parabolic interpolation everywhere,
    # then figure out where the peaks are
    # then restrict to the feasible range (fmin:fmax)
    avg = 0.5 * (S[2:] - S[:-2])

    shift = 2 * S[1:-1] - S[2:] - S[:-2]

    # Suppress divide-by-zeros.
    # Points where shift == 0 will never be selected by localmax anyway
    shift = avg / (shift + (np.abs(shift) < SMALL_FLOAT))

    # Pad back up to the same shape as S
    avg = np.pad(avg, ([1, 1], [0, 0]), mode='constant')
    shift = np.pad(shift, ([1, 1], [0, 0]), mode='constant')

    dskew = 0.5 * avg * shift

    # Pre-allocate output
    pitches = np.zeros_like(S)
    mags = np.zeros_like(S)

    # Clip to the viable frequency range
    freq_mask = ((fmin <= fft_freqs) & (fft_freqs < fmax)).reshape((-1, 1))

    # Compute the column-wise local max of S after thresholding
    # Find the argmax coordinates
    idx = np.argwhere(freq_mask &
                      pp.localmax(S * (S > (threshold * S.max(axis=0)))))

    # Store pitch and magnitude
    pitches[idx[:, 0], idx[:, 1]] = ((idx[:, 0] + shift[idx[:, 0], idx[:, 1]]) * float(sr) / n_fft)

    mags[idx[:, 0], idx[:, 1]] = (S[idx[:, 0], idx[:, 1]] + dskew[idx[:, 0], idx[:, 1]])

    return pitches, mags
