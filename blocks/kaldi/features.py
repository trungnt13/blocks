from __future__ import division, absolute_import

import os
import subprocess
from multiprocessing import cpu_count
from collections import defaultdict
from shutil import copyfile, copytree
from StringIO import StringIO
from contextlib import contextmanager

import numpy as np

from blocks.utils import (get_all_files, struct, as_tuple, exec_commands,
                          TemporaryDirectory)
from blocks.preprocessing import speech
from blocks import fuel


# ===========================================================================
# Searching for additional tools
# ===========================================================================
KALDI_PATH = os.environ['KALDI_PATH']
_tools_dir = os.listdir(os.path.join(KALDI_PATH, 'tools'))

_sph2pipe = None
for i in _tools_dir:
    if 'sph2pipe' in i and \
    os.path.isdir(os.path.join(os.path.join(KALDI_PATH, 'tools'), i)):
        _sph2pipe = os.path.abspath(os.path.join(KALDI_PATH, 'tools', i + '/' + 'sph2pipe'))


@contextmanager
def _remove_compression(file):
    # this function remove compression from make_mfcc then put it back
    # to default without any notice
    f = open(file, 'r').read()
    f = f.replace('compress=true', 'compress=false')
    _ = open(file, 'w'); _.write(f); _.close()

    yield None

    f = open(file, 'r').read()
    f = f.replace('compress=false', 'compress=true')
    _ = open(file, 'w'); _.write(f); _.close()


# ===========================================================================
# Main methods
# ===========================================================================
def sph2wav(input, output, channel=0):
    """ THis command can execute batch of sph files
    Example
    -------
    >>> from blocks import kaldi
    >>> kaldi.sph2wav(
    >>>     input=['sw02001.sph', 'sw02001.sph'],
    >>>     output=['sw02001-A.wav', 'sw02001-B.wav'],
    >>>     channel=[1, 2])

    """
    if not isinstance(input, (list, tuple)):
        input = [input]
    if not isinstance(output, (list, tuple)):
        output = [output]
    if len(input) != len(output):
        raise Exception('number of input != output.')
    channel = map(lambda x: '-c %d' % x if x == 1 or x == 2 else '',
                  as_tuple(channel, len(input), int))
    commands = []
    for i, o, c in zip(input, output, channel):
        commands.append(_sph2pipe + ' -f wav -p ' + c + ' ' + i + ' ' + o)
    exec_commands(commands)


def make_mfcc(file_lists, outpath, segments=None, utt2spk="",
    sample_frequency=16000, frame_length=25, frame_shift=10,
    window_type="povey", num_ceps=13, num_mel_bins=40,
    preemphasis_coefficient=0.97, high_freq=-200, low_freq=20,
    channel=-1, dither=1, raw_energy=True,
    remove_dc_offset=True, min_duration=0,
    snip_edges=True, subtract_mean=False, use_energy=True,
    cepstral_lifter=22, energy_floor=0,
    vtln_high=-500, vtln_low=100, vtln_map="", vtln_warp=1):
    """

    Parameters
    ----------
    file_lists : path
        a csv files for [name-path], for example:
        sw02001-A /Users/abc/tmp/sw02001-A.wav
        sw02001-B /Users/abc/tmp/sw02001-B.wav
        or path to data folder, created by kaldi
    segments : path
        segmetns file: ID wavfilename start_time(in secs) end_time(in secs) channel-id(0 or 1)
        sw02001-A_000098-001156 sw02001-A 0.0 -1
        sw02001-A_001980-002131 sw02001-A 0.0 -1
    utt2spk : string, default = ""
        Utterance to speaker_id map rspecifier (if doing VTLN and you have warps per speaker)
    sample_frequency : float, default = 16000
        Waveform data sample frequency (must match the waveform file, if specified there)
    frame_length : float, default = 25
        Frame length in milliseconds
    frame_shift : float, default = 10
        Frame shift in milliseconds
    window_type : string, default = "povey"
        Type of window ("hamming"|"hanning"|"povey"|"rectangular")
    num_ceps : int, default = 13
        Number of cepstra in MFCC computation (including C0)
    num_mel_bins : int, default = 23
        Number of triangular mel_frequency bins
    preemphasis_coefficient : float, default = 0.97
        Coefficient for use in signal preemphasis
    high_freq : float, default = -200
        High cutoff frequency for mel bins (if < 0, offset from Nyquist)
    low_freq : float, default = 20
        Low cutoff frequency for mel bins
    channel : int, default = -1
        Channel to extract (-1 -> expect mono, 0 -> left, 1 -> right)
    dither : float, default = 1
        Dithering constant (0.0 means no dither)
    raw_energy : bool, default = true
        If true, compute energy before preemphasis and windowing
    remove_dc_offset : bool, default = true
        Subtract mean from waveform on each frame
    min_duration : float, default = 0
        Minimum duration of segments to process (in seconds).
    snip_edges : bool, default = true
        If true, end effects will be handled by outputting only frames that
        completely fit in the file, and the number of frames depends on the
        frame_length.  If false, the number of frames depends only on the
        frame_shift, and we reflect the data at the ends.
    subtract_mean : bool, default = false
        Subtract mean of each feature file [CMS]; not recommended to do it this way.
    use_energy : bool, default = true
        Use energy (not C0) in MFCC computation
    cepstral_lifter : float, default = 22
        Constant that controls scaling of MFCCs
    energy_floor : float, default = 0
        Floor on energy (absolute, not relative) in MFCC computation
    vtln_high : float, default = -500
        High inflection point in piecewise linear VTLN warping function
        (if negative, offset from high_mel_freq)
    vtln_low : float, default = 100
        Low inflection point in piecewise linear VTLN warping function
    vtln_map : string, default = ""
        Map from utterance or speaker_id to vtln warp factor (rspecifier)
    vtln_warp : float, default = 1
        Vtln warp factor (only applicable if vtln_map not specified)
    """
    # ====== arguments ====== #
    boolean = lambda x: 'true' if x else 'false'
    arguments = ""
    arguments += '\n'.join([
        '--debug-mel=false',
        '--output-format=kaldi',
        '--round-to-power-of-two=true',
        '--sample-frequency=%d ' % sample_frequency,
        '--frame-length=%d' % frame_length,
        '--frame-shift=%d' % frame_shift,
        '--window-type=%s' % window_type,
        '--num-ceps=%d' % num_ceps,
        '--num-mel-bins=%d' % num_mel_bins,
        '--preemphasis-coefficient=%0.2f' % preemphasis_coefficient,
        '--high-freq=%0.2f' % high_freq,
        '--low-freq=%0.2f' % low_freq,
        '--channel=%d' % channel,
        '--dither=%0.2f' % dither,
        '--raw-energy=' + boolean(raw_energy),
        '--remove-dc-offset=' + boolean(remove_dc_offset),
        '--min-duration=%0.2f' % min_duration,
        '--snip-edges=' + boolean(snip_edges),
        '--subtract-mean=' + boolean(subtract_mean),
        '--use-energy=' + boolean(use_energy),
        '--cepstral-lifter=%0.2f' % cepstral_lifter,
        '--energy-floor=%0.2f' % energy_floor,
        '--vtln-high=%0.2f' % vtln_high,
        '--vtln-low=%0.2f' % vtln_low,
        '--vtln-warp=%0.2f' % vtln_warp,
        '--utt2spk=%s' % utt2spk if len(utt2spk) > 0 else "",
        '--vtln-map=%s' % vtln_map if len(vtln_map) > 0 else "",
    ])
    # ====== check input_paths ====== #
    if os.path.isdir(file_lists) and all(i in os.listdir(file_lists)
                                        for i in ['segments', 'wav.scp', 'spk2utt']):
        copy_mode = True # copy all files to data dir
    else:
        # construct own data dir
        copy_mode = False
        n_channel = 1
        if os.path.isdir(file_lists):
            file_lists = np.asarray(
                [(os.path.basename(i)[:-4], i) for i in get_all_files(file_lists)
                 if '.sph' in i.lower() or '.wav' in i.lower()], dtype=str)
            # sample 1 file to check n_channel
            shape = speech.read(file_lists[0][1])[0].shape
            if len(shape) == 2 and shape[-1] > 1:
                n_channel = 2
        elif os.path.isfile(file_lists) and '.scp' in file_lists:
            file_lists = np.genfromtxt(file_lists, dtype='str', delimiter=' ')
        elif (not isinstance(file_lists, (tuple, list)) and
              not isinstance(file_lists[0], (tuple, list))):
            raise Exception('file_lists can be str(path), or list of files')
        # chekc channels and convert sph to wav
        if n_channel == 2:
            _ = []
            for name, file in file_lists:
                if '.wav' == file[-4:]:
                    _.append((name, file))
                else:
                    _.append((name + '-0', _sph2pipe + ' -f wav -p -c 1 ' + file + ' |'))
                    _.append((name + '-1', _sph2pipe + ' -f wav -p -c 2 ' + file + ' |'))
            file_lists = sorted(_)
        else:
            file_lists = sorted([(i, j) if '.wav' == j[-4:]
                                 else (i, _sph2pipe + ' -f wav -p ' + j + ' |')
                                 for i, j in file_lists])
        # segmetns file: ID wavfilename start_time(in secs) end_time(in secs) channel-id(0 or 1)
        if segments is None:
            segments = np.asarray([(i, i, 0, -1) for i, j in file_lists], dtype=str)
        elif os.path.isfile(segments):
            segments = np.genfromtxt(segments, dtype='str', delimiter=' ')
        elif isinstance(segments, str):
            segments = np.genfromtxt(StringIO(segments), dtype='str', delimiter=' ')
        else:
            raise Exception('segments must be path, str, or None')
        # check spk2utt
        if len(utt2spk) == 0 or utt2spk is None:
            _ = defaultdict(list)
            for name, file, start, end in segments:
                _[file].append(name)
            spk2utt = []
            utt2spk = []
            for i, j in _.iteritems():
                spk2utt.append([i] + j)
                for k in j:
                    utt2spk.append((k, i))
            spk2utt = sorted(spk2utt)
            utt2spk = sorted(utt2spk)
    # ====== preparstre kaldi file list ====== #
    with TemporaryDirectory(add_to_path=True) as tempdir:
        # ====== link scripts ====== #
        os.symlink(os.path.join(KALDI_PATH, 'egs', 'wsj', 's5', 'utils'),
                   os.path.join(tempdir, 'utils'))
        os.symlink(os.path.join(KALDI_PATH, 'egs', 'wsj', 's5', 'steps'),
                   os.path.join(tempdir, 'steps'))
        # ====== config ====== #
        config_path = 'mfcc.conf'
        open(config_path, 'w').write(arguments)
        if copy_mode:
            copytree(file_lists, 'data')
        else:
            os.mkdir('data')
            wav_path = 'data/wav.scp'
            np.savetxt(wav_path, file_lists, fmt='%s')
            seg_path = 'data/segments'
            np.savetxt(seg_path, segments, fmt='%s')
            spk2utt_path = 'data/spk2utt'
            np.savetxt(spk2utt_path, spk2utt, fmt='%s')
            utt2spk_path = 'data/utt2spk'
            np.savetxt(utt2spk_path, utt2spk, fmt='%s')
        # ====== deploying scripts ====== #
        copyfile(os.path.join('utils/', 'parse_options.sh'), 'parse_options.sh')
        # copyfile(os.path.join('utils/', 'run.pl'), 'run.pl')
        with _remove_compression('steps/make_mfcc.sh'):
            subprocess.call('steps/make_mfcc.sh ' +
                            '--nj %d ' % (cpu_count() * 3) +
                            '--cmd utils/run.pl ' +
                            '--mfcc-config %s ' % config_path +
                            ' data data ' + outpath, shell=True)


def make_fbank(file_lists, outpath, segments=None, utt2spk="",
    sample_frequency=16000, frame_length=25, frame_shift=10, window_type="hamming",
    num_mel_bins=40, use_log_fbank=True, preemphasis_coefficient=0.97,
    high_freq=-200, low_freq=20, channel=-1, dither=1, raw_energy=True,
remove_dc_offset=True, min_duration=0, snip_edges=True, subtract_mean=False,
    use_energy=True, energy_floor=0,
    vtln_high=-500, vtln_low=100, vtln_map="", vtln_warp=1):
    """

    Parameters
    ----------
    file_lists : path
        a csv files for [name-path], for example:
        sw02001-A /Users/abc/tmp/sw02001-A.wav
        sw02001-B /Users/abc/tmp/sw02001-B.wav
        or path to data folder, created by kaldi
    segments : path
        segmetns file: ID wavfilename start_time(in secs) end_time(in secs) channel-id(0 or 1)
        sw02001-A_000098-001156 sw02001-A 0.0 -1
        sw02001-A_001980-002131 sw02001-A 0.0 -1
    utt2spk : string, default = ""
        Utterance to speaker_id map rspecifier (if doing VTLN and you have warps per speaker)
    sample_frequency : float, default = 16000
        Waveform data sample frequency (must match the waveform file, if specified there)
    frame_length : float, default = 25
        Frame length in milliseconds
    frame_shift : float, default = 10
        Frame shift in milliseconds
    window_type : string, default = "hamming"
        Type of window ("hamming"|"hanning"|"povey"|"rectangular")
    num_mel_bins : int, default = 23
        Number of triangular mel_frequency bins
    use_log_fbank : bool, default = True
        If true, produce log-filterbank, else produce linear.
    preemphasis_coefficient : float, default = 0.97
        Coefficient for use in signal preemphasis
    high_freq : float, default = -200
        High cutoff frequency for mel bins (if < 0, offset from Nyquist)
    low_freq : float, default = 20
        Low cutoff frequency for mel bins
    channel : int, default = -1
        Channel to extract (-1 -> expect mono, 0 -> left, 1 -> right)
    dither : float, default = 1
        Dithering constant (0.0 means no dither)
    raw_energy : bool, default = true
        If true, compute energy before preemphasis and windowing
    remove_dc_offset : bool, default = true
        Subtract mean from waveform on each frame
    min_duration : float, default = 0
        Minimum duration of segments to process (in seconds).
    snip_edges : bool, default = true
        If true, end effects will be handled by outputting only frames that
        completely fit in the file, and the number of frames depends on the
        frame_length.  If false, the number of frames depends only on the
        frame_shift, and we reflect the data at the ends.
    subtract_mean : bool, default = false
        Subtract mean of each feature file [CMS]; not recommended to do it this way.
    use_energy : bool, default = true
        Use energy (not C0) in MFCC computation
    energy_floor : float, default = 0
        Floor on energy (absolute, not relative) in MFCC computation
    vtln_high : float, default = -500
        High inflection point in piecewise linear VTLN warping function
        (if negative, offset from high_mel_freq)
    vtln_low : float, default = 100
        Low inflection point in piecewise linear VTLN warping function
    vtln_map : string, default = ""
        Map from utterance or speaker_id to vtln warp factor (rspecifier)
    vtln_warp : float, default = 1
        Vtln warp factor (only applicable if vtln_map not specified)
    """
    # ====== arguments ====== #
    boolean = lambda x: 'true' if x else 'false'
    arguments = ""
    arguments += '\n'.join([
        '--debug-mel=false',
        '--output-format=kaldi',
        '--round-to-power-of-two=true',
        '--sample-frequency=%d ' % sample_frequency,
        '--frame-length=%d' % frame_length,
        '--frame-shift=%d' % frame_shift,
        '--window-type=%s' % window_type,
        '--num-mel-bins=%d' % num_mel_bins,
        '--use-log-fbank=%s' % boolean(use_log_fbank),
        '--preemphasis-coefficient=%0.2f' % preemphasis_coefficient,
        '--high-freq=%0.2f' % high_freq,
        '--low-freq=%0.2f' % low_freq,
        '--channel=%d' % channel,
        '--dither=%0.2f' % dither,
        '--raw-energy=' + boolean(raw_energy),
        '--remove-dc-offset=' + boolean(remove_dc_offset),
        '--min-duration=%0.2f' % min_duration,
        '--snip-edges=' + boolean(snip_edges),
        '--subtract-mean=' + boolean(subtract_mean),
        '--use-energy=' + boolean(use_energy),
        '--energy-floor=%0.2f' % energy_floor,
        '--vtln-high=%0.2f' % vtln_high,
        '--vtln-low=%0.2f' % vtln_low,
        '--vtln-warp=%0.2f' % vtln_warp,
        '--utt2spk=%s' % utt2spk if len(utt2spk) > 0 else "",
        '--vtln-map=%s' % vtln_map if len(vtln_map) > 0 else "",
    ])
    # ====== check input_paths ====== #
    if os.path.isdir(file_lists) and all(i in os.listdir(file_lists)
                                        for i in ['segments', 'wav.scp', 'spk2utt']):
        copy_mode = True # copy all files to data dir
    else:
        # construct own data dir
        copy_mode = False
        n_channel = 1
        if os.path.isdir(file_lists):
            file_lists = np.asarray(
                [(os.path.basename(i)[:-4], i) for i in get_all_files(file_lists)
                 if '.sph' in i.lower() or '.wav' in i.lower()], dtype=str)
            # sample 1 file to check n_channel
            shape = speech.read(file_lists[0][1])[0].shape
            if len(shape) == 2 and shape[-1] > 1:
                n_channel = 2
        elif os.path.isfile(file_lists) and '.scp' in file_lists:
            file_lists = np.genfromtxt(file_lists, dtype='str', delimiter=' ')
        elif (not isinstance(file_lists, (tuple, list)) and
              not isinstance(file_lists[0], (tuple, list))):
            raise Exception('file_lists can be str(path), or list of files')
        # chekc channels and convert sph to wav
        if n_channel == 2:
            _ = []
            for name, file in file_lists:
                if '.wav' == file[-4:]:
                    _.append((name, file))
                else:
                    _.append((name + '-0', _sph2pipe + ' -f wav -p -c 1 ' + file + ' |'))
                    _.append((name + '-1', _sph2pipe + ' -f wav -p -c 2 ' + file + ' |'))
            file_lists = sorted(_)
        else:
            file_lists = sorted([(i, j) if '.wav' == j[-4:]
                                 else (i, _sph2pipe + ' -f wav -p ' + j + ' |')
                                 for i, j in file_lists])
        # segmetns file: ID wavfilename start_time(in secs) end_time(in secs) channel-id(0 or 1)
        if segments is None:
            segments = np.asarray([(i, i, 0, -1) for i, j in file_lists], dtype=str)
        elif os.path.isfile(segments):
            segments = np.genfromtxt(segments, dtype='str', delimiter=' ')
        elif isinstance(segments, str):
            segments = np.genfromtxt(StringIO(segments), dtype='str', delimiter=' ')
        else:
            raise Exception('segments must be path, str, or None')
        # check spk2utt
        if len(utt2spk) == 0 or utt2spk is None:
            _ = defaultdict(list)
            for name, file, start, end in segments:
                _[file].append(name)
            spk2utt = []
            utt2spk = []
            for i, j in _.iteritems():
                spk2utt.append([i] + j)
                for k in j:
                    utt2spk.append((k, i))
            spk2utt = sorted(spk2utt)
            utt2spk = sorted(utt2spk)
    # ====== preparstre kaldi file list ====== #
    with TemporaryDirectory(add_to_path=True) as tempdir:
        # ====== link scripts ====== #
        os.symlink(os.path.join(KALDI_PATH, 'egs', 'wsj', 's5', 'utils'),
                   os.path.join(tempdir, 'utils'))
        os.symlink(os.path.join(KALDI_PATH, 'egs', 'wsj', 's5', 'steps'),
                   os.path.join(tempdir, 'steps'))
        # ====== config ====== #
        config_path = 'fbank.conf'
        open(config_path, 'w').write(arguments)
        if copy_mode:
            copytree(file_lists, 'data')
        else:
            os.mkdir('data')
            wav_path = 'data/wav.scp'
            np.savetxt(wav_path, file_lists, fmt='%s')
            seg_path = 'data/segments'
            np.savetxt(seg_path, segments, fmt='%s')
            spk2utt_path = 'data/spk2utt'
            np.savetxt(spk2utt_path, spk2utt, fmt='%s')
            utt2spk_path = 'data/utt2spk'
            np.savetxt(utt2spk_path, utt2spk, fmt='%s')
        # ====== deploying scripts ====== #
        copyfile(os.path.join('utils/', 'parse_options.sh'), 'parse_options.sh')
        with _remove_compression('steps/make_fbank.sh'):
            subprocess.call('steps/make_fbank.sh ' +
                            '--nj %d ' % (cpu_count() * 3) +
                            '--cmd utils/run.pl ' +
                            '--fbank-config %s ' % config_path +
                            ' data data ' + outpath, shell=True)
            # os.system('cp data/make_fbank_data.*.log /Users/trungnt13/tmp/fbank_log')
