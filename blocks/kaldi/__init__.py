from __future__ import division, absolute_import

import os
import re
import subprocess
from six.moves import range, zip

import numpy as np

from blocks.utils import get_all_files, struct
from blocks import fuel

from .features import *
from .helpers import *
from .io import *

# ===========================================================================
# check KALDI_PATH
# ===========================================================================
if 'KALDI_PATH' not in os.environ:
    raise Exception('You must specify KALDI_PATH variable.')
KALDI_PATH = os.environ['KALDI_PATH']


_dirs = os.listdir(KALDI_PATH)
if 'src' not in _dirs or 'tools' not in _dirs or 'egs' not in _dirs:
    raise Exception('src, tools and egs must included in the KALDI_PATH')

# NOTE: we assume all binary are built
# add binary to PATH (copy from common path)
os.environ['PATH'] = (
    os.path.join(KALDI_PATH, 'src', 'bin') + ':' +
    os.path.join(KALDI_PATH, 'src', 'chainbin') + ':' +
    os.path.join(KALDI_PATH, 'src', 'featbin') + ':' +
    os.path.join(KALDI_PATH, 'src', 'fgmmbin') + ':' +
    os.path.join(KALDI_PATH, 'src', 'fstbin') + ':' +
    os.path.join(KALDI_PATH, 'src', 'gmmbin') + ':' +
    os.path.join(KALDI_PATH, 'src', 'ivectorbin') + ':' +
    os.path.join(KALDI_PATH, 'src', 'kwsbin') + ':' +
    os.path.join(KALDI_PATH, 'src', 'latbin') + ':' +
    os.path.join(KALDI_PATH, 'src', 'lmbin') + ':' +
    os.path.join(KALDI_PATH, 'src', 'nnet2bin') + ':' +
    os.path.join(KALDI_PATH, 'src', 'nnet3bin') + ':' +
    os.path.join(KALDI_PATH, 'src', 'nnetbin') + ':' +
    os.path.join(KALDI_PATH, 'src', 'online2bin') + ':' +
    os.path.join(KALDI_PATH, 'src', 'onlinebin') + ':' +
    os.path.join(KALDI_PATH, 'src', 'sgmm2bin') + ':' +
    os.path.join(KALDI_PATH, 'src', 'sgmmbin') + ':' +
    os.environ['PATH']
)
# ====== searching for addition tools ====== #
if not os.path.exists(os.path.join(KALDI_PATH, 'egs', 'wsj', 's5', 'utils')) or\
    not os.path.exists(os.path.join(KALDI_PATH, 'egs', 'wsj', 's5', 'utils')):
    raise Exception('Cannot find egs/wsf/s5/utils and steps in KALDI_PATH')


# ===========================================================================
# Helper
# ===========================================================================
def ali_to_phones(input_dir, output_dir, model=None, frame_shift=0.01,
                  segments=None, phones=None):
    """ ali-to-phones 1.mdl ark:1.ali ark:phones.tra

    Parameters
    ----------
    input_dir : str
        path to input dir which contains all ali.%d.gz files
    output_dir : str
        path to output dir
    model : None
        by default, looking for final.mdl in input_dir, and output_dir
    frame-shift : float, default = 0.01
        frame shift used to control the times of the ctm output
    segments : str
        path to segments file (mapping utterences => files, start, end)
    phones : str
        path to phones dictionary

    Return
    ------
    aligment.ctm : alignment for utterance with relatie times
    aligment_final.ctm : alignment for files with file times

    Note
    ----
    We search for all: "ali.%d.gz" file in input_dir for alignment
    """
    # ====== check ====== #
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if model is None:
        model = os.path.join(input_dir, 'final.mdl')
        if not os.path.exists(model):
            model = os.path.join(output_dir, 'final.mdl')
        if not os.path.exists(model):
            raise Exception('Cannot find final.mdl in both input_dir and output_dir')
    if not os.path.exists(model):
        raise Exception('Model at path: %s is not exist' % model)
    # ====== arguments ====== #
    arguments = ' '
    arguments += '--ctm-output=true '
    arguments += '--frame-shift=%0.02f' % frame_shift + ' '
    arguments += '--per-frame=false '
    arguments += '--write-lengths=false '
    arguments += '--print-args=false '
    # ====== get all ali.i.gz files ====== #
    output_file = os.path.join(output_dir, 'alignment.ctm')
    input_files = get_all_files(input_dir)
    alignment = re.compile('ali\.\d+\.gz')
    alignment_in = sorted([i for i in input_files
                           if alignment.match(os.path.basename(i)) is not None])
    alignment_out = [os.path.join(output_dir,
                                  os.path.basename(i).replace('.gz', '.ctm'))
                     for i in alignment_in]
    _file = open(output_file, 'w')
    for i, j in zip(alignment_in, alignment_out):
        ret = subprocess.call('ali-to-phones' + arguments + model + ' ' +
                        '"ark:gunzip -c %s|"' % i + ' ' +
                        '-> %s' % j, shell=True)
        _file.write(open(j, 'r').read())
        _file.flush()
        os.remove(j)
        if ret != 0:
            raise Exception('Error! Stop all ali-to-phones jobs')
    _file.close()
    # The CTM output reports start and end times relative to the utterance,
    # as opposed to the file. We will need the segments file located in data/train
    # to convert the utterance times into file times.
    if phones is not None and os.path.exists(phones):
        phones_map = {} # mapping from ID to phonemes
        for i in open(phones, 'r'):
            p, ID = i.replace('\n', '').split(' ')
            phones_map[ID] = p
    else:
        phones_map = struct()
    # The output also reports the phone ID, as opposed to the phone itself.
    # You will need the phones.txt file located in data/lang to convert the
    # phone IDs into phone symbols.
    if segments is not None and os.path.exists(segments):
        print('Converting utterances times to file times ...')
        utterances_map = {}
        output_final = open(os.path.join(output_dir, 'alignment_final.ctm'), 'w')
        for f in open(segments, 'r'):
            utt, file, start, end = f.replace('\n', '').split(' ')
            utterances_map[utt] = (file, float(start), float(end))
        for line in open(output_file, 'r'):
            utt, channel, s, duration, phoneme = line.replace('\n', '').split(' ')
            file, start, end = utterances_map[utt]
            start += float(s)
            end = start + float(duration)
            start = '%0.04f' % start; end = '%0.04f' % end
            output_final.write(file + ' ' + channel + ' ' +
                               start + ' ' + end + ' ' +
                               phones_map[phoneme] + '\n')


def ali_to_pdf(input_dir, output_dir, model=None, segments=None):
    # ====== check ====== #
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if model is None:
        model = os.path.join(input_dir, 'final.mdl')
        if not os.path.exists(model):
            model = os.path.join(output_dir, 'final.mdl')
        if not os.path.exists(model):
            raise Exception('Cannot find final.mdl in both input_dir and output_dir')
    if not os.path.exists(model):
        raise Exception('Model at path: %s is not exist' % model)
    # ====== arguments ====== #
    arguments = ' '
    arguments += '--print-args=false '
    # ====== get all ali.i.gz files ====== #
    output_file = os.path.join(output_dir, 'alignment.ctm')
    input_files = get_all_files(input_dir)
    alignment = re.compile('ali\.\d+\.gz')
    alignment_in = sorted([i for i in input_files
                           if alignment.match(os.path.basename(i)) is not None])
    alignment_out = [os.path.join(output_dir,
                                  os.path.basename(i).replace('.gz', '.txt'))
                     for i in alignment_in]
    _file = open(output_file, 'w')
    for i, j in zip(alignment_in, alignment_out):
        ret = subprocess.call('ali-to-pdf ' + arguments + ' ' + model + ' ' +
                        '"ark:gunzip -c %s|"' % i + ' ' +
                        'ark,t:%s' % j, shell=True)
        _file.write(open(j, 'r').read())
        _file.flush()
        os.remove(j)
        if ret != 0:
            raise Exception('Error! Stop all ali-to-phones jobs')
    _file.close()
    # ====== convert to files ====== #
    if segments is not None and os.path.exists(segments):
        print('Converting utterances to files ...')
        utterances_map = {}
        output_final = open(os.path.join(output_dir, 'alignment_final.ctm'), 'w')
        for f in open(segments, 'r'):
            utt, file, start, end = f.replace('\n', '').split(' ')
            utterances_map[utt] = (file, start, end)
        for line in open(output_file, 'r'):
            line = line.replace('\n', '').split(' ')
            file, start, end = utterances_map[line[0]]
            output_final.write(file + ' ' +
                               start + ' ' + end + ' ' +
                               ' '.join(line[1:]) + '\n')
