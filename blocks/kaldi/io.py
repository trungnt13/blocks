from __future__ import division, absolute_import

import os
import struct
from collections import OrderedDict

import numpy as np


def read_matrix(ark, scp=None):
    """ This method does NOT support compressed Kaldi ark file """
    # ====== check input arguments ====== #
    if not os.path.isfile(ark):
        raise Exception('Cannot file ark file at path: ' + str(ark))
    ark = os.path.abspath(ark)
    if scp is None:
        scp = ark.replace('.ark', '.scp')
    if not os.path.isfile(scp):
        raise Exception('Cannot find scp file at path: ' + str(scp))
    scp = np.genfromtxt(scp, dtype='str', delimiter=' ')
    # ====== start processing ====== #
    data = OrderedDict()
    with open(ark, 'rb') as f:
        for name, index in scp:
            path, offset = index.split(':'); offset = int(offset)
            f.seek(offset, 0)
            data_format = f.read(2)
            data_type = f.read(3)
            # ====== read row and col ====== #
            f.read(1) # remove space
            nrows = struct.unpack('<i', f.read(4))[0]
            f.read(1) # remove space
            ncols = struct.unpack('<i', f.read(4))[0]
            # ====== select data_type ====== #
            if data_type == 'FM ':
                size = 4 # bytes
            elif data_type == 'DM ':
                size = 8 # bytes
            else:
                raise ValueError('Not support data type %s' % data_type)
            dtype = 'float32' if size == 4 else 'float64'
            # ====== read text or binary ====== #
            matrix = None
            if data_format == '\0B': # Binary format
                matrix = np.frombuffer(f.read(nrows * ncols * size), dtype=dtype)
                matrix = np.reshape(matrix, (nrows, ncols))
            else: # ascii format
                rows = []
                while 1:
                    line = f.readline()
                    if len(line) == 0:
                        raise Exception # eof, should not happen!
                    if len(line.strip()) == 0:
                        continue # skip empty line
                    arr = line.strip().split()
                    if arr[-1] != ']':
                        rows.append(np.array(arr, dtype=dtype)) # not last line
                    else:
                        rows.append(np.array(arr[:-1], dtype=dtype)) # last line
                        break
                matrix = np.vstack(rows)
            data[name] = matrix
    return data
