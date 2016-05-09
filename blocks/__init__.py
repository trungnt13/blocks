"""The blocks library for parametrized Theano ops."""


# ===========================================================================
# Auto config
# ===========================================================================
def auto_config():
    import os
    import sys
    import re
    ODIN_FLAGS = os.getenv("ODIN", "")

    # ====== specific pattern ====== #
    valid_device_name = re.compile('(cuda|gpu)\d+')
    valid_cnmem_name = re.compile('(cnmem)[=]?[10]?\.\d*')
    valid_seed = re.compile('seed\D?(\d*)')

    floatX = 'float32'
    backend = 'theano'
    epsilon = 10e-8
    device = []
    cnmem = 0.
    seed = 1208251813

    s = ODIN_FLAGS.split(',')
    for i in s:
        i = i.lower().strip()
        # ====== Data type ====== #
        if 'float' in i or 'int' in i:
            floatX = i
        # ====== Backend ====== #
        elif 'theano' in i:
            backend = 'theano'
        elif 'tensorflow' in i or 'tf' in i:
            backend = 'tensorflow'
        # ====== Devices ====== #
        elif 'cpu' == i and len(device) == 0:
            device = 'cpu'
        elif 'gpu' in i or 'cuda' in i:
            if isinstance(device, str):
                raise ValueError('Device already setted to cpu')
            if i == 'gpu': i = 'gpu0'
            elif i == 'cuda': i = 'cuda0'
            if valid_device_name.match(i) is None:
                raise ValueError('Unsupport device name: %s '
                                 '(must be "cuda"|"gpu" and optional number)'
                                 ', e.g: cuda0, gpu0, cuda, gpu, ...' % i)
            device.append(i.replace('gpu', 'cuda'))
        # ====== cnmem ====== #
        elif 'cnmem' in i:
            match = valid_cnmem_name.match(i)
            if match is None:
                raise ValueError('Unsupport CNMEM format: %s. '
                                 'Valid format must be: cnmem=0.75 or cnmem=.75 '
                                 ' or cnmem.75' % str(i))

            i = i[match.start():match.end()].replace('cnmem', '').replace('=', '')
            cnmem = float(i)
        # ====== seed ====== #
        elif 'seed' in i:
            match = valid_seed.match(i)
            if match is None:
                raise ValueError('Invalid pattern for specifying seed value, '
                                 'you can try: [seed][non-digit][digits]')
            seed = int(match.group(1))

    # if DEVICE still len = 0, use cpu
    if len(device) == 0:
        device = 'cpu'
    # adject epsilon
    if floatX == 'float16':
        epsilon = 10e-5
    elif floatX == 'float32':
        epsilon = 10e-8
    elif floatX == 'float64':
        epsilon = 10e-12

    sys.stderr.write('[Auto-Config] Device : %s\n' % device)
    sys.stderr.write('[Auto-Config] Backend: %s\n' % backend)
    sys.stderr.write('[Auto-Config] FloatX : %s\n' % floatX)
    sys.stderr.write('[Auto-Config] Epsilon: %s\n' % epsilon)
    sys.stderr.write('[Auto-Config] CNMEM  : %s\n' % cnmem)
    sys.stderr.write('[Auto-Config] SEED  : %s\n' % seed)

    # ==================== create theano flags ==================== #
    if backend == 'theano':
        if device == 'cpu':
            contexts = ""
            device = "device=%s" % device
        elif len(device) == 1: # optimize for 1 gpu
            contexts = ""
            device = 'device=gpu'
        else:
            contexts = "contexts="
            contexts += ';'.join(['dev%d->cuda%d' % (i, int(_.replace('cuda', '')))
                                 for i, _ in enumerate(device)]) + ','
            # TODO: bizarre degradation in performance if not specify device=gpu
            device = 'device=gpu'
        flags = contexts + device + ",mode=FAST_RUN,floatX=%s" % floatX
        # ====== others ====== #
        flags += ',exception_verbosity=high'
        # Speedup CuDNNv4
        flags += ',dnn.conv.algo_fwd=time_once,dnn.conv.algo_bwd_filter=time_once,dnn.conv.algo_bwd_data=time_once'
        # CNMEM
        if cnmem > 0. and cnmem <= 1.:
            flags += ',lib.cnmem=%.2f,allow_gc=True' % cnmem
        os.environ['THEANO_FLAGS'] = flags
    elif backend == 'tensorflow':
        pass
    else:
        raise ValueError('Unsupport backend: ' + backend)

    class AttributeDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
    config = AttributeDict()
    config.update({'floatX': floatX, 'epsilon': epsilon,
                   'backend': backend, 'seed': seed})
    return config

# ===========================================================================
# Keras config
# ===========================================================================
autoconfig = auto_config()
import numpy
RNG_GENERATOR = numpy.random.RandomState(autoconfig.seed)

import blocks.version
__version__ = blocks.version.version
