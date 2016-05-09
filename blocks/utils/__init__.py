from __future__ import print_function
import os
import sys
import time
import contextlib
from collections import OrderedDict, deque

import shutil
from six.moves.urllib.request import urlopen
from six.moves.urllib.error import URLError, HTTPError
import tarfile

import numpy
import six
import theano
from theano import tensor
from theano import printing
from theano.gof.graph import Constant
from theano.tensor.shared_randomstreams import RandomStateSharedVariable
from theano.tensor.sharedvar import SharedVariable


def pack(arg):
    """Pack variables into a list.

    Parameters
    ----------
    arg : object
        Either a list or tuple, or any other Python object. Lists will be
        returned as is, and tuples will be cast to lists. Any other
        variable will be returned in a singleton list.

    Returns
    -------
    list
        List containing the arguments

    """
    if isinstance(arg, (list, tuple)):
        return list(arg)
    else:
        return [arg]


def unpack(arg, singleton=False):
    """Unpack variables from a list or tuple.

    Parameters
    ----------
    arg : object
        Either a list or tuple, or any other Python object. If passed a
        list or tuple of length one, the only element of that list will
        be returned. If passed a tuple of length greater than one, it
        will be cast to a list before returning. Any other variable
        will be returned as is.
    singleton : bool
        If ``True``, `arg` is expected to be a singleton (a list or tuple
        with exactly one element) and an exception is raised if this is not
        the case. ``False`` by default.

    Returns
    -------
    object
        A list of length greater than one, or any other Python object
        except tuple.

    """
    if isinstance(arg, (list, tuple)):
        if len(arg) == 1:
            return arg[0]
        else:
            if singleton:
                raise ValueError("Expected a singleton, got {}".
                                 format(arg))
            return list(arg)
    else:
        return arg


def shared_floatx_zeros_matching(shared_variable, name=None, **kwargs):
    r"""Create another shared variable with matching shape and broadcast.

    Parameters
    ----------
    shared_variable : :class:'tensor.TensorSharedVariable'
        A Theano shared variable with the desired shape and broadcastable
        flags.
    name : :obj:`str`, optional
        The name for the shared variable. Defaults to `None`.
    \*\*kwargs
        Keyword arguments to pass to the :func:`shared_floatx_zeros`
        function.

    Returns
    -------
    :class:'tensor.TensorSharedVariable'
        A new shared variable, initialized to all zeros, with the same
        shape and broadcastable flags as `shared_variable`.


    """
    if not is_shared_variable(shared_variable):
        raise ValueError('argument must be a shared variable')
    return shared_floatx_zeros(shared_variable.get_value().shape,
                               name=name,
                               broadcastable=shared_variable.broadcastable,
                               **kwargs)


def shared_floatx_zeros(shape, **kwargs):
    r"""Creates a shared variable array filled with zeros.

    Parameters
    ----------
    shape : tuple
        A tuple of integers representing the shape of the array.
    \*\*kwargs
        Keyword arguments to pass to the :func:`shared_floatx` function.

    Returns
    -------
    :class:'tensor.TensorSharedVariable'
        A Theano shared variable filled with zeros.

    """
    return shared_floatx(numpy.zeros(shape), **kwargs)


def shared_floatx_nans(shape, **kwargs):
    r"""Creates a shared variable array filled with nans.

    Parameters
    ----------
    shape : tuple
         A tuple of integers representing the shape of the array.
    \*\*kwargs
        Keyword arguments to pass to the :func:`shared_floatx` function.

    Returns
    -------
    :class:'tensor.TensorSharedVariable'
        A Theano shared variable filled with nans.

    """
    return shared_floatx(numpy.nan * numpy.zeros(shape), **kwargs)


def shared_floatx(value, name=None, borrow=False, dtype=None, **kwargs):
    r"""Transform a value into a shared variable of type floatX.

    Parameters
    ----------
    value : :class:`~numpy.ndarray`
        The value to associate with the Theano shared.
    name : :obj:`str`, optional
        The name for the shared variable. Defaults to `None`.
    borrow : :obj:`bool`, optional
        If set to True, the given `value` will not be copied if possible.
        This can save memory and speed. Defaults to False.
    dtype : :obj:`str`, optional
        The `dtype` of the shared variable. Default value is
        :attr:`config.floatX`.
    \*\*kwargs
        Keyword arguments to pass to the :func:`~theano.shared` function.

    Returns
    -------
    :class:`tensor.TensorSharedVariable`
        A Theano shared variable with the requested value and `dtype`.

    """
    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared(theano._asarray(value, dtype=dtype),
                         name=name, borrow=borrow, **kwargs)


def shared_like(variable, name=None, **kwargs):
    r"""Construct a shared variable to hold the value of a tensor variable.

    Parameters
    ----------
    variable : :class:`~tensor.TensorVariable`
        The variable whose dtype and ndim will be used to construct
        the new shared variable.
    name : :obj:`str` or :obj:`None`
        The name of the shared variable. If None, the name is determined
        based on variable's name.
    \*\*kwargs
        Keyword arguments to pass to the :func:`~theano.shared` function.

    """
    variable = tensor.as_tensor_variable(variable)
    if name is None:
        name = "shared_{}".format(variable.name)
    return theano.shared(numpy.zeros((0,) * variable.ndim,
                                     dtype=variable.dtype),
                         name=name, **kwargs)


def reraise_as(new_exc):
    """Reraise an exception as a different type or with a message.

    This function ensures that the original traceback is kept, making for
    easier debugging.

    Parameters
    ----------
    new_exc : :class:`Exception` or :obj:`str`
        The new error to be raised e.g. (ValueError("New message"))
        or a string that will be prepended to the original exception
        message

    Notes
    -----
    Note that when reraising exceptions, the arguments of the original
    exception are cast to strings and appended to the error message. If
    you want to retain the original exception arguments, please use:

    >>> try:
    ...     1 / 0
    ... except Exception as e:
    ...     reraise_as(Exception("Extra information", *e.args))
    Traceback (most recent call last):
      ...
    Exception: 'Extra information, ...

    Examples
    --------
    >>> class NewException(Exception):
    ...     def __init__(self, message):
    ...         super(NewException, self).__init__(message)
    >>> try:
    ...     do_something_crazy()
    ... except Exception:
    ...     reraise_as(NewException("Informative message"))
    Traceback (most recent call last):
      ...
    NewException: Informative message ...

    """
    orig_exc_type, orig_exc_value, orig_exc_traceback = sys.exc_info()

    if isinstance(new_exc, six.string_types):
        new_exc = orig_exc_type(new_exc)

    if hasattr(new_exc, 'args'):
        if len(new_exc.args) > 0:
            # We add all the arguments to the message, to make sure that this
            # information isn't lost if this exception is reraised again
            new_message = ', '.join(str(arg) for arg in new_exc.args)
        else:
            new_message = ""
        new_message += '\n\nOriginal exception:\n\t' + orig_exc_type.__name__
        if hasattr(orig_exc_value, 'args') and len(orig_exc_value.args) > 0:
            if getattr(orig_exc_value, 'reraised', False):
                new_message += ': ' + str(orig_exc_value.args[0])
            else:
                new_message += ': ' + ', '.join(str(arg)
                                                for arg in orig_exc_value.args)
        new_exc.args = (new_message,) + new_exc.args[1:]

    new_exc.__cause__ = orig_exc_value
    new_exc.reraised = True
    six.reraise(type(new_exc), new_exc, orig_exc_traceback)


def check_theano_variable(variable, n_dim, dtype_prefix):
    """Check number of dimensions and dtype of a Theano variable.

    If the input is not a Theano variable, it is converted to one. `None`
    input is handled as a special case: no checks are done.

    Parameters
    ----------
    variable : :class:`~tensor.TensorVariable` or convertible to one
        A variable to check.
    n_dim : int
        Expected number of dimensions or None. If None, no check is
        performed.
    dtype : str
        Expected dtype prefix or None. If None, no check is performed.

    """
    if variable is None:
        return

    if not isinstance(variable, tensor.Variable):
        variable = tensor.as_tensor_variable(variable)

    if n_dim and variable.ndim != n_dim:
        raise ValueError("Wrong number of dimensions:"
                         "\n\texpected {}, got {}".format(
                             n_dim, variable.ndim))

    if dtype_prefix and not variable.dtype.startswith(dtype_prefix):
        raise ValueError("Wrong dtype prefix:"
                         "\n\texpected starting with {}, got {}".format(
                             dtype_prefix, variable.dtype))


def is_graph_input(variable):
    """Check if variable is a user-provided graph input.

    To be considered an input the variable must have no owner, and not
    be a constant or shared variable.

    Parameters
    ----------
    variable : :class:`~tensor.TensorVariable`

    Returns
    -------
    bool
        ``True`` If the variable is a user-provided input to the graph.

    """
    return (not variable.owner and
            not isinstance(variable, SharedVariable) and
            not isinstance(variable, Constant))


def is_shared_variable(variable):
    """Check if a variable is a Theano shared variable.

    Notes
    -----
    This function excludes shared variables that store the state of Theano
    random number generators.

    """
    return (isinstance(variable, SharedVariable) and
            not isinstance(variable, RandomStateSharedVariable) and
            not hasattr(variable.tag, 'is_rng'))


def dict_subset(dict_, keys, pop=False, must_have=True):
    """Return a subset of a dictionary corresponding to a set of keys.

    Parameters
    ----------
    dict_ : dict
        The dictionary.
    keys : iterable
        The keys of interest.
    pop : bool
        If ``True``, the pairs corresponding to the keys of interest are
        popped from the dictionary.
    must_have : bool
        If ``True``, a ValueError will be raised when trying to retrieve a
        key not present in the dictionary.

    Returns
    -------
    result : ``OrderedDict``
        An ordered dictionary of retrieved pairs. The order is the same as
        in the ``keys`` argument.

    """
    not_found = object()

    def extract(k):
        if pop:
            if must_have:
                return dict_.pop(k)
            return dict_.pop(k, not_found)
        if must_have:
            return dict_[k]
        return dict_.get(k, not_found)

    result = [(key, extract(key)) for key in keys]
    return OrderedDict([(k, v) for k, v in result if v is not not_found])


def dict_union(*dicts, **kwargs):
    r"""Return union of a sequence of disjoint dictionaries.

    Parameters
    ----------
    dicts : dicts
        A set of dictionaries with no keys in common. If the first
        dictionary in the sequence is an instance of `OrderedDict`, the
        result will be OrderedDict.
    \*\*kwargs
        Keywords and values to add to the resulting dictionary.

    Raises
    ------
    ValueError
        If a key appears twice in the dictionaries or keyword arguments.

    """
    dicts = list(dicts)
    if dicts and isinstance(dicts[0], OrderedDict):
        result = OrderedDict()
    else:
        result = {}
    for d in list(dicts) + [kwargs]:
        duplicate_keys = set(result.keys()) & set(d.keys())
        if duplicate_keys:
            raise ValueError("The following keys have duplicate entries: {}"
                             .format(", ".join(str(key) for key in
                                               duplicate_keys)))
        result.update(d)
    return result


def repr_attrs(instance, *attrs):
    r"""Prints a representation of an object with certain attributes.

    Parameters
    ----------
    instance : object
        The object of which to print the string representation
    \*attrs
        Names of attributes that should be printed.

    Examples
    --------
    >>> class A(object):
    ...     def __init__(self, value):
    ...         self.value = value
    >>> a = A('a_value')
    >>> repr(a)  # doctest: +SKIP
    <blocks.utils.A object at 0x7fb2b4741a10>
    >>> repr_attrs(a, 'value')  # doctest: +SKIP
    <blocks.utils.A object at 0x7fb2b4741a10: value=a_value>

    """
    orig_repr_template = ("<{0.__class__.__module__}.{0.__class__.__name__} "
                          "object at {1:#x}")
    if attrs:
        repr_template = (orig_repr_template + ": " +
                         ", ".join(["{0}={{0.{0}}}".format(attr)
                                    for attr in attrs]))
    repr_template += '>'
    orig_repr_template += '>'
    try:
        return repr_template.format(instance, id(instance))
    except Exception:
        return orig_repr_template.format(instance, id(instance))


def put_hook(variable, hook_fn, *args):
    r"""Put a hook on a Theano variables.

    Ensures that the hook function is executed every time when the value
    of the Theano variable is available.

    Parameters
    ----------
    variable : :class:`~tensor.TensorVariable`
        The variable to put a hook on.
    hook_fn : function
        The hook function. Should take a single argument: the variable's
        value.
    \*args : list
        Positional arguments to pass to the hook function.

    """
    return printing.Print(global_fn=lambda _, x: hook_fn(x, *args))(variable)


def ipdb_breakpoint(x):
    """A simple hook function for :func:`put_hook` that runs ipdb.

    Parameters
    ----------
    x : :class:`~numpy.ndarray`
        The value of the hooked variable.

    """
    import ipdb
    ipdb.set_trace()


def print_sum(x, header=None):
    if not header:
        header = 'print_sum'
    print(header + ':', x.sum())


def print_shape(x, header=None):
    if not header:
        header = 'print_shape'
    print(header + ':', x.shape)


@contextlib.contextmanager
def change_recursion_limit(limit):
    """Temporarily changes the recursion limit."""
    old_limit = sys.getrecursionlimit()
    if old_limit < limit:
        sys.setrecursionlimit(limit)
    yield
    sys.setrecursionlimit(old_limit)


def extract_args(expected, *args, **kwargs):
    r"""Route keyword and positional arguments to a list of names.

    A frequent situation is that a method of the class gets to
    know its positional arguments only when an instance of the class
    has been created. In such cases the signature of such method has to
    be `*args, **kwargs`. The downside of such signatures is that the
    validity of a call is not checked.

    Use :func:`extract_args` if your method knows at runtime, but not
    at evaluation/compile time, what arguments it actually expects,
    in order to check that they are correctly received.

    Parameters
    ----------
    expected : list of str
        A list of strings denoting names for the expected arguments,
        in order.
    args : iterable
        Positional arguments that have been passed.
    kwargs : Mapping
        Keyword arguments that have been passed.

    Returns
    -------
    routed_args : OrderedDict
        An OrderedDict mapping the names in `expected` to values drawn
        from either `args` or `kwargs` in the usual Python fashion.

    Raises
    ------
    KeyError
        If a keyword argument is passed, the key for which is not
        contained within `expected`.
    TypeError
        If an expected argument is accounted for in both the positional
        and keyword arguments.
    ValueError
        If certain arguments in `expected` are not assigned a value
        by either a positional or keyword argument.

    """
    # Use of zip() rather than equizip() intentional here. We want
    # to truncate to the length of args.
    routed_args = dict(zip(expected, args))
    for name in kwargs:
        if name not in expected:
            raise KeyError('invalid input name: {}'.format(name))
        elif name in routed_args:
            raise TypeError("got multiple values for "
                            "argument '{}'".format(name))
        else:
            routed_args[name] = kwargs[name]
    if set(expected) != set(routed_args):
        raise ValueError('missing values for inputs: {}'.format(
                         [name for name in expected
                          if name not in routed_args]))
    return OrderedDict((key, routed_args[key]) for key in expected)


def find_bricks(top_bricks, predicate):
    """Walk the brick hierarchy, return bricks that satisfy a predicate.

    Parameters
    ----------
    top_bricks : list
        A list of root bricks to search downward from.
    predicate : callable
        A callable that returns `True` for bricks that meet the
        desired criteria or `False` for those that don't.

    Returns
    -------
    found : list
        A list of all bricks that are descendants of any element of
        `top_bricks` that satisfy `predicate`.

    """
    found = []
    visited = set()
    to_visit = deque(top_bricks)
    while len(to_visit) > 0:
        current = to_visit.popleft()
        if current not in visited:
            visited.add(current)
            if predicate(current):
                found.append(current)
            to_visit.extend(current.children)
    return found


# ===========================================================================
# Python
# ===========================================================================
class struct(object):

    '''Flexible object can be assigned any attribtues'''
    pass


class queue(object):

    """ FIFO, fast, NO thread-safe queue
    put : append to end of list
    append : append to end of list
    pop : remove data from end of list
    get : remove data from end of list
    empty : check if queue is empty
    clear : remove all data in queue
    """

    def __init__(self):
        super(queue, self).__init__()
        self._data = []
        self._idx = 0

    def put(self, value):
        self._data.append(value)

    def append(self, value):
        self._data.append(value)

    def pop(self):
        if self._idx == len(self._data):
            raise ValueError('Queue is empty')
        self._idx += 1
        return self._data[self._idx - 1]

    def get(self):
        if self._idx == len(self._data):
            raise ValueError('Queue is empty')
        self._idx += 1
        return self._data[self._idx - 1]

    def empty(self):
        if self._idx == len(self._data):
            return True
        return False

    def clear(self):
        del self._data
        self._data = []
        self._idx = 0

    def __len__(self):
        return len(self._data) - self._idx


class Progbar(object):

    '''
    This function is adpated from: https://github.com/fchollet/keras
    Original work Copyright (c) 2014-2015 keras contributors
    Modified work Copyright 2016-2017 TrungNT
    '''

    def __init__(self, target, title=''):
        '''
            @param target: total number of steps expected
        '''
        self.width = 39
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.title = title

    def update(self, current, values=[]):
        '''
            @param current: index of current step
            @param values: list of tuples (name, value_for_last_step).
            The progress bar will display averages for these values.
        '''
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        self.seen_so_far = current

        now = time.time()

        prev_total_width = self.total_width
        sys.stdout.write("\b" * prev_total_width)
        sys.stdout.write("\r")

        numdigits = int(numpy.floor(numpy.log10(self.target))) + 1
        barstr = '%s %%%dd/%%%dd [' % (self.title, numdigits, numdigits)
        bar = barstr % (current, self.target)
        prog = float(current) / self.target
        prog_width = int(self.width * prog)
        if prog_width > 0:
            bar += ('=' * (prog_width - 1))
            if current < self.target:
                bar += '>'
            else:
                bar += '='
        bar += ('.' * (self.width - prog_width))
        bar += ']'
        sys.stdout.write(bar)
        self.total_width = len(bar)

        if current:
            time_per_unit = (now - self.start) / current
        else:
            time_per_unit = 0
        eta = time_per_unit * (self.target - current)
        info = ''
        if current < self.target:
            info += ' - ETA: %ds' % eta
        else:
            info += ' - %ds' % (now - self.start)
        for k in self.unique_values:
            info += ' - %s:' % k
            if type(self.sum_values[k]) is list:
                avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                if abs(avg) > 1e-3:
                    info += ' %.4f' % avg
                else:
                    info += ' %.4e' % avg
            else:
                info += ' %s' % self.sum_values[k]

        self.total_width += len(info)
        if prev_total_width > self.total_width:
            info += ((prev_total_width - self.total_width) * " ")

        sys.stdout.write(info)
        sys.stdout.flush()

        if current >= self.target:
            sys.stdout.write("\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far + n, values)


# Under Python 2, 'urlretrieve' relies on FancyURLopener from legacy
# urllib module, known to have issues with proxy management
if sys.version_info[0] == 2:
    def urlretrieve(url, filename, reporthook=None, data=None):
        '''
        This function is adpated from: https://github.com/fchollet/keras
        Original work Copyright (c) 2014-2015 keras contributors
        '''
        def chunk_read(response, chunk_size=8192, reporthook=None):
            total_size = response.info().get('Content-Length').strip()
            total_size = int(total_size)
            count = 0
            while 1:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                count += 1
                if reporthook:
                    reporthook(count, chunk_size, total_size)
                yield chunk

        response = urlopen(url, data)
        with open(filename, 'wb') as fd:
            for chunk in chunk_read(response, reporthook=reporthook):
                fd.write(chunk)
else:
    from six.moves.urllib.request import urlretrieve


def get_file(fname, origin, untar=False):
    '''
    This function is adpated from: https://github.com/fchollet/keras
    Original work Copyright (c) 2014-2015 keras contributors
    Modified work Copyright 2016-2017 TrungNT
    '''
    datadir_base = os.path.expanduser(os.path.join('~', '.blocks'))
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.blocks')
    datadir = os.path.join(datadir_base, 'datasets')
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    if not os.path.exists(fpath):
        print('Downloading data from', origin)
        global progbar
        progbar = None

        def dl_progress(count, block_size, total_size):
            global progbar
            if progbar is None:
                progbar = Progbar(total_size)
            else:
                progbar.update(count * block_size)

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath, dl_progress)
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
        progbar = None

    if untar:
        if not os.path.exists(untar_fpath):
            print('Untaring file...')
            tfile = tarfile.open(fpath, 'r:gz')
            try:
                tfile.extractall(path=datadir)
            except (Exception, KeyboardInterrupt) as e:
                if os.path.exists(untar_fpath):
                    if os.path.isfile(untar_fpath):
                        os.remove(untar_fpath)
                    else:
                        shutil.rmtree(untar_fpath)
                raise
            tfile.close()
        return untar_fpath

    return fpath


def get_all_files(path, filter_func=None):
    ''' Recurrsively get all files in the given path '''
    file_list = []
    q = queue()
    # init queue
    if os.access(path, os.R_OK):
        for p in os.listdir(path):
            q.put(os.path.join(path, p))
    # process
    while not q.empty():
        p = q.pop()
        if os.path.isdir(p):
            if os.access(p, os.R_OK):
                for i in os.listdir(p):
                    q.put(os.path.join(p, i))
        else:
            if filter_func is not None and not filter_func(p):
                continue
            # remove dump files of Mac
            if '.DS_STORE' in p or '._' == os.path.basename(p)[:2]:
                continue
            file_list.append(p)
    return file_list


def get_module_from_path(identifier, prefix='', suffix='', path='.', exclude='',
                  prefer_compiled=False):
    ''' Algorithms:
     - Search all files in the `path` matched `prefix` and `suffix`
     - Exclude all files contain any str in `exclude`
     - Sorted all files based on alphabet
     - Load all modules based on `prefer_compiled`
     - return list of identifier found in all modules

    Parameters
    ----------
    identifier : str
        identifier of object, function or anything in script files
    prefix : str
        prefix of file to search in the `path`
    suffix : str
        suffix of file to search in the `path`
    path : str
        searching path of script files
    exclude : str, list(str)
        any files contain str in this list will be excluded
    prefer_compiled : bool
        True mean prefer .pyc file, otherwise prefer .py

    Returns
    -------
    list(object, function, ..) :
        any thing match given identifier in all found script file

    Notes
    -----
    File with multiple . character my procedure wrong results
    If the script run this this function match the searching process, a
    infinite loop may happen!
    '''
    import re
    import imp
    from inspect import getmembers
    # ====== validate input ====== #
    if exclude == '': exclude = []
    if type(exclude) not in (list, tuple, numpy.ndarray):
        exclude = [exclude]
    prefer_flag = -1
    if prefer_compiled: prefer_flag = 1
    # ====== create pattern and load files ====== #
    pattern = re.compile('^%s.*%s\.pyc?' % (prefix, suffix)) # py or pyc
    files = os.listdir(path)
    files = [f for f in files
             if pattern.match(f) and
             sum([i in f for i in exclude]) == 0]
    # ====== remove duplicated pyc files ====== #
    files = sorted(files, key=lambda x: prefer_flag * len(x)) # pyc is longer
    # .pyc go first get overrided by .py
    files = sorted({f.split('.')[0]: f for f in files}.values())
    # ====== load all modules ====== #
    modules = []
    for f in files:
        try:
            if '.pyc' in f:
                modules.append(
                    imp.load_compiled(f.split('.')[0],
                                      os.path.join(path, f))
                )
            else:
                modules.append(
                    imp.load_source(f.split('.')[0],
                                    os.path.join(path, f))
                )
        except:
            pass
    # ====== Find all identifier in modules ====== #
    ids = []
    for m in modules:
        for i in getmembers(m):
            if identifier in i:
                ids.append(i[1])
    # remove duplicate py
    return ids


def is_path(path):
    if isinstance(path, str):
        path = os.path.abspath(path)
        if os.path.exists(path):
            #the file is there
            return True
        elif os.access(os.path.dirname(path), os.W_OK):
            #the file does not exists but write privileges are given
            return True
    return False


# ===========================================================================
# Misc
# ===========================================================================
def save_wav(path, s, fs):
    from scipy.io import wavfile
    wavfile.write(path, fs, s)


def play_audio(data, fs, volumn=1, speed=1):
    ''' Play audio from numpy array.

    Parameters
    ----------
    data : numpy.ndarray
            signal data
    fs : int
            sample rate
    volumn: float
            between 0 and 1
    speed: float
            > 1 mean faster, < 1 mean slower
    '''
    import soundfile as sf
    import os

    data = numpy.asarray(data)
    if data.ndim == 1:
        channels = 1
    else:
        channels = data.shape[1]
    with sf.SoundFile('/tmp/tmp_play.wav', 'w', fs, channels,
                   subtype=None, endian=None, format=None, closefd=None) as f:
        f.write(data)
    os.system('afplay -v %f -q 1 -r %f /tmp/tmp_play.wav &' % (volumn, speed))
    raw_input('<enter> to stop audio.')
    os.system("kill -9 `ps aux | grep -v 'grep' | grep afplay | awk '{print $2}'`")
