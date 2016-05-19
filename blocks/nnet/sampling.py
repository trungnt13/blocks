"""
The :method:`pool_output_length`, :class:`Pool1D` and `Pool2D` and all its
inheritance implementation contains code from
`Lasagne <https://github.com/Lasagne/Lasagne>`_, which is covered
by the following license:

Copyright (c) 2014-2015 Lasagne contributors
All rights reserved.

LICENSE: https://github.com/Lasagne/Lasagne/blob/master/LICENSE
"""
from __future__ import division, absolute_import


from .base import NNOps, NNConfig
from blocks import backend as K
from blocks.utils import as_tuple
from blocks.utils.decorators import autoinit

