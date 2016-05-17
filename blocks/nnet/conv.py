from __future__ import division, absolute_import

import numpy as np

from .base import NNOps, NNConfig


class Conv2D(NNOps):

    # ==================== abstract method ==================== #
    def _initialize(self, *args, **kwargs):
        """ This function return NNConfig for given configuration from arg
        and kwargs
        """
        raise NotImplementedError

    def _apply(self, *args, **kwargs):
        raise NotImplementedError
