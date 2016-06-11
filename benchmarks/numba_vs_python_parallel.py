from __future__ import print_function, division

import numpy as np

from blocks import utils
from blocks import fuel
import numba

print(numba.__version__)


def bubblesort(X):
    N = len(X)
    for end in range(N, 1, -1):
        for i in range(end - 1):
            cur = X[i]
            if cur > X[i + 1]:
                tmp = X[i]
                X[i] = X[i + 1]
                X[i + 1] = tmp

original = np.arange(0.0, 20.0, 0.01, dtype='f4')
shuffled = original.copy()
np.random.shuffle(shuffled)

# ====== python ====== #
test = shuffled.copy()
with utils.UnitTimer(10):
    for i in range(10):
        bubblesort(test)
print(np.array_equal(test, original))

# ====== Numba ====== #
bubblesort_jit = numba.jit("void(f4[:])", nopython=False)(bubblesort)
test = shuffled.copy()
with utils.UnitTimer(10):
    for i in range(10):
        bubblesort_jit(test)
print(np.array_equal(test, original))

# ====== Test in parallel ====== #
np.random.shuffle(test)
mr = fuel.features.MapReduce(8)
mr.set_cache(1)
mr.add([test.copy() for i in range(24)],
       lambda x: bubblesort(x), name='Python')
mr.add([test.copy() for i in range(24)],
       lambda x: bubblesort_jit(x), name='Numba ')
mr.run()
