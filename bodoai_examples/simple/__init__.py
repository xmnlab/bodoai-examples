# Source: https://github.com/Bodo-inc/Bodo-examples/blob/master/examples/pi.py
import bodo
import numpy as np


@bodo.jit
def calc_pi(n):
    x = 2 * np.random.ranf(n) - 1
    y = 2 * np.random.ranf(n) - 1
    pi = 4 * np.sum(x ** 2 + y ** 2 < 1) / n
    return pi
