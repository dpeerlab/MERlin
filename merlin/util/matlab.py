from typing import Tuple

import numpy as np

"""
This module contains Matlab functions that do not have equivalents in
python libraries.
"""


def matlab_gauss2D(shape: Tuple[int, int] = (3, 3), sigma: float = 0.5) -> np.array:
    """2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma]).
    """
    m, n = ((ss - 1.0) / 2.0 for ss in shape)
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
