
# Author: Jose A Ruiz-Arias, jararias at uma.es

import os

import numpy as np

from .. import solar_spectrum

# References:
# Gueymard (2018) Revised composite extraterrestrial spectrum based on recent
#   solar irradiance observations. doi: 10.1016/j.solener.2018.04.067


def validate_frame(*args):
    # assert all args have 3 dimensions
    assert np.all(np.array([arg.ndim for arg in args]) == 3)


def validate_series(*args):
    # assert all args have 2 dimensions
    ndims = [arg.ndim for arg in args]
    assert np.all(np.array(ndims) == 2), f'dims mismatch: {ndims}'
    # assert all args have the same shape
    shapes = [arg.shape for arg in args]
    assert len(set(shapes)) == 1, f'shape mismatch: {shapes}'


def get_data_filename(fn):
    THIS_DIR = os.path.dirname(__file__)
    return os.path.join(THIS_DIR, '..', 'data', fn)


class SpectralTransmittance:
    def __init__(self):
        self.wvl, self.Eonl, self.Eon, self.full_Eon = solar_spectrum.read()
