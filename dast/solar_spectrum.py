
import os

import numpy as np

from . import WVL_MIN, WVL_MAX, SOLAR_SPECTRUM_FN


def get_data_filename(fn):
    THIS_DIR = os.path.dirname(__file__)
    return os.path.join(THIS_DIR, 'data', fn)


def read():

    spc_wvl, spc_Eonl = np.loadtxt(
        get_data_filename(SOLAR_SPECTRUM_FN),
        usecols=(0, 1), unpack=True)

    universe = ((spc_wvl >= WVL_MIN) & (spc_wvl <= WVL_MAX))
    wvl = spc_wvl[universe]  # in nm
    Eonl = spc_Eonl[universe]  # in W/m2/nm
    Eon = np.trapz(Eonl, wvl)  # in W/m2
    Eon_full = np.trapz(spc_Eonl, spc_wvl)
    return wvl, Eonl, Eon, Eon_full
