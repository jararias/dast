
# Author: Jose A Ruiz-Arias, jararias at uma.es

import numpy as np

from .. import airmass
from ._base import validate_frame, validate_series
from ._base import get_data_filename
from ._base import SpectralTransmittance

# References:
# SMARTS v2.9.5 code
# Chehade et al. (2013) Revised temperature-dependent ozone absorption
#   cross-section spectra (Bogumil et al.) measured with the SCHIAMACHY
#   satellite spectrometer.
#   doi: 10.5194/amt-6-3055-2013


NLOSCHMIDT = 2.6867811e19  # cm-3, number of particles in a volume of ideal gas


class OzoneTransmittance(SpectralTransmittance):

    def __init__(self, ozone_temp=223, abs_source='sciamachy'):
        super().__init__()

        assert abs_source in ('sciamachy', 'smarts')

        if abs_source == 'sciamachy':  # Chehade et al (2019)
            OZONE_TEMPS = (203, 223, 243, 273, 293)
            assert ozone_temp in OZONE_TEMPS, \
                'ozone temperature must be one of 203, 223, 243, 273 or 293'

            o3xs_column = dict(zip(OZONE_TEMPS, range(1, 6)))[ozone_temp]
            abs_wvl_uv, o3xs_uv = np.loadtxt(
                get_data_filename('SCIA_O3_Temp_cross-section_V4.1.DAT'),
                usecols=(0, o3xs_column), skiprows=20, unpack=True)

        else:  # abs_source == 'smarts'
            abs_wvl_uv, o3xs_uv = np.loadtxt(
                get_data_filename('Abs_O3UV.dat'),
                usecols=(0, 1), skiprows=1, unpack=True)

        abs_wvl_ir, abo3_ir = np.loadtxt(
            get_data_filename('Abs_O3IR.dat'),
            usecols=(0, 1), skiprows=1, unpack=True)

        o3xs_uv = np.interp(self.wvl, abs_wvl_uv, o3xs_uv, left=0., right=0.)
        abo3_ir = np.interp(self.wvl, abs_wvl_ir, abo3_ir, left=0., right=0.)
        self.abo3 = NLOSCHMIDT*o3xs_uv + abo3_ir

    def frame(self, uo, sza):
        """
        Spectral ozone (absorption) transmittance

        Inputs:
        -------
        uo: array-like, 0d or 1d
          Total-column ozone content, atm-cm (atm-cm = 1e-3 Dobson units)
        sza: array-like, 0d or 1d
          Solar zenith angle

        Output:
        -------
        Output's dimensions are (time, atmosphere, spectrum), i.e.,
        the output's shape is (len(sza), len(uo), len(wvl))
        """
        abo3 = self.abo3[None, None, :]
        am_ = np.array(airmass(sza, 'o3'), ndmin=1)[:, None, None]
        uo_ = np.array(uo, ndmin=1)[None, :, None]
        validate_frame(am_, uo_, abo3)

        tauo_l = abo3 * uo_
        return np.clip(np.exp(-am_ * tauo_l), 0., 1.)

    def series(self, uo, sza):
        """
        Spectral ozone (absorption) transmittance

        Inputs:
        -------
        uo: array-like, 0d or 1d
          Total-column ozone content, atm-cm (atm-cm = 1e-3 Dobson units)
        sza: array-like, 0d or 1d
          Solar zenith angle, with same size as uo

        Output:
        -------
        Output's dimensions are (time, spectrum), i.e., the output's
        shape is (len(sza), len(wvl))
        """
        am_ = np.array(airmass(sza, 'o3'), ndmin=1)[:, None]
        uo_ = np.array(uo, ndmin=1)[:, None]
        validate_series(am_, uo_)

        tauo_l = self.abo3[None, :] * uo_
        return np.clip(np.exp(-am_ * tauo_l), 0., 1.)
