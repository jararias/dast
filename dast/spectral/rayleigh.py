
# Author: Jose A Ruiz-Arias, jararias at uma.es

import numpy as np

from .. import airmass
from ._base import validate_frame, validate_series
from ._base import SpectralTransmittance

# References:
# Gueymard (2019) The SMARTS spectral irradiance model after 25 years: New
#   developments and validation of reference spectra.
#   doi: 10.1016/j.solener.2019.05.048


class RayleighTransmittance(SpectralTransmittance):

    def frame(self, pressure, sza):
        """
        Spectral rayleigh (scattering) transmittance

        Inputs:
        -------
        pressure: array-like, 0d or 1d
          Atmospheric pressure, hPa
        sza: array-like, 0d or 1d
          Solar zenith angle

        Output:
        -------
        Output's dimensions are (time, atmosphere, spectrum), i.e.,
        the output's shape is (len(sza), len(pressure), len(wvl))
        """
        am_ = np.array(airmass(sza, 'rayleigh'), ndmin=1)[:, None, None]
        pp0 = np.array(pressure, ndmin=1)[None, :, None] / 1013.25
        wvl_um = self.wvl[None, None, :] / 1000
        validate_frame(am_, pp0, wvl_um)

        # Eq (3) in Gueymard (2019). There is a typo in the equation.
        # It should be the reciprocal of the parenthesis
        taur_l = pp0 / (117.3405 * wvl_um**4 - 1.5107 * wvl_um**2 +
                        0.017535 - 8.7743E-4 / wvl_um**2)
        return np.clip(np.exp(-am_ * taur_l), 0., 1.)

    def series(self, pressure, sza):
        """
        Spectral rayleigh (scattering) transmittance

        Inputs:
        -------
        pressure: array-like, 0d or 1d
          Atmospheric pressure, hPa
        sza: array-like, 0d or 1d
          Solar zenith angle, with same shape as pressure

        Output:
        -------
        Output's dimensions are (time, spectrum), i.e., the output's
        shape is (len(sza), len(wvl))
        """
        am_ = np.array(airmass(sza, 'rayleigh'), ndmin=1)[:, None]
        pp0 = np.array(pressure, ndmin=1)[:, None] / 1013.25
        validate_series(am_, pp0)

        # Eq (3) in Gueymard (2019). There is a typo in the equation.
        # It should be the reciprocal of the parenthesis
        wvl_um = self.wvl[None, :] / 1000
        taur_l = pp0 / (117.3405 * wvl_um**4 - 1.5107 * wvl_um**2 +
                        0.017535 - 8.7743E-4 / wvl_um**2)
        return np.clip(np.exp(-am_ * taur_l), 0., 1.)
