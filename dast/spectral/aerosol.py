
# Author: Jose A Ruiz-Arias, jararias at uma.es

import numpy as np

from .. import airmass
from ._base import validate_frame, validate_series
from ._base import SpectralTransmittance


class AerosolTransmittance(SpectralTransmittance):

    def frame(self, beta, alpha, sza):
        """
        Spectral aerosol (extinction) transmittance

        Inputs:
        -------
        beta: array-like, 0d or 1d
          Angstrom's turbidity, i.e., aerosol optical depth at 1 micron
        alpha: array-like, 0d or 1d.
          Angstrom's exponent, with same size as beta
        sza: array-like, 0d or 1d
          Solar zenith angle

        Output:
        -------
        Output's dimensions are (time, atmosphere, spectrum), i.e.,
        the output's shape is (len(sza), len(beta), len(wvl))
        """
        assert np.array(beta).shape == np.array(alpha).shape
        am_ = np.array(airmass(sza, 'aerosol'), ndmin=1)[:, None, None]
        b = np.array(beta, ndmin=1)[None, :, None]
        a = np.array(alpha, ndmin=1)[None, :, None]
        wvl_um = self.wvl[None, None, :] / 1000.
        validate_frame(am_, b, a, wvl_um)

        taua_l = (b / (wvl_um**a))  # Angstrom's law
        return np.clip(np.exp(-am_ * taua_l), 0., 1.)

    def series(self, beta, alpha, sza):
        """
        Spectral aerosol (extinction) transmittance

        Inputs:
        -------
        beta: array-like, 0d or 1d
          Angstrom's turbidity, i.e., aerosol optical depth at 1 micron
        alpha: array-like, 0d or 1d.
          Angstrom's exponent, with same size as beta
        sza: array-like, 0d or 1d
          Aerosol optical airmass, with same size as beta and alpha

        Output:
        -------
        Output's dimensions are (time, spectrum), i.e., the output's
        shape is (len(sza), len(wvl))
        """
        am_ = np.array(airmass(sza, 'aerosol'), ndmin=1)[:, None]
        b = np.array(beta, ndmin=1)[:, None]
        a = np.array(alpha, ndmin=1)[:, None]
        validate_series(am_, b, a)

        wvl_um = self.wvl[None, :] / 1000.
        taua_l = (b / (wvl_um**a))  # Angstrom's law
        return np.clip(np.exp(-am_ * taua_l), 0., 1.)
