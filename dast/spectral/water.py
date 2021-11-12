
# Author: Jose A Ruiz-Arias, jararias at uma.es

import warnings

import numpy as np
from scipy.interpolate import interp1d

from .. import airmass
from ._base import validate_frame, validate_series
from ._base import get_data_filename
from ._base import SpectralTransmittance

# References:
# SMARTS v2.9.5 code


NLOSCHMIDT = 2.6867811e19  # cm-3, number of particles in a volume of ideal gas


def warningfilter(action, category=RuntimeWarning):
    def warning_deco(func):
        def func_wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter(action, category)
                return func(*args, **kwargs)
        return func_wrapper
    return warning_deco


class WaterTransmittance(SpectralTransmittance):

    def __init__(self):

        super().__init__()

        column_names = (
            'wvl', 'abs',
            'iband',
            'ifitw', 'bwa0', 'bwa1', 'bwa2',
            'ifitm', 'bma0', 'bma1', 'bma2',
            'ifitmw', 'bmwa0', 'bmwa1', 'bmwa2',
            'bpa1', 'bpa2'
        )

        data = np.loadtxt(get_data_filename('Abs_H2O.dat'), skiprows=1)
        self.h2o = dict(zip(column_names, data.T))

    @warningfilter('ignore')
    def _calculate_Bw(self, pw):
        iband = self.h2o['iband']
        pw0 = np.full(iband.shape, 4.11467)
        pw0[iband == 2] = 2.92232
        pw0[iband == 3] = 1.41642
        pw0[iband == 4] = 0.41612
        pw0[iband == 5] = 0.05663
        pww0 = pw - pw0

        bwa0 = self.h2o['bwa0']
        bwa1 = self.h2o['bwa1']
        bwa2 = self.h2o['bwa2']
        Bw = (1. + bwa0 * pww0 + bwa1 * pww0**2)

        ifitw = self.h2o['ifitw']
        ifitw = ifitw * np.ones_like(Bw, dtype=ifitw.dtype)
        Bw[ifitw == 1] = (Bw / (1. + bwa2 * pww0))[ifitw == 1]
        Bw[ifitw == 2] = (Bw / (1. + bwa2 * (pww0**2)))[ifitw == 2]
        Bw[ifitw == 6] = (bwa0 + bwa1 * pww0)[ifitw == 6]

        h2oabs = self.h2o['abs']
        h2oabs = h2oabs * np.ones_like(Bw, dtype=h2oabs.dtype)
        Bw[h2oabs <= 0.] = 1.
        return np.clip(Bw, 0.05, 7.0)

    @warningfilter('ignore')
    def _calculate_Bm(self, am):
        am1 = am - 1.
        am12 = am1**2

        bma0 = self.h2o['bma0']
        bma1 = self.h2o['bma1']
        bma2 = self.h2o['bma2']
        # Bm = np.ones((bma1.shape[0], am.shape[1], 1))
        Bm = np.ones(tuple([1]*bma1.ndim))

        ifitm = self.h2o['ifitm']
        ifitm = ifitm * np.ones_like(Bm, dtype=ifitm.dtype)
        Bm = np.where(ifitm == 0, bma1*(am**bma2), Bm)
        Bmx = (1. + bma0*am1 + bma1*am12) / (1. + bma2*am1)
        Bm = np.where(ifitm == 1, Bmx, Bm)
        Bmx = (1. + bma0*am1 + bma1*am12) / (1. + bma2*am12)
        Bm = np.where(ifitm == 2, Bmx, Bm)
        Bmx = (1. + bma0*am1 + bma1*am12) / (1. + bma2*am1**.5)
        Bm = np.where(ifitm == 3, Bmx, Bm)
        Bmx = (1. + bma0*am1**.25) / (1. + bma2*am1**.1)
        Bm = np.where(ifitm == 5, Bmx, Bm)

        h2oabs = self.h2o['abs']
        h2oabs = h2oabs * np.ones_like(Bm, dtype=h2oabs.dtype)
        Bm[h2oabs <= 0.] = 1.
        return np.clip(Bm, 0.05, 7.0)

    @warningfilter('ignore')
    def _calculate_Bmw(self, pw, am):
        Bw = self._calculate_Bw(pw)
        Bm = self._calculate_Bm(am)
        Bmw = Bm * Bw

        ifitm = self.h2o['ifitm']
        ifitm = ifitm * np.ones_like(Bmw, dtype=ifitm.dtype)
        ifitmw = self.h2o['ifitmw']
        ifitmw = ifitmw * np.ones_like(Bmw, dtype=ifitmw.dtype)
        absh2o = self.h2o['abs']
        absh2o = absh2o * np.ones_like(Bmw, dtype=absh2o.dtype)
        Bw = Bw * np.ones(Bmw.shape)
        Bm = Bm * np.ones(Bmw.shape)

        cond1 = np.abs(Bw-1) < 1e-6
        cond2 = ((ifitm != 0) | (absh2o <= 0.)) & (np.abs(Bm - 1.) < 1e-6)
        cond3 = ((ifitm == 0) | (absh2o <= 0.)) & (Bm > 0.968) & (Bm < 1.0441)
        cond4 = (ifitmw == -1) | (absh2o <= 0.)
        cond = cond1 | cond2 | cond3 | cond4

        iband = self.h2o['iband']
        iband = iband * np.ones_like(Bmw, dtype=iband.dtype)
        w0 = 4.11467 * np.ones_like(Bmw, dtype='float')
        w0[iband == 2] = 2.92232
        w0[iband == 3] = 1.41642
        w0[iband == 4] = 0.41612
        w0[iband == 5] = 0.05663

        amw = am*(pw/w0)
        amw1 = amw - 1.
        amw12 = amw1**2
        bmwa0 = self.h2o['bmwa0']
        bmwa1 = self.h2o['bmwa1']
        bmwa2 = self.h2o['bmwa2']

        Bmwx = np.ones(Bmw.shape)
        universe = (ifitmw == 0) & (absh2o > 0.)
        dummy = bmwa1*(amw**bmwa2)
        Bmwx[universe] = dummy[universe]
        universe = (ifitmw == 1) & (absh2o > 0.)
        dummy = (1. + bmwa0*amw1 + bmwa1*amw12) / (1. + bmwa2*amw1)
        Bmwx[universe] = dummy[universe]
        universe = (ifitmw == 2) & (absh2o > 0.)
        dummy = (1. + bmwa0*amw1 + bmwa1*amw12) / (1. + bmwa2*amw12)
        Bmwx[universe] = dummy[universe]

        Bmw = np.where(cond, Bmw, Bmwx)
        return np.clip(Bmw, 0.05, 7)

    @warningfilter('ignore')
    def _calculate_Bp(self, pw, pr, am):
        bpa1 = self.h2o['bpa1']
        bpa2 = self.h2o['bpa2']

        pwm = pw*am
        pp0 = pr / 1013.25
        pp01 = np.maximum(0.65, pp0)
        pp02 = pp01**2
        qp = 1. - pp0
        qp1 = np.minimum(0.35, qp)
        qp2 = qp1**2

        absh2o = self.h2o['abs']
        absh2o = absh2o * np.ones(am.shape) * np.ones(pw.shape)
        iband = self.h2o['iband']
        iband = iband * np.ones(absh2o.shape)

        Bp = (1. + 0.1623*qp) * np.ones(iband.shape)
        universe = (iband == 2) & (absh2o > 0.)
        Bpx = (1. + 0.08721*qp1) * np.ones(iband.shape)
        Bp[universe] = Bpx[universe]
        universe = (iband == 3) & (absh2o > 0.)
        A = (1. - bpa1*qp1 - bpa2*qp2) * np.ones(iband.shape)
        Bp[universe] = A[universe]
        universe = (iband == 4) & (absh2o > 0.)
        B = 1. - pwm*np.exp(-0.63486 + 6.9149*pp01 - 13.853*pp02)
        Bp[universe] = (A*B)[universe]
        universe = (iband == 5) & (absh2o > 0.)
        B = 1. - pwm*np.exp(8.9243 - 18.197*pp01 + 2.4141*pp02)
        Bp[universe] = (A*B)[universe]

        Bp[(np.abs(qp*np.ones(Bp.shape)) < 1e-5) | (absh2o <= 0)] = 1
        return np.clip(Bp, 0.3, 1.7)

    def frame(self, pw, sza, pressure=1013.25):
        """
        Spectral water (absorption) transmittance

        Inputs:
        -------
        pw: array-like, 0d or 1d
          Precipitable water, cm (i.e., total-column water vapor content)
        sza: array-like, 0d or 1d
          Solar zenith angle
        pressure: float
          Atmospheric pressure at the computation level, hPa

        Output:
        -------
        Output's dimensions are (time, atmosphere, spectrum), i.e.,
        the output's shape is (len(sza), len(pw), len(wvl))
        """
        am_ = np.array(airmass(sza, 'h2o'), ndmin=1)[:, None, None]
        pw_ = np.array(pw, ndmin=1)[None, :, None]
        pr = np.array(pressure, ndmin=1)[None, :, None]
        validate_frame(am_, pw_, pr)

        for column_name, column_value in self.h2o.items():
            self.h2o[column_name] = column_value[None, None, :]

        Bmw = self._calculate_Bmw(pw_, am_)
        Bp = self._calculate_Bp(pw_, pr, am_)
        pwm = (pw_*am_)**0.9426
        tauw_l = Bmw*Bp * self.h2o['abs']*pwm

        tauw_l = interp1d(
            self.h2o['wvl'][0, 0, :], tauw_l, kind='linear',
            bounds_error=False, fill_value=0., axis=-1)(self.wvl)

        for column_name, column_value in self.h2o.items():
            self.h2o[column_name] = column_value[0, 0, :]

        return np.clip(np.exp(-tauw_l), 0., 1.)

    def series(self, pw, sza, pressure=1013.25):
        """
        Spectral water (absorption) transmittance

        Inputs:
        -------
        pw: array-like, 0d or 1d
          Precipitable water, cm (i.e., total-column water vapor content)
        sza: array-like, 0d or 1d
          Solar zenith angle, with same shape as pw
        pressure: float
          Atmospheric pressure at the computation level, hPa

        Output:
        -------
        Output's dimensions are (time, spectrum), i.e., the output's
        shape is (len(sza), len(wvl))
        """
        am_ = np.array(airmass(sza, 'h2o'), ndmin=1)[:, None]
        pw_ = np.array(pw, ndmin=1)[:, None]
        validate_series(am_, pw_)

        pr = np.full(am_.shape, pressure)

        for column_name, column_value in self.h2o.items():
            self.h2o[column_name] = column_value[None, :]

        Bmw = self._calculate_Bmw(pw_, am_)
        Bp = self._calculate_Bp(pw_, pr, am_)
        pwm = (pw_*am_)**0.9426
        tauw_l = Bmw*Bp * self.h2o['abs']*pwm

        tauw_l = interp1d(
            self.h2o['wvl'][0, :], tauw_l, kind='linear',
            bounds_error=False, fill_value=0., axis=-1)(self.wvl)

        for column_name, column_value in self.h2o.items():
            self.h2o[column_name] = column_value[0, :]

        return np.clip(np.exp(-tauw_l), 0., 1.)
