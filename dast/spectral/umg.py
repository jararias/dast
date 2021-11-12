
# Author: Jose A Ruiz-Arias, jararias at uma.es

import numpy as np

from .. import airmass
from ._base import validate_frame, validate_series
from ._base import get_data_filename
from ._base import SpectralTransmittance

# References:
# SMARTS v2.9.5 code


NLOSCHMIDT = 2.6867811e19  # cm-3, number of particles in a volume of ideal gas


def interp(x, xp, yp):
    return np.interp(x, xp, yp, left=0., right=0.)


class UMGTransmittance(SpectralTransmittance):

    """

    USSA atmosphere!!

    """

    def __init__(self, co2_ppm=395., tair=15., with_trace_gases=True):
        assert np.isscalar(tair)
        assert np.isscalar(co2_ppm)

        super().__init__()

        self.co2_ppm = co2_ppm
        self.tair = tair
        self.with_trace_gases = with_trace_gases

        def read_gas(gas_name):
            data = np.loadtxt(
                get_data_filename(f'Abs_{gas_name}.dat'), skiprows=1)
            gas_wvl = data[:, 0]
            if data.shape[1] == 2:
                return interp(self.wvl, gas_wvl, data[:, 1])
            else:
                return [
                    interp(self.wvl, gas_wvl, data[:, k])
                    for k in range(1, data.shape[1])
                ]

        # absorption coeffs | cross sections
        self.o2abs = read_gas('O2')
        self.n2abs = read_gas('N2')
        self.coabs = read_gas('CO')
        self.co2abs = read_gas('CO2')
        self.ch4abs = read_gas('CH4')
        self.o4abs = 1e-46 * read_gas('O4')
        self.n2oabs = read_gas('N2O')

        if self.with_trace_gases is True:
            self.nh3abs = read_gas('NH3')
            self.noabs = read_gas('NO')
            sigma, b0 = read_gas('NO2')
            self.no2abs = NLOSCHMIDT*(sigma + b0*(228.7-220.))
            sigma, b0 = read_gas('SO2U')
            self.so2abs = NLOSCHMIDT*(sigma + b0*(247.-213))
            self.so2abs += read_gas('SO2I')
            xs, b0 = read_gas('HNO3')
            self.hno3abs = 1e-20*NLOSCHMIDT*xs*np.exp(1e-3*b0*(234.2-298.))
            xs, b0 = read_gas('NO3')
            self.no3abs = NLOSCHMIDT*(xs+b0*(225.3-230.))
            self.hno2abs = NLOSCHMIDT*read_gas('HNO2')
            xs, b0 = read_gas('CH2O')
            self.ch2oabs = NLOSCHMIDT*(xs+b0*(264.-293.))
            self.broabs = NLOSCHMIDT*read_gas('BrO')
            xs, b0, b1 = read_gas('ClNO')
            TCl = 230.  # K
            self.clnoabs = xs*NLOSCHMIDT*(1.+b0*(TCl-296)+b1*(TCl-296)**2)

    def frame(self, pressure, sza):
        """
        Spectral molecular (absorption) transmittance

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
        pp0 = np.array(pressure, ndmin=1)[None, :, None] / 1013.25
        sza_ = np.array(sza, ndmin=1)[:, None, None]
        validate_frame(sza_, pp0)
        taug_l = self._optical_depth(pp0, sza_, 'frame')
        return np.clip(np.exp(-taug_l), 0., 1.)

    def series(self, pressure, sza):
        """
        Spectral molecular (absorption) transmittance

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
        pp0 = np.array(pressure, ndmin=1)[:, None] / 1013.25
        sza_ = np.array(sza, ndmin=1)[:, None]
        validate_series(sza_, pp0)
        taug_l = self._optical_depth(pp0, sza_, 'series')
        return np.clip(np.exp(-taug_l), 0., 1.)

    def _optical_depth(self, pp0, sza, mode):

        tt0 = np.zeros_like(pp0) + (self.tair + 273.15) / 273.15  # noqa

        if mode == 'frame':
            taug_l = np.zeros((len(sza), pp0.size, len(self.wvl)))

        if mode == 'series':
            taug_l = np.zeros((len(sza), len(self.wvl)))

        def getam(constituent):
            return airmass(sza, constituent)

        def getabs(constituent):
            if mode == 'frame':
                return getattr(self, f'{constituent}abs')[None, None, :]
            elif mode == 'series':
                return getattr(self, f'{constituent}abs')[None, :]

        # Uniformly Mixed Gases

        # 1. Oxygen, O2
        abundance = 1.67766e5 * pp0
        taug_l += getabs('o2') * abundance * getam('o2')
        # 2. Methane, CH4
        abundance = 1.3255 * (pp0 ** 1.0574)
        taug_l += getabs('ch4') * abundance * getam('ch4')
        # 3. Carbon Monoxide, CO
        abundance = .29625 * (pp0**2.4480) * \
            np.exp(.54669 - 2.4114 * pp0 + .65756 * (pp0**2))
        taug_l += getabs('co') * abundance * getam('co')
        # 4. Nitrous Oxide, N2O
        abundance = .24730 * (pp0**1.0791)
        taug_l += getabs('n2o') * abundance * getam('n2o')
        # 5. Carbon Dioxide, CO2
        abundance = 0.802685 * self.co2_ppm * pp0
        taug_l += getabs('co2') * abundance * getam('co2')
        # 6. Nitrogen, N2
        abundance = 3.8269 * (pp0**1.8374)
        taug_l += getabs('n2') * abundance * getam('n2')
        # 7. Oxygen-Oxygen, O4
        abundance = 1.8171e4 * (NLOSCHMIDT**2) * (pp0**1.7984) / (tt0**.344)
        taug_l += getabs('o4') * abundance * getam('o2')

        # Misc. Trace Gases

        if self.with_trace_gases is True:
            # 1. Nitric Acid, HNO3
            abundance = 1e-4*3.637*(pp0**0.12319)
            taug_l += getabs('hno3') * abundance * getam('hno3')
            # 2. Nitrogen Dioxide, NO2
            abundance = 1e-4*np.minimum(1.8599+0.18453*pp0, 41.771*pp0)
            taug_l += getabs('no2') * abundance * getam('no2')
            # 3. Nitrogen Trioxide, NO3
            abundance = 5e-5
            taug_l += getabs('no3') * abundance * getam('no3')
            # 4. Nitric Oxide, NO
            abundance = 1e-4*np.minimum(0.74307+2.4015*pp0, 57.079*pp0)
            taug_l += getabs('no') * abundance * getam('no')
            # 5. Sulfur Dioxide, SO2
            abundance = 1e-4*0.11133*(pp0**.812) * np.exp(
                .81319+3.0557*(pp0**2)-1.578*(pp0**3))
            taug_l += getabs('so2') * abundance * getam('so2')
            # 6. Ozone, O3
            # ...implemented as independent transmittance
            # 7. Ammonia, NH3
            lpp0 = np.log(pp0)
            abundance = np.exp(
                - 8.6499 + 2.1947*lpp0 - 2.5936*(lpp0**2)
                - 1.819*(lpp0**3) - 0.65854*(lpp0**4))
            taug_l += getabs('nh3') * abundance * getam('nh3')
            # 8. Bromine Monoxide, BrO
            abundance = 2.5e-6
            taug_l += getabs('bro') * abundance * getam('bro')
            # 9. Formaldehyde, CH2O
            abundance = 3e-4
            taug_l += getabs('ch2o') * abundance * getam('ch2o')
            # 10. Nitrous Acid, HNO2
            abundance = 1e-4
            taug_l += getabs('hno2') * abundance * getam('hno2')
            # 11. Chlorine Nitrate, ClNO3
            abundance = 1.2e-4
            taug_l += getabs('clno') * abundance * getam('clno')

        return taug_l

#     def series(self, pressure, am=None, sza=None):
#         """
#         Spectral molecular (absorption) transmittance

#         Inputs:
#         -------
#         pressure: array-like, 0d or 1d
#           Atmospheric pressure, hPa
#         am: array-like, 0d or 1d
#           Atmospheric optical airmass, with same shape as pressure
#         sza: array-like, 0d or 1d
#           Solar zenith angle, with same shape as pressure

#         Output:
#         -------
#         Output's dimensions are (time, spectrum), i.e., the output's
#         shape is (len(am), len(wvl))
#         """
#         pp0 = np.array(pressure, ndmin=1)[:, None] / 1013.25
#         if sza is None:
#             am_ = np.array(am, ndmin=1)[:, None]
#             validate_frame(am_, pp0)
#             n_steps = len(am_)
#         else:
#             sza_ = np.array(sza, ndmin=1)[:, None]
#             validate_frame(sza_, pp0)
#             n_steps = len(sza_)

#         tt0 = np.zeros_like(pp0) + (self.tair + 273.15) / 273.15

#         taug_l = np.zeros((n_steps, len(self.wvl)))

#         def getam(constituent):
#             if sza is None:
#                 return am_
#             return airmass(sza_, constituent)

#         # Uniformly Mixed Gases

#         # 1. Oxygen, O2
#         abundance = 1.67766e5 * pp0
#         taug_l += self.o2abs[None, :] * abundance * getam('o2')
#         # 2. Methane, CH4
#         abundance = 1.3255 * (pp0 ** 1.0574)
#         taug_l += self.ch4abs[None, :] * abundance * getam('ch4')
#         # 3. Carbon Monoxide, CO
#         abundance = .29625 * (pp0**2.4480) * \
#             np.exp(.54669 - 2.4114 * pp0 + .65756 * (pp0**2))
#         taug_l += self.coabs[None, :] * abundance * getam('co')
#         # 4. Nitrous Oxide, N2O
#         abundance = .24730 * (pp0**1.0791)
#         taug_l += self.n2oabs[None, :] * abundance * getam('n2o')
#         # 5. Carbon Dioxide, CO2
#         abundance = 0.802685 * self.co2_ppm * pp0
#         taug_l += self.co2abs[None, :] * abundance * getam('co2')
#         # 6. Nitrogen, N2
#         abundance = 3.8269 * (pp0**1.8374)
#         taug_l += self.n2abs[None, :] * abundance * getam('n2')
#         # 7. Oxygen-Oxygen, O4
#         abundance = 1.8171e4 * (NLOSCHMIDT**2) * (pp0**1.7984) / (tt0**.344)
#         taug_l += self.o4xs[None, :] * 1e-46 * abundance * getam('o2')

#         # Misc. Trace Gases

#         # 1. Nitric Acid, HNO3
#         abundance = 1e-4*3.637*(pp0**0.12319)
#         taug_l += self.hno3abs[None, :] * abundance
#         # 2. Nitrogen Dioxide, NO2
#         abundance = 1e-4*np.minimum(1.8599+0.18453*pp0, 41.771*pp0)
#         taug_l += self.no2abs[None, :] * abundance
#         # 3. Nitrogen Trioxide, NO3
#         abundance = 5e-5
#         taug_l += self.no3abs[None, :] * abundance
#         # 4. Nitric Oxide, NO
#         abundance = 1e-4*np.minimum(0.74307+2.4015*pp0, 57.079*pp0)
#         taug_l += self.noabs[None, :] * abundance
#         # 5. Sulfur Dioxide, SO2
#         abundance = 1e-4*0.11133*(pp0**.812) * np.exp(
#             .81319+3.0557*(pp0**2)-1.578*(pp0**3))
#         taug_l += self.so2abs[None, :] * abundance
#         # 6. Ozone, O3
#         # ...implemented as independent transmittance
#         # 7. Ammonia, NH3
#         lpp0 = np.log(pp0)
#         abundance = np.exp(-8.6499 + 2.1947*lpp0 - 2.5936*(lpp0**2)
#                            - 1.819*(lpp0**3) - 0.65854*(lpp0**4))
#         taug_l += self.nh3abs[None, :] * abundance
#         # 8. Bromine Monoxide, BrO
#         # ...not implemented
#         # 9. Formaldehyde, CH2O
#         abundance = 3e-4
#         taug_l += self.ch2oabs[None, :] * abundance
#         # 10. Nitrous Acid, HNO2
#         abundance = 1e-4
#         taug_l += self.hno2abs[None, :] * abundance
#         # 11. Chlorine Nitrate, ClNO3
#         # ...not implemented

#         return np.clip(np.exp(-am_ * taug_l), 0., 1.)
