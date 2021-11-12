
##############################################################################
# author: Jose A Ruiz-Arias, University of Malaga                            #
# email: jararias at uma.es, jose.ruiz-arias at solargis.com                 #
# date: 11.11.2021                                                           #
##############################################################################

##############################################################################
# BEFORE RUNNING THIS SCRIPT, YOU NEED TO DOWNLOAD SMARTS V2.9.5 TO LOCAL.   #
# THE CODE IS FREELY AVAILABLE AT:                                           #
#                                                                            #
#     https://www.nrel.gov/grid/solar-resource/smarts.html                   #
#                                                                            #
# THEN, UNCOMPRESS AND SET THE VARIABLE SMARTS_DIR (SEE BELOW) TO THE DIR    #
# WHERE SMARTS IS UNCOMPRESSED.                                              #
# IN ADDITION, SMARTS SHOULD BE CONFIGURED IN BATCH MODE. TO THAT AIM, EDIT  #
# THE SOURCE FILE Source_code/smarts295.f, COMMENT THE CODE LINE 188         #
# (batch=.FALSE.) AND UNCOMMENT THE CODE LINE 189 (batch=.TRUE.)             #
# THIS SCRIPT HAS BEEN TESTED WITH THE VERSION FOR LINUX.                    #
##############################################################################

import numpy as np
import pylab as pl
from scipy.optimize import root_scalar

from dast import pysmarts as pysm
from dast import spectral, airmass


SMARTS_DIR = '../SMARTS_295_Linux'

SMARTS_INP_TEMPLATE = """
'Template input for custom run'	!Card 1 Comment
1                        !Card 2 ISPR
$pressure 0. 0.          !Card 2a Pressure, altitude, height
1                        !Card 3 IATMOS
'USSA'                   !Card 3a Atmos
0                        !Card 4 IH2O
$pwater                  !Card 4a Precipitable water
0                        !Card 5 IO3
0 $ozone                 !Card 5a Ialt, AbO3
1                        !Card 6 IGAS
370.0                    !Card 7 CO2 amount (ppm)
0                        !Card 7a ISPCTR
'USER'                   !Card 8 Aeros (aerosol model)
$alpha $alpha 0.92 0.65  !Card 8a User inputs for ALPHA1, ALPHA2, OMEGL, GG
1                        !Card 9 ITURB
$beta                    !Card 9a Turbidity coeff. (BETA)
38                       !Card 10 IALBDX
0                        !Card 10b ITILT
290 4000 1.0 1361.1      !Card 11 Min & max wvls; sun-earth dist cor; sol const
2                        !Card 12 IPRT
290 4000 .5              !Card12a Min & max wvls to print; printing step size
9                        !Card12b Number of Variables to Print
1 2 15 16 17 18 19 20 21 !Card12c Variable codes
0                        !Card 13 ICIRC
0                        !Card 14 ISCAN
0                        !Card 15 ILLUM
0                        !Card 16 IUV
2                        !Card 17 IMASS
$am                      !Card 17a Air mass
"""


def retrieve_sza(am):
    return root_scalar(lambda sza: am - airmass(sza), bracket=[0, 90]).root


def symmetric_colormap(pc, max_value=None):
    vmin, vmax = pc.get_clim()
    vmax = max(abs(vmin), abs(vmax))
    if max_value is not None:
        vmax = max(max_value, vmax)
    pc.set_clim(-vmax, vmax)


if __name__ == '__main__':

    min_am = 1   # minimum molecular airmass
    max_am = 20  # maximum molecular airmass
    n_ams = 6    # number of molecular airmasses

    pressure = 1013.25  # atmospheric pressure, hPa
    ozone = 0.3         # total-column ozone content, atm-cm
    pwater = 1.4        # precipitable water, cm
    beta = 0.12         # Angstrom's turbidity parameter
    alpha = 1.3         # Angstrom's exponent parameter

    dast_method = 'frame'  # 'series'  # dast's accessor method

    # initalize and setup

    pysm.build(SMARTS_DIR)

    ozo_t = spectral.OzoneTransmittance(abs_source='smarts')
    ray_t = spectral.RayleighTransmittance()
    umg_t = spectral.UMGTransmittance(co2_ppm=370)
    wat_t = spectral.WaterTransmittance()
    aer_t = spectral.AerosolTransmittance()

    atmos = dict(
        pressure=pressure, ozone=ozone,
        pwater=pwater, beta=beta, alpha=alpha
    )

    wvl = ozo_t.wvl
    ams = np.logspace(np.log10(min_am), np.log10(max_am), n_ams)
    szas = np.array([retrieve_sza(am) for am in ams])

    # simulate

    smarts, _ = pysm.run_batch(
        SMARTS_INP_TEMPLATE, {'am': ams}, atmos, wvl
    )

    dast = {}

    if dast_method == 'frame':
        dast['Tol'] = ozo_t.frame(atmos['ozone'], szas)[:, 0, :]
        dast['Trl'] = ray_t.frame(atmos['pressure'], szas)[:, 0, :]
        dast['Tgl'] = umg_t.frame(atmos['pressure'], szas)[:, 0, :]
        dast['Twl'] = wat_t.frame(
            atmos['pwater'], szas, atmos['pressure'])[:, 0, :]
        dast['Tal'] = aer_t.frame(
            atmos['beta'], atmos['alpha'], szas)[:, 0, :]

    if dast_method == 'series':
        dast['Tol'] = ozo_t.series(
            np.full(len(szas), atmos['ozone']), szas)
        dast['Trl'] = ray_t.series(
            np.full(len(szas), atmos['pressure']), szas)
        dast['Tgl'] = umg_t.series(
            np.full(len(szas), atmos['pressure']), szas)
        dast['Twl'] = wat_t.series(
            np.full(len(szas), atmos['pwater']), szas)
        dast['Tal'] = aer_t.series(
            np.full(len(szas), atmos['beta']),
            np.full(len(szas), atmos['alpha']), szas)

    # plot

    for tr_name in ('Tol', 'Trl', 'Tgl', 'Twl', 'Tal'):
        if tr_name not in dast:
            continue

        pl.figure(tr_name, figsize=(16, 6))

        # plot spectra
        pl.subplot2grid((1, 5), (0, 0), 1, 3)
        for k in range(len(ams)):
            pl.plot(wvl, smarts[tr_name][k], ls='-', lw=2, color='k')
            pl.plot(wvl, dast[tr_name][k], ls='-',
                    lw=1, label=f'am={ams[k]:.2f}')
        pl.ylabel(tr_name, fontsize=18)
        pl.xlabel('Wavelength (nm)', fontsize=18)
        pl.tick_params(labelsize=14)
        pl.title('SMARTS is black. Dast is colored', fontsize=18)
        pl.legend()

        # plot differences ("errors")
        pl.subplot2grid((1, 5), (0, 3), 1, 2)
        diff = dast[tr_name] - smarts[tr_name]
        cmap = pl.cm.RdBu_r
        pc = pl.pcolormesh(wvl, ams, diff, cmap=cmap, shading='auto',
                           vmin=-.01, vmax=.01)
        cb = pl.colorbar()
        cb.set_ticks(np.arange(-0.01, 0.011, 0.0025))
        pl.xlabel('Wavelength (nm)', fontsize=18)
        pl.ylabel('Airmass', fontsize=18)
        pl.tick_params(labelsize=14)
        pl.title(f'{tr_name}: dast - smarts295', fontsize=18)

        pl.tight_layout()

    pl.show()

    pysm.wipeout()
