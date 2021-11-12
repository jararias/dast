
##############################################################################
# author: Jose A Ruiz-Arias, University of Malaga                            #
# email: jararias at uma.es, jose.ruiz-arias at solargis.com                 #
# date: 11.11.2021                                                           #
##############################################################################

import os

import numpy as np
from scipy.optimize import root_scalar

from dast import spectral, airmass

from progress.bar import Bar


TARGET_DIR = '../results'


def retrieve_sza(am):
    return root_scalar(lambda sza: am - airmass(sza), bracket=[0, 90]).root


def make_dataset():

    AM = np.logspace(0., np.log10(20), 30)
    PR = [800, 1013]  # [~2000 m, and sea level]
    UO = [0.3]
    PW = np.linspace(0., 5., 81)
    BETA = np.linspace(0., 1.2, 97)
    ALPHA = [0.3, 1.3, 2.3]

    SZA = np.array([retrieve_sza(am) for am in AM])

    out_shape = (len(AM), len(PR), len(UO), len(PW), len(BETA), len(ALPHA))
    ones = np.ones(out_shape)

    print(f'Shape: {out_shape} ({np.prod(out_shape)} elements)')

    # spectral transmittances
    ozo_t = spectral.OzoneTransmittance()
    ray_t = spectral.RayleighTransmittance()
    umg_t = spectral.UMGTransmittance()
    wat_t = spectral.WaterTransmittance()
    aer_t = spectral.AerosolTransmittance()

    print('Computing spectral transmittances...')
    Tol = ozo_t.frame(UO, SZA)[:, None, :, None, None, None]
    Trl = ray_t.frame(PR, SZA)[:, :, None, None, None, None]
    Tgl = umg_t.frame(PR, SZA)[:, :, None, None, None, None]
    Twl = wat_t.frame(PW, SZA)[:, None, None, :, None, None]
    Tal = np.concatenate(
        [
            aer_t.frame(BETA, np.full(BETA.shape, alpha_k), SZA)[:, :, None, :]
            for alpha_k in ALPHA
        ],
        axis=2
    )[:, None, None, None, :, :]

    # solar spectrum
    wvl, Eonl = ozo_t.wvl, ozo_t.Eonl
    Eon = np.trapz(Eonl, wvl)

    def integrate(transmittance, band=None):
        if band is None:
            return np.trapz(Eonl*transmittance, wvl)
        return np.trapz(Eonl[band]*transmittance[..., band], wvl[band])

    # INDEPENDENT INTEGRATION SCHEME ######################################
    print('Computing independent broadband transmittances...')
    To_indep = (integrate(Tol) / Eon) * ones
    Tr_indep = (integrate(Trl) / Eon) * ones
    Tg_indep = (integrate(Tgl) / Eon) * ones
    Tw_indep = (integrate(Twl) / Eon) * ones
    Ta_indep = (integrate(Tal) / Eon) * ones
    T_indep = To_indep*Tr_indep*Tg_indep*Tw_indep*Ta_indep

    # 2-BAND INDEPENDENT INTEGRATION SCHEME ################################
    print('Computing 2-band independent broadband transmittances...')

    # parameters for the 2-band independent transmittance scheme
    band1 = (wvl >= 290) & (wvl <= 700)
    band2 = (wvl >= 700) & (wvl <= 4000)
    Eon1 = np.trapz(Eonl[band1], wvl[band1])
    f1 = Eon1 / Eon
    Eon2 = np.trapz(Eonl[band2], wvl[band2])
    f2 = Eon2 / Eon

    To1_indep = (integrate(Tol, band1) / Eon1) * ones
    Tr1_indep = (integrate(Trl, band1) / Eon1) * ones
    Tg1_indep = (integrate(Tgl, band1) / Eon1) * ones
    Tw1_indep = (integrate(Twl, band1) / Eon1) * ones
    Ta1_indep = (integrate(Tal, band1) / Eon1) * ones
    To2_indep = (integrate(Tol, band2) / Eon2) * ones
    Tr2_indep = (integrate(Trl, band2) / Eon2) * ones
    Tg2_indep = (integrate(Tgl, band2) / Eon2) * ones
    Tw2_indep = (integrate(Twl, band2) / Eon2) * ones
    Ta2_indep = (integrate(Tal, band2) / Eon2) * ones
    T_indep2b = (
        f1*To1_indep*Tr1_indep*Tg1_indep*Tw1_indep*Ta1_indep +
        f2*To2_indep*Tr2_indep*Tg2_indep*Tw2_indep*Ta2_indep
    )

    # INTERDEPENDENT INTEGRATION SCHEME ###################################
    print('Computing interdependent broadband transmittances...')
    To_inter = (integrate(Tol) / Eon) * ones
    Tr_inter = (integrate(Tol*Trl) / integrate(Tol)) * ones
    Tg_inter = (integrate(Tol*Trl*Tgl) / integrate(Tol*Trl)) * ones
    Tw_inter = (integrate(Tol*Trl*Tgl*Twl) / integrate(Tol*Trl*Tgl)) * ones

    # the calculation of Ta_inter can exhaust the memory so I
    # split the calculation to ensure that I have enough memory

    step = 20
    weight = Eonl*Tol*Trl*Tgl*Twl
    Ta_inter = np.zeros(Tw_inter.shape)
    steps = np.arange(0, len(wvl), step)
    bar = Bar('Ta_inter progress:', max=len(steps), suffix='%(percent)d%%')
    for kmin in steps:
        kmax = min(kmin + step, len(wvl))
        Ta_inter += np.trapz(
            weight[..., kmin: kmax + 1] * Tal[..., kmin: kmax + 1],
            wvl[kmin: kmax + 1]
        )
        bar.next()
    Ta_inter = Ta_inter / np.trapz(weight, wvl)
    print()

    T_inter = To_inter*Tr_inter*Tg_inter*Tw_inter*Ta_inter

    # PRESCRIBED INTERDEPENDENT INTEGRATION SCHEME ########################
    print('Computing prescribed interdependent broadband transmittances...')

    # parameters for the centered interdependent scheme
    UO_CINTER = 0.3
    PR_CINTER = 1013.25
    PW_CINTER = 1.4

    Tolp = ozo_t.frame(UO_CINTER, SZA)[:, None, :, None, None, None]
    Trlp = ray_t.frame(PR_CINTER, SZA)[:, :, None, None, None, None]
    Tglp = umg_t.frame(PR_CINTER, SZA)[:, :, None, None, None, None]
    Twlp = wat_t.frame(PW_CINTER, SZA)[:, None, None, :, None, None]

    To_cinter = (integrate(Tol) / Eon) * ones
    Tr_cinter = (integrate(Tolp*Trl) / integrate(Tolp)) * ones
    Tg_cinter = (integrate(Tolp*Trlp*Tgl) / integrate(Tolp*Trlp)) * ones
    Tw_cinter = (
        integrate(Tolp*Trlp*Tglp*Twl) / integrate(Tolp*Trlp*Tglp)
    ) * ones

    step = 20
    weights = Eonl*Tolp*Trlp*Tglp*Twlp
    Ta_cinter = np.zeros(Tw_cinter.shape)
    steps = np.arange(0, len(wvl), step)
    bar = Bar('Ta_cinter progress:', max=len(steps), suffix='%(percent)d%%')
    for kmin in steps:
        kmax = min(kmin + step, len(wvl))
        Ta_cinter += np.trapz(
            weights[..., kmin: kmax + 1] * Tal[..., kmin: kmax + 1],
            wvl[kmin: kmax + 1]
        )
        bar.next()
    Ta_cinter = Ta_cinter / np.trapz(weights, wvl)
    print()

    T_cinter = To_cinter*Tr_cinter*Tg_cinter*Tw_cinter*Ta_cinter

    # HYBRID INTEGRATION SCHEME ###########################################
    print('Computing hybrid broadband transmittances...')

    To1_inter = (
        integrate(Tol, band1) /
        Eon1
    ) * ones
    Tr1_inter = (
        integrate(Tolp*Trl, band1) /
        integrate(Tolp, band1)
    ) * ones
    Tg1_inter = (
        integrate(Tolp*Trlp*Tgl, band1) /
        integrate(Tolp*Trlp, band1)
    ) * ones
    Tw1_inter = (
        integrate(Tolp*Trlp*Tglp*Twl, band1) /
        integrate(Tolp*Trlp*Tglp, band1)
    ) * ones
    Ta1_inter = (
        integrate(Tolp*Trlp*Tglp*Twlp*Tal, band1) /
        integrate(Tolp*Trlp*Tglp*Twlp, band1)
    ) * ones
    To2_inter = (
        integrate(Tol, band2) /
        Eon2
    ) * ones
    Tr2_inter = (
        integrate(Tolp*Trl, band2) /
        integrate(Tolp, band2)
    ) * ones
    Tg2_inter = (
        integrate(Tolp*Trlp*Tgl, band2) /
        integrate(Tolp*Trlp, band2)
    ) * ones
    Tw2_inter = (
        integrate(Tolp*Trlp*Tglp*Twl, band2) /
        integrate(Tolp*Trlp*Tglp, band2)
    ) * ones
    Ta2_inter = (
        integrate(Tolp*Trlp*Tglp*Twlp*Tal, band2) /
        integrate(Tolp*Trlp*Tglp*Twlp, band2)
    ) * ones
    T_inter_hybrid = (
        f1*To1_inter*Tr1_inter*Tg1_inter*Tw1_inter*Ta1_inter +
        f2*To2_inter*Tr2_inter*Tg2_inter*Tw2_inter*Ta2_inter
    )

    # SERIALIZE THE RESULTS ###############################################

    target_dir = os.path.join(TARGET_DIR, 'lut_transmittance')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    np.savez_compressed(
        f'{target_dir}/indep_transmittances.npz',
        To=To_indep, Tr=Tr_indep, Tg=Tg_indep, Tw=Tw_indep,
        Ta=Ta_indep, T=T_indep, am=AM, pressure=PR, uo=UO,
        pw=PW, beta=BETA, alpha=ALPHA)

    np.savez_compressed(
        f'{target_dir}/indep2b_transmittances.npz',
        To1=To1_indep, To2=To2_indep, Tr1=Tr1_indep, Tr2=Tr2_indep,
        Tg1=Tg1_indep, Tg2=Tg2_indep, Tw1=Tw1_indep, Tw2=Tw2_indep,
        Ta1=Ta1_indep, Ta2=Ta2_indep, T=T_indep2b, am=AM,
        pressure=PR, uo=UO, pw=PW, beta=BETA, alpha=ALPHA)

    np.savez_compressed(
        f'{target_dir}/inter_transmittances.npz',
        To=To_inter, Tr=Tr_inter, Tg=Tg_inter, Tw=Tw_inter,
        Ta=Ta_inter, T=T_inter, am=AM, pressure=PR, uo=UO,
        pw=PW, beta=BETA, alpha=ALPHA)

    np.savez_compressed(
        f'{target_dir}/cinter_transmittances.npz',
        To=To_cinter, Tr=Tr_cinter, Tg=Tg_cinter, Tw=Tw_cinter,
        Ta=Ta_cinter, T=T_cinter, am=AM, pressure=PR, uo=UO,
        pw=PW, beta=BETA, alpha=ALPHA)

    np.savez_compressed(
        f'{target_dir}/hybrid_transmittances.npz',
        To1=To1_inter, To2=To2_inter, Tr1=Tr1_inter, Tr2=Tr2_inter,
        Tg1=Tg1_inter, Tg2=Tg2_inter, Tw1=Tw1_inter, Tw2=Tw2_inter,
        Ta1=Ta1_inter, Ta2=Ta2_inter, T=T_inter_hybrid, am=AM,
        pressure=PR, uo=UO, pw=PW, beta=BETA, alpha=ALPHA)


if __name__ == '__main__':

    make_dataset()
