
##############################################################################
# author: Jose A Ruiz-Arias, University of Malaga                            #
# email: jararias at uma.es, jose.ruiz-arias at solargis.com                 #
# date: 11.11.2021                                                           #
##############################################################################

import os

import numpy as np
import pylab as pl
from scipy.optimize import root_scalar
from matplotlib.patches import Ellipse
from matplotlib.font_manager import FontProperties

from dast import spectral, airmass


TARGET_DIR = '../figures'


def retrieve_sza(am):
    return root_scalar(lambda sza: am - airmass(sza), bracket=[0, 90]).root


def fuente(ttf):
    def inner(**kwargs):
        return FontProperties(fname=ttf, **kwargs)
    return inner


arial = fuente('fonts/arial.ttf')
arial_bold = fuente('fonts/arialbd.ttf')


def plot_molineaux_and_ineichen_1996():

    # temperate rural taken from Molineaux and Ineichen, 1996
    cases = {                 # uo     P    Pw  Beta  Alpha
        'temperate rural':    (0.3, 1000., 2. , 0.1 ,   1.3),  # noqa
        'high altitude':      (0.3,  600., 0.5, 0.05,   1.3),  # noqa
        'tropical polluted':  (0.3, 1000., 4.5, 0.4 ,   2.1),  # noqa
        'dry desert':         (0.3, 1000., 0.5, 0.4 ,   0.3)   # noqa
    }

    am = np.logspace(0, np.log10(6), 20)
    sza = np.array([retrieve_sza(am_) for am_ in am])

    ozo_t = spectral.OzoneTransmittance()
    ray_t = spectral.RayleighTransmittance()
    umg_t = spectral.UMGTransmittance()
    wat_t = spectral.WaterTransmittance()
    aer_t = spectral.AerosolTransmittance()

    wvl = ozo_t.wvl  # [290, 4000] um
    Eonl = ozo_t.Eonl  # [290, 4000] um
    Eon = ozo_t.Eon  # integral of Eonl in [290, 4000] um

    pl.figure(1, figsize=(18, 13.5), dpi=300)

    # ozo_color = '#984ea3'
    # ray_color = '#377eb8'
    # umg_color = '#4daf4a'
    # wat_color = '#ff7f00'
    # aer_color = '#e41a1c'

    # parameters for the 2-band independent transmittance scheme
    band1 = (wvl >= 290) & (wvl <= 700)
    Eon1 = np.trapz(Eonl[band1], wvl[band1])
    f1 = Eon1 / Eon
    band2 = (wvl >= 700) & (wvl <= 4000)
    Eon2 = np.trapz(Eonl[band2], wvl[band2])
    f2 = Eon2 / Eon

    # parameteres for the centered interdependent transmittance scheme
    Tolp = ozo_t.frame(0.3, sza)
    Trlp = ray_t.frame(1013., sza)
    Tglp = umg_t.frame(1013., sza)
    Twlp = wat_t.frame(1.4, sza)

    def integrate(transmittance, band=None):
        if band is None:
            return np.trapz(Eonl*transmittance, wvl)
        return np.trapz(Eonl[band]*transmittance[..., band], wvl[band])

    for n_case, (case_name, case_values) in enumerate(cases.items()):

        pl.subplot(2, 2, n_case+1)

        uo, P, pw, beta, alpha = case_values

        # spectral transmittances
        Tol = ozo_t.frame(uo, sza)
        Trl = ray_t.frame(P, sza)
        Tgl = umg_t.frame(P, sza)
        Twl = wat_t.frame(pw, sza)
        Tal = aer_t.frame(beta, alpha, sza)

        # INTERDEPENDENT BROADBAND TRANSMITTANCES...
        kwargs = dict(ls='-', lw=3, marker='', color='k')
        a, b = integrate(Tol), integrate(1)
        T = a / b
        pl.semilogx(am, T, **kwargs)
        a, b = integrate(Tol*Trl), a
        T = T * (a / b)
        pl.semilogx(am, T, **kwargs)
        a, b = integrate(Tol*Trl*Tgl), a
        T = T * (a / b)
        pl.semilogx(am, T, **kwargs)
        a, b = integrate(Tol*Trl*Tgl*Twl), a
        T = T * (a / b)
        pl.semilogx(am, T, **kwargs)
        a, b = integrate(Tol*Trl*Tgl*Twl*Tal), a
        T = T * (a / b)
        pl.semilogx(am, T, **kwargs)

        if n_case == 2:
            pl.text(0., -0.22, 'Interdependent', transform=pl.gca().transAxes,
                    ha='left', va='center', fontproperties=arial_bold(size=20),
                    color=kwargs['color'])

        # INDEPENDENT BROADBAND TRANSMITTANCES...
        kwargs = dict(ls='-', lw=1.2, marker='')
        kwargs.update({'color': 'red'})
        T = integrate(Tol) / Eon
        pl.semilogx(am, T, **kwargs)
        T = T * (integrate(Trl) / Eon)
        pl.semilogx(am, T, **kwargs)
        T = T * (integrate(Tgl) / Eon)
        pl.semilogx(am, T, **kwargs)
        T = T * (integrate(Twl) / Eon)
        pl.semilogx(am, T, **kwargs)
        T = T * (integrate(Tal) / Eon)
        pl.semilogx(am, T, **kwargs)

        if n_case == 2:
            pl.text(0.31, -0.22, 'Independent', transform=pl.gca().transAxes,
                    ha='left', va='center', fontproperties=arial(size=20),
                    color=kwargs['color'])

        # centered interdependent broadband transmittances...
        kwargs.update({'color': 'green'})
        T = integrate(Tol) / integrate(1)
        pl.semilogx(am, T, **kwargs)
        Tl = Tolp
        T = T * (integrate(Tl*Trl) / integrate(Tl))
        pl.semilogx(am, T, **kwargs)
        Tl = Tl * Trlp
        T = T * (integrate(Tl*Tgl) / integrate(Tl))
        pl.semilogx(am, T, **kwargs)
        Tl = Tl * Tglp
        T = T * (integrate(Tl*Twl) / integrate(Tl))
        pl.semilogx(am, T, **kwargs)
        Tl = Tl * Twlp
        T = T * (integrate(Tl*Tal) / integrate(Tl))
        pl.semilogx(am, T, **kwargs)

        if n_case == 2:
            pl.text(0.56, -0.22, 'Prescribed Interdependent', ha='left',
                    va='center', transform=pl.gca().transAxes,
                    fontproperties=arial(size=20), color=kwargs['color'])

        # 2-BAND INDEPENDENT BROADBAND TRANSMITTANCES...
        kwargs.update({'color': 'blue'})
        To1 = integrate(Tol, band1) / Eon1
        To2 = integrate(Tol, band2) / Eon2
        pl.semilogx(am, f1*To1 + f2*To2, **kwargs)
        Tr1 = integrate(Trl, band1) / Eon1
        Tr2 = integrate(Trl, band2) / Eon2
        pl.semilogx(am, f1*To1*Tr1 + f2*To2*Tr2, **kwargs)
        Tg1 = integrate(Tgl, band1) / Eon1
        Tg2 = integrate(Tgl, band2) / Eon2
        pl.semilogx(am, f1*To1*Tr1*Tg1 + f2*To2*Tr2*Tg2, **kwargs)
        Tw1 = integrate(Twl, band1) / Eon1
        Tw2 = integrate(Twl, band2) / Eon2
        pl.semilogx(am, f1*To1*Tr1*Tg1*Tw1 + f2*To2*Tr2*Tg2*Tw2, **kwargs)
        Ta1 = integrate(Tal, band1) / Eon1
        Ta2 = integrate(Tal, band2) / Eon2
        pl.semilogx(
            am, f1*To1*Tr1*Tg1*Tw1*Ta1 + f2*To2*Tr2*Tg2*Tw2*Ta2, **kwargs
        )

        if n_case == 2:
            pl.text(1.05, -0.22, '2-band Independent', ha='left',
                    va='center', transform=pl.gca().transAxes,
                    fontproperties=arial(size=20), color=kwargs['color'])

        # HYBRID BROADBAND TRANSMITTANCES...
        kwargs.update({'color': 'orange'})
        To1 = integrate(Tol, band1) / Eon1
        To2 = integrate(Tol, band2) / Eon2
        T = f1*To1 + f2*To2
        pl.semilogx(am, T, **kwargs)
        Tl = Tolp
        Tr1 = integrate(Tl*Trl, band1) / integrate(Tl, band1)
        Tr2 = integrate(Tl*Trl, band2) / integrate(Tl, band2)
        T = f1*To1*Tr1 + f2*To2*Tr2
        pl.semilogx(am, T, **kwargs)
        Tl = Tolp*Trlp
        Tg1 = integrate(Tl*Tgl, band1) / integrate(Tl, band1)
        Tg2 = integrate(Tl*Tgl, band2) / integrate(Tl, band2)
        T = f1*To1*Tr1*Tg1 + f2*To2*Tr2*Tg2
        pl.semilogx(am, T, **kwargs)
        Tl = Tolp*Trlp*Tglp
        Tw1 = integrate(Tl*Twl, band1) / integrate(Tl, band1)
        Tw2 = integrate(Tl*Twl, band2) / integrate(Tl, band2)
        T = f1*To1*Tr1*Tg1*Tw1 + f2*To2*Tr2*Tg2*Tw2
        pl.semilogx(am, T, **kwargs)
        Tl = Tolp*Trlp*Tglp*Twlp
        Ta1 = integrate(Tl*Tal, band1) / integrate(Tl, band1)
        Ta2 = integrate(Tl*Tal, band2) / integrate(Tl, band2)
        T = f1*To1*Tr1*Tg1*Tw1*Ta1 + f2*To2*Tr2*Tg2*Tw2*Ta2
        pl.semilogx(am, T, **kwargs)

        if n_case == 2:
            pl.text(1.42, -0.22, 'Hybrid', ha='left',
                    va='center', transform=pl.gca().transAxes,
                    fontproperties=arial(size=20), color=kwargs['color'])

        # plot decorations...

        pl.title(case_name.capitalize(), x=0.02, ha='left',
                 fontproperties=arial(size=16))

        text_box = (
            f'P={P:.0f} hPa\n'r'u$_o$='f'{uo:.1f} atm-cm\n'r'p$_w$='
            f'{pw:.1f} cm\n'r'$\beta$='f'{beta:.2f}  '
            r'$\alpha$='f'{alpha:.1f}')

        pl.text(
            0.012, 0.016, text_box, transform=pl.gca().transAxes, ha='left',
            va='bottom', multialignment='left', fontproperties=arial(size=15),
            bbox=dict(facecolor='w', edgecolor='k', linewidth=.7, pad=3),
            zorder=1000)

        pl.text(
            -0.10, 0.98, f"({'abcd'[n_case]})", ha='center', va='center',
            transform=pl.gca().transAxes, fontproperties=arial_bold(size=18))

        pl.xlim(left=1, right=6)
        pl.ylim(bottom=0, top=1.01)  # , 0.45)

        xticks = np.arange(1.2, 6.81, 0.2)
        pl.gca().xaxis.set_ticks(xticks, minor=True)
        pl.gca().xaxis.set_ticklabels(['']*len(xticks), minor=True)
        pl.gca().xaxis.set_ticks(np.arange(1., 7.), minor=False)
        pl.gca().xaxis.set_ticklabels(
            [f'{x:.0f}' for x in np.arange(1., 7.)], minor=False)
        pl.tick_params(
            axis='x', which='major', direction='inout',
            top=True, length=10, width=1.2, color='0.4'
        )
        pl.tick_params(
            axis='x', which='minor', direction='in',
            top=True, length=3, width=1.2, color='0.4'
        )
        pl.xlim(1, 6)

        pl.gca().yaxis.set_ticks(np.linspace(0., 1., 6), minor=False)
        pl.gca().yaxis.set_ticks(np.linspace(0., 1., 51), minor=True)
        pl.tick_params(
            axis='y', which='major', direction='inout',
            right=True, length=10, width=1.2, color='0.4'
        )
        pl.tick_params(
            axis='y', which='minor', direction='in',
            right=True, length=3, width=1.2, color='0.4'
        )

        pl.grid(axis='y', which='major', color='0.8', ls='-', lw=1.2)
        pl.grid(axis='y', which='minor', color='0.8', dashes=(5, 2))

        for spine in pl.gca().spines.values():
            spine.set_color('0.4')
            spine.set_lw(1.2)

        pl.setp(
            pl.getp(pl.gca(), 'xticklabels'), fontproperties=arial(size=18)
        )

        pl.setp(
            pl.getp(pl.gca(), 'yticklabels'), fontproperties=arial(size=18)
        )

        pl.xlabel(
            'Optical air mass',
            labelpad=12, fontproperties=arial(size=20)
        )

        pl.ylabel(
            'Broadband transmittance',
            fontproperties=arial(size=20), labelpad=14
        )

        if n_case == 2:
            # leg = pl.legend(
            #     loc='upper left', bbox_to_anchor=(-.01, -.18),
            #     fontsize=15, ncol=5, frameon=False)
            # for obj in pl.findobj(leg):
            #     if isinstance(obj, pl.matplotlib.lines.Line2D):
            #         obj.set_color('k')

            # o+r+g+w+a
            transform = pl.gca().transAxes
            ellipse = Ellipse(
                xy=(.81, .075), width=0.03, height=0.08,
                angle=0., facecolor='none', edgecolor='k',
                linewidth=1.5, transform=transform, zorder=1000)
            pl.gca().add_artist(ellipse)
            pl.arrow(
                0.75, 0.17, 0.03, -0.035, transform=transform,
                head_width=0.015, linewidth=1.0, color='k', zorder=1000)
            pl.text(
                0.65, 0.19, 'o+r+g+w+a', transform=transform,
                multialignment='left', fontproperties=arial(size=18),
                zorder=1000)

            # o+r+g+w
            ellipse = Ellipse(
                xy=(.925, .455), width=0.03, height=0.12,
                angle=0., facecolor='none', edgecolor='k',
                linewidth=1.5, transform=transform, zorder=1000)
            pl.gca().add_artist(ellipse)
            pl.arrow(
                0.85, 0.36, 0.04, 0.03, transform=transform,
                head_width=0.015, linewidth=1.0, color='k', zorder=1000)
            pl.text(
                0.71, 0.34, 'o+r+g+w', transform=transform,
                multialignment='left', fontproperties=arial(size=18),
                zorder=1000)

            # o+r+g
            pl.arrow(
                0.63, 0.67, 0.07, 0.02, transform=transform,
                head_width=0.015, linewidth=1.0, color='k', zorder=1000)
            pl.text(
                0.53, 0.665, 'o+r+g', transform=transform,
                multialignment='left', fontproperties=arial(size=18),
                zorder=1000)

            # o+r
            pl.arrow(
                0.89, 0.78, -0.06, -0.03, transform=transform,
                head_width=0.015, linewidth=1.0, color='k', zorder=1000)
            pl.text(
                0.905, 0.785, 'o+r', transform=transform,
                multialignment='left', fontproperties=arial(size=18),
                zorder=1000)

            # o
            pl.arrow(
                0.66, 0.88, -0.04, 0.04, transform=transform,
                head_width=0.015, linewidth=1.0, color='k', zorder=1000)
            pl.text(
                0.67, 0.86, 'o', transform=transform,
                multialignment='left', fontproperties=arial(size=18),
                zorder=1000)

    pl.subplots_adjust(
        left=0.07, right=0.98, bottom=0.11, top=0.97,
        wspace=0.23, hspace=0.23)

    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    pl.savefig(f'{TARGET_DIR}/cumulative_transmittance.png', dpi=300)
    pl.close()


if __name__ == '__main__':

    plot_molineaux_and_ineichen_1996()
