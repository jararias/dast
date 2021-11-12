
##############################################################################
# author: Jose A Ruiz-Arias, University of Malaga                            #
# email: jararias at uma.es, jose.ruiz-arias at solargis.com                 #
# date: 11.11.2021                                                           #
##############################################################################

import os

import numpy as np
import pylab as pl
from matplotlib.font_manager import FontProperties

from dast import solar_spectrum


SOURCE_DIR = '../results/lut_transmittance'
TARGET_DIR = '../figures'


def fuente(ttf):
    def inner(**kwargs):
        return FontProperties(fname=ttf, **kwargs)
    return inner


arial = fuente('fonts/arial.ttf')
arial_bold = fuente('fonts/arialbd.ttf')

_, _, _, Eon = solar_spectrum.read()

data = np.load(f'{SOURCE_DIR}/indep_transmittances.npz')
indep = {k: data[k] for k in data.keys()}

data = np.load(f'{SOURCE_DIR}/indep2b_transmittances.npz')
indep2b = {k: data[k] for k in data.keys()}

data = np.load(f'{SOURCE_DIR}/inter_transmittances.npz')
inter = {k: data[k] for k in data.keys()}

data = np.load(f'{SOURCE_DIR}/cinter_transmittances.npz')
cinter = {k: data[k] for k in data.keys()}

data = np.load(f'{SOURCE_DIR}/hybrid_transmittances.npz')
hybrid = {k: data[k] for k in data.keys()}


def plot_error(pressure_value, am_value):

    uo_value = 0.3

    n_pressure = np.argwhere(indep['pressure'] == pressure_value).item()
    n_uo = np.argwhere(indep['uo'] == uo_value).item()
    n_am = np.argmin(np.abs(indep['am'] - am_value)).item()

    pressure = indep['pressure'][n_pressure]
    uo = indep['uo'][n_uo]

    alpha = indep['alpha']
    beta = indep['beta']
    pw = indep['pw']

    pl.figure(1, figsize=(7, 9), dpi=600)
    pl.subplots_adjust(bottom=0.13, top=0.965, left=0.12,
                       right=0.98, wspace=0.08, hspace=0.08)

    kwargs = {
        'cmap': pl.cm.get_cmap('RdBu_r', 64),
        'vmin': -40,
        'vmax': 40
    }

    alpha_indices = (0, 1, 2)
    x = np.linspace(pw[0], pw[-1], len(pw) + 1)
    y = np.linspace(beta[0], beta[-1], len(beta) + 1)

    xticks_major = np.arange(0., 5.01, 1.)
    xticks_minor = np.arange(0.5, 4.6, 1.)
    xticklabels = [f'{x:.1f}' for x in xticks_major]
    xticklabels[-1] = None

    yticks_major = np.arange(0., 1.21, 0.2)
    yticks_minor = np.arange(.1, 1.11, 0.2)

    frame_color = '0.4'
    grid_color = '0.7'

    scheme_label = [
        'Independent',
        '2-band Independent',
        'Prescribed interdep.',
        'Hybrid'
    ]

    for n_row, ds in enumerate((indep, indep2b, cinter, hybrid)):
        residue6d = Eon*(ds['T'] - inter['T'])

        for n_col, index in enumerate(alpha_indices):

            ax = pl.subplot2grid((4, 3), (n_row, n_col), 1, 1)

            residue2d = residue6d[n_am, n_pressure, n_uo, :, :, index].T
            pc = ax.pcolormesh(x, y, residue2d, **kwargs)

            ax.xaxis.set_ticks(xticks_major, minor=False)
            ax.xaxis.set_ticklabels(xticklabels)
            ax.xaxis.set_ticks(xticks_minor, minor=True)
            ax.set_xlim(xticks_major[0], xticks_major[-1])
            pl.setp(pl.getp(ax, 'xticklabels'), fontproperties=arial(size=8))

            ax.yaxis.set_ticks(yticks_major, minor=False)
            ax.yaxis.set_ticks(yticks_minor, minor=True)
            ax.set_ylim(yticks_major[0], yticks_major[-1])
            pl.setp(pl.getp(ax, 'yticklabels'), fontproperties=arial(size=8))

            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color(frame_color)

            ax.grid(which='both', lw=0.5, color=grid_color, ls=':')

            ax.tick_params(
                which='both', color=frame_color, width=0.5,
                direction='inout', right=True, top=True,
                labelleft=False, labelbottom=False)

            font_color = '0.9' if (n_row, n_col) in ((0, 2),) else 'k'
            ax.text(
                0.5, 0.975, r'$\alpha$'f'={alpha[index]:.1f}',
                ha='center', va='top', transform=ax.transAxes,
                color=font_color, fontproperties=arial(size=8)
            )

            ax.text(
                0.98, 0.98, f"({'abcdefghijkl'[n_row*3 + n_col]})",
                ha='right', va='top', transform=ax.transAxes,
                color=font_color, fontproperties=arial_bold(size=7)
            )

            if n_col == 0:
                ax.tick_params(labelleft=True)
                ylabel = u'\u00c5'+'ngstr'+u'\u00f6'+'m\'s turbidity'
                ax.set_ylabel(ylabel, fontproperties=arial(size=9), labelpad=8)
                ax.text(
                    -0.38, 0.5, f'{scheme_label[n_row]} scheme',
                    rotation='vertical', transform=ax.transAxes,
                    ha='center', va='center', fontproperties=arial(size=9)
                )

            if n_row == 3:
                ax.tick_params(labelbottom=True)
                ax.set_xlabel(
                    'Precipitable water (cm)', labelpad=6,
                    fontproperties=arial(size=9)
                )

            if n_row == 3 and n_col == 0:
                pos = ax.get_position()
                cax = pl.axes([pos.x0, 0.055, 0.4, 0.01])
                cb = pl.colorbar(
                    pc, cax=cax, pad=0.01, orientation='horizontal'
                )
                cb.outline.set_edgecolor('0.4')
                cb.outline.set_linewidth(0.5)
                cax.tick_params(which='both', width=0.5, color='0.4')
                pl.setp(cax.get_xticklabels(), fontproperties=arial(size=8))

                cax.xaxis.set_ticks(np.arange(-40, 41, 10), minor=False)
                cax.xaxis.set_ticks(np.arange(-35, 36, 10), minor=True)
                cb.set_label(
                    r'$E_{bn}$ approximation error: '
                    r'$\hat E_{bn} - E_{bn}$, in W/m$^2$',
                    fontproperties=arial(size=8), rotation=0,
                    labelpad=4)

            if n_row == 3 and n_col == 2:
                pos = ax.get_position()
                cax = pl.axes([pos.x0 - 0.125, 0.055, 0.4, 0.01])
                cb = pl.colorbar(
                    pc, cax=cax, pad=0.01, orientation='horizontal'
                )
                cb.outline.set_edgecolor('0.4')
                cb.outline.set_linewidth(0.5)
                cax.tick_params(which='both', width=0.5, color='0.4')
                pl.setp(cax.get_xticklabels(), fontproperties=arial(size=8))

                xticks = np.r_[-2.8:2.9:0.7]
                cax.xaxis.set_ticks((Eon/100)*xticks)
                cax.xaxis.set_ticklabels([f'{x:.1f}' for x in xticks])
                cb.set_label(
                    r'Relative $E_{bn}$ approximation error: '
                    r'($\hat E_{bn} - E_{bn}$) / $E_{on}$, in %',
                    fontproperties=arial(size=8), rotation=0,
                    labelpad=4)

    title = (f'P={pressure:.0f} hPa     '
             r'u$_o$='f'{uo:.2f} atm-cm     '
             f'am={am_value:.1f}')
    pl.suptitle(title, x=0.55, y=0.988, fontproperties=arial(size=9))

    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    fname = f'{TARGET_DIR}/error_beta-pw_P{pressure:.0f}_am{am_value:.1f}.png'
    print(f'Saving to {fname}')
    pl.savefig(fname, dpi=600)
    pl.close()


if __name__ == '__main__':

    plot_error(1013., 1.2)
    plot_error(1013., 3.0)
