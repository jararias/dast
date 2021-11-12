
##############################################################################
# author: Jose A Ruiz-Arias, University of Malaga                            #
# email: jararias at uma.es, jose.ruiz-arias at solargis.com                 #
# date: 11.11.2021                                                           #
##############################################################################

from importlib import import_module

import numpy as np  # noqa
import pylab as pl
import cartopy.feature as cfeat
import cartopy.crs as ccrs
import netCDF4

from matplotlib.font_manager import FontProperties


SOURCE_DIR = '../results/ww_transmittance'
TARGET_DIR = '../figures'

tools = import_module('4_make_ww_data')


def fuente(ttf):
    def inner(**kwargs):
        return FontProperties(fname=ttf, **kwargs)
    return inner


arial = fuente('fonts/arial.ttf')
arial_bold = fuente('fonts/arialbd.ttf')


def symmetric_colormap(pc):
    vmin, vmax = pc.get_clim()
    vmax = max(abs(vmin), abs(vmax))
    pc.set_clim(-vmax, vmax)


def plot_instantaneous_error_map(time_index=12):
    variables, _ = tools.read_netcdf(
        f'{SOURCE_DIR}/total_transmittance_20200701.nc4'
    )

    times = netCDF4.num2date(
        variables['time']['values'],
        units=variables['time']['attributes']['units'],
        only_use_cftime_datetimes=True,
        only_use_python_datetimes=True
    )
    lons = variables['lon']['values']
    lats = variables['lat']['values']
    Tinter = variables['inter']['values'][time_index]

    diff = {}
    diff['indep'] = variables['indep']['values'][time_index] - Tinter
    diff['indep2b'] = variables['indep2b']['values'][time_index] - Tinter
    diff['interp'] = variables['interp']['values'][time_index] - Tinter
    diff['hybrid'] = variables['hybrid']['values'][time_index] - Tinter

    pl.figure('instantaneous map', figsize=(4.0, 7.4), dpi=600)
    pl.subplots_adjust(bottom=.02, top=.98, left=.03, right=.98, hspace=.15)
    print(times[time_index])

    title = {
        'indep': 'Independent scheme',
        'indep2b': '2-band independent scheme',
        'interp': 'Prescribed interdependent scheme',
        'hybrid': 'Hybrid scheme'
    }

    majorticks = {
        'indep': np.arange(-4, 4.1, 2),
        'indep2b': np.arange(-1.5, 1.51, 0.5),
        'interp': np.arange(-0.9, 1., 0.3),
        'hybrid': np.arange(-0.3, 0.31, 0.1)
    }

    minorticks = {
        'indep': [],
        'indep2b': [],
        'interp': [],
        'hybrid': []
    }

    cmap = pl.cm.get_cmap('RdBu_r')
    tr_names = ('indep', 'indep2b', 'interp', 'hybrid')
    for k, tr_name in enumerate(tr_names):
        z = 100 * diff[tr_name]
        # print(f'mean d{tr_name}={np.nanpercentile(np.abs(z), 90)}')
        ax = pl.subplot(4, 1, k+1, projection=ccrs.PlateCarree())
        pc = ax.pcolormesh(lons, lats, z, cmap=cmap)
        symmetric_colormap(pc)
        ax.coastlines(color='0.4', linewidth=0.3)
        ax.set_global()

        cb = pl.colorbar(pc, pad=0.02)
        cb.outline.set_edgecolor('0.4')
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(which='both', width=0.5, color='0.4')
        pl.setp(cb.ax.get_yticklabels(), fontproperties=arial(size=7))
        cb.ax.yaxis.set_ticks(majorticks[tr_name], minor=False)
        cb.ax.yaxis.set_ticks(minorticks[tr_name], minor=True)
        cb.set_label(
            r'$E_{bn}$ approximation error (%)', rotation=270,
            fontproperties=arial(size=8), labelpad=14)

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('0')

        ax.text(0.01, 1.01, title[tr_name], transform=ax.transAxes,
                ha='left', va='bottom', fontproperties=arial(size=8))

        ax.text(0.99, 1.01, times[time_index].strftime('%Y-%m-%d %H:%M UTC'),
                transform=ax.transAxes, ha='right', va='bottom',
                fontproperties=arial(size=4))

    pl.tight_layout()
    pl.savefig(f'{TARGET_DIR}/instantaneous_error_map.png', dpi=600)
    pl.close()


def plot_mean_error_map():
    variables, _ = tools.read_netcdf(
        f'{SOURCE_DIR}/total_transmittance_2020.nc4'
    )

    data = {}
    lats = variables.get('lat').get('values')
    lons = variables.get('lon').get('values')
    for tr_name in ('indep', 'indep2b', 'interp', 'hybrid'):
        data[tr_name] = variables[tr_name]['values'][0]
        for stat in ('mbe', 'sde', 'p66', 'p90'):
            label = f'{stat}_{tr_name}'
            data[label] = variables[label]['values'][0]

    pl.figure('mbe', figsize=(8, 3.7), dpi=600)
    pl.subplots_adjust(bottom=.01, top=.98, left=0, right=.98,
                       hspace=.2, wspace=.15)

    title = {
        'indep': 'Independent scheme',
        'indep2b': '2-band independent scheme',
        'interp': 'Prescribed interdependent scheme',
        'hybrid': 'Hybrid scheme'
    }

    majorticks = {
        'indep': np.arange(-1.5, 1.51, 0.5),
        'indep2b': np.arange(-0.6, 0.61, 0.2),
        'interp': np.arange(-0.6, 0.61, 0.2),
        'hybrid': np.arange(-0.18, 0.2, 0.06)
    }

    minorticks = {
        'indep': [],
        'indep2b': [],
        'interp': [],
        'hybrid': []
    }

    cmap = pl.cm.get_cmap('RdBu_r')

    for k, tr_name in enumerate(('indep', 'indep2b', 'interp', 'hybrid')):
        z = 100 * data[f'mbe_{tr_name}']  # (\hat Ebn - Ebn) / Eon, en %
        # print(f'{tr_name}: {np.nanpercentile(np.abs(z), 90)}')

        ax = pl.subplot(2, 2, k+1, projection=ccrs.PlateCarree())
        pc = ax.pcolormesh(lons, lats, z, cmap=cmap)
        symmetric_colormap(pc)
        ax.add_feature(cfeat.OCEAN, zorder=100, linewidth=0.3,
                       edgecolor='0.4', facecolor='w')
        ax.coastlines(color='0.4', linewidth=0.3)
        ax.set_global()

        cb = pl.colorbar(pc, pad=0.02)
        cb.outline.set_edgecolor('0.4')
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(which='both', width=0.5, color='0.4')
        pl.setp(cb.ax.get_yticklabels(), fontproperties=arial(size=7))
        cb.ax.yaxis.set_ticks(majorticks[tr_name], minor=False)
        cb.ax.yaxis.set_ticks(minorticks[tr_name], minor=True)
        cb.set_label(
            r'$E_{bn}$ approximation error (%)', rotation=270,
            fontproperties=arial(size=8), labelpad=14)

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('0')

        ax.text(0.01, 1.01, title[tr_name], transform=ax.transAxes,
                ha='left', va='bottom', fontproperties=arial(size=8))

    pl.tight_layout()
    pl.savefig(f'{TARGET_DIR}/mean_error_map.png', dpi=600)
    pl.close()


def plot_error_distribution():

    def fit_predict(distr_name, x_fit, x_predict):
        from scipy import stats
        # distr_name = 'laplace', 'cauchy', 'laplace_asymmetric'...
        distr = getattr(stats, distr_name)
        pdf = distr.pdf(x_predict, *distr.fit(x_fit))
        return pdf

    tr_names = ('indep', 'indep2b', 'interp', 'hybrid')

    data = {}
    wildcard = f'{SOURCE_DIR}/total_transmittance_2020????.nc4'
    with netCDF4.MFDataset(wildcard) as cdf:
        inter = cdf.variables['inter'][:]
        for tr_name in tr_names:
            print(f'Reading {tr_name}...')
            values = np.ravel(cdf.variables[tr_name][:] - inter)
            data[tr_name] = np.compress(~np.isnan(values), values)

    # q = np.arange(0, 101, 10)
    # print(' '*8 + '  '.join([f'{x: 5.0f}' for x in q]))
    # for tr_name in tr_names:
    #     z = 100 * data.get(tr_name)
    #     p = np.percentile(np.abs(z), q)
    #     print(f'{tr_name:8s}' + '  '.join([f'{x: 5.2f}' for x in p]))

    print('Calculating PDFs...')
    pdf_data = {}
    for tr_name in tr_names:
        z = 100*data.get(tr_name)
        pdf, bin_edges = np.histogram(
            z, bins=np.linspace(-1.5, 1.5, 500), density=True
        )
        bin_centers = bin_edges[:-1] + 0.5*np.diff(bin_edges)
        pdf_data[tr_name] = (bin_centers, pdf)

    # np.savez_compressed('pdf_data', **pdf_data)
    # print('Reading PDFs...')
    # pdf_data = dict(np.load('pdf_data.npz'))

    pl.figure('error distribution', figsize=(4, 3), dpi=600)
    pl.subplots_adjust(left=.13, right=.965, bottom=.155, top=.98)

    label = {
        'indep': r'\textsc{indep}',
        'indep2b': r'\textsc{indep-2b}',
        'interp': r'\textsc{inter-p}',
        'hybrid': r'\textsc{hybrid}'
    }

    for tr_name in tr_names:
        ax = pl.subplot(111)
        bin_centers, pdf = pdf_data[tr_name]
        zorder = 100 if tr_name == 'hybrid' else 101
        kwargs = dict(ls='-', lw=0.8, marker='', label=label[tr_name])
        line, = ax.plot(bin_centers, pdf, zorder=zorder, **kwargs)
        if tr_name == 'hybrid':
            ax.fill_between(bin_centers, pdf, zorder=0.99, ec='none',
                            fc=line.get_color(), alpha=0.2)
        ax.set_ylim(0, 13)
        ax.set_xlim(left=-1.5, right=1.5)
        pl.setp(ax.get_xticklabels(), fontproperties=arial(size=9))
        pl.setp(ax.get_yticklabels(), fontproperties=arial(size=9))

        ax.tick_params(width=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        ax.set_xlabel(r'$E_{bn}$ approximation error (%)',
                      labelpad=6, fontproperties=arial(size=10))
        ax.set_ylabel('Probability Distribution',
                      labelpad=6, fontproperties=arial(size=10))

    pl.matplotlib.rcParams['text.usetex'] = True
    pl.legend(fontsize=8, frameon=False)
    pl.matplotlib.rcParams['text.usetex'] = False

    pl.savefig(f'{TARGET_DIR}/error_distribution.png', dpi=600)
    pl.close()


if __name__ == '__main__':

    # plot_instantaneous_error_map()
    # plot_mean_error_map()
    plot_error_distribution()
