
##############################################################################
# author: Jose A Ruiz-Arias, University of Malaga                            #
# email: jararias at uma.es, jose.ruiz-arias at solargis.com                 #
# date: 11.11.2021                                                           #
##############################################################################

from importlib import import_module
from datetime import datetime

import numpy as np
import netCDF4


SOURCE_DIR = '../results/ww_transmittance'

tools = import_module('4_make_ww_data')
init_netcdf = tools.init_netcdf
add_variable = tools.add_variable

Tr = {}
wildcard = f'{SOURCE_DIR}/total_transmittance_2020????.nc4'
with netCDF4.MFDataset(wildcard) as cdf:
    lats = cdf.variables['lat'][:]
    lons = cdf.variables['lon'][:]
    Tr['inter'] = cdf.variables['inter'][:]
    Tr['indep'] = cdf.variables['indep'][:]
    Tr['indep2b'] = cdf.variables['indep2b'][:]
    Tr['interp'] = cdf.variables['interp'][:]
    Tr['hybrid'] = cdf.variables['hybrid'][:]

fname = f'{SOURCE_DIR}/total_transmittance_2020.nc4'

init_netcdf([datetime(2020, 1, 1, 0, 0, 0)], lats, lons, fname)

for tr_name, tr_values in Tr.items():
    add_variable(
        tr_name, np.nanmean(tr_values, axis=0, keepdims=True),
        {
            'description': f'total transmittance, {tr_name} integr',
            'units': '-'
        },
        fname
    )

    if tr_name == 'inter':
        continue

    diff = tr_values - Tr['inter']

    add_variable(
        f'mbe_{tr_name}', np.nanmean(diff, axis=0, keepdims=True),
        {
            'description': f'mean {tr_name} minus inter total transmittances',
            'units': '-'
        },
        fname
    )

    add_variable(
        f'sde_{tr_name}', np.nanstd(diff, axis=0, keepdims=True),
        {
            'description': (
                f'Std dev of {tr_name} minus inter total transmittances'),
            'units': '-'
        },
        fname
    )

    add_variable(
        f'p66_{tr_name}',
        np.nanpercentile(np.abs(diff), q=66, axis=0, keepdims=True),
        {
            'description': (
                f'p66 of abs of {tr_name} minus inter total transmittances'),
            'units': '-'
        },
        fname
    )

    add_variable(
        f'p90_{tr_name}',
        np.nanpercentile(np.abs(diff), q=90, axis=0, keepdims=True),
        {
            'description': (
                f'p90 of abs of {tr_name} minus inter total transmittances'),
            'units': '-'
        },
        fname
    )
