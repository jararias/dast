
##############################################################################
# author: Jose A Ruiz-Arias, University of Malaga                            #
# email: jararias at uma.es, jose.ruiz-arias at solargis.com                 #
# date: 11.11.2021                                                           #
##############################################################################

import os
import sys  # noqa

import pandas as pd
import numpy as np
import netCDF4

from progress.bar import Bar

# import sunpos
from dast import spectral


SOURCE_DIR = 'atmosphere'
TARGET_DIR = '../results/ww_transmittance'


def read_netcdf(fn):
    if not os.path.exists(fn):
        raise ValueError('missing file {}'.format(fn))

    cdf_variables = {}
    cdf_attributes = {}

    with netCDF4.Dataset(fn, mode='r') as cdf:

        dimensions = list(cdf.dimensions.keys())
        variables = list(
            set(cdf.variables.keys()).difference(dimensions)
        )

        for var_name in dimensions + variables:
            var_dict = {}
            var_obj = cdf.variables.get(var_name)
            var_dict['dimensions'] = var_obj.dimensions
            var_dict['values'] = var_obj[:]
            var_dict['attributes'] = {}
            if var_name in dimensions:
                var_dict['unlimited'] = \
                    cdf.dimensions[var_name].isunlimited()
            for attr_name in var_obj.ncattrs():
                var_dict['attributes'][attr_name] = \
                    getattr(var_obj, attr_name)
            cdf_variables.update({var_name: var_dict})

        for attr_name in cdf.ncattrs():
            cdf_attributes[attr_name] = getattr(cdf, attr_name)

    return cdf_variables, cdf_attributes


def regrid(grid_x, grid_y, grid_z, x, y):
    # transformation to the segment (0,1)x(0,1)
    def normalize(v, grid):
        return (v - grid[0]) / (grid[-1] - grid[0])
    ycoords = normalize(grid_y, grid_y)
    xcoords = normalize(grid_x, grid_x)
    yinterp = normalize(y, grid_y)
    xinterp = normalize(x, grid_x)

    zvalues = grid_z
    if np.ma.is_masked(zvalues):
        zvalues = np.where(zvalues.mask, np.nan, zvalues.data)
    assert zvalues.ndim >= 2, \
        'grid_val must have at least ndim=2. Got {}'.format(zvalues.ndim)

    def clip(k, kmax):
        return np.clip(k, 0, kmax)

    j1 = ((grid_y.size - 1) * yinterp).astype(int)
    i1 = ((grid_x.size - 1) * xinterp).astype(int)
    jmax, imax = grid_y.size - 1, grid_x.size - 1
    Axy = (ycoords[clip(j1 + 1, jmax)] - ycoords[clip(j1, jmax)]) * \
        (xcoords[clip(i1 + 1, imax)] - xcoords[clip(i1, imax)])
    A11 = (ycoords[clip(j1 + 1, jmax)] - yinterp) * \
        (xcoords[clip(i1 + 1, imax)] - xinterp) / Axy
    A12 = (ycoords[clip(j1 + 1, jmax)] - yinterp) * \
        (xinterp - xcoords[clip(i1, imax)]) / Axy
    A21 = (yinterp - ycoords[clip(j1, jmax)]) * \
        (xcoords[clip(i1 + 1, imax)] - xinterp) / Axy
    A22 = (yinterp - ycoords[clip(j1, jmax)]) * \
        (xinterp - xcoords[clip(i1, imax)]) / Axy
    return (
        zvalues[..., clip(j1, jmax), clip(i1, imax)] * A11 +
        zvalues[..., clip(j1, jmax), clip(i1 + 1, imax)] * A12 +
        zvalues[..., clip(j1 + 1, jmax), clip(i1, imax)] * A21 +
        zvalues[..., clip(j1 + 1, jmax), clip(i1 + 1, imax)] * A22
    )


def read_file(fname, reduced_grid=True):
    variables, _ = read_netcdf(fname)
    units = variables.get('time').get('attributes').get('units')
    times = netCDF4.num2date(
        variables['time'].get('values'), units=units,
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True)
    lats = variables.get('lat').get('values')
    lons = variables.get('lon').get('values')

    ncvar = list(
        set(variables.keys()).difference(['time', 'lat', 'lon'])
    )[0]
    values = variables.get(ncvar).get('values')

    if ncvar == 'TO3':
        values *= 1e-3  # atm-cm
    if ncvar == 'PS':
        values *= 1e-2  # hPa
    if ncvar == 'TQV':
        values *= 1e-1  # cm

    if reduced_grid is True:
        # # 3-hourly, for testing
        # times = times[::8]
        # values = values[::8]

        # # 5-degrees
        # rlons = np.linspace(-177.5, 177.5, 72)
        # rlats = np.linspace(-85., 85., 35)
        # xlons, xlats = np.meshgrid(rlons, rlats)
        # values = regrid(lons, lats, values, xlons, xlats)
        # lons, lats = rlons, rlats

        # 2-degree
        rlons = np.linspace(-179., 179., 180)
        rlats = np.linspace(-89., 89., 90)
        xlons, xlats = np.meshgrid(rlons, rlats)
        values = regrid(lons, lats, values, xlons, xlats)
        lons, lats = rlons, rlats

        # # 1-degree
        # rlons = np.linspace(-179.5, 179.5, 360)
        # rlats = np.linspace(-89.5, 89.5, 180)
        # xlons, xlats = np.meshgrid(rlons, rlats)
        # values = regrid(lons, lats, values, xlons, xlats)
        # lons, lats = rlons, rlats

    return times, lats, lons, values


def read_atmosphere(day):
    print(f'Reading atmosphere, day {day:%Y-%m-%d}')

    fname = os.path.join(
        SOURCE_DIR, f'ozone/merra2/hourly/{day:%Y}',
        f'merra2_ozone_hourly_{day:%Y%m%d}.nc4')
    times, lats, lons, ozone = read_file(fname)

    fname = os.path.join(
        SOURCE_DIR, f'pressure/merra2/hourly/{day:%Y}',
        f'merra2_pressure_hourly_{day:%Y%m%d}.nc4')
    times, lats, lons, pressure = read_file(fname)

    fname = os.path.join(
        SOURCE_DIR, f'pwater/merra2/hourly/{day:%Y}',
        f'merra2_pwater_hourly_{day:%Y%m%d}.nc4')
    times, lats, lons, pwater = read_file(fname)

    fname = os.path.join(
        SOURCE_DIR, f'aerosol/merra2/hourly/{day:%Y}',
        f'merra2_aer_beta_hourly_{day:%Y%m%d}.nc4')
    times, lats, lons, aer_beta = read_file(fname)

    fname = os.path.join(
        SOURCE_DIR, f'aerosol/merra2/hourly/{day:%Y}',
        f'merra2_aer_ang_exp_hourly_{day:%Y%m%d}.nc4')
    times, lats, lons, aer_alpha = read_file(fname)

    fname = os.path.join(
        SOURCE_DIR,
        f'sunpos/hourly/{day:%Y}/sunpos_hourly_{day:%Y%m%d}.nc4')
    times, lats, lons, cosz = read_file(fname)
    # cosz = sunpos.regular_grid(times, lats, lons).cosz

    atmos = {
        'cosz': cosz,
        'ozone': ozone,
        'pressure': pressure,
        'pwater': pwater,
        'aer_beta': aer_beta,
        'aer_alpha': aer_alpha
    }

    return times, lats, lons, atmos


def calculate_transmittance(atmos):

    # central values for the prescribed interdependent integration scheme
    UO_CINTER = 0.3  # cm
    PR_CINTER = 1013.25  # hPa
    PW_CINTER = 1.3  # cm

    sza_max = 90.
    sza = np.degrees(np.arccos(atmos['cosz']))
    sza[sza > sza_max] = np.nan

    ozo_t = spectral.OzoneTransmittance()
    ray_t = spectral.RayleighTransmittance()
    umg_t = spectral.UMGTransmittance()
    wat_t = spectral.WaterTransmittance()
    aer_t = spectral.AerosolTransmittance()

    wvl = ozo_t.wvl
    band1 = (wvl >= 290) & (wvl <= 700)
    band2 = (wvl >= 700) & (wvl <= 4000)

    Eonl = ozo_t.Eonl[None, :]
    Eon = np.trapz(Eonl, wvl)
    Eon1 = np.trapz(Eonl[:, band1], wvl[band1])
    Eon2 = np.trapz(Eonl[:, band2], wvl[band2])

    def integrate(transmittance, band=None):
        if band is None:
            return np.trapz(Eonl*transmittance, wvl)
        return np.trapz(Eonl[:, band]*transmittance[..., band], wvl[band])

    n_times, n_lats, n_lons = sza.shape

    Tr = {}
    Tr['inter'] = np.ones((n_times, n_lats, n_lons))
    Tr['indep'] = np.ones((n_times, n_lats, n_lons))
    Tr['indep2b'] = np.ones((n_times, n_lats, n_lons))
    Tr['interp'] = np.ones((n_times, n_lats, n_lons))
    Tr['hybrid'] = np.ones((n_times, n_lats, n_lons))

    pbar = Bar('Calculating transmittances:', max=n_times)
    for k in range(n_times):
        sza_ = np.ravel(sza[k])

        # spectral transmittances
        Tol = ozo_t.series(np.ravel(atmos['ozone'][k]), sza_)
        Trl = ray_t.series(np.ravel(atmos['pressure'][k]), sza_)
        Tgl = umg_t.series(np.ravel(atmos['pressure'][k]), sza_)
        Twl = wat_t.series(np.ravel(atmos['pwater'][k]), sza_)
        Tal = aer_t.series(
            np.ravel(atmos['aer_beta'][k]),
            np.ravel(atmos['aer_alpha'][k]),
            sza_)

        uo = np.full(n_lats*n_lons, UO_CINTER)
        pr = np.full(n_lats*n_lons, PR_CINTER)
        pw = np.full(n_lats*n_lons, PW_CINTER)

        Tolp = ozo_t.series(uo, np.ravel(sza[k]))
        Trlp = ray_t.series(pr, np.ravel(sza[k]))
        Tglp = umg_t.series(pr, np.ravel(sza[k]))
        Twlp = wat_t.series(pw, np.ravel(sza[k]))

        # INTERDEPENDENT INTEGRATION SCHEME
        # It is identical to Eq (3)
        Tr['inter'][k] = np.reshape(
            integrate(Tol*Trl*Tgl*Twl*Tal) / Eon,
            (n_lats, n_lons)
        )

        # INDEPENDENT INTEGRATION SCHEME
        Tindep_ = np.ones(n_lats*n_lons)
        Tindep_ = Tindep_ * (integrate(Tol) / Eon)
        Tindep_ = Tindep_ * (integrate(Trl) / Eon)
        Tindep_ = Tindep_ * (integrate(Tgl) / Eon)
        Tindep_ = Tindep_ * (integrate(Twl) / Eon)
        Tindep_ = Tindep_ * (integrate(Tal) / Eon)
        Tr['indep'][k] = np.reshape(Tindep_, (n_lats, n_lons))
        del Tindep_

        # 2-BAND INDEPENDENT INTEGRATION SCHEME
        Tindep2b_1 = np.ones(n_lats*n_lons)
        Tindep2b_1 = Tindep2b_1 * (integrate(Tol, band1) / Eon1)
        Tindep2b_1 = Tindep2b_1 * (integrate(Trl, band1) / Eon1)
        Tindep2b_1 = Tindep2b_1 * (integrate(Tgl, band1) / Eon1)
        Tindep2b_1 = Tindep2b_1 * (integrate(Twl, band1) / Eon1)
        Tindep2b_1 = Tindep2b_1 * (integrate(Tal, band1) / Eon1)
        Tindep2b_2 = np.ones(n_lats*n_lons)
        Tindep2b_2 = Tindep2b_2 * (integrate(Tol, band2) / Eon2)
        Tindep2b_2 = Tindep2b_2 * (integrate(Trl, band2) / Eon2)
        Tindep2b_2 = Tindep2b_2 * (integrate(Tgl, band2) / Eon2)
        Tindep2b_2 = Tindep2b_2 * (integrate(Twl, band2) / Eon2)
        Tindep2b_2 = Tindep2b_2 * (integrate(Tal, band2) / Eon2)
        Tr['indep2b'][k] = np.reshape(
            (Eon1 / Eon) * Tindep2b_1 + (Eon2 / Eon) * Tindep2b_2,
            (n_lats, n_lons)
        )
        del Tindep2b_1, Tindep2b_2

        # PRESCRIBED INTERDEPENDENT INTEGRATION SCHEME
        Tcinter_ = np.ones(n_lats*n_lons)
        Tcinter_ = Tcinter_ * (integrate(Tol) / Eon)
        Tcinter_ = Tcinter_ * (
            integrate(Tolp*Trl) /
            integrate(Tolp)
        )
        Tcinter_ = Tcinter_ * (
            integrate(Tolp*Trlp*Tgl) /
            integrate(Tolp*Trlp)
        )
        Tcinter_ = Tcinter_ * (
            integrate(Tolp*Trlp*Tglp*Twl) /
            integrate(Tolp*Trlp*Tglp)
        )
        Tcinter_ = Tcinter_ * (
            integrate(Tolp*Trlp*Tglp*Twlp*Tal) /
            integrate(Tolp*Trlp*Tglp*Twlp)
        )
        Tr['interp'][k] = np.reshape(Tcinter_, (n_lats, n_lons))
        del Tcinter_

        # HYBRID INTEGRATION SCHEME
        Thybrid_1 = np.ones(n_lats*n_lons)
        Thybrid_2 = np.ones(n_lats*n_lons)

        Thybrid_1 = Thybrid_1 * (
            integrate(Tol, band1) / Eon1
        )
        Thybrid_2 = Thybrid_2 * (
            integrate(Tol, band2) / Eon2
        )

        t = Tolp
        Thybrid_1 = Thybrid_1 * (
            integrate(t*Trl, band1) / integrate(t, band1)
        )
        Thybrid_2 = Thybrid_2 * (
            integrate(t*Trl, band2) / integrate(t, band2)
        )

        t = t*Trlp
        Thybrid_1 = Thybrid_1 * (
            integrate(t*Tgl, band1) / integrate(t, band1)
        )
        Thybrid_2 = Thybrid_2 * (
            integrate(t*Tgl, band2) / integrate(t, band2)
        )

        t = t*Tglp
        Thybrid_1 = Thybrid_1 * (
            integrate(t*Twl, band1) / integrate(t, band1)
        )
        Thybrid_2 = Thybrid_2 * (
            integrate(t*Twl, band2) / integrate(t, band2)
        )

        t = t*Twlp
        Thybrid_1 = Thybrid_1 * (
            integrate(t*Tal, band1) / integrate(t, band1)
        )
        Thybrid_2 = Thybrid_2 * (
            integrate(t*Tal, band2) / integrate(t, band2)
        )
        Tr['hybrid'][k] = np.reshape(
            (Eon1 / Eon) * Thybrid_1 + (Eon2 / Eon) * Thybrid_2,
            (n_lats, n_lons)
        )
        del Thybrid_1, Thybrid_2

        pbar.next()
    print()

    return Tr


def init_netcdf(times, lats, lons, fname=None):

    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    if fname is None:
        day = times[0]
        fname = os.path.join(
            TARGET_DIR, f'total_transmittance_{day:%Y%m%d}.nc4'
        )
    print(f'Initializing netcdf: {fname}')

    with netCDF4.Dataset(fname, mode='w', format='NETCDF4_CLASSIC') as cdf:
        cdf.createDimension('time', None)
        cdf.createDimension('lat', len(lats))
        cdf.createDimension('lon', len(lons))

        var = cdf.createVariable('time', 'f', ('time'), zlib=True)
        time_units = 'hours since 1970-01-01'
        var[:] = netCDF4.date2num(times, units=time_units)
        setattr(var, 'description', 'universal time')
        setattr(var, 'units', time_units)

        var = cdf.createVariable('lat', 'f', ('lat',), zlib=True)
        var[:] = lats
        setattr(var, 'description', 'latitude')
        setattr(var, 'units', 'degrees')

        var = cdf.createVariable('lon', 'f', ('lon',), zlib=True)
        var[:] = lons
        setattr(var, 'description', 'longitude')
        setattr(var, 'units', 'degrees')

    return fname


def add_variable(varname, values, attributes, fname):

    kwargs = dict(datatype='f', dimensions=('time', 'lat', 'lon'), zlib=True)

    with netCDF4.Dataset(fname, mode='a', format='NETCDF4_CLASSIC') as cdf:
        var = cdf.createVariable(varname, **kwargs)
        var[:] = values
        for key, val in attributes.items():
            setattr(var, key, val)


ATTRIBUTES = {
    'inter': {
        'description': (
            'total transmittance, interdependent integration scheme'),
        'units': '-'
    },
    'indep': {
        'description': (
            'total transmittance, independent integration scheme'),
        'units': '-'
    },
    'indep2b': {
        'description': (
            'total transmittance, two-band independent integration scheme'),
        'units': '-'
    },
    'interp': {
        'description': (
            'total transmittance, prescribed interdependent '
            'integration scheme'),
        'units': '-'
    },
    'hybrid': {
        'description': (
            'total transmittance, hybrid scheme'),
        'units': '-'
    },
}

if __name__ == '__main__':

    start_day = sys.argv[1]  # '20200101'
    end_day = sys.argv[2]  # '20200131'

    days = pd.date_range(start_day, end_day, freq='D')
    for day in days:
        times, lats, lons, atmos = read_atmosphere(day)
        tr_dict = calculate_transmittance(atmos)
        fname = init_netcdf(times, lats, lons)
        for tr_name, tr_values in tr_dict.items():
            attributes = ATTRIBUTES[tr_name]
            add_variable(tr_name, tr_values, attributes, fname)
