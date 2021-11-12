
##############################################################################
# author: Jose A Ruiz-Arias, University of Malaga                            #
# email: jararias at uma.es, jose.ruiz-arias at solargis.com                 #
# date: 11.11.2021                                                           #
##############################################################################

import os
import warnings
import subprocess
from string import Template

import numpy as np
import pandas as pd


SMARTS_SRC = 'smarts295.f'
SMARTS_EXE = 'smarts.exe'


def build(smarts_dir):

    if not os.path.exists(smarts_dir):
        msg = f'missing directory {smarts_dir}'
        this_dir = os.path.dirname(__file__)
        smarts_dir = os.path.join(this_dir, smarts_dir)
        warnings.warn(f'{msg}. Trying {smarts_dir}')
        if not os.path.exists(smarts_dir):
            raise ValueError(f'missing directory {smarts_dir}')

    source_dir = os.path.join(smarts_dir, 'Source_code')
    if not os.path.exists(source_dir):
        raise ValueError(f'missing directory {source_dir}')

    source_file = os.path.join(source_dir, SMARTS_SRC)
    if not os.path.exists(source_file):
        raise ValueError(f'missing source code {source_file}')

    print('Building SMARTS2...', end='', flush=True)
    compile_cmd = f'gfortran -o {SMARTS_EXE} {source_file}'
    subprocess.run(compile_cmd.split(), capture_output=True)

    for dir_name in ('Albedo', 'Gases', 'Solar'):
        if os.path.exists(f'./{dir_name}'):
            os.remove(f'./{dir_name}')
        os.symlink(os.path.join(smarts_dir, dir_name), f'./{dir_name}')
    print(' done!')


def clean():
    for out_type in ('inp', 'out', 'ext', 'spc'):
        fname = f'smarts295.{out_type}.txt'
        if os.path.exists(fname):
            os.remove(fname)


def wipeout():
    clean()
    if os.path.exists(SMARTS_EXE):
        os.remove(SMARTS_EXE)
    for link_name in ('Albedo', 'Gases', 'Solar'):
        if os.path.exists(link_name):
            os.remove(link_name)


def run(template, clean_outputs=True, **kwargs):

    template = Template(template)
    smarts_inp = template.safe_substitute(**kwargs)

    print(smarts_inp, file=open('smarts295.inp.txt', 'w'))
    subprocess.run(f'./{SMARTS_EXE}', capture_output=True)

    data = pd.read_csv('smarts295.ext.txt', delimiter='\s+', index_col=0)
    data = data.rename(
        columns={
            'Extraterrestrial_spectrm': 'Eonl',
            'Direct_normal_irradiance': 'Ebnl',
            'RayleighScat_trnsmittnce': 'Trl',
            'Ozone_totl_transmittance': 'Tol',
            'Trace_gas__transmittance': 'Tcl',
            'WaterVapor_transmittance': 'Twl',
            'Mixed_gas__transmittance': 'Tgl',
            'Aerosol_tot_transmittnce': 'Tal',
            'Direct_rad_transmittance': 'Ttl'
        }
    )
    data.index.name = 'wvl'

    data['Tgl'] = data['Tgl'] * data['Tcl']
    data.pop('Tcl')

    if clean_outputs is True:
        clean()

    return data


def run_batch(template, batch_param, atmos, wvl=None):
    kwargs = atmos.copy()

    res = {}
    name = list(batch_param.keys())[0]
    values = batch_param.get(name)
    for k, value in enumerate(values):
        print(
            f'Running SMARTS2 with {name}={value:g} '
            f'(run {k+1} of {len(values)})'
        )
        kwargs.update({name: value})
        r = run(template, **kwargs)
        wvl = r.index if wvl is None else wvl
        for tr_name in ('Tol', 'Trl', 'Tgl', 'Twl', 'Tal'):
            if tr_name not in res:
                res[tr_name] = np.zeros((len(values), len(wvl)))
            res[tr_name][k, :] = np.interp(wvl, r.index, r[tr_name])
    return res, wvl
