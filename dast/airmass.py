
import numpy as np


def airmass(sza, constituent='rayleigh'):
    coefs = {
        'rayleigh': [0.48353, 0.095846,  96.741, -1.754 ],  # noqa
        'aerosol' : [0.16851, 0.18198 ,  95.318, -1.9542],  # noqa
        'o3':       [1.0651 , 0.6379  , 101.8  , -2.2694],  # noqa
        'h2o':      [0.10648, 0.11423 ,  93.781, -1.9203],  # noqa
        'o2':       [0.65779, 0.064713,  96.974, -1.8084],  # noqa
        'ch4':      [0.49381, 0.35569 ,  98.23 , -2.1616],  # noqa
        'co':       [0.505  , 0.063191,  95.899, -1.917 ],  # noqa
        'n2o':      [0.61696, 0.060787,  96.632, -1.8279],  # noqa
        'co2':      [0.65786, 0.064688,  96.974, -1.8083],  # noqa
        'n2':       [0.38155, 8.871e-05, 95.195, -1.8053],  # noqa
        'hno3':     [1.044  , 0.78456 , 103.15 , -2.4794],  # noqa
        'no2':      [1.1212 , 1.6132  , 111.55 , -3.2629],  # noqa
        'no':       [0.77738, 0.11075 , 100.34 , -1.5794],  # noqa
        'so2':      [0.63454, 0.00992 ,  95.804, -2.0573],  # noqa
        'nh3':      [0.32101, 0.010793,  94.337, -2.0548]   # noqa
    }

    coefs['no3'] = coefs['no2'].copy()
    coefs['bro'] = coefs['o3'].copy()
    coefs['ch2o'] = coefs['n2o'].copy()
    coefs['hno2'] = coefs['hno3'].copy()
    coefs['clno'] = coefs['no2'].copy()
    coefs['ozone'] = coefs['o3'].copy()
    coefs['water'] = coefs['h2o'].copy()

    p = coefs.get(constituent.lower(), None)
    if p is None:
        raise ValueError(f'unknown constituent {constituent}')

    cosz = np.cos(np.radians(sza))
    return 1. / (cosz + p[0]*(sza**p[1])*(p[2]-sza)**p[3])
