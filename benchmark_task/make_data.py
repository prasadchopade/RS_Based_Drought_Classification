"""Builds scene.nc for the task, plus truth.json with the answers for the grader.

The answers come straight from the arrays generated here, not from the drought
package, so the grader doesn't depend on the code it might be used to check.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

DIVISOR = 10000
MISSING_FLAG = -9999
L = 0.428


def build(seed=7):
    rng = np.random.default_rng(seed)
    # North to south on purpose, and greener towards the equator so weighting matters
    lat = np.arange(45.0, 4.0, -2.5)
    lon = np.arange(70.0, 91.0, 2.5)
    months = pd.date_range('2022-01-01', periods=24, freq='MS')
    shape = (months.size, lat.size, lon.size)

    season = 0.08 * np.sin(2 * np.pi * (months.month.values - 4) / 12)
    index = 0.72 - 0.012 * (lat - 5.0)
    index = index[None, :, None] + season[:, None, None] + rng.normal(0, 0.01, shape)
    red = np.clip(0.08 + rng.normal(0, 0.005, shape), 0.02, None)
    nir = np.clip(red * (1 + index) / (1 - index), None, 1.0)

    raw_red = np.round(red * DIVISOR).astype('int16')
    raw_nir = np.round(nir * DIVISOR).astype('int16')
    # Gaps hit each band separately, and more often at the greener southern pixels
    gap_chance = 0.04 + 0.2 * (45.0 - lat)[None, :, None] / 40.0
    raw_red[rng.random(shape) < gap_chance] = MISSING_FLAG
    raw_nir[rng.random(shape) < gap_chance] = MISSING_FLAG
    return lat, lon, months, raw_red, raw_nir, rng


def answers(lat, lon, months, raw_red, raw_nir):
    valid = (raw_red != MISSING_FLAG) & (raw_nir != MISSING_FLAG)
    red, nir = raw_red / DIVISOR, raw_nir / DIVISOR
    savi = (nir - red) / (nir + red + L) * (1 + L)
    weights = np.broadcast_to(np.cos(np.deg2rad(lat))[None, :, None], valid.shape)
    return {
        'tolerance': {'mean_savi': 0.001, 'valid_fraction': 0.001},
        'months': [
            {
                'month': month.strftime('%Y-%m'),
                'mean_savi': float((savi[t] * weights[t])[valid[t]].sum() / weights[t][valid[t]].sum()),
                'valid_fraction': float(weights[t][valid[t]].sum() / weights[t].sum()),
            }
            for t, month in enumerate(months)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('outdir')
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    lat, lon, months, raw_red, raw_nir, rng = build(args.seed)
    (outdir / 'truth.json').write_text(json.dumps(answers(lat, lon, months, raw_red, raw_nir), indent=1))

    # Months are written out of order, the way a badly merged archive ends up
    order = rng.permutation(months.size)
    band_attrs = {'reflectance_divisor': DIVISOR, 'missing_flag': MISSING_FLAG}
    xr.Dataset(
        {
            'red': (('time', 'lat', 'lon'), raw_red[order], band_attrs),
            'nir': (('time', 'lat', 'lon'), raw_nir[order], band_attrs),
        },
        coords={'time': months[order], 'lat': lat, 'lon': lon},
    ).to_netcdf(outdir / 'scene.nc')


if __name__ == '__main__':
    main()
