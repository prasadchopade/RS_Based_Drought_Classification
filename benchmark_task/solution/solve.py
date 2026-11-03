"""Reference solution. Usage: python solve.py scene.nc results.csv"""
import sys

import numpy as np
import pandas as pd
import xarray as xr

L = 0.428

scene_path, results_path = sys.argv[1:3]
ds = xr.open_dataset(scene_path).sortby('time')

valid = (ds['red'] != ds['red'].attrs['missing_flag']) & (ds['nir'] != ds['nir'].attrs['missing_flag'])
red = ds['red'].where(valid) / ds['red'].attrs['reflectance_divisor']
nir = ds['nir'].where(valid) / ds['nir'].attrs['reflectance_divisor']
savi = (nir - red) / (nir + red + L) * (1 + L)

weights = xr.ones_like(valid.isel(time=0, drop=True), dtype='float64') * np.cos(np.deg2rad(ds['lat']))
pd.DataFrame({
    'month': pd.DatetimeIndex(ds['time'].values).strftime('%Y-%m'),
    'mean_savi': savi.weighted(weights).mean(('lat', 'lon')).values,
    'valid_fraction': ((valid * weights).sum(('lat', 'lon')) / weights.sum()).values,
}).to_csv(results_path, index=False)
