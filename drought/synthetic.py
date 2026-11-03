"""A made-up Sentinel-2 style red/NIR cube with a drought we planted ourselves.

The real dataset isn't in this repo, and checking a pipeline needs a known answer
anyway. Nothing here is real satellite data. The area matches the Rajasthan box
in the SAVI download script.
"""
import numpy as np
import pandas as pd
import xarray as xr

LAT_RANGE = (26.5, 26.9)
LON_RANGE = (71.0, 71.5)
YEARS = range(2018, 2024)

DROUGHT_START, DROUGHT_END = '2021-07-01', '2021-10-31'
DROUGHT_LAT = (26.60, 26.80)
DROUGHT_LON = (71.10, 71.35)
DROUGHT_FACTOR = 0.6


def _axis(low, high, size):
    step = (high - low) / size
    return low + step * (np.arange(size) + 0.5)


def drought_truth(times, lat, lon):
    """True where and when the planted drought is happening."""
    in_time = (times >= np.datetime64(DROUGHT_START)) & (times <= np.datetime64(DROUGHT_END))
    in_lat = (lat >= DROUGHT_LAT[0]) & (lat <= DROUGHT_LAT[1])
    in_lon = (lon >= DROUGHT_LON[0]) & (lon <= DROUGHT_LON[1])
    return in_time[:, None, None] & in_lat[None, :, None] & in_lon[None, None, :]


def make_cube(seed=42, n_lat=40, n_lon=50):
    rng = np.random.default_rng(seed)
    lat = _axis(*LAT_RANGE, n_lat)[::-1]
    lon = _axis(*LON_RANGE, n_lon)
    # Three looks a month, roughly what Sentinel-2 gets after a bad pass or two
    times = pd.DatetimeIndex([
        pd.Timestamp(year, month, day)
        for year in YEARS for month in range(1, 13) for day in (1, 11, 21)
    ])
    shape = (times.size, n_lat, n_lon)

    # Dry baseline with a monsoon green-up peaking around late August
    base = 0.18 + 0.06 * (lon - lon.min()) / np.ptp(lon)
    peak = np.exp(-0.5 * ((times.day_of_year.values - 240) / 40) ** 2)
    year_effect = dict(zip(YEARS, rng.normal(0, 0.01, len(YEARS))))
    wet_year = np.array([year_effect[year] for year in times.year])
    clean = base[None, None, :] + 0.30 * peak[:, None, None] + wet_year[:, None, None]

    index = clean + rng.normal(0, 0.01, shape)
    index = np.where(drought_truth(times.values, lat, lon), index * DROUGHT_FACTOR, index)
    index = np.clip(index, -0.95, 0.95)

    red = 0.10 + rng.normal(0, 0.01, (n_lat, n_lon)) + rng.normal(0, 0.002, shape)
    nir = red * (1 + index) / (1 - index)

    # Cloud gaps, a lot worse during the monsoon
    cloud_chance = np.where(np.isin(times.month, (7, 8, 9)), 0.4, 0.1)
    cloudy = rng.random(shape) < cloud_chance[:, None, None]
    red[cloudy] = np.nan
    nir[cloudy] = np.nan

    # A few broken retrievals, the pipeline has to catch these
    nir[rng.random(shape) < 0.0005] = 1.3
    red[rng.random(shape) < 0.0005] = -0.02

    dims = ('time', 'lat', 'lon')
    ds = xr.Dataset(
        {
            'red': (dims, red.astype('float32'), {'long_name': 'red reflectance (B04)', 'units': '1'}),
            'nir': (dims, nir.astype('float32'), {'long_name': 'near infrared reflectance (B08)', 'units': '1'}),
            'crs': ((), 0, {'epsg_code': 'EPSG:4326', 'grid_mapping_name': 'latitude_longitude'}),
        },
        coords={
            'time': times,
            'lat': ('lat', lat, {'standard_name': 'latitude', 'units': 'degrees_north'}),
            'lon': ('lon', lon, {'standard_name': 'longitude', 'units': 'degrees_east'}),
        },
        attrs={
            'Conventions': 'CF-1.8',
            'title': 'Synthetic red/NIR cube with a planted drought',
            'source': f'drought.synthetic.make_cube(seed={seed}), not real satellite data',
        },
    )
    return ds


def write_cube(ds, path):
    # Stored the way reflectance products usually are, as scaled integers with a fill value
    packed = {'dtype': 'int16', 'scale_factor': 1e-4, '_FillValue': -9999, 'zlib': True}
    ds.to_netcdf(path, encoding={'red': packed, 'nir': packed})
