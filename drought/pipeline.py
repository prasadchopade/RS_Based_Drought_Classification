"""Takes a red/NIR cube, writes the drought products and a report on what happened."""
import hashlib
import json
import platform
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import indices
from .validation import validate_cube

NO_DATA_TIFF = 255
SOFTWARE = ('numpy', 'pandas', 'xarray', 'netCDF4', 'rasterio')


class InvalidInput(Exception):
    """The input failed validation, the report says which checks and why."""


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def analyse(ds):
    """Red and NIR in, monthly NDVI, SAVI, anomaly, VCI and drought class out."""
    red = indices.mask_reflectance(ds['red'])
    nir = indices.mask_reflectance(ds['nir'])

    monthly_ndvi = indices.monthly_composite(indices.ndvi(red, nir))
    monthly_savi = indices.monthly_composite(indices.savi(red, nir))
    anomaly = indices.standardized_anomaly(monthly_ndvi)

    products = xr.Dataset({
        'ndvi': monthly_ndvi.astype('float32').assign_attrs(long_name='monthly maximum NDVI', units='1'),
        'savi': monthly_savi.astype('float32').assign_attrs(long_name='monthly maximum SAVI', units='1'),
        'ndvi_anomaly': anomaly.astype('float32').assign_attrs(
            long_name='NDVI anomaly in robust standard deviations from the calendar month median', units='1'),
        'vci': indices.vegetation_condition_index(monthly_ndvi).astype('float32').assign_attrs(
            long_name='vegetation condition index', units='percent'),
        'drought_class': indices.classify(anomaly).assign_attrs(
            long_name='drought class', flag_values=[0, 1, 2], flag_meanings='no_drought mild_drought strong_drought'),
    })
    for name in products.data_vars:
        products[name].attrs['grid_mapping'] = 'crs'
    products['crs'] = ds['crs']
    products['lat'].attrs.update(ds['lat'].attrs)
    products['lon'].attrs.update(ds['lon'].attrs)
    products.attrs.update(
        Conventions='CF-1.8',
        title='Monthly drought indicators',
        source=ds.attrs.get('source', 'unknown'),
        history='Built with python -m drought run',
    )
    return products


def _write_netcdf(products, path):
    encoding = {name: {'zlib': True} for name in products.data_vars if name != 'crs'}
    encoding['drought_class'] = {'zlib': True, '_FillValue': None}
    products.to_netcdf(path, encoding=encoding)


def _write_csv(summary, path):
    table = summary.to_dataframe()
    table.insert(0, 'month', pd.DatetimeIndex(table.index).strftime('%Y-%m'))
    table.round(6).to_csv(path, index=False, lineterminator='\n')
    return table


def _write_geotiff(classes, month, path):
    import rasterio
    from rasterio.transform import from_origin

    # GeoTIFFs read north-up, so the first row has to be the northernmost
    layer = classes.sel(time=month).sortby('lat', ascending=False).sortby('lon')
    lat, lon = layer['lat'].values, layer['lon'].values
    dy, dx = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
    data = layer.values.astype('int16')
    data[data == indices.NO_DATA_CLASS] = NO_DATA_TIFF

    profile = dict(
        driver='GTiff', height=data.shape[0], width=data.shape[1], count=1, dtype='uint8',
        crs='EPSG:4326', transform=from_origin(lon[0] - dx / 2, lat[0] + dy / 2, dx, dy),
        nodata=NO_DATA_TIFF, compress='deflate',
    )
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data.astype('uint8'), 1)
        dst.update_tags(month=str(month)[:7], **{f'class_{k}': v for k, v in indices.CLASS_NAMES.items()})


def _plot_overview(summary, classes, month, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    mild, strong, none, no_data = '#fab219', '#d03b3b', '#e1e0d9', '#fcfcfb'
    ink, muted = '#52514e', '#898781'

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7.5), height_ratios=[1, 1.5], layout='constrained')

    times = pd.DatetimeIndex(summary['time'].values)
    strong_pct = summary['strong_fraction'].fillna(0).values * 100
    mild_pct = summary['mild_fraction'].fillna(0).values * 100
    top.bar(times, strong_pct, width=24, color=strong, label='Strong drought')
    top.bar(times, mild_pct, width=24, bottom=strong_pct, color=mild, label='Mild drought')
    top.set_title('Share of the area in drought each month', loc='left', color=ink)
    top.set_ylabel('% of valid area', color=ink)
    top.grid(axis='y', color=none, linewidth=0.8)
    top.set_axisbelow(True)
    top.spines[['top', 'right', 'left']].set_visible(False)
    top.tick_params(colors=muted, length=0)
    top.legend(frameon=False, loc='upper left', labelcolor=ink)

    layer = classes.sel(time=month)
    bottom.pcolormesh(
        layer['lon'], layer['lat'], layer.values,
        cmap=ListedColormap([no_data, none, mild, strong]), vmin=-1.5, vmax=2.5, shading='nearest',
    )
    bottom.set_title(f'Drought class, {str(month)[:7]}', loc='left', color=ink)
    bottom.set_xlabel('Longitude (degrees east)', color=ink)
    bottom.set_ylabel('Latitude (degrees north)', color=ink)
    bottom.set_aspect(1 / np.cos(np.deg2rad(float(layer['lat'].mean()))))
    bottom.tick_params(colors=muted, length=0)
    bottom.spines[:].set_visible(False)
    bottom.legend(
        handles=[
            Patch(color=none, label='No drought'), Patch(color=mild, label='Mild drought'),
            Patch(color=strong, label='Strong drought'),
            Patch(facecolor=no_data, edgecolor=muted, label='No valid data'),
        ],
        frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1), labelcolor=ink,
    )
    fig.savefig(path, dpi=130, facecolor='white')
    plt.close(fig)


def run(input_path, outdir, map_month=None):
    """Validate, analyse and write everything to outdir. Raises InvalidInput on a failed check."""
    input_path, outdir = Path(input_path), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    report = {
        'input': {'file': input_path.name, 'sha256': sha256_file(input_path)},
        'software': {'python': platform.python_version(), **{name: metadata.version(name) for name in SOFTWARE}},
        'parameters': {
            'savi_l': indices.SAVI_L,
            'ndvi_noise_floor': indices.NDVI_NOISE_FLOOR,
            'mild_threshold': indices.MILD_THRESHOLD,
            'strong_threshold': indices.STRONG_THRESHOLD,
            'min_months_in_a_row': 2,
        },
    }

    with xr.open_dataset(input_path) as opened:
        ds = opened.load()
    report['checks'] = validate_cube(ds)

    failed = [check for check in report['checks'] if check['status'] == 'fail']
    if failed:
        report['status'] = 'failed validation'
        (outdir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        reasons = '; '.join(f"{check['check']}: {check['detail']}" for check in failed)
        raise InvalidInput(reasons)

    products = analyse(ds)
    classes = products['drought_class']
    summary = indices.regional_summary(products['ndvi'], classes)

    if map_month is None:
        month = products['time'].values[-1]
    else:
        month = np.datetime64(f'{map_month}-01', 'ns')
        if month not in products['time'].values:
            raise ValueError(f'{map_month} is not in the data')

    csv_path = outdir / 'regional_timeseries.csv'
    _write_netcdf(products, outdir / 'drought_indices.nc')
    table = _write_csv(summary, csv_path)
    _write_geotiff(classes, month, outdir / 'drought_class.tif')
    _plot_overview(summary, classes, month, outdir / 'overview.png')

    affected = (table['mild_fraction'] + table['strong_fraction']).fillna(0)
    report['status'] = 'ok'
    report['summary'] = {
        'months': int(products['time'].size),
        'map_month': str(month)[:7],
        'pixel_months_without_data': round(float((classes == indices.NO_DATA_CLASS).mean()), 6),
        'worst_month': str(table.loc[affected.idxmax(), 'month']),
        'worst_month_area_in_drought': round(float(affected.max()), 6),
    }
    # The NetCDF bytes can differ between library versions, so fingerprint the data instead
    report['outputs'] = {
        'regional_timeseries_sha256': sha256_file(csv_path),
        'drought_class_array_sha256': hashlib.sha256(classes.values.tobytes()).hexdigest(),
    }
    (outdir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return products, report


def detection_scores(flagged, truth, valid):
    """Precision and recall for flagged pixel-months, over pixels that had data."""
    flagged, truth = flagged & valid, truth & valid
    hits = int((flagged & truth).sum())
    false_alarms = int((flagged & ~truth).sum())
    misses = int((~flagged & truth).sum())
    quiet = int((valid & ~truth).sum())
    return {
        'precision': round(hits / max(hits + false_alarms, 1), 4),
        'recall': round(hits / max(hits + misses, 1), 4),
        'false_alarm_rate': round(false_alarms / max(quiet, 1), 4),
    }
