"""Input checks. Bad coordinates or units should stop a run, not quietly skew it."""
import json

import numpy as np

REFLECTANCE_UNITS = ('1', 'reflectance')
MIN_YEARS_PER_MONTH = 3
MAX_MISSING_FRACTION = 0.5


def parse_coordinates(text):
    """Turn '[[lon, lat], ...]' into a closed polygon ring, or raise ValueError."""
    try:
        points = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        raise ValueError('Coordinates must look like [[71.0, 26.5], [71.0, 26.9], [71.5, 26.9]]') from None
    if not isinstance(points, list):
        raise ValueError('Coordinates must be a list of [lon, lat] pairs')

    ring = []
    for point in points:
        is_pair = isinstance(point, list) and len(point) == 2
        if not is_pair or not all(type(value) in (int, float) for value in point):
            raise ValueError(f'Not a [lon, lat] pair: {point}')
        lon, lat = point
        # NaN fails these comparisons too, json.loads lets it through
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError(f'Out of range, expected [lon, lat] in degrees: {point}')
        ring.append([float(lon), float(lat)])

    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    if len(ring) < 4:
        raise ValueError('A polygon needs at least 3 different points')

    # Shoelace formula, zero area means the points are all on one line
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(ring, ring[1:]))
    if area == 0:
        raise ValueError('These points do not enclose any area')
    return ring


def _result(check, status, detail):
    return {'check': check, 'status': status, 'detail': detail}


def _check_axis(values, name, low, high):
    if values.size < 2 or not np.all(np.isfinite(values)):
        return _result(name, 'fail', 'needs at least 2 finite values')
    if values.min() < low or values.max() > high:
        return _result(name, 'fail', f'values run from {values.min()} to {values.max()}, expected {low} to {high}')
    steps = np.diff(values)
    if not (np.all(steps > 0) or np.all(steps < 0)):
        return _result(name, 'fail', 'must be strictly increasing or decreasing')
    if not np.allclose(steps, steps[0], rtol=1e-3, atol=0):
        return _result(name, 'fail', 'spacing is not regular')
    return _result(name, 'pass', f'{values.size} values, step {abs(steps[0]):.5f} degrees')


def _check_crs(ds):
    epsg = ds['crs'].attrs.get('epsg_code') if 'crs' in ds.variables else ds.attrs.get('crs')
    if str(epsg).upper() in ('EPSG:4326', '4326'):
        return _result('crs', 'pass', 'EPSG:4326')
    return _result('crs', 'fail', f'expected EPSG:4326 lat/lon, found {epsg}')


def _check_time(ds):
    times = ds['time'].values
    if times.size < 2 or not np.issubdtype(times.dtype, np.datetime64):
        return _result('time', 'fail', 'needs at least 2 decoded timestamps')
    if np.any(np.diff(times) <= np.timedelta64(0, 'ns')):
        return _result('time', 'fail', 'timestamps must be sorted with no duplicates')

    # An anomaly needs a few years of the same calendar month to compare against
    months = times.astype('datetime64[M]').astype(int)
    years_per_month = [len({month // 12 for month in months if month % 12 == m}) for m in range(12)]
    thinnest = min(years_per_month)
    if thinnest < MIN_YEARS_PER_MONTH:
        return _result('time', 'warn', f'a calendar month only has {thinnest} year(s) of data, anomalies will be weak')
    return _result('time', 'pass', f'{times.size} timestamps, at least {thinnest} years per calendar month')


def _check_band(ds, name):
    if name not in ds:
        return [_result(f'{name}_units', 'fail', 'band is missing')]
    band = ds[name]
    units = band.attrs.get('units')
    if str(units) not in REFLECTANCE_UNITS:
        return [_result(f'{name}_units', 'fail', f"expected reflectance (units '1'), found {units}")]

    values = band.values
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return [_result(f'{name}_range', 'fail', 'band has no valid data')]
    if np.median(valid) > 1.5:
        return [_result(f'{name}_range', 'fail', 'median is above 1.5, this looks like raw digital numbers')]

    bad = np.mean((valid < 0) | (valid > 1))
    missing = 1 - valid.size / values.size
    checks = [_result(f'{name}_range', 'warn' if bad else 'pass', f'{bad:.4%} of pixels outside 0-1, these get masked')]
    status = 'warn' if missing > MAX_MISSING_FRACTION else 'pass'
    checks.append(_result(f'{name}_missing', status, f'{missing:.1%} missing'))
    return checks


def validate_cube(ds):
    """Check a red/NIR cube. Returns one result per check, status is pass, warn or fail."""
    results = [_check_crs(ds)]
    results.append(_check_axis(ds['lat'].values, 'lat', -90, 90) if 'lat' in ds else _result('lat', 'fail', 'missing'))
    results.append(_check_axis(ds['lon'].values, 'lon', -180, 180) if 'lon' in ds else _result('lon', 'fail', 'missing'))
    results.append(_check_time(ds) if 'time' in ds else _result('time', 'fail', 'missing'))
    for band in ('red', 'nir'):
        results.extend(_check_band(ds, band))
    return results
