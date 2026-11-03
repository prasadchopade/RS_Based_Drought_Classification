import numpy as np
import pytest

from drought import synthetic
from drought.validation import parse_coordinates, validate_cube


def statuses(ds):
    return {check['check']: check['status'] for check in validate_cube(ds)}


@pytest.fixture(scope='module')
def good_cube():
    return synthetic.make_cube()


def test_parse_coordinates_closes_the_ring():
    ring = parse_coordinates('[[71.0, 26.5], [71.0, 26.9], [71.5, 26.9]]')
    assert ring == [[71.0, 26.5], [71.0, 26.9], [71.5, 26.9], [71.0, 26.5]]


def test_parse_coordinates_keeps_a_closed_ring_as_is():
    closed = '[[71, 26], [71, 27], [72, 27], [71, 26]]'
    assert len(parse_coordinates(closed)) == 4


@pytest.mark.parametrize('text', [
    "__import__('os').system('echo hacked')",
    '',
    'not json',
    '{"lon": 71, "lat": 26}',
    '[1, 2, 3]',
    '[[71, 26, 5], [71, 27, 5], [72, 27, 5]]',
    '[["71", "26"], ["71", "27"], ["72", "27"]]',
    '[[true, false], [71, 27], [72, 27]]',
    '[[71, 26], [71, 27]]',
    '[[200, 26], [71, 27], [72, 27]]',
    '[[71, 95], [71, 27], [72, 27]]',
    '[[NaN, 26], [71, 27], [72, 27]]',
    '[[Infinity, 26], [71, 27], [72, 27]]',
    '[[71, 26], [72, 27], [73, 28]]',
])
def test_parse_coordinates_rejects(text):
    with pytest.raises(ValueError):
        parse_coordinates(text)


def test_good_cube_has_no_failures(good_cube):
    assert 'fail' not in statuses(good_cube).values()


def test_broken_retrievals_only_warn(good_cube):
    result = statuses(good_cube)
    assert result['red_range'] == 'warn'
    assert result['nir_range'] == 'warn'


def test_wrong_crs_fails(good_cube):
    ds = good_cube.copy()
    ds['crs'] = ds['crs'].assign_attrs(epsg_code='EPSG:32643')
    assert statuses(ds)['crs'] == 'fail'


def test_missing_crs_fails(good_cube):
    assert statuses(good_cube.drop_vars('crs'))['crs'] == 'fail'


def test_latitude_out_of_range_fails(good_cube):
    ds = good_cube.assign_coords(lat=good_cube['lat'] + 80)
    assert statuses(ds)['lat'] == 'fail'


def test_uneven_grid_fails(good_cube):
    lon = good_cube['lon'].values.copy()
    lon[10] += 0.004
    assert statuses(good_cube.assign_coords(lon=lon))['lon'] == 'fail'


def test_unsorted_time_fails(good_cube):
    assert statuses(good_cube.isel(time=np.arange(good_cube.sizes['time'])[::-1]))['time'] == 'fail'


def test_duplicate_time_fails(good_cube):
    times = good_cube['time'].values.copy()
    times[1] = times[0]
    assert statuses(good_cube.assign_coords(time=times))['time'] == 'fail'


def test_short_record_warns(good_cube):
    assert statuses(good_cube.sel(time='2018'))['time'] == 'warn'


def test_raw_digital_numbers_fail(good_cube):
    ds = good_cube.copy()
    ds['red'] = ds['red'] * 10000
    assert statuses(ds)['red_range'] == 'fail'


def test_wrong_units_fail(good_cube):
    ds = good_cube.copy()
    ds['nir'] = ds['nir'].assign_attrs(units='W m-2')
    assert statuses(ds)['nir_units'] == 'fail'


def test_missing_band_fails(good_cube):
    assert statuses(good_cube.drop_vars('nir'))['nir_units'] == 'fail'


def test_mostly_empty_band_warns(good_cube):
    ds = good_cube.copy()
    ds['red'] = ds['red'].where(ds['time'] > np.datetime64('2022-01-01'))
    assert statuses(ds)['red_missing'] == 'warn'
