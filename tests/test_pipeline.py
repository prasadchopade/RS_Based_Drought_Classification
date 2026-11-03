"""Runs the whole pipeline on the synthetic cube, where the right answer is known."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr

from drought import pipeline, synthetic


@pytest.fixture(scope='module')
def scene(tmp_path_factory):
    path = tmp_path_factory.mktemp('input') / 'scene.nc'
    synthetic.write_cube(synthetic.make_cube(), path)
    return path


@pytest.fixture(scope='module')
def result(scene, tmp_path_factory):
    outdir = tmp_path_factory.mktemp('output')
    products, report = pipeline.run(scene, outdir, map_month='2021-08')
    return products, report, outdir


def truth_for(products):
    return synthetic.drought_truth(products['time'].values, products['lat'].values, products['lon'].values)


@pytest.mark.parametrize('seed', [42, 1, 2, 3])
def test_planted_drought_comes_back(seed):
    products = pipeline.analyse(synthetic.make_cube(seed=seed))
    classes = products['drought_class'].values
    scores = pipeline.detection_scores(classes >= 1, truth_for(products), classes != -1)
    assert scores['recall'] >= 0.85
    assert scores['precision'] >= 0.75
    assert scores['false_alarm_rate'] <= 0.01


def test_drought_lands_in_the_right_months(result):
    products, _, outdir = result
    table = pd.read_csv(outdir / 'regional_timeseries.csv').set_index('month')
    affected = table['mild_fraction'] + table['strong_fraction']
    assert set(affected.nlargest(4).index) == {'2021-07', '2021-08', '2021-09', '2021-10'}


def test_anomaly_beats_vci_on_false_alarms(result):
    # With six years, every pixel and month has a year that scores a VCI of 0 by definition
    products, _, _ = result
    classes = products['drought_class'].values
    truth, valid = truth_for(products), classes != -1
    vci = np.nan_to_num(products['vci'].values, nan=100)
    assert pipeline.detection_scores(vci < 20, truth, valid)['false_alarm_rate'] > 0.15
    assert pipeline.detection_scores(classes >= 1, truth, valid)['false_alarm_rate'] < 0.01


def test_broken_retrievals_do_not_reach_the_products(result):
    products, _, _ = result
    ndvi = products['ndvi'].values
    assert np.nanmax(ndvi) < 0.9
    assert np.nanmin(ndvi) > 0


def test_values_survive_scale_and_fill_encoding(scene):
    # Written as int16 with a scale factor, so a round trip is only good to about 1e-4
    original = synthetic.make_cube()
    with xr.open_dataset(scene) as reloaded:
        np.testing.assert_allclose(reloaded['nir'].values, original['nir'].values, atol=1e-4)
        assert np.array_equal(np.isnan(reloaded['nir'].values), np.isnan(original['nir'].values))


def test_months_with_no_data_stay_missing(result):
    products, report, _ = result
    classes = products['drought_class'].values
    assert np.array_equal(classes == -1, np.isnan(products['ndvi'].values))
    assert 0 < report['summary']['pixel_months_without_data'] < 0.1


def test_netcdf_has_cf_metadata(result):
    _, _, outdir = result
    with xr.open_dataset(outdir / 'drought_indices.nc') as ds:
        assert ds.attrs['Conventions'] == 'CF-1.8'
        assert 'not real satellite data' in ds.attrs['source']
        assert ds['ndvi'].attrs['units'] == '1'
        assert ds['vci'].attrs['units'] == 'percent'
        assert ds['drought_class'].attrs['flag_meanings'] == 'no_drought mild_drought strong_drought'
        assert ds['lat'].attrs['units'] == 'degrees_north'
        assert ds['lon'].attrs['units'] == 'degrees_east'
        assert ds['crs'].attrs['epsg_code'] == 'EPSG:4326'
        assert ds['ndvi'].attrs['grid_mapping'] == 'crs'
        assert ds.sizes['time'] == 72


def test_csv_format(result):
    _, _, outdir = result
    table = pd.read_csv(outdir / 'regional_timeseries.csv')
    assert list(table.columns) == ['month', 'mean_ndvi', 'valid_fraction', 'mild_fraction', 'strong_fraction']
    assert len(table) == 72
    assert table['month'].iloc[0] == '2018-01' and table['month'].iloc[-1] == '2023-12'
    assert table['month'].is_monotonic_increasing
    for column in ('valid_fraction', 'mild_fraction', 'strong_fraction'):
        assert table[column].between(0, 1).all()


def test_geotiff_is_georeferenced_correctly(result):
    products, _, outdir = result
    layer = products['drought_class'].sel(time='2021-08-01')
    with rasterio.open(outdir / 'drought_class.tif') as tif:
        assert tif.crs.to_epsg() == 4326
        assert tif.nodata == 255
        assert tif.tags()['month'] == '2021-08'
        assert tif.bounds.left == pytest.approx(synthetic.LON_RANGE[0], abs=1e-6)
        assert tif.bounds.right == pytest.approx(synthetic.LON_RANGE[1], abs=1e-6)
        assert tif.bounds.bottom == pytest.approx(synthetic.LAT_RANGE[0], abs=1e-6)
        assert tif.bounds.top == pytest.approx(synthetic.LAT_RANGE[1], abs=1e-6)

        # Every pixel read back by coordinate must match the array, this catches a flipped raster
        data = tif.read(1)
        for lat in layer['lat'].values:
            for lon in layer['lon'].values[::7]:
                row, col = tif.index(lon, lat)
                expected = int(layer.sel(lat=lat, lon=lon))
                assert data[row, col] == (255 if expected == -1 else expected)


def test_report_records_provenance(scene, result):
    _, report, outdir = result
    assert report['status'] == 'ok'
    assert report['input']['sha256'] == pipeline.sha256_file(scene)
    assert report['software']['xarray'] and report['software']['rasterio']
    assert report['parameters']['min_months_in_a_row'] == 2
    assert report['outputs']['regional_timeseries_sha256'] == pipeline.sha256_file(outdir / 'regional_timeseries.csv')
    assert json.loads((outdir / 'report.json').read_text()) == report


def test_same_input_gives_same_output(scene, result, tmp_path):
    _, first, _ = result
    _, second = pipeline.run(scene, tmp_path, map_month='2021-08')
    assert second['outputs'] == first['outputs']
    assert second['input'] == first['input']


def test_same_seed_same_cube_different_seed_different_cube():
    a, b, c = synthetic.make_cube(seed=5), synthetic.make_cube(seed=5), synthetic.make_cube(seed=6)
    assert a['red'].identical(b['red'])
    assert not a['red'].identical(c['red'])


def test_failed_validation_stops_the_run_and_explains(tmp_path):
    bad = synthetic.make_cube()
    bad['red'] = bad['red'] * 10000
    path = tmp_path / 'bad.nc'
    bad.to_netcdf(path)
    outdir = tmp_path / 'out'
    with pytest.raises(pipeline.InvalidInput, match='digital numbers'):
        pipeline.run(path, outdir)
    assert json.loads((outdir / 'report.json').read_text())['status'] == 'failed validation'
    assert not (outdir / 'drought_indices.nc').exists()


def test_unknown_map_month_is_an_error(scene, tmp_path):
    with pytest.raises(ValueError, match='1999-01'):
        pipeline.run(scene, tmp_path, map_month='1999-01')


def test_cli_exit_codes(scene, tmp_path):
    good = subprocess.run([sys.executable, '-m', 'drought', 'run', scene, '--outdir', tmp_path / 'ok'])
    assert good.returncode == 0

    bad = synthetic.make_cube()
    bad['nir'] = bad['nir'].assign_attrs(units='W m-2')
    bad_path = tmp_path / 'bad.nc'
    bad.to_netcdf(bad_path)
    failed = subprocess.run(
        [sys.executable, '-m', 'drought', 'run', bad_path, '--outdir', tmp_path / 'bad'],
        capture_output=True, text=True,
    )
    assert failed.returncode == 2
    assert 'nir_units' in failed.stderr
