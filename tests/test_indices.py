"""Index and anomaly maths, checked against values worked out by hand."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from drought import indices


def series(values, times=None):
    """A single pixel over time."""
    times = times if times is not None else pd.date_range('2018-07-01', periods=len(values), freq='YS-JUL')
    return xr.DataArray(
        np.array(values, dtype='float64')[:, None, None],
        dims=('time', 'lat', 'lon'),
        coords={'time': times, 'lat': [26.0], 'lon': [71.0]},
    )


def test_ndvi_known_value():
    assert indices.ndvi(xr.DataArray(0.1), xr.DataArray(0.5)).item() == pytest.approx(0.4 / 0.6, abs=1e-9)


def test_ndvi_is_missing_when_both_bands_are_zero():
    assert np.isnan(indices.ndvi(xr.DataArray(0.0), xr.DataArray(0.0)).item())


def test_savi_known_value():
    assert indices.savi(xr.DataArray(0.1), xr.DataArray(0.5)).item() == pytest.approx(0.4 * 1.428 / 1.028, abs=1e-9)


def test_mask_reflectance_drops_impossible_values():
    masked = indices.mask_reflectance(xr.DataArray([-0.02, 0.0, 0.5, 1.0, 1.3]))
    assert np.isnan(masked.values[[0, 4]]).all()
    assert masked.values[1:4].tolist() == [0.0, 0.5, 1.0]


def test_composite_takes_the_max_and_ignores_gaps():
    times = pd.to_datetime(['2020-01-01', '2020-01-11', '2020-01-21', '2020-02-01', '2020-02-11'])
    composite = indices.monthly_composite(series([np.nan, 0.3, 0.5, np.nan, np.nan], times))
    assert composite.values[0, 0, 0] == pytest.approx(0.5)
    assert np.isnan(composite.values[1, 0, 0])


def test_anomaly_matches_hand_calculation():
    # Median 0.32, MAD 0.02, so the scale is 1.4826 * 0.02
    anomaly = indices.standardized_anomaly(series([0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.10]))
    assert anomaly.values[-1, 0, 0] == pytest.approx(-0.22 / (1.4826 * 0.02), abs=1e-6)
    assert anomaly.values[2, 0, 0] == pytest.approx(0.0, abs=1e-9)


def test_anomaly_noise_floor_stops_tiny_changes_looking_extreme():
    # MAD here is about 0.001, far smaller than the sensor can really resolve
    anomaly = indices.standardized_anomaly(series([0.300, 0.301, 0.302, 0.303, 0.295]))
    assert anomaly.values[-1, 0, 0] == pytest.approx((0.295 - 0.301) / indices.NDVI_NOISE_FLOOR, abs=1e-6)


def test_vci_runs_from_0_to_100():
    vci = indices.vegetation_condition_index(series([0.2, 0.3, 0.4]))
    assert vci.values[:, 0, 0].tolist() == pytest.approx([0.0, 50.0, 100.0])


def monthly(values):
    return series(values, pd.date_range('2020-01-01', periods=len(values), freq='MS'))


def test_classify_needs_two_low_months_in_a_row():
    classes = indices.classify(monthly([0, -5, 0, 0, -2.5, -5, 0]))
    assert classes.values[:, 0, 0].tolist() == [0, 0, 0, 0, 1, 2, 0]


def test_classify_thresholds_include_their_edges():
    classes = indices.classify(monthly([-2.0, -2.0, -3.0, -3.0, -1.99, -1.99]))
    assert classes.values[:, 0, 0].tolist() == [1, 1, 2, 2, 0, 0]


def test_classify_marks_missing_data():
    classes = indices.classify(monthly([np.nan, -5, -5, np.nan]))
    assert classes.values[:, 0, 0].tolist() == [indices.NO_DATA_CLASS, 2, 2, indices.NO_DATA_CLASS]


def test_area_weights_follow_cosine_of_latitude():
    field = xr.DataArray(
        np.ones((1, 3, 1)), dims=('time', 'lat', 'lon'),
        coords={'time': [np.datetime64('2020-01-01')], 'lat': [0.0, 60.0, 80.0], 'lon': [10.0]},
    )
    weights = indices.area_weights(field)
    assert weights.values[:, 0].tolist() == pytest.approx([1.0, 0.5, np.cos(np.deg2rad(80))])


def test_regional_summary_is_area_weighted():
    # Two latitudes, cos(60) = 0.5, so the equator pixel counts twice as much
    coords = {'time': [np.datetime64('2020-01-01')], 'lat': [0.0, 60.0], 'lon': [10.0]}
    ndvi = xr.DataArray(np.array([[[0.6], [0.3]]]), dims=('time', 'lat', 'lon'), coords=coords)
    classes = xr.DataArray(np.array([[[2], [0]]], dtype='int8'), dims=('time', 'lat', 'lon'), coords=coords)
    summary = indices.regional_summary(ndvi, classes)
    assert summary['mean_ndvi'].item() == pytest.approx((0.6 * 1 + 0.3 * 0.5) / 1.5)
    assert summary['strong_fraction'].item() == pytest.approx(1 / 1.5)
    assert summary['valid_fraction'].item() == pytest.approx(1.0)


def test_regional_summary_leaves_missing_pixels_out_of_the_share():
    coords = {'time': [np.datetime64('2020-01-01')], 'lat': [0.0, 60.0], 'lon': [10.0]}
    ndvi = xr.DataArray(np.array([[[0.6], [np.nan]]]), dims=('time', 'lat', 'lon'), coords=coords)
    classes = xr.DataArray(np.array([[[2], [-1]]], dtype='int8'), dims=('time', 'lat', 'lon'), coords=coords)
    summary = indices.regional_summary(ndvi, classes)
    assert summary['mean_ndvi'].item() == pytest.approx(0.6)
    assert summary['strong_fraction'].item() == pytest.approx(1.0)
    assert summary['valid_fraction'].item() == pytest.approx(1 / 1.5)
