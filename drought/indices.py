"""Vegetation indices and the monthly drought signal built on top of them."""
import numpy as np
import xarray as xr

# Same soil brightness factor as the SAVI evalscript
SAVI_L = 0.428
# Stops the anomaly blowing up in months where NDVI barely changes between years
NDVI_NOISE_FLOOR = 0.02
MILD_THRESHOLD = -2.0
STRONG_THRESHOLD = -3.0
NO_DATA_CLASS = -1
# Same numbering the Streamlit app and the SVM use
CLASS_NAMES = {0: 'no drought', 1: 'mild drought', 2: 'strong drought'}


def mask_reflectance(band):
    return band.where((band >= 0) & (band <= 1))


def ndvi(red, nir):
    total = nir + red
    return ((nir - red) / total).where(total > 0)


def savi(red, nir, L=SAVI_L):
    return (nir - red) / (nir + red + L) * (1 + L)


def monthly_composite(index):
    """Highest value per pixel per month. Cloud and haze pull NDVI down, so the max is the cleanest look."""
    return index.resample(time='1MS').max(skipna=True)


def standardized_anomaly(monthly, noise_floor=NDVI_NOISE_FLOOR):
    """How far each month sits from its usual value for that calendar month.

    Uses median and MAD because the drought years are part of the record, and a
    mean and standard deviation would get dragged towards them.
    """
    median = monthly.groupby('time.month').median(skipna=True)
    deviation = monthly.groupby('time.month') - median
    mad = abs(deviation).groupby('time.month').median(skipna=True)
    scale = (1.4826 * mad).clip(min=noise_floor)
    return (deviation.groupby('time.month') / scale).drop_vars('month')


def vegetation_condition_index(monthly):
    """Kogan's VCI, 0 is the worst this month has been at this pixel and 100 the best."""
    grouped = monthly.groupby('time.month')
    low, high = grouped.min(skipna=True), grouped.max(skipna=True)
    above_low = monthly.groupby('time.month') - low
    spread = (high - low).where(high > low)
    return 100 * (above_low.groupby('time.month') / spread).drop_vars('month')


def classify(anomaly, mild=MILD_THRESHOLD, strong=STRONG_THRESHOLD):
    """Drought has to last, so a pixel needs two low months in a row before it counts."""
    below = anomaly <= mild
    lasting = below & (below.shift(time=1, fill_value=False) | below.shift(time=-1, fill_value=False))
    classes = xr.where(lasting & (anomaly <= strong), 2, xr.where(lasting, 1, 0))
    return classes.where(anomaly.notnull(), NO_DATA_CLASS).astype('int8')


def area_weights(field):
    """Grid cells shrink towards the poles, so weight each one by cos(latitude)."""
    weights = np.cos(np.deg2rad(field['lat']))
    return xr.ones_like(field.isel(time=0, drop=True), dtype='float64') * weights


def regional_summary(ndvi_monthly, classes):
    """One row per month with area-weighted NDVI and the share of the area in each class."""
    weights = area_weights(classes)
    valid = classes != NO_DATA_CLASS
    valid_area = (valid * weights).sum(('lat', 'lon'))
    return xr.Dataset({
        'mean_ndvi': ndvi_monthly.weighted(weights).mean(('lat', 'lon')),
        'valid_fraction': valid_area / weights.sum(),
        'mild_fraction': ((classes == 1) * weights).sum(('lat', 'lon')) / valid_area.where(valid_area > 0),
        'strong_fraction': ((classes == 2) * weights).sum(('lat', 'lon')) / valid_area.where(valid_area > 0),
    })
