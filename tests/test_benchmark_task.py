"""A benchmark task is only useful if the right answer passes and lazy ones fail."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

TASK = Path(__file__).parent.parent / 'benchmark_task'
L = 0.428


@pytest.fixture(scope='module')
def task_dir(tmp_path_factory):
    outdir = tmp_path_factory.mktemp('task')
    subprocess.run([sys.executable, TASK / 'make_data.py', outdir], check=True)
    return outdir


def grade(results, truth):
    return subprocess.run([sys.executable, TASK / 'tests' / 'grade.py', results, truth], capture_output=True, text=True)


def attempt(task_dir, path, scale=True, mask=True, weight=True, sort=True):
    """A solution with one of the traps left in, to check the grader catches it."""
    ds = xr.open_dataset(task_dir / 'scene.nc')
    if sort:
        ds = ds.sortby('time')
    valid = (ds['red'] != -9999) & (ds['nir'] != -9999)
    red, nir = ds['red'].astype('float64'), ds['nir'].astype('float64')
    if mask:
        red, nir = red.where(valid), nir.where(valid)
    if scale:
        red, nir = red / 10000, nir / 10000
    savi = (nir - red) / (nir + red + L) * (1 + L)

    cosine = np.cos(np.deg2rad(ds['lat'])) if weight else xr.ones_like(ds['lat'], dtype='float64')
    weights = xr.ones_like(valid.isel(time=0, drop=True), dtype='float64') * cosine
    pd.DataFrame({
        'month': pd.DatetimeIndex(ds['time'].values).strftime('%Y-%m'),
        'mean_savi': savi.weighted(weights).mean(('lat', 'lon')).values,
        'valid_fraction': ((valid * weights).sum(('lat', 'lon')) / weights.sum()).values,
    }).to_csv(path, index=False)


def test_reference_solution_passes(task_dir, tmp_path):
    results = tmp_path / 'results.csv'
    subprocess.run([sys.executable, TASK / 'solution' / 'solve.py', task_dir / 'scene.nc', results], check=True)
    outcome = grade(results, task_dir / 'truth.json')
    assert outcome.returncode == 0, outcome.stdout
    assert 'PASS' in outcome.stdout


def test_correct_attempt_passes_too(task_dir, tmp_path):
    attempt(task_dir, tmp_path / 'results.csv')
    assert grade(tmp_path / 'results.csv', task_dir / 'truth.json').returncode == 0


@pytest.mark.parametrize('trap', ['scale', 'mask', 'weight', 'sort'])
def test_each_shortcut_fails(task_dir, tmp_path, trap):
    attempt(task_dir, tmp_path / 'results.csv', **{trap: False})
    outcome = grade(tmp_path / 'results.csv', task_dir / 'truth.json')
    assert outcome.returncode == 1, f'skipping {trap} should fail the grader'


def test_missing_results_file_fails_cleanly(task_dir, tmp_path):
    outcome = grade(tmp_path / 'nothing.csv', task_dir / 'truth.json')
    assert outcome.returncode == 1
    assert 'could not read' in outcome.stdout


def test_wrong_columns_fail(task_dir, tmp_path):
    results = tmp_path / 'results.csv'
    pd.DataFrame({'month': ['2022-01'], 'savi': [0.5]}).to_csv(results, index=False)
    assert grade(results, task_dir / 'truth.json').returncode == 1
