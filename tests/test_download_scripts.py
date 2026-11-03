"""Runs the Copernicus download scripts against a fake API, no credentials or network."""
import runpy
import sys
import types
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
MONTHS = 6 * 12


class FakeSession:
    requests = []

    def __init__(self, client=None):
        pass

    def fetch_token(self, **kwargs):
        return {'access_token': 'fake'}

    def post(self, url, json):
        FakeSession.requests.append((url, json))
        return types.SimpleNamespace(status_code=200, content=b'image')


@pytest.fixture
def fake_api(tmp_path, monkeypatch):
    FakeSession.requests = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('COPERNICUS_CLIENT_ID', 'id')
    monkeypatch.setenv('COPERNICUS_CLIENT_SECRET', 'secret')
    monkeypatch.setitem(sys.modules, 'oauthlib.oauth2', types.SimpleNamespace(BackendApplicationClient=lambda client_id: None))
    monkeypatch.setitem(sys.modules, 'requests_oauthlib', types.SimpleNamespace(OAuth2Session=FakeSession))
    return tmp_path


def time_ranges():
    for _, body in FakeSession.requests:
        window = body['input']['data'][0]['dataFilter']['timeRange']
        yield datetime.fromisoformat(window['from'].rstrip('Z')), datetime.fromisoformat(window['to'].rstrip('Z'))


def test_ndvi_script_writes_one_file_per_region_and_month(fake_api):
    runpy.run_path(str(ROOT / 'copernicus_ndvi_dataset_download.py'))
    assert len(list(fake_api.glob('*.png'))) == 4 * MONTHS
    assert len(FakeSession.requests) == 4 * MONTHS
    assert FakeSession.requests[0][0] == 'https://sh.dataspace.copernicus.eu/api/v1/process'


def test_savi_script_writes_one_file_per_month(fake_api):
    runpy.run_path(str(ROOT / 'copernicus_savi_dataset_download.py'))
    assert len(list((fake_api / 'rajasthan_savi').glob('*.png'))) == MONTHS
    assert (fake_api / 'rajasthan_savi' / 'rajasthan_savi_2018_01.png').exists()
    assert (fake_api / 'rajasthan_savi' / 'rajasthan_savi_2023_12.png').exists()


def test_time_windows_cover_the_whole_period_without_gaps(fake_api):
    runpy.run_path(str(ROOT / 'copernicus_savi_dataset_download.py'))
    windows = list(time_ranges())
    assert windows[0][0] == datetime(2018, 1, 1)
    assert windows[-1][1] == datetime(2024, 1, 1)
    assert all(previous[1] == current[0] for previous, current in zip(windows, windows[1:]))


@pytest.mark.parametrize('script', ['copernicus_ndvi_dataset_download.py', 'copernicus_savi_dataset_download.py'])
def test_scripts_need_credentials_from_the_environment(fake_api, monkeypatch, script):
    monkeypatch.delenv('COPERNICUS_CLIENT_ID')
    with pytest.raises(KeyError, match='COPERNICUS_CLIENT_ID'):
        runpy.run_path(str(ROOT / script))
