"""Drives the real Streamlit app with a small stand-in model, no network involved."""
import io
from pathlib import Path

import gdown
import joblib
import numpy as np
import pytest
import streamlit as st
from PIL import Image
from sklearn.svm import SVC
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parent.parent / 'app.py')
FEATURES = 512 * 512 * 3
GOOD_COORDINATES = '[[71.0, 26.5], [71.0, 26.9], [71.5, 26.9]]'


@pytest.fixture(scope='module')
def stand_in_model(tmp_path_factory):
    path = tmp_path_factory.mktemp('model') / 'svm_model.pkl'
    X = np.random.default_rng(0).random((3, FEATURES))
    joblib.dump(SVC(kernel='linear').fit(X, [0, 1, 2]), path)
    return path


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ('COPERNICUS_CLIENT_ID', 'COPERNICUS_CLIENT_SECRET', 'MODEL_SHA256'):
        monkeypatch.delenv(name, raising=False)
    st.cache_resource.clear()
    return tmp_path


@pytest.fixture
def app(workdir, stand_in_model):
    (workdir / 'svm_model.pkl').write_bytes(stand_in_model.read_bytes())
    return AppTest.from_file(APP, default_timeout=60).run()


def image_bytes(mode, fmt='PNG'):
    buffer = io.BytesIO()
    Image.new(mode, (64, 64), color=128 if mode == 'L' else None).save(buffer, format=fmt)
    return buffer.getvalue()


def retrieve(app, text):
    app.radio[0].set_value('Use Coordinates').run()
    app.text_input[0].set_value(text).run()
    return app.button[0].click().run()


def test_app_starts_with_a_good_model(app):
    assert not app.exception
    assert not app.error
    assert app.success[0].value == 'Model loaded successfully!'
    assert app.title[0].value == 'Drought Image Classification App'


def test_failed_download_shows_an_error_instead_of_crashing(workdir, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError('no network')
    monkeypatch.setattr(gdown, 'download', fail)
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert 'no network' in app.error[0].value
    assert not app.title


def test_model_with_wrong_checksum_is_never_loaded(workdir, stand_in_model, monkeypatch):
    (workdir / 'svm_model.pkl').write_bytes(stand_in_model.read_bytes())
    monkeypatch.setenv('MODEL_SHA256', '0' * 64)
    def fail(*args, **kwargs):
        raise AssertionError('should not unpickle')
    monkeypatch.setattr(joblib, 'load', fail)
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert 'does not match MODEL_SHA256' in app.error[0].value


@pytest.mark.parametrize('mode', ['RGB', 'RGBA', 'L', 'P'])
def test_any_image_mode_gets_a_prediction(app, mode):
    app.file_uploader[0].set_value(('scene.png', image_bytes(mode), 'image/png')).run()
    assert not app.exception
    assert not app.error
    assert any(m.value.startswith('Prediction: **') for m in app.markdown)


def test_corrupt_upload_shows_an_error(app):
    app.file_uploader[0].set_value(('scene.png', b'not an image', 'image/png')).run()
    assert not app.exception
    assert app.error


def test_code_in_the_coordinates_box_never_runs(app, workdir, monkeypatch):
    monkeypatch.setenv('COPERNICUS_CLIENT_ID', 'x')
    monkeypatch.setenv('COPERNICUS_CLIENT_SECRET', 'x')
    retrieve(app, "__import__('pathlib').Path('hacked').touch()")
    assert not app.exception
    assert 'Coordinates must look like' in app.error[0].value
    assert not (workdir / 'hacked').exists()


def test_out_of_range_coordinates_are_rejected(app):
    retrieve(app, '[[200, 26], [71, 27], [72, 27]]')
    assert 'Out of range' in app.error[0].value


def test_missing_credentials_are_explained(app):
    retrieve(app, GOOD_COORDINATES)
    assert 'COPERNICUS_CLIENT_ID' in app.error[0].value


def test_end_date_before_start_date_is_rejected(app, monkeypatch):
    monkeypatch.setenv('COPERNICUS_CLIENT_ID', 'x')
    monkeypatch.setenv('COPERNICUS_CLIENT_SECRET', 'x')
    app.radio[0].set_value('Use Coordinates').run()
    app.text_input[0].set_value(GOOD_COORDINATES).run()
    app.date_input[0].set_value('2024-02-01').run()
    app.date_input[1].set_value('2024-01-01').run()
    app.button[0].click().run()
    assert "can't be before" in app.error[0].value
