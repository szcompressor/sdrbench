"""End-to-end against the real Hugging Face mirror and Globus (run with `pytest -m online`)."""
import hashlib

import numpy as np
import pytest

import sdrbench

pytestmark = pytest.mark.online
SMALL = ("exaalt", "2869440/xx.f32")  # 11 MB file from a 60 MB archive


def test_hf_download_matches_catalog(tmp_path):
    f = sdrbench.files(*SMALL)[0]
    p = sdrbench.download(*SMALL, cache=tmp_path)
    assert hashlib.sha256(p.read_bytes()).hexdigest() == f.sha256


def test_load_shape_and_dtype(tmp_path):
    x = sdrbench.load(*SMALL, cache=tmp_path)
    assert x.shape == (2869440,) and x.dtype == np.dtype("<f4")
    assert np.isfinite(x).all()


def test_globus_and_hf_give_identical_bytes(tmp_path):
    a = sdrbench.download(*SMALL, source="hf", cache=tmp_path)
    b = sdrbench.download(*SMALL, source="globus", cache=tmp_path)
    assert a.read_bytes() == b.read_bytes()


def test_multidimensional_load(tmp_path):
    x = sdrbench.load("exaalt", "copper/dataset1-5423x3137.x.f32.dat", cache=tmp_path)
    assert x.shape == (5423, 3137)
    # positions move little between consecutive time steps (rows) -> confirms C order
    assert np.abs(np.diff(x[:50], axis=0)).mean() < np.abs(np.diff(x[:50], axis=1)).mean()
