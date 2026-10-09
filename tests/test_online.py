"""End-to-end against the real Hugging Face mirror and Globus (run with `pytest -m online`)."""
import hashlib

import numpy as np
import pytest

import sdrbench

pytestmark = pytest.mark.online


def test_hf_download_matches_catalog(tmp_path):
    f = sdrbench.dataset("exaalt").field("xx")  # 11 MB file from a 60 MB archive
    p = f.download(cache=tmp_path, source="hf")
    assert hashlib.sha256(p.read_bytes()).hexdigest() == f.sha256


def test_default_api(tmp_path):
    ds = sdrbench.dataset("exaalt", cache=tmp_path)
    assert ds.variant == "small" and set(ds.fields) == {"xx", "yy", "zz", "vx", "vy", "vz"}
    x = ds["xx"]
    assert x.shape == (2869440,) and x.dtype == np.dtype("<f4") and np.isfinite(x).all()


def test_globus_and_hf_give_identical_bytes(tmp_path):
    f = sdrbench.dataset("exaalt").field("vx")
    assert f.download(cache=tmp_path, source="hf").read_bytes() == f.download(cache=tmp_path, source="globus").read_bytes()


def test_multidimensional_c_order(tmp_path):
    x = sdrbench.load("exaalt", "dataset1.x", variant="copper", cache=tmp_path)
    assert x.shape == (5423, 3137)
    # atoms move little between consecutive time steps (axis 0) -> C order, time slowest
    assert np.abs(np.diff(x[:50], axis=0)).mean() < np.abs(np.diff(x[:50], axis=1)).mean()


def test_card_example_runs(tmp_path, monkeypatch, capsys):
    """The usage example on the Hugging Face dataset card works as written."""
    pytest.importorskip("pysz")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sync"))
    import sync
    monkeypatch.setenv("SDRBENCH_CACHE", str(tmp_path))
    d = sdrbench.catalog()["datasets"]["exaalt"]
    exec(compile(sync.card_python("exaalt", d), "card", "exec"), {})
    assert "ratio" in capsys.readouterr().out


def test_save_to_directory_from_hf(tmp_path):
    f = sdrbench.dataset("exaalt").field("vy")
    p = f.download(tmp_path / "data", source="hf")
    assert p == tmp_path / "data" / "2869440" / "vy.f32" and p.is_file() and not p.is_symlink()
    assert hashlib.sha256(p.read_bytes()).hexdigest() == f.sha256


def test_qmcpack_default_is_preconditioned(tmp_path):
    """Default QMCPACK variant = (288, 115, 69, 69): the stored native file, transposed."""
    pre = sdrbench.dataset("qmcpack", cache=tmp_path)
    assert pre.variant == "preconditioned" and pre.variants == ["preconditioned", "original"]
    x = pre["einspline"]
    native = sdrbench.dataset("qmcpack", "original", cache=tmp_path)["einspline"]
    assert x.shape == (288, 115, 69, 69) and native.shape == (115, 69, 69, 288)
    assert x.flags.c_contiguous and np.array_equal(x, np.asarray(native).transpose(3, 0, 1, 2))
    # a plain reshape of the stored bytes would NOT give this
    assert not np.array_equal(x, np.asarray(native).reshape(288, 115, 69, 69))
