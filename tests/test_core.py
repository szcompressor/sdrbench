import numpy as np
import pytest

import sdrbench
import sdrbench.core as core
from sdrbench.cli import main as cli


def test_listing(fake_dataset):
    assert sdrbench.datasets() == ["fake"]
    assert sdrbench.variants("fake") == ["v1"]
    assert [f.path for f in sdrbench.files("fake", arrays_only=True)] == ["v1/a.f32", "v1/sub/b.d64"]
    assert [f.path for f in sdrbench.files("fake", "*.d64")] == ["v1/sub/b.d64"]
    with pytest.raises(KeyError, match="unknown dataset"):
        sdrbench.info("nope")


def test_load_from_hf_by_name_and_path(fake_dataset):
    x = sdrbench.load("fake", "a.f32", cache=fake_dataset["cache"])
    assert x.shape == (2, 3, 4) and x.dtype == np.dtype("<f4")
    np.testing.assert_array_equal(x, fake_dataset["a"])
    y = sdrbench.load("fake", "v1/sub/b.d64", mmap=False, cache=fake_dataset["cache"])
    np.testing.assert_array_equal(y, fake_dataset["b"])
    assert fake_dataset["hf_calls"] == ["v1/a.f32", "v1/sub/b.d64"]


def test_globus_matches_hf(fake_dataset):
    g = sdrbench.download("fake", "v1/sub/b.d64", source="globus", cache=fake_dataset["cache"])
    assert g.read_bytes() == (fake_dataset["hub"] / "v1/sub/b.d64").read_bytes()
    assert g == fake_dataset["cache"] / "globus" / "fake" / "v1" / "sub" / "b.d64"
    # second call reuses the unpacked archive (would fail if it re-downloaded a missing URL)
    core.catalog()["datasets"]["fake"]["variants"]["v1"]["archive"]["url"] = "file:///nonexistent.tar.gz"
    x = sdrbench.load("fake", "a.f32", source="globus", cache=fake_dataset["cache"])
    np.testing.assert_array_equal(x, fake_dataset["a"])


def test_sha_mismatch_detected(fake_dataset):
    sdrbench.download("fake", "a.f32", source="globus", cache=fake_dataset["cache"])
    p = fake_dataset["cache"] / "globus" / "fake" / "v1" / "a.f32"
    p.write_bytes(b"\x00" * p.stat().st_size)
    with pytest.raises(IOError, match="sha256 mismatch"):
        sdrbench.download("fake", "a.f32", source="globus", cache=fake_dataset["cache"])


def test_errors(fake_dataset):
    with pytest.raises(ValueError, match="matches 3 files"):
        sdrbench.download("fake", "*.*", cache=fake_dataset["cache"])
    with pytest.raises(ValueError, match="not a raw array"):
        sdrbench.load("fake", "notes.txt", cache=fake_dataset["cache"])
    with pytest.raises(ValueError, match="source must be"):
        sdrbench.download("fake", "a.f32", source="ftp")


def test_cli(fake_dataset, capsys):
    assert cli(["list"]) == 0
    assert "fake" in capsys.readouterr().out
    assert cli(["files", "fake"]) == 0
    out = capsys.readouterr().out
    assert "v1/a.f32" in out and "2x3x4" in out
    assert cli(["download", "fake", "*.f32", "--cache", str(fake_dataset["cache"])]) == 0
    assert cli(["download", "fake", "*.nothing"]) == 1
