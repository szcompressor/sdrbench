import numpy as np
import pytest

import sdrbench
import sdrbench.core as core
from sdrbench.cli import main as cli


def test_list_and_fields(fake_dataset):
    assert sdrbench.list() == ["fake"]
    ds = sdrbench.dataset("fake")
    assert ds.variant == "v1" and ds.variants == ["v1", "v2", "v1t"]
    assert ds.fields == ["a", "sub/b"]                 # suffixes stripped, text files are not fields
    assert [f.name for f in ds.files] == ["a", "sub/b", "notes.txt"]
    assert len(ds) == 2 and "a" in ds and list(ds) == ["a", "sub/b"]
    assert "2 fields" in repr(ds)
    assert sdrbench.dataset("fake", "v2").fields == ["a32", "a64"]       # names come from the catalog
    assert sdrbench.dataset("fake/v2").variant == "v2"


def test_getitem_returns_c_order_array(fake_dataset):
    ds = sdrbench.dataset("fake", cache=fake_dataset["cache"])
    x = ds["a"]
    assert isinstance(x, np.memmap) and x.shape == (2, 3, 4) and x.dtype == np.dtype("<f4")
    np.testing.assert_array_equal(x, fake_dataset["a"])
    np.testing.assert_array_equal(sdrbench.load("fake", "sub/b", mmap=False), fake_dataset["b"])
    # file name and repo path work too
    assert ds.field("a.f32") is ds.field("a") and ds.field("v1/sub/b.d64").name == "sub/b"
    assert dict(ds.items()).keys() == {"a", "sub/b"}


def test_fallback_to_globus_when_hf_fails(fake_dataset, monkeypatch):
    def broken(f, cache):
        raise OSError("HF is down")
    monkeypatch.setattr(core, "_download_hf", broken)
    with pytest.warns(UserWarning, match="falling back"):
        x = sdrbench.load("fake", "a", cache=fake_dataset["cache"])
    np.testing.assert_array_equal(x, fake_dataset["a"])
    assert (fake_dataset["cache"] / "globus" / "fake" / "v1" / "a.f32").exists()
    with pytest.raises(OSError, match="HF is down"):
        sdrbench.dataset("fake").field("a").download(source="hf")


def test_globus_bytes_equal_hf(fake_dataset):
    f = sdrbench.dataset("fake").field("sub/b")
    g = f.download(fake_dataset["cache"], source="globus")
    assert g.read_bytes() == (fake_dataset["hub"] / "v1/sub/b.d64").read_bytes()
    # archive is unpacked once and reused
    core.catalog()["datasets"]["fake"]["variants"]["v1"]["archive"]["url"] = "file:///nonexistent.tar.gz"
    assert sdrbench.dataset("fake").field("a").download(fake_dataset["cache"], source="globus").exists()


def test_globus_sha_mismatch_detected(fake_dataset):
    f = sdrbench.dataset("fake").field("a")
    p = f.download(fake_dataset["cache"], source="globus")
    p.write_bytes(b"\x00" * p.stat().st_size)
    with pytest.raises(IOError, match="sha256 mismatch"):
        f.download(fake_dataset["cache"], source="globus")


def test_errors(fake_dataset):
    with pytest.raises(KeyError, match="unknown dataset"):
        sdrbench.dataset("nope")
    with pytest.raises(KeyError, match="no variant"):
        sdrbench.dataset("fake", "v9")
    with pytest.raises(KeyError, match="no field 'zzz'"):
        sdrbench.dataset("fake")["zzz"]
    with pytest.raises(ValueError, match="not a raw array"):
        sdrbench.dataset("fake").field("notes.txt").load()
    with pytest.raises(ValueError, match="source must be"):
        sdrbench.dataset("fake").field("a").download(source="ftp")


def test_cli(fake_dataset, capsys):
    assert cli(["list"]) == 0
    assert "fake/v1" in capsys.readouterr().out
    assert cli(["info", "fake"]) == 0
    out = capsys.readouterr().out
    assert "2x3x4" in out and "<f4" in out
    assert cli(["download", "fake", "a", "--cache", str(fake_dataset["cache"])]) == 0
    assert cli(["download", "fake", "nope"]) == 1


def test_derived_transposed_variant(fake_dataset, monkeypatch):
    t = sdrbench.dataset("fake", "v1t", cache=fake_dataset["cache"])["a"]
    assert t.shape == (4, 2, 3) and t.flags.c_contiguous
    np.testing.assert_array_equal(t, fake_dataset["a"].transpose(2, 0, 1))
    def broken(f, cache):
        raise OSError("down")
    monkeypatch.setattr(core, "_download_hf", broken)
    with pytest.warns(UserWarning):
        g = sdrbench.dataset("fake", "v1t").field("a").load(fake_dataset["cache"])
    np.testing.assert_array_equal(g, t)
    assert (fake_dataset["cache"] / "globus" / "fake" / "v1" / "a.f32").exists()  # base variant's folder
