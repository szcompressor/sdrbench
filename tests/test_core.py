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
    def broken(f, cache, local_dir=None):
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
    g = f.download(cache=fake_dataset["cache"], source="globus")
    assert g.read_bytes() == (fake_dataset["hub"] / "v1/sub/b.d64").read_bytes()
    # archive is unpacked once and reused
    core.catalog()["datasets"]["fake"]["variants"]["v1"]["archive"]["url"] = "file:///nonexistent.tar.gz"
    assert sdrbench.dataset("fake").field("a").download(cache=fake_dataset["cache"], source="globus").exists()


def test_globus_sha_mismatch_detected(fake_dataset):
    f = sdrbench.dataset("fake").field("a")
    p = f.download(cache=fake_dataset["cache"], source="globus")
    p.write_bytes(b"\x00" * p.stat().st_size)
    with pytest.raises(IOError, match="sha256 mismatch"):
        f.download(cache=fake_dataset["cache"], source="globus")


def test_errors(fake_dataset):
    with pytest.raises(KeyError, match="unknown dataset"):
        sdrbench.dataset("nope")
    with pytest.raises(KeyError, match="unknown fake variant 'v9'"):
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
    out_dir = fake_dataset["cache"].parent / "out"
    assert cli(["download", "fake", "a", "sub/b", "-o", str(out_dir)]) == 0
    assert (out_dir / "v1" / "a.f32").is_file() and (out_dir / "v1" / "sub" / "b.d64").is_file()
    assert cli(["download", "fake", "nope"]) == 1


def test_derived_transposed_variant(fake_dataset, monkeypatch):
    t = sdrbench.dataset("fake", "v1t", cache=fake_dataset["cache"])["a"]
    assert t.shape == (4, 2, 3) and t.flags.c_contiguous
    np.testing.assert_array_equal(t, fake_dataset["a"].transpose(2, 0, 1))
    def broken(f, cache, local_dir=None):
        raise OSError("down")
    monkeypatch.setattr(core, "_download_hf", broken)
    with pytest.warns(UserWarning):
        g = sdrbench.dataset("fake", "v1t").field("a").load(fake_dataset["cache"])
    np.testing.assert_array_equal(g, t)
    assert (fake_dataset["cache"] / "globus" / "fake" / "v1" / "a.f32").exists()  # base variant's folder


def test_case_insensitive_lookup(fake_dataset, monkeypatch):
    assert sdrbench.dataset("FAKE").name == "fake" and sdrbench.dataset("Fake/V2").variant == "v2"
    ds = sdrbench.dataset("fake")
    assert ds.field("A") is ds.field("a") and ds.field("SUB/B").name == "sub/b"
    assert ds.fields == ["a", "sub/b"]              # original names are what is listed
    # two names differing only in case: exact matches win, other spellings are ambiguous
    v1 = fake_dataset["catalog"]["datasets"]["fake"]["variants"]["v1"]["files"]
    v1 += [{**v1[0], "path": "v1/QV.f32", "name": "QV"}, {**v1[0], "path": "v1/Qv.f32", "name": "Qv"}]
    ds = sdrbench.dataset("fake")
    assert ds.field("QV").path == "v1/QV.f32" and ds.field("Qv").path == "v1/Qv.f32"
    with pytest.raises(KeyError, match="ambiguous"):
        ds.field("qv")


def test_save_plain_files_to_directory(fake_dataset, tmp_path):
    ds = sdrbench.dataset("fake", cache=fake_dataset["cache"])
    p = ds.field("a").download(tmp_path / "data")
    assert p == tmp_path / "data" / "v1" / "a.f32" and not p.is_symlink()
    assert p.read_bytes() == fake_dataset["a"].tobytes()
    paths = ds.download(tmp_path / "all")
    assert sorted(x.relative_to(tmp_path / "all").as_posix() for x in paths) == ["v1/a.f32", "v1/notes.txt", "v1/sub/b.d64"]
    # Globus source: file is placed in the directory too, original name, verified bytes
    g = ds.field("sub/b").download(tmp_path / "g", source="globus")
    assert g == tmp_path / "g" / "v1" / "sub" / "b.d64" and g.read_bytes() == fake_dataset["b"].tobytes()
    assert not list((tmp_path / "g").rglob("*.partial"))
