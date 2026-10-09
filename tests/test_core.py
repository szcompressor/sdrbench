import numpy as np
import pytest

import sdrbench
import sdrbench._core as core
from sdrbench.cli import main as cli

REAL_DOWNLOAD_HF = core._download_hf  # captured before the fixture replaces it


def test_list_and_fields(fake_dataset):
    assert sdrbench.list() == ["fake"]
    ds = sdrbench.dataset("fake")
    assert ds.variant == "v1" and ds.variants == ["v1", "v2", "series", "v1t"]
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


def test_globus_cache_is_verified_once_and_repaired(fake_dataset, monkeypatch):
    f = sdrbench.dataset("fake").field("a")
    p = f.download(cache=fake_dataset["cache"], source="globus")
    hashes = []
    real = core.sha256_file
    monkeypatch.setattr(core, "sha256_file", lambda q: hashes.append(q) or real(q))
    f.download(cache=fake_dataset["cache"], source="globus")
    assert hashes == []                        # verified marker: no re-hash of big files
    p.write_bytes(b"\x00" * p.stat().st_size)  # corrupt the cached copy
    q = f.download(cache=fake_dataset["cache"], source="globus")
    assert q.read_bytes() == fake_dataset["a"].tobytes()  # unpacked again and fixed


def test_globus_archive_not_matching_catalog_raises(fake_dataset):
    entry = fake_dataset["catalog"]["datasets"]["fake"]["variants"]["v1"]["files"][0]
    entry["sha256"] = "0" * 64
    with pytest.raises(IOError, match="does not match the catalog"):
        sdrbench.dataset("fake").field("a").download(cache=fake_dataset["cache"], source="globus")


def test_hf_download_is_checked_against_catalog(fake_dataset, monkeypatch, tmp_path):
    bad = tmp_path / "bad.f32"
    bad.write_bytes(b"\x00" * 96)
    import types
    fake_hub = types.SimpleNamespace(hf_hub_download=lambda **k: str(bad))
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_hub)
    monkeypatch.setattr(core, "_download_hf", REAL_DOWNLOAD_HF)
    with pytest.raises(IOError, match="sha256 mismatch"):
        sdrbench.dataset("fake").field("a").download(source="hf")


def test_no_fallback_on_local_disk_errors_or_offline(fake_dataset, monkeypatch):
    import errno as _errno
    def full(f, cache, local_dir=None):
        raise OSError(_errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(core, "_download_hf", full)
    with pytest.raises(OSError, match="No space"):
        sdrbench.dataset("fake").field("a").download()
    def down(f, cache, local_dir=None):
        raise ConnectionError("unreachable")
    monkeypatch.setattr(core, "_download_hf", down)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(ConnectionError):
        sdrbench.dataset("fake").field("a").download()


def test_contains_and_non_array_getitem(fake_dataset):
    ds = sdrbench.dataset("fake")
    assert "A" in ds and "a.f32" in ds and "notes.txt" not in ds and "zzz" not in ds
    with pytest.raises(KeyError, match="not an array field"):
        ds["notes.txt"]


def test_type_hints_resolve():
    import typing
    assert typing.get_type_hints(sdrbench.list)["return"] == typing.List[str]
    typing.get_type_hints(sdrbench.Dataset.download)


def test_errors(fake_dataset):
    with pytest.raises(KeyError, match="unknown dataset"):
        sdrbench.dataset("nope")
    with pytest.raises(KeyError, match="unknown fake variant 'v9'"):
        sdrbench.dataset("fake", "v9")
    with pytest.raises(KeyError, match="no field 'zzz'"):
        sdrbench.dataset("fake")["zzz"]
    with pytest.raises(ValueError, match="not a raw array"):
        sdrbench.dataset("fake").field("notes.txt").load()
    with pytest.raises(KeyError, match="not an array field"):
        sdrbench.dataset("fake")["notes.txt"]
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
        g = sdrbench.dataset("fake", "v1t").field("a").load(cache=fake_dataset["cache"])
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


def test_parallel_download_and_single_globus_unpack(fake_dataset, monkeypatch, tmp_path):
    import threading
    import time
    import sdrbench._core as core_mod
    seen, active, peak = [], [0], [0]
    real = fake_dataset_hf = core_mod._download_hf
    lock = threading.Lock()

    def slow_hf(f, cache, local_dir=None):
        with lock:
            active[0] += 1; peak[0] = max(peak[0], active[0])
        time.sleep(0.2)
        with lock:
            active[0] -= 1
        seen.append(f.path)
        return real(f, cache, local_dir)
    monkeypatch.setattr(core_mod, "_download_hf", slow_hf)
    paths = sdrbench.dataset("fake").download(tmp_path / "d", workers=3)
    assert peak[0] >= 2 and len(paths) == 3                       # really concurrent
    assert [p.name for p in paths] == ["a.f32", "b.d64", "notes.txt"]  # order preserved

    # Globus: parallel callers of one archive download and unpack it exactly once
    calls = []
    real_fetch = core_mod.fetch_archive
    def counting_fetch(*a, **k):
        calls.append(a[0]); time.sleep(0.2); return real_fetch(*a, **k)
    monkeypatch.setattr(core_mod, "fetch_archive", counting_fetch)
    paths = sdrbench.dataset("fake", cache=tmp_path / "c").download(tmp_path / "g", source="globus", workers=3)
    assert len(calls) == 1 and all(p.exists() for p in paths)


def test_series_steps_and_tuple_keys(fake_dataset):
    ds = sdrbench.dataset("fake", "series", cache=fake_dataset["cache"])
    assert ds.variables == ["P", "A", "B", "C"] and ds.steps == ["01", "02", "5"]
    s = ds.series("P")
    assert s.steps == ["01", "02"] and len(s) == 2 and s["02"].name == "P/02"
    np.testing.assert_array_equal(s.stack(), np.stack([fake_dataset["p1"], fake_dataset["p2"]]))
    np.testing.assert_array_equal(ds["P", "01"], fake_dataset["p1"])
    np.testing.assert_array_equal(s.concatenate(axis=1), np.concatenate([fake_dataset["p1"], fake_dataset["p2"]], 1))
    with pytest.raises(KeyError, match="did you mean 'P'"):
        ds.series("PP")


def test_component_views_are_zero_copy_slices(fake_dataset):
    ds = sdrbench.dataset("fake", "series", cache=fake_dataset["cache"])
    for i, c in enumerate("ABC"):
        x = ds[f"{c}/5"]
        assert isinstance(x, np.memmap) and x.shape == (2, 2)
        np.testing.assert_array_equal(x, fake_dataset["comp"][i])
    f = ds.field("B/5")
    assert f.stored_shape == (3, 2, 2) and f.shape_fastest_first == (2, 2)
    np.testing.assert_array_equal(f.load(mmap=False), fake_dataset["comp"][1])


def test_save_materialises_views(fake_dataset, tmp_path):
    f = sdrbench.dataset("fake", "v1t").field("a")
    out = f.save(tmp_path / "a_pre.f32")
    np.testing.assert_array_equal(np.fromfile(out, "<f4").reshape(f.shape), fake_dataset["a"].transpose(2, 0, 1))
    g = sdrbench.dataset("fake", "series").field("C/5").save(tmp_path / "c.d64")
    np.testing.assert_array_equal(np.fromfile(g, "<f8").reshape(2, 2), fake_dataset["comp"][2])


def test_local_root_is_used_before_downloading(fake_dataset, monkeypatch, tmp_path):
    # an untarred SDRBench copy keeps the archive's top folder: found by name + size
    raw = tmp_path / "lustre" / "SDRBENCH-FAKE"
    raw.mkdir(parents=True)
    (raw / "a.f32").write_bytes(fake_dataset["a"].tobytes())
    def no_network(*a, **k):
        raise AssertionError("must not download")
    monkeypatch.setattr(core, "_download_hf", no_network)
    x = sdrbench.dataset("fake", root=tmp_path / "lustre")["a"]
    np.testing.assert_array_equal(x, fake_dataset["a"])
    # an earlier ds.download(dir) round-trips through root=dir; $SDRBENCH_DATA works too
    dl = tmp_path / "dl" / "v1" / "sub"
    dl.mkdir(parents=True)
    (dl / "b.d64").write_bytes(fake_dataset["b"].tobytes())
    monkeypatch.setenv("SDRBENCH_DATA", str(tmp_path / "dl"))
    np.testing.assert_array_equal(sdrbench.dataset("fake")["sub/b"], fake_dataset["b"])
    # a local file with the wrong size is ignored (release the memmap first: Windows cannot
    # rewrite a file that is mapped)
    del x
    import gc
    gc.collect()
    (raw / "a.f32").write_bytes(b"short")
    core._name_index.cache_clear()
    with pytest.warns(UserWarning, match="falling back"):   # local copy ignored -> HF (broken here) -> Globus
        x = sdrbench.dataset("fake", root=tmp_path / "lustre", cache=fake_dataset["cache"])["a"]
    np.testing.assert_array_equal(x, fake_dataset["a"])


def test_mapping_list_and_exports(fake_dataset):
    import collections.abc
    ds = sdrbench.dataset("fake")
    assert isinstance(ds, collections.abc.Mapping) and ds.get("zzz") is None and "a" in ds.keys()
    assert sdrbench.list(variants=True) == ["fake/v1", "fake/v2", "fake/series", "fake/v1t"]
    assert "list" not in sdrbench.__all__ and sdrbench.list() == ["fake"]
    assert [f.name for f in ds.extra_files] == ["notes.txt"]
    assert "fields" in repr(ds) and "<f4 2x3x4" in repr(ds.field("a"))


def test_errors_guide_the_user(fake_dataset):
    ds = sdrbench.dataset("fake")
    with pytest.raises(KeyError, match="did you mean 'a'"):
        ds["aa"]
    with pytest.raises(KeyError, match="is in variant 'series'"):
        ds["P/01"]
    with pytest.raises(KeyError, match="did you mean 'fake'"):
        sdrbench.dataset("fak")


def test_folder_name_is_a_variant_alias(fake_dataset):
    cat = fake_dataset["catalog"]["datasets"]["fake"]["variants"]
    cat["v2"]["folder"] = "512x512x512"
    assert sdrbench.dataset("fake", "512x512x512").variant == "v2"


def test_citation(fake_dataset):
    c = sdrbench.dataset("fake").citation()
    assert "zhao2020sdrbench" in c and "fake/v1" in c and "sdrbench==" in c
