import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

import sdrbench._core as core


def make_tar(path, members, top="SDRBENCH-FAKE"):
    """members: {relative name: bytes}; written under a single top-level dir plus macOS junk."""
    with tarfile.open(path, "w:gz") as tf:
        for name, data in {**members, "._junk": b"resource fork"}.items():
            ti = tarfile.TarInfo(f"{top}/{name}")
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return hashlib.md5(path.read_bytes()).hexdigest()


def make_zip(path, members):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
        zf.writestr("__MACOSX/._x", b"junk")
    return hashlib.md5(path.read_bytes()).hexdigest()


@pytest.fixture
def fake_dataset(tmp_path, monkeypatch):
    """A one-dataset catalog whose Globus archive is a local file:// tarball."""
    a = np.arange(2 * 3 * 4, dtype="<f4").reshape(2, 3, 4)
    b = np.linspace(0, 1, 10, dtype="<f8")
    p1, p2 = np.full((2, 2), 1, "<f4"), np.full((2, 2), 2, "<f4")
    comp = np.arange(3 * 2 * 2, dtype="<f8").reshape(3, 2, 2)
    members = {"a.f32": a.tobytes(), "sub/b.d64": b.tobytes(), "notes.txt": b"hello",
               "P02.f32": p2.tobytes(), "P01.f32": p1.tobytes(), "t5.d64": comp.tobytes()}
    tgz = tmp_path / "fake.tar.gz"
    md5 = make_tar(tgz, members)
    sha = {k: hashlib.sha256(v).hexdigest() for k, v in members.items()}
    cat = {"schema": 1, "datasets": {"fake": {
        "title": "Fake", "description": "test", "source": "tests", "repo": "sdrbench/fake",
        "variants": {"v1": {
            "archive": {"url": tgz.as_uri(), "bytes": tgz.stat().st_size, "md5": md5},
            "files": [
                {"path": "v1/a.f32", "name": "a", "bytes": a.nbytes, "sha256": sha["a.f32"], "dtype": "<f4", "shape": [2, 3, 4]},
                {"path": "v1/sub/b.d64", "name": "sub/b", "bytes": b.nbytes, "sha256": sha["sub/b.d64"], "dtype": "<f8", "shape": [10]},
                {"path": "v1/notes.txt", "name": None, "bytes": 5, "sha256": sha["notes.txt"], "dtype": None, "shape": None},
            ]},
            "v2": {  # names come from the catalog
                "archive": {"url": "file:///unused.tar.gz", "bytes": 0, "md5": None},
                "files": [
                    {"path": "v2/a.f32", "name": "a32", "bytes": 4, "sha256": "0" * 64, "dtype": "<f4", "shape": [1]},
                    {"path": "v2/a.d64", "name": "a64", "bytes": 8, "sha256": "0" * 64, "dtype": "<f8", "shape": [1]},
                ]},
            "series": {  # time series P/01, P/02 and a stacked file split into components A, B, C
                "archive": {"url": tgz.as_uri(), "bytes": tgz.stat().st_size, "md5": md5}, "folder": "v1",
                "files": [
                    {"path": "v1/P01.f32", "name": "P/01", "var": "P", "step": "01", "bytes": 16, "sha256": sha["P01.f32"], "dtype": "<f4", "shape": [2, 2]},
                    {"path": "v1/P02.f32", "name": "P/02", "var": "P", "step": "02", "bytes": 16, "sha256": sha["P02.f32"], "dtype": "<f4", "shape": [2, 2]},
                ] + [{"path": "v1/t5.d64", "name": f"{c}/5", "var": c, "step": "5", "bytes": 96, "sha256": sha["t5.d64"],
                      "dtype": "<f8", "shape": [2, 2], "file_shape": [3, 2, 2], "index": i} for i, c in enumerate("ABC")]},
            "v1t": {  # derived layout of v1/a.f32: transpose (2, 0, 1)
                "archive": {"url": tgz.as_uri(), "bytes": tgz.stat().st_size, "md5": md5}, "folder": "v1", "transpose": [2, 0, 1],
                "files": [{"path": "v1/a.f32", "name": "a", "bytes": a.nbytes, "sha256": sha["a.f32"],
                           "dtype": "<f4", "shape": [4, 2, 3], "file_shape": [2, 3, 4], "transpose": [2, 0, 1]}]}}}}}
    monkeypatch.setattr(core, "catalog", lambda: cat)

    # stand-in for huggingface_hub.hf_hub_download: serve the same bytes from a local folder
    hub = tmp_path / "hub"
    for name, data in members.items():
        (hub / "v1" / name).parent.mkdir(parents=True, exist_ok=True)
        (hub / "v1" / name).write_bytes(data)
    calls = []

    def fake_hf(f, cache, local_dir=None):
        calls.append(f.path)
        if local_dir is not None:  # mimic hf_hub_download(local_dir=...): real file at <dir>/<path>
            dest = Path(local_dir) / f.path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((hub / f.path).read_bytes())
            return dest
        return hub / f.path

    monkeypatch.setattr(core, "_download_hf", fake_hf)
    return {"comp": comp, "p1": p1, "p2": p2, "a": a, "b": b, "cache": tmp_path / "cache", "hf_calls": calls, "catalog": cat, "hub": hub}


def write_catalog(path, cat):
    path.write_text(json.dumps(cat))
