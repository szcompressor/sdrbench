import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

import sdrbench.core as core


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
    members = {"a.f32": a.tobytes(), "sub/b.d64": b.tobytes(), "notes.txt": b"hello"}
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
            "v1t": {  # derived layout of v1/a.f32: transpose (2, 0, 1)
                "archive": {"url": tgz.as_uri(), "bytes": tgz.stat().st_size, "md5": md5}, "archive_variant": "v1",
                "files": [{"path": "v1/a.f32", "name": "a", "bytes": a.nbytes, "sha256": sha["a.f32"],
                           "dtype": "<f4", "shape": [4, 2, 3], "transpose": [2, 0, 1]}]}}}}}
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
    return {"a": a, "b": b, "cache": tmp_path / "cache", "hf_calls": calls, "catalog": cat, "hub": hub}


def write_catalog(path, cat):
    path.write_text(json.dumps(cat))
