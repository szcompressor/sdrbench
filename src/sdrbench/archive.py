"""Download and unpack SDRBench archives.

The Hugging Face mirror is built with exactly this code, so a file fetched
from Globus ends up at the same relative path, with the same bytes, as the
file stored on Hugging Face.
"""
from __future__ import annotations

import hashlib
import http.client
import os
import shutil
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Tuple

CHUNK = 1 << 22
JUNK = ("._", ".DS_Store", "__MACOSX")


def _is_junk(name: str) -> bool:
    return any(part.startswith(JUNK) for part in Path(name).parts)


def sha256_file(path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


class _HashingReader:
    """File-like wrapper that hashes and counts everything read through it."""

    def __init__(self, raw):
        self.raw = raw
        self.md5 = hashlib.md5()
        self.nbytes = 0

    def read(self, n: int = -1) -> bytes:
        b = self.raw.read(n)
        self.md5.update(b)
        self.nbytes += len(b)
        return b

    def drain(self) -> None:
        while self.read(CHUNK):
            pass


class _ResumingResponse:
    """Readable HTTP body that reconnects with a Range request when the connection drops,
    so multi-GB downloads survive transient network failures.

    The full size must be known (Globus sends GET bodies chunked, without Content-Length, so it
    comes from HEAD); without it a dropped connection cannot be told apart from the end of the
    file, so we refuse to download instead of risking a silently truncated archive."""

    def __init__(self, url: str, retries: int = 10, timeout: float = 120):
        self.url, self.retries, self.timeout = url, retries, timeout
        self.pos = 0
        self.total = self._head_length()
        self.resp = self._connect()

    def _head_length(self) -> int:
        last = None
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(self.url, method="HEAD", headers={"User-Agent": "sdrbench"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    n = r.headers.get("Content-Length")
                    if n is not None:
                        return int(n)
                    last = "no Content-Length in HEAD response"
            except urllib.error.HTTPError as e:
                if e.code not in (429, 502, 503, 504):  # only temporary server errors are retried
                    raise
                last = e
            except (OSError, http.client.HTTPException, ValueError) as e:
                last = e
            time.sleep(min(60, 2 ** attempt))
        raise IOError(f"{self.url}: cannot determine the archive size ({last}); refusing an unverifiable download")

    def _connect(self):
        headers = {"User-Agent": "sdrbench"}
        if self.pos:
            headers["Range"] = f"bytes={self.pos}-"
        resp = urllib.request.urlopen(urllib.request.Request(self.url, headers=headers), timeout=self.timeout)
        if self.pos:
            rng = resp.headers.get("Content-Range", "")
            if resp.status != 206 or not rng.startswith(f"bytes {self.pos}-"):
                resp.close()
                raise IOError(f"{self.url}: server did not resume at byte {self.pos} ({resp.status} {rng!r})")
        return resp

    def read(self, n: int = -1) -> bytes:
        for attempt in range(self.retries + 1):
            try:
                b = self.resp.read(n)
                if not b and n != 0 and self.pos < self.total:
                    # body ended early: a dropped connection (with chunked encoding this can look
                    # like a clean end of stream)
                    raise http.client.IncompleteRead(b"", self.total - self.pos)
                self.pos += len(b)
                if self.pos > self.total:
                    raise IOError(f"{self.url}: received more than the {self.total} bytes announced")
                return b
            except (OSError, http.client.HTTPException) as e:
                if isinstance(e, IOError) and "announced" in str(e) or attempt == self.retries:
                    raise
                time.sleep(min(60, 2 ** attempt))
                try:
                    self.resp.close()
                except Exception:
                    pass
                try:
                    self.resp = self._connect()
                except (OSError, http.client.HTTPException):
                    continue  # try again on the next attempt
        raise IOError(f"{self.url}: download failed at byte {self.pos} of {self.total}")

    def close(self):
        self.resp.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _open_url(url: str):
    if url.startswith(("http://", "https://")):
        return _ResumingResponse(url)
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "sdrbench"}), timeout=120)


def _safe_target(root: Path, name: str) -> Path:
    target = (root / name).resolve()
    if os.path.commonpath([str(root.resolve()), str(target)]) != str(root.resolve()):
        raise ValueError(f"unsafe path in archive: {name}")
    return target


def _extract_tar_stream(stream, root: Path) -> None:
    with tarfile.open(fileobj=stream, mode="r|*") as tf:
        for m in tf:
            if _is_junk(m.name) or not (m.isfile() or m.isdir()):
                continue
            target = _safe_target(root, m.name)
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(m)
            with open(target, "wb") as out:
                shutil.copyfileobj(src, out, CHUNK)


def _extract_zip(path: Path, root: Path) -> None:
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if _is_junk(info.filename) or info.is_dir():
                continue
            target = _safe_target(root, info.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, CHUNK)


def fetch_archive(url: str, dest, expected_md5: Optional[str] = None) -> Tuple[str, int]:
    """Stream ``url`` (a .tar.gz or .zip) into directory ``dest``.

    If all files of the archive live under a single top-level directory, its contents are
    placed directly in ``dest`` (the same rule the Hugging Face mirror uses). File contents
    are never modified. The whole archive must arrive: its size is checked against the
    server's, and its md5 against ``expected_md5`` when given.
    Returns ``(md5 of the archive, archive size in bytes)``.
    """
    import tempfile

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=dest.name + ".", suffix=".partial", dir=dest.parent))
    try:
        with _open_url(url) as resp:
            reader = _HashingReader(resp)
            if url.endswith(".zip"):
                zpath = tmp / "_archive.zip"
                with open(zpath, "wb") as out:
                    shutil.copyfileobj(reader, out, CHUNK)
                files = tmp / "files"
                _extract_zip(zpath, files)
                zpath.unlink()
            else:
                files = tmp / "files"
                files.mkdir()
                _extract_tar_stream(reader, files)
                reader.drain()
            total = getattr(resp, "total", None)
        if total is not None and reader.nbytes != total:
            raise IOError(f"{url}: got {reader.nbytes} of {total} bytes")
        md5 = reader.md5.hexdigest()
        if expected_md5 and md5 != expected_md5:
            raise IOError(f"md5 mismatch for {url}: got {md5}, expected {expected_md5}")
        tops = {p.relative_to(files).parts[0] for p in files.rglob("*") if p.is_file()}
        nested = {p.relative_to(files).parts[0] for p in files.rglob("*") if p.is_file()
                  and len(p.relative_to(files).parts) > 1}
        src = files / next(iter(tops)) if len(tops) == 1 and tops == nested else files
        shutil.rmtree(dest, ignore_errors=True)
        os.replace(src, dest)
        return md5, reader.nbytes
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
