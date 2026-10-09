import io
import tarfile

import pytest

from sdrbench.archive import fetch_archive, sha256_file
from conftest import make_tar, make_zip


def test_tar_single_top_dir_is_flattened_and_junk_dropped(tmp_path):
    md5 = make_tar(tmp_path / "x.tar.gz", {"f1.f32": b"\x00" * 16, "d/f2.d64": b"\x01" * 8})
    got_md5, n = fetch_archive((tmp_path / "x.tar.gz").as_uri(), tmp_path / "out")
    assert got_md5 == md5 and n == (tmp_path / "x.tar.gz").stat().st_size
    files = sorted(p.relative_to(tmp_path / "out").as_posix() for p in (tmp_path / "out").rglob("*") if p.is_file())
    assert files == ["d/f2.d64", "f1.f32"]
    assert (tmp_path / "out" / "f1.f32").read_bytes() == b"\x00" * 16


def test_tar_without_top_dir_kept_as_is(tmp_path):
    path = tmp_path / "flat.tar.gz"
    with tarfile.open(path, "w:gz") as tf:
        for name in ("a.f32", "b.f32"):
            ti = tarfile.TarInfo(name); ti.size = 4
            tf.addfile(ti, io.BytesIO(b"abcd"))
    fetch_archive(path.as_uri(), tmp_path / "out")
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["a.f32", "b.f32"]


def test_zip(tmp_path):
    md5 = make_zip(tmp_path / "x.zip", {"top/a_20x2_double": b"\x02" * 320})
    got, _ = fetch_archive((tmp_path / "x.zip").as_uri(), tmp_path / "out")
    assert got == md5
    assert [p.name for p in (tmp_path / "out").rglob("*") if p.is_file()] == ["a_20x2_double"]
    assert not (tmp_path / "x.zip.partial").exists()


def test_md5_mismatch_raises_and_leaves_nothing(tmp_path):
    make_tar(tmp_path / "x.tar.gz", {"a": b"1"})
    with pytest.raises(IOError, match="md5 mismatch"):
        fetch_archive((tmp_path / "x.tar.gz").as_uri(), tmp_path / "out", expected_md5="0" * 32)
    assert not (tmp_path / "out").exists() and not (tmp_path / "out.partial").exists()


def test_path_traversal_rejected(tmp_path):
    path = tmp_path / "evil.tar.gz"
    with tarfile.open(path, "w:gz") as tf:
        ti = tarfile.TarInfo("../../escape"); ti.size = 1
        tf.addfile(ti, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unsafe path"):
        fetch_archive(path.as_uri(), tmp_path / "out")
    assert not (tmp_path.parent / "escape").exists()


def test_sha256_file(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"abc")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_download_resumes_after_connection_drop(tmp_path):
    """A server that cuts the connection mid-body; the reader must resume with Range."""
    import http.server
    import threading
    from conftest import make_tar
    md5 = make_tar(tmp_path / "x.tar.gz", {"big.f32": bytes(range(256)) * 4000})
    blob = (tmp_path / "x.tar.gz").read_bytes()
    hits = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()

        def do_GET(self):
            rng = self.headers.get("Range")
            start = int(rng.split("=")[1].rstrip("-")) if rng else 0
            hits.append(start)
            body = blob[start:]
            self.send_response(206 if rng else 200)
            self.send_header("Content-Length", str(len(body)))
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{len(blob) - 1}/{len(blob)}")
            self.end_headers()
            if len(hits) == 1:  # first request: send part of the body, then drop the connection
                self.wfile.write(body[: len(body) // 3])
                self.wfile.flush()
                self.connection.shutdown(2)
                return
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        got, n = fetch_archive(f"http://127.0.0.1:{srv.server_port}/x.tar.gz", tmp_path / "out")
    finally:
        srv.shutdown()
    assert got == md5 and n == len(blob)
    assert len(hits) >= 2 and hits[0] == 0 and hits[1] > 0
    assert (tmp_path / "out" / "big.f32").read_bytes() == bytes(range(256)) * 4000


def test_resume_with_chunked_encoding_and_clean_close(tmp_path):
    """Globus sends chunked bodies without Content-Length; a connection closed cleanly between
    chunks must still be detected (via HEAD's Content-Length) and resumed."""
    import http.server
    import threading
    from conftest import make_tar
    md5 = make_tar(tmp_path / "x.tar.gz", {"big.f32": bytes(range(256)) * 8000})
    blob = (tmp_path / "x.tar.gz").read_bytes()
    gets = []

    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()

        def do_GET(self):
            rng = self.headers.get("Range")
            start = int(rng.split("=")[1].rstrip("-")) if rng else 0
            gets.append(start)
            body = blob[start:] if len(gets) > 1 else blob[start:len(blob) // 2]
            self.send_response(206 if rng else 200)
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{len(blob) - 1}/{len(blob)}")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.end_headers()
            for i in range(0, len(body), 4096):
                piece = body[i:i + 4096]
                self.wfile.write(b"%x\r\n" % len(piece) + piece + b"\r\n")
            if len(gets) > 1:
                self.wfile.write(b"0\r\n\r\n")  # proper end only on the resumed request
            self.wfile.flush()

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        got, n = fetch_archive(f"http://127.0.0.1:{srv.server_port}/x.tar.gz", tmp_path / "out")
    finally:
        srv.shutdown()
    assert got == md5 and n == len(blob) and gets[0] == 0 and 0 < gets[1] <= len(blob) // 2  # resumed mid-file
    assert (tmp_path / "out" / "big.f32").read_bytes() == bytes(range(256)) * 8000


def _serve(handler):
    import http.server
    import threading
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_refuses_download_when_size_unknown(tmp_path):
    """No HEAD Content-Length + chunked GET: truncation would be undetectable -> refuse."""
    import http.server
    from conftest import make_tar
    make_tar(tmp_path / "x.tar.gz", {"a.f32": b"1" * 100})
    blob = (tmp_path / "x.tar.gz").read_bytes()

    class H(http.server.BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.send_response(404); self.end_headers()

        def do_GET(self):
            self.send_response(200); self.send_header("Content-Length", str(len(blob))); self.end_headers()
            self.wfile.write(blob)

        def log_message(self, *a):
            pass
    srv = _serve(H)
    try:
        with pytest.raises(Exception):
            fetch_archive(f"http://127.0.0.1:{srv.server_port}/x.tar.gz", tmp_path / "out")
    finally:
        srv.shutdown()
    assert not (tmp_path / "out").exists()


def test_flatten_ignores_empty_directories(tmp_path):
    """Same rule as the mirror: only files decide whether the top-level folder is dropped."""
    path = tmp_path / "e.tar.gz"
    with tarfile.open(path, "w:gz") as tf:
        ti = tarfile.TarInfo("TOP/a.f32"); ti.size = 4
        tf.addfile(ti, io.BytesIO(b"abcd"))
        d = tarfile.TarInfo("EMPTY"); d.type = tarfile.DIRTYPE
        tf.addfile(d)
    fetch_archive(path.as_uri(), tmp_path / "out")
    assert (tmp_path / "out" / "a.f32").exists()


def test_concurrent_processes_same_archive(tmp_path):
    """Two processes unpacking the same archive into one cache must not clobber each other."""
    import subprocess
    import sys
    from conftest import make_tar
    members = {f"f{i:03d}.f32": bytes([i % 256]) * 20000 for i in range(80)}
    make_tar(tmp_path / "m.tar.gz", members)
    code = ("import sys; from sdrbench.core import _globus_lock; from sdrbench.archive import fetch_archive\n"
            "with _globus_lock(sys.argv[2]):\n fetch_archive(sys.argv[1], sys.argv[2])")
    procs = [subprocess.Popen([sys.executable, "-c", code, (tmp_path / "m.tar.gz").as_uri(), str(tmp_path / "out")])
             for _ in range(3)]
    assert all(p.wait() == 0 for p in procs)
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == sorted(members)
