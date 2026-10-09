"""Integrity of the bundled catalog and of the rules that generate it."""
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sync"))
import sync  # noqa: E402

ITEM = {"<f4": 4, "<f8": 8, "<i4": 4, "<i8": 8, "<u1": 1, "<u2": 2, "<u8": 8}
import sdrbench  # noqa: E402

CATALOG = sdrbench.catalog()  # the installed package's catalog (works for wheel tests too)
CFG = json.loads((ROOT / "sync" / "datasets.json").read_text())


def all_files():
    for d, ds in CATALOG["datasets"].items():
        for v, vv in ds["variants"].items():
            for f in vv["files"]:
                yield d, vv.get("folder", v), f


def test_catalog_not_empty():
    assert CATALOG["datasets"]


def test_every_array_size_matches_shape():
    for d, v, f in all_files():
        assert f["path"].startswith(v + "/"), f
        assert len(f["sha256"]) == 64
        if f["dtype"]:
            stored = f.get("file_shape") or f["shape"]
            assert math.prod(stored) * ITEM[f["dtype"]] == f["bytes"], (d, f["path"])
            if f.get("index") is not None:  # component view: one slice along axis 0 of the stored file
                assert list(stored[1:]) == f["shape"] and 0 <= f["index"] < stored[0]
            if f.get("transpose"):
                assert sorted(f["shape"]) == sorted(stored)
        else:
            assert f["shape"] is None


def test_paths_unique_and_repos_named():
    for d, ds in CATALOG["datasets"].items():
        # every stored file belongs to exactly one (non-derived) variant; components share their file
        paths = [p for v in ds["variants"].values() if "transpose" not in v for p in {f["path"] for f in v["files"]}]
        assert len(paths) == len(set(paths)), d
        assert ds["repo"] == f"sdrbench/{d}"


def test_catalog_is_reproducible_from_state():
    """The committed catalog equals what sync/sync.py builds from sync/state.json."""
    state = json.loads((ROOT / "sync" / "state.json").read_text())
    built, errors = sync.build_catalog(CFG, state, strict=False)
    assert errors == []
    assert built == CATALOG


def test_no_excluded_dataset_is_mirrored():
    mirrored = {vv["archive"] for ds in CFG["datasets"].values() for vv in ds["variants"].values() if "archive" in vv}
    assert not mirrored & set(CFG["excluded"])
    assert "brown-samples" not in CATALOG["datasets"]
    assert not any("NSTX" in vv["archive"]["url"] for ds in CATALOG["datasets"].values() for vv in ds["variants"].values())


# --- rule engine -----------------------------------------------------------------
RULES = [{"glob": "*.f32", "dtype": "<f4", "shape": [2, 3]},
         {"glob": "*.d64", "dtype": "<f8", "shape": "1d"},
         {"glob": "*.bp", "dtype": None, "shape": None}]


def entry(path, nbytes, rules=RULES, folder="v"):
    out = sync.file_entries("ds", folder, {"path": f"{folder}/{path}", "bytes": nbytes, "sha256": "0" * 64}, rules)
    return [(e["dtype"], e["shape"], e["name"]) for e in out]


def test_rules_exact_and_1d():
    assert entry("a.f32", 24) == [("<f4", [2, 3], "a")]
    assert entry("sub/b.d64", 80) == [("<f8", [10], "sub/b")]
    assert entry("c.bp", 123) == [(None, None, None)]
    assert entry("README.txt", 7) == [(None, None, None)]


def test_rules_reject_wrong_size_and_unknown_files():
    with pytest.raises(sync.CatalogError, match="!="):
        entry("a.f32", 28)
    with pytest.raises(sync.CatalogError, match="not a multiple"):
        entry("b.d64", 81)
    with pytest.raises(sync.CatalogError, match="no dtype/shape rule"):
        entry("mystery.dat", 8)
    with pytest.raises(sync.CatalogError, match="unsupported dtype"):
        entry("x.f32", 8, rules=[{"glob": "*.f32", "dtype": "<c8", "shape": [1]}])


def test_rules_match_folder_with_slash():
    assert entry("xx.f32", 24, folder="multi-timesteps/1840") == [("<f4", [2, 3], "xx")]


def test_var_step_identity():
    rule = {"pattern": r"(\w+)f(\d+)\.f32", "var": "{0}", "step": "{1}"}
    assert sync.identity("Pf07.f32", rule) == ("P", "07")
    assert sync.field_name("P", "07") == "P/07" and sync.field_name("T", None) == "T"
    assert sync.default_var("CLDHGH_1_1800_3600.f32") == "CLDHGH"
    assert sync.default_var("T-98x1200x1200.f32") == "T"
    assert sync.default_var("temperature.f32") == "temperature"
    with pytest.raises(sync.CatalogError, match="does not match pattern"):
        sync.identity("other.f32", rule)


def test_components_split_axis0_into_views():
    rules = [{"glob": "*.d64", "dtype": "<f8", "shape": [3, 2, 2], "pattern": r"t(\d)\.d64", "step": "{0}",
              "components": ["A", "B", "C"]}]
    out = sync.file_entries("ds", "v", {"path": "v/t5.d64", "bytes": 96, "sha256": "0" * 64}, rules)
    assert [(e["name"], e["index"], e["shape"], e["file_shape"]) for e in out] == [
        ("A/5", 0, [2, 2], [3, 2, 2]), ("B/5", 1, [2, 2], [3, 2, 2]), ("C/5", 2, [2, 2], [3, 2, 2])]


def test_select_exclude():
    assert sync.selected("QGRAUPf01.f32", {"select": "QGRAUPf*"})
    assert not sync.selected("QCLOUDf01.f32", {"select": "QGRAUPf*"})
    assert not sync.selected("631-tst.bin.f32", {"exclude": "*-tst.bin.f32"})


def test_field_names_unique_in_catalog():
    for d, ds in CATALOG["datasets"].items():
        for v, vv in ds["variants"].items():
            names = [f["name"] for f in vv["files"] if f["name"]]
            assert len(names) == len(set(names)), (d, v)
            assert all(f["name"] for f in vv["files"] if f["dtype"]), (d, v)


def test_descriptions_mention_no_dims_or_unknown_variants():
    """Descriptions are prose: no folder names / dims strings used as variant names."""
    import re
    for d, ds in CFG["datasets"].items():
        for v, vv in ds["variants"].items():
            text = vv.get("description", "")
            for folder in {x.get("folder") for x in ds["variants"].values()} - {None}:
                if re.search(r"\d+x\d+", folder):
                    assert f"variant {folder}" not in text and f"'{folder}'" not in text, (d, v, folder)


def test_page_link_parsing():
    base = "https://g.example/raw-data/"
    html = (f'<a href="{base}A/x.tar.gz">x</a> <a href="{base}A/y-property.txt">p</a>'
            f'<a href="{base}B/z.zip">z</a> <a href="https://other/q.tar.gz">o</a>')
    arch, txt = sync.page_links(html, base)
    assert arch == ["A/x.tar.gz", "B/z.zip"] and txt == ["A/y-property.txt"]


def test_cards_render_for_every_dataset():
    for name, d in CATALOG["datasets"].items():
        card = sync.render_card(name, d)
        assert card.startswith("---\nlicense: other")
        assert d["title"] in card and "zhao2020sdrbench" in card
        if d.get("acknowledgment"):
            assert d["acknowledgment"] in card


# --- streaming mirror (dry: no Hugging Face API) -----------------------------------
def test_stream_mirror_layout_matches_fetch_archive(tmp_path):
    from conftest import make_tar, make_zip
    from sdrbench._archive import fetch_archive, sha256_file
    members = {"a.f32": b"\x01" * 400, "d/b.d64": b"\x02" * 800}
    for maker, name in ((make_tar, "x.tar.gz"), (make_zip, "x.zip")):
        src = tmp_path / name
        md5 = maker(src, members) if maker is make_tar else maker(src, {f"TOP/{k}": v for k, v in members.items()})
        files, got_md5, n = sync.stream_mirror(None, "sdrbench/x", "var", src.as_uri(), tmp_path / "scratch")
        assert got_md5 == md5 and n == src.stat().st_size
        fetch_archive(src.as_uri(), tmp_path / "ref" / name)
        ref = sorted((f"var/{p.relative_to(tmp_path / 'ref' / name).as_posix()}", p.stat().st_size, sha256_file(p))
                     for p in (tmp_path / "ref" / name).rglob("*") if p.is_file())
        assert sorted((f["path"], f["bytes"], f["sha256"]) for f in files) == ref
        assert not (tmp_path / "scratch").exists()


def test_stream_mirror_restarts_without_flattening(tmp_path):
    import io
    import tarfile
    src = tmp_path / "multi.tar.gz"
    with tarfile.open(src, "w:gz") as tf:
        for name in ("A/one.f32", "B/two.f32"):
            ti = tarfile.TarInfo(name); ti.size = 4
            tf.addfile(ti, io.BytesIO(b"abcd"))
    files, _, _ = sync.stream_mirror(None, "sdrbench/x", "v", src.as_uri(), tmp_path / "s")
    assert sorted(f["path"] for f in files) == ["v/A/one.f32", "v/B/two.f32"]


def test_variant_names_have_no_dimension_strings():
    import re
    for d, ds in CATALOG["datasets"].items():
        for v in ds["variants"]:
            assert not re.search(r"\d+x\d+", v), (d, v)


def test_core_never_calls_builtin_list():
    """core.py defines list() (sdrbench.list); a stray list(...) call there would call it."""
    import ast
    src = (ROOT / "src" / "sdrbench" / "_core.py").read_text() if (ROOT / "src").exists() else None
    if src is None:
        pytest.skip("source tree not available (testing an installed wheel)")
    calls = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "list"]
    assert all(c.args == [] and all(k.arg == "variants" for k in c.keywords) for c in calls), "use [*x] instead of list(x) in _core.py"


class FakeHF:
    """Records commits; pretends the repo already holds `existing` files."""
    def __init__(self, existing):
        self.existing, self.commits = existing, []

    def list_repo_files(self, repo, repo_type=None):
        return self.existing

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        from huggingface_hub import CommitOperationDelete
        self.commits.append(([o.path_in_repo for o in operations if not isinstance(o, CommitOperationDelete)],
                             [o.path_in_repo for o in operations if isinstance(o, CommitOperationDelete)]))


def _tar(tmp_path, n):
    from conftest import make_tar
    path = tmp_path / f"t{n}.tar.gz"
    make_tar(path, {f"f{i}.f32": bytes([i]) * 4096 for i in range(n)})
    return path


def test_truncated_archive_deletes_nothing(tmp_path):
    src = _tar(tmp_path, 5)
    api = FakeHF([f"v/f{i}.f32" for i in range(5)])
    with pytest.raises(IOError, match="nothing deleted"):
        sync.stream_mirror(api, "sdrbench/x", "v", src.as_uri(), tmp_path / "s",
                           expected_bytes=src.stat().st_size + 1000)
    assert all(not dels for _, dels in api.commits)


def test_mass_deletion_refused_unless_allowed(tmp_path):
    src = _tar(tmp_path, 2)
    api = FakeHF([f"v/f{i}.f32" for i in range(10)])
    with pytest.raises(IOError, match="refusing"):
        sync.stream_mirror(api, "sdrbench/x", "v", src.as_uri(), tmp_path / "s", expected_bytes=src.stat().st_size)
    api = FakeHF([f"v/f{i}.f32" for i in range(10)])
    sync.stream_mirror(api, "sdrbench/x", "v", src.as_uri(), tmp_path / "s2", expected_bytes=src.stat().st_size,
                       max_delete_fraction=1.0)
    assert sorted(api.commits[-1][1]) == sorted(f"v/f{i}.f32" for i in range(2, 10))


def test_normal_update_removes_only_stale_files(tmp_path):
    src = _tar(tmp_path, 8)
    api = FakeHF([f"v/f{i}.f32" for i in range(8)] + ["v/old.f32", "w/other.f32"])
    files, _, n = sync.stream_mirror(api, "sdrbench/x", "v", src.as_uri(), tmp_path / "s",
                                     expected_bytes=src.stat().st_size)
    assert n == src.stat().st_size and len(files) == 8
    assert api.commits[-1][1] == ["v/old.f32"]   # other variants untouched


def test_catalog_pins_revisions():
    built, _ = sync.build_catalog(CFG, {}, strict=False, revisions={"nyx": "abc123"})
    state = json.loads((ROOT / "sync" / "state.json").read_text())
    built, _ = sync.build_catalog(CFG, state, strict=False, revisions={"nyx": "abc123"})
    assert built["datasets"]["nyx"]["revision"] == "abc123"
    assert "revision" not in built["datasets"]["cesm-atm"]


def test_catalog_refreshes_only_requested_revisions(monkeypatch, tmp_path):
    """Card commits must not re-pin datasets: only --refresh'ed ones get a new revision."""
    import argparse
    revs = tmp_path / "revisions.json"
    revs.write_text(json.dumps({"nyx": "old-nyx", "hacc": "old-hacc"}))
    monkeypatch.setattr(sync, "REVISIONS", revs)
    monkeypatch.setattr(sync, "CATALOG", tmp_path / "catalog.json")
    asked = []
    monkeypatch.setattr(sync, "fetch_revisions", lambda c: asked.append([*c["datasets"]]) or {d: "new-" + d for d in c["datasets"]})
    sync.cmd_catalog(argparse.Namespace(lenient=True, refresh=None))
    assert asked == [] and json.loads(revs.read_text()) == {"nyx": "old-nyx", "hacc": "old-hacc"}
    sync.cmd_catalog(argparse.Namespace(lenient=True, refresh=["nyx"]))
    assert asked == [["nyx"]] and json.loads(revs.read_text()) == {"nyx": "new-nyx", "hacc": "old-hacc"}
