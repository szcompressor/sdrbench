"""Integrity of the bundled catalog and of the rules that generate it."""
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sync"))
import sync  # noqa: E402

ITEM = {"<f4": 4, "<f8": 8, "<i4": 4, "<i8": 8, "<u1": 1}
import sdrbench  # noqa: E402

CATALOG = sdrbench.catalog()  # the installed package's catalog (works for wheel tests too)
CFG = json.loads((ROOT / "sync" / "datasets.json").read_text())


def all_files():
    for d, ds in CATALOG["datasets"].items():
        for v, vv in ds["variants"].items():
            for f in vv["files"]:
                yield d, v, f


def test_catalog_not_empty():
    assert CATALOG["datasets"]


def test_every_array_size_matches_shape():
    for d, v, f in all_files():
        assert f["path"].startswith(v + "/"), f
        assert len(f["sha256"]) == 64
        if f["dtype"]:
            assert math.prod(f["shape"]) * ITEM[f["dtype"]] == f["bytes"], (d, f["path"])
        else:
            assert f["shape"] is None


def test_paths_unique_and_repos_named():
    for d, ds in CATALOG["datasets"].items():
        paths = [f["path"] for v in ds["variants"].values() for f in v["files"]]
        assert len(paths) == len(set(paths)), d
        assert ds["repo"] == f"sdrbench/{d}"


def test_catalog_is_reproducible_from_state():
    """The committed catalog equals what sync/sync.py builds from sync/state.json."""
    state = json.loads((ROOT / "sync" / "state.json").read_text())
    built, errors = sync.build_catalog(CFG, state, strict=False)
    assert errors == []
    assert built == CATALOG


def test_no_excluded_dataset_is_mirrored():
    mirrored = {vv["archive"] for ds in CFG["datasets"].values() for vv in ds["variants"].values()}
    assert not mirrored & set(CFG["excluded"])
    assert "brown-samples" not in CATALOG["datasets"]
    assert not any("NSTX" in vv["archive"]["url"] for ds in CATALOG["datasets"].values() for vv in ds["variants"].values())


# --- rule engine -----------------------------------------------------------------
RULES = [{"glob": "*.f32", "dtype": "<f4", "shape": [2, 3]},
         {"glob": "*.d64", "dtype": "<f8", "shape": "1d"},
         {"glob": "*.bp", "dtype": None, "shape": None}]


def entry(path, nbytes, rules=RULES, variant="v"):
    return sync.file_entry("ds", variant, {"path": f"{variant}/{path}", "bytes": nbytes}, rules)


def test_rules_exact_and_1d():
    assert entry("a.f32", 24) == ("<f4", [2, 3])
    assert entry("sub/b.d64", 80) == ("<f8", [10])
    assert entry("c.bp", 123) == (None, None)
    assert entry("README.txt", 7) == (None, None)


def test_rules_reject_wrong_size_and_unknown_files():
    with pytest.raises(sync.CatalogError, match="!="):
        entry("a.f32", 28)
    with pytest.raises(sync.CatalogError, match="not a multiple"):
        entry("b.d64", 81)
    with pytest.raises(sync.CatalogError, match="no dtype/shape rule"):
        entry("mystery.dat", 8)


def test_rules_match_variant_with_slash():
    assert entry("xx.f32", 24, variant="multi-timesteps/1840") == ("<f4", [2, 3])


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
