#!/usr/bin/env python3
"""Keep the Hugging Face mirror of SDRBench in sync with the Globus originals.

    python sync/sync.py check                 # what changed on Globus / the SDRBench page?
    python sync/sync.py mirror --changed      # re-mirror changed archives to Hugging Face
    python sync/sync.py catalog               # rebuild src/sdrbench/catalog.json from sync/state.json
    python sync/sync.py cards                 # (re)write dataset cards on Hugging Face
    python sync/sync.py verify                # check dtypes/shapes against SDRBench property files

Globus stays the source of truth. state.json records, per archive, the HTTP
size/ETag/Last-Modified we mirrored plus the md5 of the archive and sha256 of
every extracted file.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
from sdrbench.archive import fetch_archive, sha256_file  # noqa: E402

DATASETS = HERE / "datasets.json"
STATE = HERE / "state.json"
CATALOG = ROOT / "src" / "sdrbench" / "catalog.json"
ORG = os.environ.get("SDRBENCH_HF_ORG", "sdrbench")
ARCHIVE_RE = re.compile(r'href="([^"]+\.(?:tar\.gz|tgz|zip|bp))"')
TEXT_RE = re.compile(r'href="([^"]+\.txt)"')
NON_ARRAY = (".txt", ".h5", ".hdf5", ".bp", ".zip", ".py", ".md", ".json", ".csv", ".xml", ".sh", ".c", ".h", ".cpp")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_json(p, default=None):
    return json.loads(Path(p).read_text()) if Path(p).exists() else default


def save_json(p, obj):
    Path(p).write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")


def cfg():
    return load_json(DATASETS)


def archives(c):
    """(dataset, variant, archive_rel) for every mirrored archive."""
    for d, ds in c["datasets"].items():
        for v, vv in ds["variants"].items():
            if "archive" in vv:  # derived views have no archive of their own
                yield d, v, vv["archive"]


def http_head(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "sdrbench-sync"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            h = r.headers
            return {"status": r.status, "bytes": int(h.get("Content-Length", -1)),
                    "etag": h.get("ETag"), "last_modified": h.get("Last-Modified")}
    except urllib.error.HTTPError as e:
        return {"status": e.code}


def page_links(page_html, base):
    arch = {html.unescape(u) for u in ARCHIVE_RE.findall(page_html)}
    txt = {html.unescape(u) for u in TEXT_RE.findall(page_html)}
    rel = lambda us: sorted(u[len(base):] for u in us if u.startswith(base))
    return rel(arch), rel(txt)


def fetch_text(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "sdrbench-sync"}), timeout=120) as r:
        return r.read().decode("utf-8", "replace")


# ---------------------------------------------------------------- check
def cmd_check(a):
    c, state = cfg(), load_json(STATE, {})
    base = c["globus_base"]
    arch_links, _ = page_links(fetch_text(c["page"]), base)
    mapped = {rel for _, _, rel in archives(c)}
    unmapped = [r for r in arch_links if r not in mapped and r not in c["excluded"]]
    removed = sorted(r for r in mapped if r not in arch_links)
    changed = []
    for d, v, rel in archives(c):
        head = http_head(base + rel)
        old = state.get(rel)
        if head.get("status") != 200:
            changed.append({"archive": rel, "reason": f"HTTP {head.get('status')}"})
        elif old is None:
            changed.append({"archive": rel, "reason": "not mirrored yet"})
        else:
            diffs = [k for k in ("bytes", "etag", "last_modified")
                     if old.get(k) is not None and head.get(k) is not None and old[k] != head[k]]
            if diffs:
                changed.append({"archive": rel, "reason": "changed " + ",".join(diffs)})
    report = {"changed": changed, "new_on_page": unmapped, "gone_from_page": removed}
    print(json.dumps(report, indent=1))
    if a.output:
        save_json(a.output, report)
    gh = os.environ.get("GITHUB_OUTPUT")
    if gh:
        with open(gh, "a") as f:
            f.write(f"changed={'true' if changed else 'false'}\n")
            f.write(f"attention={'true' if (unmapped or removed) else 'false'}\n")
    return 0


# ---------------------------------------------------------------- mirror
def upload_dir(api, repo, local, path_in_repo, message, delete=True):
    for attempt in range(5):
        try:
            api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(local),
                              path_in_repo=path_in_repo, commit_message=message,
                              delete_patterns="*" if delete else None)
            return
        except Exception as e:  # network hiccups on multi-GB uploads
            log(f"  upload attempt {attempt + 1} failed: {e}")
            time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"upload failed: {repo}/{path_in_repo}")


def mirror_metadata(api, c, dataset, work, dry):
    """Original property/template/README .txt files -> <repo>/metadata/."""
    base, gdir = c["globus_base"], c["datasets"][dataset]["globus_dir"]
    _, txts = page_links(fetch_text(c["page"]), base)
    mine = [t for t in txts if t.rsplit("/", 1)[0] == gdir]
    out = work / dataset / "metadata"
    out.mkdir(parents=True, exist_ok=True)
    for t in mine:
        (out / t.rsplit("/", 1)[1]).write_bytes(urllib.request.urlopen(base + t, timeout=120).read())
    if not dry and mine:
        upload_dir(api, f"{ORG}/{dataset}", out, "metadata", "Update SDRBench metadata files")
    return len(mine)


class _Restart(Exception):
    pass


def _members(url, scratch):
    """Yield (name, local_tmp_path) for every regular file of a .tar.gz/.zip, one at a time,
    so disk usage stays at one file (zip archives must be stored whole first).
    Also returns md5/size of the archive through the generator's StopIteration value."""
    import tarfile
    import zipfile
    from sdrbench.archive import _HashingReader, _is_junk, _open_url, CHUNK
    scratch.mkdir(parents=True, exist_ok=True)
    with _open_url(url) as resp:
        reader = _HashingReader(resp)
        if url.endswith(".zip"):
            zpath = scratch / "_archive.zip"
            with open(zpath, "wb") as out:
                shutil.copyfileobj(reader, out, CHUNK)
            with zipfile.ZipFile(zpath) as zf:
                for info in zf.infolist():
                    if info.is_dir() or _is_junk(info.filename):
                        continue
                    tmp = scratch / f"m{time.time_ns()}"
                    with zf.open(info) as src, open(tmp, "wb") as out:
                        shutil.copyfileobj(src, out, CHUNK)
                    yield info.filename, tmp
            zpath.unlink()
        else:
            with tarfile.open(fileobj=reader, mode="r|*") as tf:
                for m in tf:
                    if not m.isfile() or _is_junk(m.name):
                        continue
                    tmp = scratch / f"m{time.time_ns()}"
                    with open(tmp, "wb") as out:
                        shutil.copyfileobj(tf.extractfile(m), out, CHUNK)
                    yield m.name, tmp
            reader.drain()
    return reader.md5.hexdigest(), reader.nbytes


def _commit(api, repo, ops, message):
    for attempt in range(5):
        try:
            api.create_commit(repo_id=repo, repo_type="dataset", operations=ops, commit_message=message)
            return
        except Exception as e:  # network hiccups on multi-GB uploads
            log(f"   commit attempt {attempt + 1} failed: {e}")
            time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"commit to {repo} failed")


def stream_mirror(api, repo, variant, url, scratch, batch_bytes=8e9, flatten=True):
    """Mirror one archive into <repo>/<variant>/ without unpacking it all: each member is
    written to a temp file, hashed, and committed in batches, then deleted. Files that are no
    longer in the archive are removed from the repo. Returns (files, md5, archive bytes).

    Like sdrbench.archive.fetch_archive, a single top-level directory is dropped. That is
    only known at the end, so we assume it from the first member and restart without
    flattening in the (never yet seen) case that a later member breaks the assumption."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    top, files, batch, nbatch = None, [], [], 0
    gen = _members(url, scratch)
    try:
        while True:
            try:
                name, tmp = next(gen)
            except StopIteration as stop:
                md5, nbytes = stop.value
                break
            parts = Path(name).parts
            if flatten and top is None:
                top = parts[0] if len(parts) > 1 else ""
            if flatten and top and (len(parts) < 2 or parts[0] != top):
                raise _Restart()
            rel = "/".join(parts[1:] if flatten and top else parts)
            size = tmp.stat().st_size
            files.append({"path": f"{variant}/{rel}", "bytes": size, "sha256": sha256_file(tmp)})
            batch.append((f"{variant}/{rel}", tmp))
            nbatch += size
            if api and (nbatch >= batch_bytes or len(batch) >= 50):
                _commit(api, repo, [CommitOperationAdd(p, str(t)) for p, t in batch],
                        f"Sync {variant} from SDRBench ({len(files)} files so far)")
            if not api or nbatch >= batch_bytes or len(batch) >= 50:
                for _, t in batch:
                    t.unlink()
                batch, nbatch = [], 0
    except _Restart:
        gen.close()
        shutil.rmtree(scratch, ignore_errors=True)
        log("   archive has several top-level entries; restarting without flattening")
        return stream_mirror(api, repo, variant, url, scratch, batch_bytes, flatten=False)
    if api:
        new = {f["path"] for f in files}
        old = [p for p in api.list_repo_files(repo, repo_type="dataset") if p.startswith(variant + "/")]
        ops = [CommitOperationAdd(p, str(t)) for p, t in batch]
        ops += [CommitOperationDelete(p) for p in old if p not in new]
        if ops:
            _commit(api, repo, ops, f"Sync {variant} from SDRBench ({len(files)} files)")
    for _, t in batch:
        t.unlink(missing_ok=True)
    shutil.rmtree(scratch, ignore_errors=True)
    return files, md5, nbytes


def cmd_mirror(a):
    c = cfg()
    state_path = Path(a.state_out) if a.state_out else STATE
    state = load_json(STATE, {})
    out_state = {} if a.state_out else state
    base = c["globus_base"]
    todo = list(archives(c))
    if a.changed:
        rep = load_json(a.changed)
        if rep is None:
            print("--changed needs the JSON written by `check --output`", file=sys.stderr)
            return 2
        want = {x["archive"] for x in rep["changed"]}
        todo = [t for t in todo if t[2] in want]
    if a.only:
        todo = [t for t in todo if t[0] in a.only or t[2] in a.only]
    api = None
    if not a.dry:
        from huggingface_hub import HfApi
        api = HfApi()
    work = Path(a.workdir or tempfile.mkdtemp(prefix="sdrbench-"))
    work.mkdir(parents=True, exist_ok=True)
    skipped, done_ds = [], set()
    for dataset, variant, rel in todo:
        url = base + rel
        head = http_head(url)
        if head.get("status") != 200:
            log(f"skip {rel}: HTTP {head.get('status')}")
            skipped.append(rel)
            continue
        if api and dataset not in done_ds:
            api.create_repo(f"{ORG}/{dataset}", repo_type="dataset", exist_ok=True)
        log(f"== {dataset}/{variant} <- {url} ({head['bytes']/1e9:.2f}GB, {shutil.disk_usage(work).free/1e9:.0f}GB free)")
        t0 = time.time()
        files, md5, n = stream_mirror(api, f"{ORG}/{dataset}", variant, url, work / "scratch")
        log(f"   {len(files)} files, archive md5 {md5}, {time.time() - t0:.0f}s")
        out_state[rel] = {"dataset": dataset, "variant": variant, "url": url, "bytes": n, "md5": md5,
                          "etag": head.get("etag"), "last_modified": head.get("last_modified"),
                          "files": files, "synced": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        save_json(state_path if a.state_out else (work / "state.dry.json" if a.dry else STATE), out_state)
        if dataset not in done_ds:
            mirror_metadata(api, c, dataset, work, a.dry)
            done_ds.add(dataset)
    if not a.workdir:
        shutil.rmtree(work, ignore_errors=True)
    if not a.dry and not a.state_out:
        cmd_catalog(a)
    if skipped:
        log("NOT mirrored (needs attention):", *skipped)
        return 3 if a.fail_on_skip else 0
    return 0


def cmd_merge(a):
    """Merge state fragments written by parallel `mirror --state-out` jobs into state.json."""
    state = load_json(STATE, {})
    for p in a.fragments:
        state.update(load_json(p, {}))
    save_json(STATE, state)
    log(f"state.json: {len(state)} archives after merging {len(a.fragments)} fragments")
    return 0


def cmd_plan(a):
    """Datasets to mirror (JSON list for a CI matrix) from `check --output`."""
    rep = load_json(a.check, {"changed": []})
    by_arch = {rel: d for d, _, rel in archives(cfg())}
    ds = sorted({by_arch[x["archive"]] for x in rep["changed"] if x["archive"] in by_arch} | set(a.extra or []))
    print(json.dumps(ds))
    gh = os.environ.get("GITHUB_OUTPUT")
    if gh:
        with open(gh, "a") as f:
            f.write(f"datasets={json.dumps(ds)}\n")
    return 0


# ---------------------------------------------------------------- catalog
ITEMSIZE = {"<f4": 4, "<f8": 8, "<i4": 4, "<i8": 8, "<u1": 1, "<u2": 2, "<u8": 8}


class CatalogError(ValueError):
    pass


def match_rule(path, rules):
    """First rule whose glob matches the file name (or its path inside the variant)."""
    import fnmatch
    name = path.rsplit("/", 1)[-1]
    for r in rules:
        if fnmatch.fnmatch(name, r["glob"]) or fnmatch.fnmatch(path, r["glob"]):
            return r
    return None


DTYPE_SUFFIXES = (".bin.f32", ".pre.f32.dat", ".f32.dat", ".bin.d64", ".f32", ".f64", ".d64", ".u16", ".u64", ".i64", ".dat", ".bin",
                  ".raw", "_double", "_float", "_int32_t", "_int64_t")
TRAILING_DIMS = re.compile(r"[_-]\d+(?:[x_]\d+)+$")  # _1_1800_3600, -98x1200x1200, _115_69_69_288


def field_name(rel, rule):
    """Short field name: the physical variable, without dtype/dimension decorations.
    A rule may give 'pattern' (regex on the path inside the variant) and 'name' (str.format
    template over the groups); otherwise suffixes and trailing dimensions are stripped."""
    if rule and rule.get("pattern"):
        m = re.fullmatch(rule["pattern"], rel)
        if not m:
            raise CatalogError(f"{rel}: does not match name pattern {rule['pattern']!r}")
        return rule["name"].format(*m.groups(), **m.groupdict())
    n = rel
    for suf in DTYPE_SUFFIXES:
        if n.endswith(suf) and len(n) > len(suf):
            n = n[: -len(suf)]
            break
    head, _, tail = n.rpartition("/")
    tail = TRAILING_DIMS.sub("", tail) or tail
    return f"{head}/{tail}" if head else tail


def file_entry(dataset, variant, f, rules):
    """(dtype, shape, name) for one file from the variant's explicit rules; sizes must agree exactly."""
    rel = f["path"][len(variant) + 1:] if f["path"].startswith(variant + "/") else f["path"]
    r = match_rule(rel, rules)
    name = f["path"].rsplit("/", 1)[-1]
    if r is None:
        if name.lower().endswith(NON_ARRAY):
            return None, None, None
        raise CatalogError(f"{dataset}/{f['path']}: no dtype/shape rule (add one to sync/datasets.json)")
    if r.get("dtype") is None:
        return None, None, None
    item = ITEMSIZE[r["dtype"]]
    shape = r["shape"]
    if shape == "1d":
        if f["bytes"] % item:
            raise CatalogError(f"{dataset}/{f['path']}: {f['bytes']} bytes is not a multiple of {item}")
        shape = [f["bytes"] // item]
    if math.prod(shape) * item != f["bytes"]:
        raise CatalogError(f"{dataset}/{f['path']}: shape {shape} x {item}B != {f['bytes']} bytes")
    return r["dtype"], list(shape), field_name(rel, r)


def build_catalog(c, state, strict=True):
    out = {"schema": 1, "org": ORG, "globus_base": c["globus_base"], "page": c["page"], "datasets": {}}
    errors = []
    for d, ds in c["datasets"].items():
        entry = {k: ds[k] for k in ("title", "description", "source", "acknowledgment", "citation_extra") if k in ds}
        entry["repo"] = f"{ORG}/{d}"
        entry["variants"] = {}
        for v, vv in ds["variants"].items():
            if "view_of" in vv:
                continue  # added below, once the base variant exists
            st = state.get(vv["archive"])
            if not st:
                continue
            fl = []
            for f in st["files"]:
                try:
                    dt, shape, fname = file_entry(d, v, f, vv.get("rules", []))
                except CatalogError as e:
                    errors.append(str(e))
                    dt, shape, fname = None, None, None
                fl.append({"path": f["path"], "name": fname, "bytes": f["bytes"], "sha256": f["sha256"],
                           "dtype": dt, "shape": shape})
            names = [x["name"] for x in fl if x["name"]]
            dups = sorted({n for n in names if names.count(n) > 1})
            if dups:
                errors.append(f"{d}/{v}: duplicate field names {dups} (add a 'pattern'/'name' rule)")
            entry["variants"][v] = {"archive": {"url": st["url"], "bytes": st["bytes"], "md5": st["md5"]},
                                    "files": fl}
        for v, vv in ds["variants"].items():
            base = entry["variants"].get(vv.get("view_of"))
            if "view_of" not in vv or base is None:
                continue
            axes = vv["transpose"]
            entry["variants"][v] = {
                "archive": base["archive"], "archive_variant": vv["view_of"], "note": vv.get("note"),
                "files": [{**f, "shape": [f["shape"][i] for i in axes], "transpose": axes}
                          for f in base["files"] if f["dtype"] and len(f["shape"]) == len(axes)]}
        if entry["variants"]:
            out["datasets"][d] = entry
    if errors and strict:
        raise CatalogError("\n".join(errors))
    return out, errors


def cmd_catalog(a):
    cat, errors = build_catalog(cfg(), load_json(STATE, {}), strict=not getattr(a, "lenient", False))
    for e in errors:
        log("RULE ERROR", e)
    CATALOG.write_text(json.dumps(cat, indent=1) + "\n")
    nfiles = sum(len(v["files"]) for d in cat["datasets"].values() for v in d["variants"].values())
    log(f"catalog: {len(cat['datasets'])} datasets, {nfiles} files -> {CATALOG.relative_to(ROOT)}")
    return 0


# ---------------------------------------------------------------- cards
SDRBENCH_BIBTEX = """@inproceedings{zhao2020sdrbench,
  title     = {{SDRBench}: Scientific Data Reduction Benchmark for Lossy Compressors},
  author    = {Zhao, Kai and Di, Sheng and Liang, Xin and Li, Sihuan and Tao, Dingwen and Chen, Zizhong and Cappello, Franck},
  booktitle = {2020 IEEE International Conference on Big Data (Big Data)},
  pages     = {2716--2724},
  year      = {2020}
}"""


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or u == "TB":
            return f"{n:.1f} {u}" if u != "B" else f"{n} B"
        n /= 1000


def _example(name, d):
    """(variant, field) used in the card examples: the smallest array of the default variant."""
    v0 = next(iter(d["variants"]))
    arrays = [f for f in d["variants"][v0]["files"] if f["dtype"] and len(f["shape"]) > 1] or \
             [f for f in d["variants"][v0]["files"] if f["dtype"]]
    return v0, (min(arrays, key=lambda f: f["bytes"]) if arrays else None)


def card_python(name, d):
    """The runnable usage example shown on the card (also executed by the online tests)."""
    v0, f = _example(name, d)
    if f is None:
        return f'import sdrbench\nds = sdrbench.dataset("{name}")\npaths = ds.download()\n'
    shape = tuple(f["shape"])
    return f"""import numpy as np
import sdrbench
from pysz import sz, szConfig, szErrorBoundMode

ds = sdrbench.dataset("{name}")         # default variant: {v0}
print(ds.fields)
x = ds["{f['name']}"]                     # numpy array, dtype {f['dtype']}, shape {shape}

# compress with SZ3 (pysz) at a 1e-3 value-range-relative error bound
conf = szConfig()
conf.errorBoundMode = szErrorBoundMode.REL
conf.relErrorBound = 1e-3
compressed, ratio = sz.compress(np.ascontiguousarray(x), conf)
y, _ = sz.decompress(compressed, x.dtype.type, x.shape)
max_err, psnr, nrmse = sz.verify(np.asarray(x), y)
print(f"ratio {{ratio:.1f}}x, PSNR {{psnr:.1f}} dB, max error {{max_err:.3g}}")
"""


def render_card(name, d):
    total = sum(f["bytes"] for v in d["variants"].values() if "archive_variant" not in v for f in v["files"])
    sections = []
    for v, vv in d["variants"].items():
        arr = [f for f in vv["files"] if f["dtype"]]
        other = [f for f in vv["files"] if not f["dtype"]]
        head = f"### `{v}`" + (" (default)" if v == next(iter(d["variants"])) else "")
        lines = [head, ""]
        if vv.get("note"):
            lines += [vv["note"], ""]
        if "archive_variant" in vv:
            lines += [f"Derived layout: no extra files; computed on load from `{vv['archive_variant']}/`.", ""]
        else:
            lines += [f"{len(vv['files'])} files, {human(sum(f['bytes'] for f in vv['files']))}, "
                      f"from [{vv['archive']['url'].rsplit('/', 1)[-1]}]({vv['archive']['url']}).", ""]
        if arr:
            lines += ["| Field | dtype | Shape (C order) | File |", "|---|---|---|---|"]
            shown = arr if len(arr) <= 30 else arr[:12]
            lines += [f"| `{f['name']}` | `{f['dtype']}` | {' x '.join(map(str, f['shape']))} | `{f['path']}` |" for f in shown]
            if len(arr) > len(shown):
                lines += [f"| ... {len(arr) - len(shown)} more | | | |"]
        if other:
            lines += ["", "Other files: " + ", ".join(f"`{f['path']}`" for f in other[:10])
                      + (" ..." if len(other) > 10 else "")]
        sections.append("\n".join(lines))
    extra = ""
    if d.get("acknowledgment"):
        extra += f"\n**Acknowledgment requested by the data provider:** {d['acknowledgment']}\n"
    if d.get("citation_extra"):
        extra += f"\n**Citation requested by the data provider:** {d['citation_extra']}\n"
    v0, f = _example(name, d)
    plain = ""
    if f:
        plain = f"""
Without the package, any file can be read with `huggingface_hub` and numpy:

```python
from huggingface_hub import hf_hub_download
import numpy as np
p = hf_hub_download("{d['repo']}", "{f['path']}", repo_type="dataset")
x = np.fromfile(p, dtype="{f['dtype']}").reshape({tuple(f['shape'])})
```
"""
    others = [v for v in d["variants"] if v != v0]
    variant_line = (f'\nOther variants: `sdrbench.dataset("{name}", "{others[0]}")`'
                    + (f" (all: {', '.join(f'`{v}`' for v in others)})" if len(others) > 1 else "") + ".\n") if others else ""
    return f"""---
license: other
license_name: sdrbench
license_link: https://sdrbench.github.io/
pretty_name: "SDRBench: {d['title']}"
viewer: false
tags:
- scientific-data
- lossy-compression
- sdrbench
- hpc
- sz3
---

# SDRBench — {d['title']}

{d['description']}

This repository is an **unmodified mirror** of the {d['title']} dataset from
[SDRBench]({CAT_PAGE}), the Scientific Data Reduction Benchmark. The originals are hosted by
Argonne National Laboratory on Globus; every archive was unpacked and its files uploaded
byte-for-byte (sha256 verified). `metadata/` holds the original SDRBench property and template
files. The mirror is kept in sync automatically by
[szcompressor/sdrbench](https://github.com/szcompressor/sdrbench).

- **Data source:** {d['source']}
- **Total size:** {human(total)}
- **Format:** raw little-endian binary, C order (slowest dimension first)
{extra}
## Usage

```bash
pip install "sdrbench[sz3]"     # sdrbench + pysz (SZ3)
```

```python
{card_python(name, d).rstrip()}
```
{variant_line}
Files are downloaded on first use and cached; if Hugging Face is unreachable the package falls
back to the original archive on Globus.
{plain}
## Contents

{chr(10).join(chr(10) + s for s in sections)}

## Citation

If you use this dataset, please cite SDRBench{' and the data provider (see above)' if extra else ''}:

```bibtex
{SDRBENCH_BIBTEX}
```

## License and terms

The data are distributed under the same terms as on the [SDRBench website]({CAT_PAGE}).
Rights remain with the original data providers listed above.
"""


CAT_PAGE = "https://sdrbench.github.io/datasets.html"


def cmd_cards(a):
    cat = json.loads(CATALOG.read_text())
    api = None
    if not a.dry:
        from huggingface_hub import HfApi
        api = HfApi()
    outdir = Path(a.outdir) if a.outdir else None
    for name, d in cat["datasets"].items():
        if a.only and name not in a.only:
            continue
        card = render_card(name, d)
        if outdir:
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / f"{name}.md").write_text(card)
        if api:
            api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                            repo_id=f"{ORG}/{name}", repo_type="dataset",
                            commit_message="Update dataset card")
            log("card ->", f"{ORG}/{name}")
    return 0


# ---------------------------------------------------------------- verify
PROP_RE = re.compile(r"The property of (\S+?)\s*:\s*.*?The first 10 values are:\s*(.*?)\.\.\.\..*?numOfElem\s*=\s*(\d+)",
                     re.S)


def read_range(repo, path, start, length):
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers
    url = hf_hub_url(repo, path, repo_type="dataset")
    req = urllib.request.Request(url, headers={**build_hf_headers(),
                                               "Range": f"bytes={start}-{start + length - 1}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    if len(data) != length:
        raise IOError(f"range read of {path} returned {len(data)} bytes, wanted {length}")
    return data


def roughness(block, shape):
    """Mean |first difference| along every axis, relative to the data's spread."""
    import numpy as np
    k = block.size // math.prod(shape[1:])
    x = block[: k * math.prod(shape[1:])].reshape((k, *shape[1:])).astype(np.float64)
    spread = np.mean(np.abs(x - x.mean())) or 1.0
    return float(np.mean([np.mean(np.abs(np.diff(x, axis=ax))) for ax in range(x.ndim) if x.shape[ax] > 1]) / spread)


def layout_check(repo, f, max_bytes=256 << 20):
    """Is the catalog shape smoother than every other ordering of the same dimensions?"""
    import itertools
    import numpy as np
    shape = tuple(f["shape"])
    alts = sorted({p for p in itertools.permutations(shape) if p != shape})
    if not alts:
        return None  # all dimensions equal: ordering cannot be wrong
    item = ITEMSIZE[f["dtype"]]
    slab = max(math.prod(s[1:]) for s in [shape, *alts])
    n = min(4 * slab, f["bytes"] // item, max_bytes // item)
    if n < 2 * slab:
        return None
    start = ((f["bytes"] // item - n) // 2) * item  # middle of the file (first frames may be placeholders)
    block = np.frombuffer(read_range(repo, f["path"], start, n * item), dtype=f["dtype"])
    if not np.isfinite(block).all() or block.std() == 0:
        block = np.nan_to_num(block)
        if block.std() == 0:
            return None
    score = roughness(block, shape)
    best_alt = min(alts, key=lambda s: roughness(block, s))
    return score, roughness(block, best_alt), best_alt


def cmd_verify(a):
    """1) first 10 values and element counts from SDRBench property files vs. the catalog
       dtype/shape applied to the bytes on Hugging Face; 2) for every folder of non-cubic arrays,
       most files must be smoother in the catalog's C-order shape than in any other ordering
       (all files of one folder share a layout; a few fields such as top-of-atmosphere solar
       flux are too regular for this test and are reported as 'odd')."""
    from huggingface_hub import HfApi, hf_hub_download

    cat = json.loads(CATALOG.read_text())
    api = HfApi()
    problems, nprop, nlayout = [], 0, 0
    for name, d in cat["datasets"].items():
        if a.only and name not in a.only:
            continue
        # derived (transposed) variants are checked through their base files
        arrays = [f for v in d["variants"].values() if "archive_variant" not in v for f in v["files"] if f["dtype"]]
        byname = {}
        for f in arrays:
            byname.setdefault(f["path"].rsplit("/", 1)[-1], []).append(f)
        for p in [p for p in api.list_repo_files(d["repo"], repo_type="dataset") if p.endswith(".txt")]:
            text = Path(hf_hub_download(d["repo"], p, repo_type="dataset")).read_text(errors="replace")
            for fname, vals, nelem in PROP_RE.findall(text):
                expect = [float(x) for x in vals.split()][:10]
                cands = byname.get(fname, [])
                if not cands:
                    continue
                # several variants can hold a file of the same name: compare with the one(s) whose
                # element count matches the property file
                same_n = [f for f in cands if math.prod(f["shape"]) == int(nelem)]
                if not same_n:
                    nprop += 1
                    problems.append(f"{name}/{fname} (property: no file with numOfElem={nelem})")
                    print(f"[property] MISMATCH {name}/{fname}: no file with numOfElem={nelem} "
                          f"(catalog shapes {[f['shape'] for f in cands]})")
                    continue
                for f in same_n:
                    nprop += 1
                    item = ITEMSIZE[f["dtype"]]
                    raw = read_range(d["repo"], f["path"], 0, len(expect) * item)
                    import numpy as np
                    got = np.frombuffer(raw, dtype=f["dtype"]).astype(float).tolist()
                    ok = all(math.isclose(g, e, rel_tol=1e-4, abs_tol=2e-6) for g, e in zip(got, expect))
                    print(f"[property] {'ok' if ok else 'MISMATCH':8s} {name}/{f['path']}  shape={f['shape']} "
                          f"numOfElem={nelem}" + ("" if ok else f"  first values {got[:3]} vs {expect[:3]}"))
                    if not ok:
                        problems.append(f"{name}/{f['path']} (property)")
        groups = {}
        for f in arrays:
            groups.setdefault((f["path"].rsplit("/", 1)[0], tuple(f["shape"])), []).append(f)
        for (folder, shape), fs in groups.items():
            sample = fs if a.all else fs[:: max(1, len(fs) // 5)][:5]  # up to 5 files per group
            votes = []
            for f in sample:
                res = layout_check(d["repo"], f)
                if res is None:
                    continue
                nlayout += 1
                score, alt_score, alt = res
                votes.append(score < alt_score)
                print(f"[layout]   {'ok' if votes[-1] else 'odd':8s} {name}/{f['path']}  {f['shape']} roughness={score:.3f}"
                      f"  best other order {list(alt)}={alt_score:.3f}")
            if votes and sum(votes) * 2 <= len(votes):
                problems.append(f"{name}/{folder} shape {list(shape)}: only {sum(votes)}/{len(votes)} files favour it (layout)")
    print(f"\n{nprop} property checks, {nlayout} layout checks, {len(problems)} problems")
    for p in problems:
        print("  PROBLEM", p)
    return 1 if problems else 0


# ---------------------------------------------------------------- import from the initial lab run
def cmd_import(a):
    """Bootstrap state.json from the per-archive JSON records of the initial mirror run."""
    c, state = cfg(), load_json(STATE, {})
    by_variant = {(d, v): rel for d, v, rel in archives(c)}
    for p in sorted(Path(a.dir).glob("*.json")):
        r = json.loads(p.read_text())
        rel = by_variant[(r["repo"], r["subdir"])]
        head = http_head(c["globus_base"] + rel)
        state[rel] = {"dataset": r["repo"], "variant": r["subdir"], "url": r["source_url"],
                      "bytes": r["archive_bytes"], "md5": r["archive_md5"],
                      "etag": head.get("etag"), "last_modified": head.get("last_modified"),
                      "files": r["files"], "synced": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(p.stat().st_mtime))}
    save_json(STATE, state)
    log(f"state.json: {len(state)} archives")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("check"); s.add_argument("--output")
    s = sub.add_parser("mirror")
    s.add_argument("--changed", metavar="CHECK_JSON", help="only archives listed by `check --output`")
    s.add_argument("--only", nargs="*", help="dataset names or archive paths")
    s.add_argument("--workdir")
    s.add_argument("--dry", action="store_true", help="download and hash, but do not upload")
    s.add_argument("--fail-on-skip", action="store_true")
    s.add_argument("--state-out", help="write state of mirrored archives to this file instead of state.json")
    s = sub.add_parser("merge"); s.add_argument("fragments", nargs="+")
    s = sub.add_parser("plan"); s.add_argument("check"); s.add_argument("--extra", nargs="*")
    s = sub.add_parser("catalog"); s.add_argument("--lenient", action="store_true", help="report rule errors instead of failing")
    s = sub.add_parser("cards"); s.add_argument("--only", nargs="*"); s.add_argument("--dry", action="store_true"); s.add_argument("--outdir")
    s = sub.add_parser("verify"); s.add_argument("--only", nargs="*"); s.add_argument("--all", action="store_true")
    s = sub.add_parser("import"); s.add_argument("dir")
    a = p.parse_args(argv)
    return {"check": cmd_check, "mirror": cmd_mirror, "catalog": cmd_catalog, "cards": cmd_cards,
            "verify": cmd_verify, "import": cmd_import, "merge": cmd_merge, "plan": cmd_plan}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
