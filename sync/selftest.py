#!/usr/bin/env python3
"""Exercise the published package on every file of every dataset.

    python sync/selftest.py --workdir /big/disk/tmp [--only nyx hacc] [--jobs 4]

For each variant and file, through the public API (the installed `sdrbench` package):
  * download to a plain directory (Field.download(dir)) and check the sha256 against the catalog
  * for arrays: ds[field] must have the catalog dtype and C-order shape, be C-contiguous, and its
    min/max must match the SDRBench property file when one lists the file; NaN/Inf are counted
  * derived (transposed) variants must equal the transpose of their stored files
  * archives smaller than --globus-max-gb are also fetched from Globus (forced fallback path) and
    every file must be byte-identical to the Hugging Face copy
Files are deleted after checking, so disk use stays at a few files. Writes selftest-report.json.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import sdrbench
from sdrbench.archive import sha256_file

PROP = re.compile(r"The property of (\S+?)\s*:.*?min\s*=\s*(\S+)\s*max\s*=\s*(\S+)", re.S)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def property_ranges(ds_name, work):
    """{file name: (min, max)} from the SDRBench property files of a dataset (text files on HF)."""
    out = {}
    for v in sdrbench.dataset(ds_name).variants:
        d = sdrbench.dataset(ds_name, v)
        for f in d.files:
            if f.filename.endswith(".txt") and "propert" in f.filename.lower():
                text = f.download(work / "props").read_text(errors="replace")
                for fname, lo, hi in PROP.findall(text):
                    try:
                        out.setdefault(fname, set()).add((float(lo), float(hi)))
                    except ValueError:
                        pass
    meta = sdrbench.catalog()["datasets"][ds_name]
    # metadata/ folder (not part of any variant): fetch through huggingface_hub directly
    from huggingface_hub import HfApi, hf_hub_download
    for p in HfApi().list_repo_files(meta["repo"], repo_type="dataset"):
        if p.startswith("metadata/") and "propert" in p.lower():
            text = Path(hf_hub_download(meta["repo"], p, repo_type="dataset", local_dir=str(work / "props"))).read_text(errors="replace")
            for fname, lo, hi in PROP.findall(text):
                try:
                    out.setdefault(fname, set()).add((float(lo), float(hi)))
                except ValueError:
                    pass
    return out


def _range(x, chunk=1 << 26):
    """min, max over finite values and the number of non-finite values, in bounded memory."""
    flat = x.reshape(-1)
    lo, hi, nbad = math.inf, -math.inf, 0
    for i in range(0, flat.size, chunk):
        c = np.asarray(flat[i:i + chunk])
        if c.dtype.kind == "f":
            ok = np.isfinite(c)
            nbad += int(c.size - np.count_nonzero(ok))
            c = c[ok]
        if c.size:
            lo, hi = min(lo, float(c.min())), max(hi, float(c.max()))
    return (lo if lo != math.inf else math.nan), (hi if hi != -math.inf else math.nan), nbad


def check_field(ds, f, work, ranges, plain_max=200e6):
    """Check one file through the public API; returns a result dict (None for derived layouts)."""
    if f.transpose:  # derived layout: checked against its stored file in check_view
        return None
    r = {"dataset": ds.name, "variant": ds.variant, "field": f.name, "path": f.path, "ok": True, "issues": []}
    tag = f"{ds.name}-{ds.variant}-{f.path}".replace("/", "_")
    dest, cache = work / f"plain-{tag}", work / f"cache-{tag}"
    api = sdrbench.dataset(ds.name, ds.variant, cache=cache)  # private cache per check
    try:
        if not f.dtype or f.nbytes <= plain_max:  # Field.download(dir): plain file, original name
            p = f.download(dest)
            if p.resolve() != (dest / f.path).resolve() or p.is_symlink() or sha256_file(p) != f.sha256:
                r["issues"].append(f"download(dir) gave {p} with wrong name/content")
        if f.dtype:
            x = api[f.name]  # public API: memmap with the catalog dtype and shape
            if x.dtype != np.dtype(f.dtype) or tuple(x.shape) != tuple(f.shape) or not x.flags.c_contiguous:
                r["issues"].append(f"ds[{f.name!r}] returned {x.dtype} {x.shape}")
            if sha256_file(api.field(f.name).download(cache=cache)) != f.sha256:  # file behind the array
                r["issues"].append("sha256 mismatch")
            lo, hi, nbad = _range(x)
            if nbad:
                r["nonfinite"] = nbad
            r["min"], r["max"] = lo, hi
            exp = ranges.get(f.filename)
            if exp:
                if any(math.isclose(lo, a, rel_tol=1e-4, abs_tol=1e-5) and math.isclose(hi, b, rel_tol=1e-4, abs_tol=1e-5)
                       for a, b in exp):
                    r["property_checked"] = True
                else:
                    r["issues"].append(f"min/max {lo:.6g}/{hi:.6g} != property file {sorted(exp)}")
            del x
    except Exception as e:  # report and keep going
        r["issues"].append(f"{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(dest, ignore_errors=True)
        shutil.rmtree(cache, ignore_errors=True)
    r["ok"] = not r["issues"]
    return r


def check_view(ds, f, work):
    """A transposed variant must equal the transpose of the stored file."""
    r = {"dataset": ds.name, "variant": ds.variant, "field": f.name, "path": f.path, "ok": True, "issues": []}
    base = None
    for v in sdrbench.dataset(ds.name).variants:
        b = sdrbench.dataset(ds.name, v)
        for g in b.files:
            if g.path == f.path and not g.transpose:
                base = g
    if base is None:
        r["issues"].append("stored file of the derived layout not found in any variant")
        r["ok"] = False
        return r
    cache = work / f"cache-view-{ds.name}-{ds.variant}"
    try:
        x = sdrbench.dataset(ds.name, ds.variant, cache=cache)[f.name]
        s = np.asarray(base.load(cache=cache)).transpose(f.transpose)
        if x.shape != tuple(f.shape) or not np.array_equal(x, s) or not x.flags.c_contiguous:
            r["issues"].append("derived layout does not equal the transposed stored file")
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    r["ok"] = not r["issues"]
    return r


def check_globus(name, variant, work):
    """Force the Globus path for a whole (small) archive; bytes must equal Hugging Face."""
    ds = sdrbench.dataset(name, variant)
    r = {"dataset": name, "variant": variant, "check": "globus", "ok": True, "issues": []}
    gdir, hdir = work / f"g-{name}-{variant}", work / f"h-{name}-{variant}"
    try:
        for f in ds.files:
            if f.transpose:
                continue
            g = f.download(gdir, cache=work / "gcache", source="globus")
            h = f.download(hdir, cache=work / "hcache", source="hf")
            if sha256_file(g) != sha256_file(h) or sha256_file(g) != f.sha256:
                r["issues"].append(f"{f.path}: Globus and HF bytes differ")
    except Exception as e:
        r["issues"].append(f"{type(e).__name__}: {e}")
    finally:
        for d in (gdir, hdir, work / "gcache", work / "hcache"):
            shutil.rmtree(d, ignore_errors=True)
    r["ok"] = not r["issues"]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--globus-max-gb", type=float, default=3.0)
    a = ap.parse_args()
    work = Path(a.workdir)
    work.mkdir(parents=True, exist_ok=True)
    results = []
    for name in sdrbench.list():
        if a.only and name not in a.only:
            continue
        ranges = {k: v for k, v in property_ranges(name, work).items()}
        for variant in sdrbench.dataset(name).variants:
            ds = sdrbench.dataset(name, variant)
            log(f"== {name}/{variant}: {len(ds.files)} files, {ds.nbytes / 1e9:.1f} GB")
            with ThreadPoolExecutor(a.jobs) as pool:
                rs = [r for r in pool.map(lambda f: check_field(ds, f, work, ranges), ds.files) if r]
            rs += [check_view(ds, f, work) for f in ds.files if f.transpose]
            arch = sdrbench.catalog()["datasets"][name]["variants"][variant]
            if "transpose" not in arch and arch["archive"]["bytes"] < a.globus_max_gb * 1e9:
                rs.append(check_globus(name, variant, work))
            bad = [r for r in rs if not r["ok"]]
            log(f"   {len(rs) - len(bad)}/{len(rs)} ok, {sum(r.get('property_checked', False) for r in rs)} min/max "
                f"checked against property files, {sum(1 for r in rs if r.get('nonfinite'))} with NaN/Inf")
            for r in bad:
                log("   FAIL", r["path"] if "path" in r else r, r["issues"])
            results += rs
    Path("selftest-report.json").write_text(json.dumps(results, indent=1))
    bad = [r for r in results if not r["ok"]]
    log(f"TOTAL {len(results)} checks, {len(bad)} failures")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
