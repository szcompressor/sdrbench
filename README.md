# sdrbench

The [SDRBench](https://sdrbench.github.io/) scientific datasets as numpy arrays, served from the
[Hugging Face mirror](https://huggingface.co/sdrbench) (byte-for-byte copies of the original
archives, pinned and sha256-verified), with automatic fallback to the original Globus archives at
Argonne.

```bash
pip install sdrbench            # or "sdrbench[sz3]" to also get pysz (SZ3)
```

```python
import sdrbench

sdrbench.list()                          # ['basilisk-turbulence', 'cesm-atm', 'exaalt', ...]
sdrbench.list(variants=True)             # ['basilisk-turbulence/small', ..., 'nyx/original', 'nyx/log', ...]

nyx = sdrbench.dataset("nyx")            # default variant; sdrbench.dataset("nyx", "log")
nyx.fields                               # ['baryon_density', 'dark_matter_density', 'temperature', ...]
t = nyx["temperature"]                   # numpy memmap, float32, shape (512, 512, 512), C order
sdrbench.load("cesm-atm", "CLDHGH")      # one-liner

# time series: fields are "var/step"
hur = sdrbench.dataset("hurricane-isabel", "P")
hur.steps                                # ['01', ..., '48']
p = hur.series("P")                      # ordered sequence of fields
x = hur["P", "07"]                       # one step
block = p.stack(slice(0, 4))             # (4, 100, 500, 500) copy

# stacked files are split into zero-copy views per variable
T = sdrbench.dataset("s3d")["T/1.1000E-03"]      # (500, 500, 500), component 6 of the stored file

# raw files to a directory (original SDRBench names, downloaded in parallel)
nyx.download("data/")                    # -> data/512x512x512/*.f32
sdrbench.dataset("nyx", root="data/")    # later: use local files first (also $SDRBENCH_DATA)

print(nyx.citation())                    # BibTeX + provider request + exact data version
```

Compress with SZ3 through [pysz](https://pypi.org/project/pysz/):

```python
import numpy as np
from pysz import sz, szConfig, szErrorBoundMode

x = sdrbench.dataset("scale-letkf")["T"]
conf = szConfig()
conf.errorBoundMode = szErrorBoundMode.REL
conf.relErrorBound = 1e-3
compressed, ratio = sz.compress(np.ascontiguousarray(x), conf)
y, _ = sz.decompress(compressed, x.dtype.type, x.shape)
print(ratio, sz.verify(np.asarray(x), y))  # ratio, (max error, PSNR, NRMSE)
```

Command line:

```bash
sdrbench list                                   # every dataset/variant with field count and size
sdrbench info hurricane-isabel/P                # description, fields, dtype, shape, files
sdrbench download nyx temperature -o data/      # plain files under data/<repo path>, -j parallel
sdrbench path cesm-atm CLDHGH                   # local path (downloads if needed), e.g. for `sz3 -i`
sdrbench cite qmcpack
```

## How it works

- **Datasets, variants, fields.** A dataset (`nyx`) has variants with descriptive names (`original`,
  `log`; `cesm-atm`: `2d`, `2d-cleared`, `3d`; `hacc`: `medium`, `big`, `sbig`, `region1`...;
  `hurricane-isabel`: `snapshot`, `P`, `QCLOUD`, ...). The first is the default. The SDRBench / Hugging
  Face folder name also works as a variant name (`dataset("nyx", "512x512x512")`).
- **Field names are physical variables** (`CLDHGH`, `T`, `temperature`), or `var/step` for time series
  and slabs (`P/07`, `xx/00042`, `density/31`); `ds.variables`, `ds.steps` and `ds.series(var)` expose the
  structure. The original SDRBench file name also works (`ds["CLDHGH_1_1800_3600.f32"]`), lookups are
  case-insensitive, and errors suggest close matches. Non-array files are in `ds.extra_files`.
- **dtype and shape come from the catalog, never from names. Shapes are C order** (slowest first; the
  convention of the SZ3 test-suite table): `np.fromfile(path, dtype).reshape(field.stored_shape)` reads
  any stored file. `field.shape_fastest_first` is the order command-line compressors expect.
- **Views, no copies of the data.** QMCPACK's default variant `preconditioned` (288 x 115 x 69 x 69, the
  layout of the SDRBench examples) is computed on load by transposing the stored native file (variant
  `original`); S3D's stacked files are split into per-variable memmap views. `field.save(path)` writes
  the array exactly as loaded.
- **Downloads** come from Hugging Face, pinned to the commit this release's catalog was built from and
  checked against the catalog sha256, and are cached (`SDRBENCH_CACHE` moves the cache). Local copies are
  used first when `root=` / `$SDRBENCH_DATA` is given (an untarred SDRBench directory works). If Hugging
  Face cannot deliver, the package falls back to the original archive on Globus (whole archive, md5 and
  sha256 verified, size shown in a warning; resumes dropped connections). `source="globus"` forces it.
- Arrays are read-only memmaps; `np.array(x)` (or `load(mmap=False)`) gives a writable copy, e.g. for
  `torch.from_numpy`.

## Datasets

| Dataset | Hugging Face | Variants | Source |
|---|---|---|---|
| CESM-ATM | [sdrbench/cesm-atm](https://huggingface.co/datasets/sdrbench/cesm-atm) | 2d, 2d-cleared, 3d | climate (SNL) |
| EXAALT | [sdrbench/exaalt](https://huggingface.co/datasets/sdrbench/exaalt) | small, copper-1/2, helium-1/2 | molecular dynamics |
| Hurricane ISABEL | [sdrbench/hurricane-isabel](https://huggingface.co/datasets/sdrbench/hurricane-isabel) | snapshot, P, U, ..., CLOUD_log10, ... | weather (NCAR, IEEE Vis 2004) |
| EXAFEL | [sdrbench/exafel](https://huggingface.co/datasets/sdrbench/exafel) | small, large, assembled | LCLS X-ray images |
| HACC | [sdrbench/hacc](https://huggingface.co/datasets/sdrbench/hacc) | medium, big, sbig, region1-6 | cosmology particles |
| NYX | [sdrbench/nyx](https://huggingface.co/datasets/sdrbench/nyx) | original, log | cosmology |
| NWChem | [sdrbench/nwchem](https://huggingface.co/datasets/sdrbench/nwchem) | default, f32 | quantum chemistry |
| SCALE-LETKF | [sdrbench/scale-letkf](https://huggingface.co/datasets/sdrbench/scale-letkf) | original, log | weather (RIKEN) |
| QMCPACK | [sdrbench/qmcpack](https://huggingface.co/datasets/sdrbench/qmcpack) | preconditioned, original | quantum Monte Carlo |
| Miranda | [sdrbench/miranda](https://huggingface.co/datasets/sdrbench/miranda) | small, big | turbulence (LLNL) |
| S3D | [sdrbench/s3d](https://huggingface.co/datasets/sdrbench/s3d) | default | combustion (SNL) |
| Basilisk-Turbulence | [sdrbench/basilisk-turbulence](https://huggingface.co/datasets/sdrbench/basilisk-turbulence) | small, large | 2D turbulence |
| XGC | [sdrbench/xgc](https://huggingface.co/datasets/sdrbench/xgc) | raw, adios | fusion (PPPL) |

`sdrbench list` and `sdrbench info` show fields and sizes. NSTX GPI is not mirrored (its owner asks to
be contacted before results are published); HACC region 5 is not available on Globus (HTTP 404).

Please cite SDRBench and the data provider named on each dataset card (`sdrbench cite <dataset>`), and
report the `sdrbench` version you used: a release always reads the same data.

## Keeping the mirror in sync

Globus remains the source of truth. `.github/workflows/release.yml` runs weekly:

1. `sync/sync.py check` HEADs every archive linked from the SDRBench page and compares size, ETag and
   Last-Modified with `sync/state.json`; archives that appear on, or disappear from, the page open an issue.
2. Changed archives are re-mirrored in parallel jobs (`sync.py mirror`). Archives are streamed: each file is
   unpacked, hashed and committed to `sdrbench/<dataset>` in batches and then deleted, so a runner only needs
   room for the largest single file (~17 GB); the job frees disk like the SZ3 CI. Files removed from an
   archive are removed from the repo, but only if the whole archive arrived and at most 25% of a variant
   goes (`--allow-delete` overrides). The same command works on any machine:
   `HF_TOKEN=... python sync/sync.py mirror --only <dataset>`.
3. `sync.py catalog` regenerates `src/sdrbench/catalog.json` (pinned to the new Hugging Face revisions),
   `sync.py verify` checks dtypes, element counts and dimension order against the data, `sync.py cards`
   refreshes the dataset cards and file tables, and a new patch version is tagged and published to PyPI.

### Adding a dataset

1. Add an entry to `sync/datasets.json` (see its `_comment`): title, provider, science description, and per
   variant the Hugging Face `folder`, Globus `archive`, optional `select`/`exclude`, and `rules` mapping file
   globs to dtype, C-order shape and `var`/`step` name templates.
2. `python sync/sync.py mirror --only <dataset> --dry --workdir /tmp/w` downloads and hashes without uploading
   (writes `/tmp/w/state.dry.json`); fix rules until `sync.py catalog` builds without errors (it refuses files
   without a rule or with a size that does not match the shape).
3. Mirror for real, then `sync.py catalog`, `sync.py verify --only <dataset>`, `sync.py cards --only <dataset>`,
   and `python sync/selftest.py --only <dataset> --workdir /big/tmp` (every field through the package).

## Development

```bash
pip install -e ".[test,sz3]"
pytest                       # offline tests
pytest -m online             # end-to-end against Hugging Face and Globus
python sync/sync.py verify   # data-level check of every dtype/shape (network)
python sync/selftest.py --workdir /big/tmp   # every field of every dataset through the package
```
