# sdrbench

Python access to the [SDRBench](https://sdrbench.github.io/) scientific datasets, served from the
[Hugging Face mirror](https://huggingface.co/sdrbench) or from the original Globus archives at
Argonne.

```bash
pip install sdrbench            # or "sdrbench[sz3]" to also get pysz (SZ3)
```

```python
import sdrbench

sdrbench.list()                          # ['basilisk-turbulence', 'cesm-atm', 'exaalt', ...]
nyx = sdrbench.dataset("nyx")            # default variant; sdrbench.dataset("nyx", "log") for the log fields
nyx.fields                               # ['baryon_density', 'dark_matter_density', 'temperature', ...]
t = nyx["temperature"]                   # numpy memmap, float32, shape (512, 512, 512)
for name, x in nyx.items(): ...          # downloads each field when reached
sdrbench.load("cesm-atm", "CLDHGH")      # one-liner

# save the raw files (original SDRBench names) to a directory of your choice
nyx.field("temperature").download("data/")   # -> data/512x512x512/temperature.f32 (repo path, original name)
nyx.download("data/")                        # every file of the variant
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
sdrbench list
sdrbench info hurricane-isabel/P
sdrbench download nyx temperature -o data/     # plain files under data/<repo path>
```

- **Field names** are the physical variables (`CLDHGH`, `T`, `temperature`); dtype and shape come
  from the catalog, not from the file name. The original SDRBench file name also works
  (`ds["CLDHGH_1_1800_3600.f32"]`).
- **Shapes are C order** (slowest dimension first), the convention of the SZ3 test-suite table, so
  `np.fromfile(path, dtype).reshape(field.shape)` is right for every stored file. Derived layouts
  (QMCPACK `preconditioned`) are the exception: their file is the stored original, so use
  `.reshape(field.stored_shape).transpose(field.transpose)`, or simply `ds[name]`.
- **Downloads** come from Hugging Face, pinned to the commit the catalog was built from and checked
  against its sha256, and are cached (`SDRBENCH_CACHE` moves the cache). If Hugging Face cannot
  deliver, the package falls back to the original SDRBench archive on Globus (whole archive,
  verified by md5 and sha256; a warning shows its size). `field.download(source="globus")` forces
  Globus. `ds.download(dir)` fetches files in parallel (`workers=`, CLI `-j`).
- Files on Hugging Face are byte-for-byte the files inside the SDRBench archives (a single top-level
  folder inside an archive is dropped). Derived layouts are computed on load and not stored:
  QMCPACK's default variant `preconditioned` (288 x 115 x 69 x 69) is the transpose of the stored
  native file, which stays available as variant `original`.
- **Variants** have descriptive names (`nyx`: `original`, `log`; `cesm-atm`: `2d`, `2d-cleared`,
  `3d`; `hacc`: `medium`, `big`, `sbig`, `region1`...; `hurricane-isabel`: `snapshot`, `P`, `QCLOUD`,
  ...). The first one is the default; shapes are metadata (`field.shape`), never part of a name.

## Datasets

| Dataset | Hugging Face | Source |
|---|---|---|
| CESM-ATM | [sdrbench/cesm-atm](https://huggingface.co/datasets/sdrbench/cesm-atm) | climate (SNL) |
| EXAALT | [sdrbench/exaalt](https://huggingface.co/datasets/sdrbench/exaalt) | molecular dynamics |
| Hurricane ISABEL | [sdrbench/hurricane-isabel](https://huggingface.co/datasets/sdrbench/hurricane-isabel) | weather (NCAR, IEEE Vis 2004) |
| EXAFEL | [sdrbench/exafel](https://huggingface.co/datasets/sdrbench/exafel) | LCLS images |
| HACC | [sdrbench/hacc](https://huggingface.co/datasets/sdrbench/hacc) | cosmology particles |
| NYX | [sdrbench/nyx](https://huggingface.co/datasets/sdrbench/nyx) | cosmology |
| NWChem | [sdrbench/nwchem](https://huggingface.co/datasets/sdrbench/nwchem) | quantum chemistry |
| SCALE-LETKF | [sdrbench/scale-letkf](https://huggingface.co/datasets/sdrbench/scale-letkf) | weather (RIKEN) |
| QMCPACK | [sdrbench/qmcpack](https://huggingface.co/datasets/sdrbench/qmcpack) | quantum Monte Carlo |
| Miranda | [sdrbench/miranda](https://huggingface.co/datasets/sdrbench/miranda) | turbulence (LLNL) |
| S3D | [sdrbench/s3d](https://huggingface.co/datasets/sdrbench/s3d) | combustion (SNL) |
| Basilisk-Turbulence | [sdrbench/basilisk-turbulence](https://huggingface.co/datasets/sdrbench/basilisk-turbulence) | 2D turbulence |
| XGC | [sdrbench/xgc](https://huggingface.co/datasets/sdrbench/xgc) | fusion (PPPL) |

NSTX GPI is not mirrored (its owner asks to be contacted before results are published); use the
[SDRBench page](https://sdrbench.github.io/datasets.html) for it.

Please cite SDRBench and the data provider named on each dataset card.

## Keeping the mirror in sync

Globus remains the source of truth. `.github/workflows/release.yml` runs weekly:

1. `sync/sync.py check` HEADs every archive linked from the SDRBench page and compares size,
   ETag and Last-Modified with `sync/state.json`; it also reports archives that appear on, or
   disappear from, the page.
2. Changed datasets are re-mirrored in parallel jobs (`sync.py mirror`). Archives are streamed:
   each file is unpacked, hashed and committed to `sdrbench/<dataset>` in batches and then
   deleted, so a runner only needs room for the largest single file (~17 GB); the job frees disk
   like the SZ3 CI. Files removed from an archive are removed from the repo too. The same command
   works on any machine: `HF_TOKEN=... python sync/sync.py mirror --only <dataset>`.
3. `sync.py catalog` regenerates `src/sdrbench/catalog.json`, `sync.py verify` checks dtypes and
   shapes against the data, `sync.py cards` refreshes the dataset cards, and a new patch version
   is tagged and published to PyPI.

New datasets need an entry in `sync/datasets.json` with explicit dtype/shape rules; the catalog
build refuses files without a rule or whose size does not match their shape.

## Development

```bash
pip install -e ".[test]"
pytest                 # offline tests
pytest -m online       # end-to-end against Hugging Face and Globus
python sync/sync.py verify   # data-level check of every dtype/shape (needs network)
```
