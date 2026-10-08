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
nyx = sdrbench.dataset("nyx")            # default variant; sdrbench.dataset("nyx", "512x512x512_log")
nyx.fields                               # ['baryon_density', 'dark_matter_density', 'temperature', ...]
t = nyx["temperature"]                   # numpy memmap, float32, shape (512, 512, 512)
for name, x in nyx.items(): ...          # downloads each field when reached
sdrbench.load("cesm-atm", "CLDHGH")      # one-liner
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
sdrbench info hurricane-isabel/Pf
sdrbench download nyx temperature
```

- **Field names** are the physical variables (`CLDHGH`, `T`, `temperature`); dtype and shape come
  from the catalog, not from the file name. The original SDRBench file name also works
  (`ds["CLDHGH_1_1800_3600.f32"]`).
- **Shapes are C order** (slowest dimension first), the convention of the SZ3 test-suite table, so
  `np.fromfile(path, dtype).reshape(shape)` is always right.
- **Downloads** come from Hugging Face and are cached; if that fails the package falls back to the
  original SDRBench archive on Globus (verified by sha256). `field.download(source="globus")`
  forces Globus.
- Files on Hugging Face are byte-for-byte the files inside the SDRBench archives (a single top-level
  folder inside an archive is dropped). Derived layouts such as QMCPACK `288x115x69x69`
  (preconditioned) are computed on load and are not stored.

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
