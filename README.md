# sdrbench

Python access to the [SDRBench](https://sdrbench.github.io/) scientific datasets, served from the
[Hugging Face mirror](https://huggingface.co/sdrbench) or from the original Globus archives at
Argonne.

```bash
pip install sdrbench
```

```python
import sdrbench

sdrbench.datasets()                     # ['basilisk-turbulence', 'cesm-atm', 'exaalt', ...]
sdrbench.files("nyx")                   # FileInfo(path, bytes, sha256, dtype, shape) for every file
x = sdrbench.load("nyx", "512x512x512/temperature.f32")   # numpy memmap, shape (512, 512, 512)
p = sdrbench.download("hurricane-isabel", "100x500x500/Pf48.bin.f32")          # local path
p = sdrbench.download("hurricane-isabel", "Pf48.bin.f32", source="globus")      # same bytes, from Globus
```

Command line:

```bash
sdrbench list
sdrbench files cesm-atm '*CLDHGH*'
sdrbench download qmcpack --source globus
```

- **Shapes are C order** (slowest dimension first), the same convention as the SZ3 test-suite
  dataset table, so `np.fromfile(p, dtype).reshape(shape)` is always correct.
- `source="hf"` downloads single files through `huggingface_hub` (cached in the usual HF cache).
  `source="globus"` downloads the original SDRBench archive once, unpacks it to
  `~/.cache/sdrbench/globus` (override with `cache=` or `SDRBENCH_CACHE`), and verifies the
  file's sha256 against the catalog.
- Files on Hugging Face are byte-for-byte the files inside the SDRBench archives; only a single
  top-level folder inside each archive is dropped.

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
2. Changed archives are re-mirrored (`sync.py mirror`): download, unpack, sha256, upload to
   `sdrbench/<dataset>` (files removed from an archive are removed from the repo too).
   Archives too large for a GitHub runner fail the job; run the same command on a big machine:
   `HF_TOKEN=... python sync/sync.py mirror --only <dataset>`.
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
