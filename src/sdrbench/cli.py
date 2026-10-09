import argparse
import sys

from . import core


def _size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sdrbench", description="SDRBench scientific datasets")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list datasets and their variants")
    pi = sub.add_parser("info", help="fields of a dataset (NAME or NAME/VARIANT)")
    pi.add_argument("dataset")
    pd = sub.add_parser("download", help="download a dataset variant, or some of its fields")
    pd.add_argument("dataset")
    pd.add_argument("fields", nargs="*", help="field names (default: all files)")
    pd.add_argument("-o", "--output", default=None,
                    help="save plain files under this directory as <dir>/<repo path> (default: keep in cache)")
    pd.add_argument("--cache", default=None, help="cache directory (default: HF cache / ~/.cache/sdrbench)")
    pd.add_argument("-j", "--workers", type=int, default=8, help="parallel downloads (default 8)")
    a = p.parse_args(argv)

    try:
        if a.cmd == "list":
            for name in core.list():
                d = core.dataset(name)
                print(f"{name:22s} {d.title}")
                for v in d.variants:
                    dv = core.dataset(name, v)
                    print(f"    {name}/{v:30s} {len(dv):4d} fields {_size(dv.nbytes):>10s}")
        elif a.cmd == "info":
            d = core.dataset(a.dataset)
            print(d.title, "-", d.description)
            print("source:", d.source_info)
            for f in d.files:
                shape = "x".join(map(str, f.shape)) if f.shape else ""
                print(f"  {f.name:40s} {f.dtype or '':4s} {shape:20s} {_size(f.nbytes):>10s}")
        elif a.cmd == "download":
            d = core.dataset(a.dataset, cache=a.cache)
            for p in d.download(a.output, fields=a.fields or None, workers=a.workers):
                print(p)
    except KeyError as e:
        print(e.args[0], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
