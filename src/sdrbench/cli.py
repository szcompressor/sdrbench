import argparse
import sys

from . import core


def _size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1000


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sdrbench", description="SDRBench datasets on Hugging Face / Globus")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list datasets")
    pf = sub.add_parser("files", help="list files of a dataset")
    pf.add_argument("dataset")
    pf.add_argument("pattern", nargs="?", default="*")
    pd = sub.add_parser("download", help="download files matching a pattern")
    pd.add_argument("dataset")
    pd.add_argument("pattern", nargs="?", default="*")
    pd.add_argument("--source", choices=core.SOURCES, default="hf")
    pd.add_argument("--cache", default=None, help="cache directory (default ~/.cache/sdrbench)")
    a = p.parse_args(argv)

    if a.cmd == "list":
        for name in core.datasets():
            d = core.info(name)
            total = sum(f["bytes"] for v in d["variants"].values() for f in v["files"])
            print(f"{name:22s} {_size(total):>9s}  {d['title']}")
    elif a.cmd == "files":
        for f in core.files(a.dataset, a.pattern):
            shape = "x".join(map(str, f.shape)) if f.shape else ""
            print(f"{f.path:60s} {_size(f.bytes):>9s}  {f.dtype or '':4s} {shape}")
    elif a.cmd == "download":
        matched = core.files(a.dataset, a.pattern)
        if not matched:
            print("no files match", file=sys.stderr)
            return 1
        for f in matched:
            print(core.download(a.dataset, f.path, source=a.source, cache=a.cache))
    return 0


if __name__ == "__main__":
    sys.exit(main())
