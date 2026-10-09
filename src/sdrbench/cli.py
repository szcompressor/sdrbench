import argparse
import sys

from . import _core as core
from ._core import _human


def main(argv=None) -> int:
    from . import __version__

    p = argparse.ArgumentParser(prog="sdrbench", description="SDRBench scientific datasets")
    p.add_argument("--version", action="version", version=f"sdrbench {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list datasets and their variants")
    pi = sub.add_parser("info", help="fields of a dataset variant (NAME or NAME/VARIANT)")
    pi.add_argument("dataset")
    pd = sub.add_parser("download", help="download a dataset variant, or some of its fields")
    pd.add_argument("dataset")
    pd.add_argument("fields", nargs="*", help="field names (default: all files)")
    pd.add_argument("-o", "--output", default=None,
                    help="save plain files under this directory as <dir>/<repo path> (default: keep in cache)")
    pd.add_argument("--cache", default=None, help="cache directory (default: HF cache / ~/.cache/sdrbench)")
    pd.add_argument("-j", "--workers", type=int, default=8, help="parallel downloads (default 8)")
    pp = sub.add_parser("path", help="print the local path of a field's file (downloading it if needed)")
    pp.add_argument("dataset")
    pp.add_argument("field")
    pc = sub.add_parser("cite", help="BibTeX for a dataset")
    pc.add_argument("dataset")
    a = p.parse_args(argv)

    try:
        if a.cmd == "list":
            rows = []
            for name in core.list():
                for v in core.dataset(name).variants:
                    d = core.dataset(name, v)
                    rows.append((f"{name}/{v}", f"{len(d)} fields", _human(d.nbytes), d.title))
            w = max(len(r[0]) for r in rows)
            for r in rows:
                print(f"{r[0]:{w}s}  {r[1]:>11s}  {r[2]:>9s}  {r[3]}")
        elif a.cmd == "info":
            d = core.dataset(a.dataset)
            print(f"{d.title} - {d.name}/{d.variant}  ({len(d)} fields, {_human(d.nbytes)})")
            print(d.description)
            if d.note:
                print("Note:", d.note)
            print("Provider:", d.provider)
            others = [v for v in d.variants if v != d.variant]
            if others:
                print("Other variants:", ", ".join(others))
            for f in d.files:
                shape = "x".join(map(str, f.shape)) if f.shape else "(file)"
                print(f"  {f.name:32s} {f.dtype or '':5s} {shape:22s} {_human(f.nbytes):>10s}  {f.path}")
        elif a.cmd == "download":
            d = core.dataset(a.dataset, cache=a.cache)
            for path in d.download(a.output, fields=a.fields or None, workers=a.workers):
                print(path)
        elif a.cmd == "path":
            print(core.dataset(a.dataset).field(a.field).download())
        elif a.cmd == "cite":
            print(core.dataset(a.dataset).citation())
    except KeyError as e:
        print(e.args[0], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
