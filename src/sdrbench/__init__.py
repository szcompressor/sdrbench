"""SDRBench scientific datasets as numpy arrays.

    >>> import sdrbench
    >>> sdrbench.list()                      # datasets; sdrbench.list(variants=True) for all variants
    >>> nyx = sdrbench.dataset("nyx")
    >>> nyx.fields
    >>> t = nyx["temperature"]               # memmap, shape (512, 512, 512), C order
    >>> p = sdrbench.dataset("hurricane-isabel", "P").series("P")   # 48 time steps

Files come from the Hugging Face mirror (https://huggingface.co/sdrbench), pinned to the
revision this release was built from; if Hugging Face cannot deliver, from the original
SDRBench archives on Globus. Local copies can be used via ``root=`` or ``$SDRBENCH_DATA``.
"""
from ._core import Dataset, Field, Series, cache_dir, catalog, dataset, list, load

try:
    from ._version import __version__
except ImportError:  # source checkout without a build
    __version__ = "0.0.0"

# `list` is deliberately not exported by `from sdrbench import *` (it would shadow the builtin);
# use sdrbench.list().
__all__ = ["Dataset", "Field", "Series", "cache_dir", "catalog", "dataset", "load"]
