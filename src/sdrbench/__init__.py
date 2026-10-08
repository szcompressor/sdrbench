"""SDRBench scientific datasets as numpy arrays.

    >>> import sdrbench
    >>> sdrbench.list()
    >>> nyx = sdrbench.dataset("nyx")
    >>> nyx.fields
    >>> t = nyx["temperature"]          # shape (512, 512, 512), C order

Files come from the Hugging Face mirror (https://huggingface.co/sdrbench); if that fails
they are fetched from the original SDRBench archives on Globus.
"""
from .core import Dataset, Field, cache_dir, catalog, dataset, list, load

try:
    from ._version import __version__
except ImportError:  # source checkout without a build
    __version__ = "0.0.0"

__all__ = ["Dataset", "Field", "cache_dir", "catalog", "dataset", "list", "load"]
