"""Access SDRBench scientific datasets from Hugging Face or the original Globus archives.

    >>> import sdrbench
    >>> sdrbench.datasets()
    >>> x = sdrbench.load("nyx", "512x512x512/temperature.f32")   # numpy array, shape (512, 512, 512)
"""
from .core import FileInfo, cache_dir, catalog, datasets, download, files, info, load, variants

try:
    from ._version import __version__
except ImportError:  # running from a source checkout without a build
    __version__ = "0.0.0"

__all__ = ["FileInfo", "cache_dir", "catalog", "datasets", "download", "files", "info", "load", "variants"]
