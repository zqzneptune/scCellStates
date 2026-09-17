"""Discover reproducible continuous cell states across biological samples."""

from importlib.metadata import PackageNotFoundError, version

from sccellstates.io import AnnDataSummary, InputError, get_matrix, validate_anndata

try:
    __version__ = version("sccellstates")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "AnnDataSummary",
    "InputError",
    "__version__",
    "get_matrix",
    "validate_anndata",
]
