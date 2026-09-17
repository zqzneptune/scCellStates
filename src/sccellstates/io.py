"""AnnData-first input contracts for scCellStates."""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
from scipy import sparse

Matrix = np.ndarray | sparse.spmatrix


class InputError(ValueError):
    """Raised when an input violates the scCellStates data contract."""


@dataclass(frozen=True)
class AnnDataSummary:
    """Validated dimensions and sample information for an AnnData object."""

    n_obs: int
    n_vars: int
    n_samples: int
    sample_key: str
    layer: str | None
    is_sparse: bool


def get_matrix(adata: ad.AnnData, *, layer: str | None = None) -> Matrix:
    """Return `.X` or an explicitly selected layer without copying or densifying."""
    if not isinstance(adata, ad.AnnData):
        raise TypeError("adata must be an anndata.AnnData object")
    if layer is None:
        matrix = adata.X
    else:
        try:
            matrix = adata.layers[layer]
        except KeyError:
            raise InputError(f"layer {layer!r} is not present in adata.layers") from None
    if matrix is None:
        location = "adata.X" if layer is None else f"adata.layers[{layer!r}]"
        raise InputError(f"{location} is empty")
    return matrix


def _validate_values(matrix: Matrix) -> None:
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if values.ndim != 1 and np.asarray(matrix).ndim != 2:
        raise InputError("the selected expression matrix must be two-dimensional")
    if not np.issubdtype(values.dtype, np.number):
        raise InputError("the selected expression matrix must be numeric")
    if not np.isfinite(values).all():
        raise InputError("the selected expression matrix contains NaN or infinite values")
    if (values < 0).any():
        raise InputError("the selected expression matrix contains negative values")


def validate_anndata(
    adata: ad.AnnData,
    *,
    sample_key: str,
    layer: str | None = None,
    min_samples: int = 1,
) -> AnnDataSummary:
    """Validate an AnnData object without changing it.

    Parameters
    ----------
    adata
        Cells-by-genes annotated matrix.
    sample_key
        Column in ``adata.obs`` identifying independent biological samples.
    layer
        Expression layer to validate. ``None`` selects ``adata.X``.
    min_samples
        Minimum number of distinct samples required by the caller. Use two or
        more for cross-sample recurrence analyses.
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError("adata must be an anndata.AnnData object")
    if min_samples < 1:
        raise ValueError("min_samples must be at least 1")
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise InputError("adata must contain at least one cell and one gene")
    if not adata.obs_names.is_unique:
        raise InputError("adata.obs_names must be unique")
    if not adata.var_names.is_unique:
        raise InputError("adata.var_names must be unique")
    if sample_key not in adata.obs:
        raise InputError(f"sample key {sample_key!r} is not present in adata.obs")

    samples = adata.obs[sample_key]
    if samples.isna().any():
        raise InputError(f"adata.obs[{sample_key!r}] contains missing sample identifiers")
    normalized = samples.astype("string").str.strip()
    if normalized.eq("").any():
        raise InputError(f"adata.obs[{sample_key!r}] contains empty sample identifiers")
    n_samples = int(normalized.nunique())
    if n_samples < min_samples:
        raise InputError(
            f"at least {min_samples} biological samples are required; found {n_samples}"
        )

    matrix = get_matrix(adata, layer=layer)
    if matrix.shape != adata.shape:
        raise InputError(
            f"selected matrix shape {matrix.shape} does not match AnnData shape {adata.shape}"
        )
    _validate_values(matrix)
    return AnnDataSummary(
        n_obs=adata.n_obs,
        n_vars=adata.n_vars,
        n_samples=n_samples,
        sample_key=sample_key,
        layer=layer,
        is_sparse=sparse.issparse(matrix),
    )
