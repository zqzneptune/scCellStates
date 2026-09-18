"""Common contracts and diagnostics for fixed-vocabulary state projection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import anndata as ad
import numpy as np
from scipy import sparse

from sccellstates.io import get_matrix
from sccellstates.programs import ProgramSet
from sccellstates.state import StateError


@dataclass(frozen=True)
class ProjectorSpec:
    """Declared capabilities of one state projector."""

    name: str
    requires_raw_counts: bool = False
    requires_training: bool = False
    supports_sparse: bool = True
    supports_uncertainty: bool = False
    deterministic: bool = True
    supports_cpu: bool = True
    supports_gpu: bool = False


@dataclass(frozen=True)
class StateResult:
    """Standard result returned by fixed-vocabulary state projectors."""

    usages: np.ndarray
    normalized_states: np.ndarray
    cell_names: tuple[str, ...]
    sample_id: str
    vocabulary: ProgramSet
    reconstructed_features: np.ndarray | None
    projection_error: np.ndarray
    relative_error: np.ndarray
    feature_coverage: float
    observed_feature_coverage: np.ndarray
    uncertainty: np.ndarray | None
    projector_name: str
    projector_version: str
    parameters: Mapping[str, object]
    vocabulary_id: str
    missing_features: tuple[str, ...] = ()
    extra_features: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def states(self) -> np.ndarray:
        """Compatibility alias for the normalized state coordinates."""
        return self.normalized_states

    @property
    def cell_ids(self) -> tuple[str, ...]:
        """Portable-contract alias for the projected cell identifiers."""
        return self.cell_names

    @property
    def projector_parameters(self) -> Mapping[str, object]:
        """Portable-contract alias for fitted projector parameters."""
        return self.parameters

    @property
    def provenance(self) -> Mapping[str, object]:
        """Compatibility view of projector and vocabulary provenance."""
        return {
            "workflow": "fixed_vocabulary_projection",
            "method": self.projector_name,
            "projector_version": self.projector_version,
            "parameters": dict(self.parameters),
            "vocabulary_id": self.vocabulary_id,
        }


@dataclass(frozen=True)
class ProjectionBenchmark:
    """Comparable results from applying several projectors to the same inputs."""

    results: Mapping[str, tuple[StateResult, ...]]
    methods: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "methods", tuple(self.methods))
        object.__setattr__(
            self, "results", {name: tuple(self.results[name]) for name in self.methods}
        )


class StateProjector:
    """Interface implemented by all fixed-vocabulary projectors."""

    spec = ProjectorSpec("base")

    def __init__(self, vocabulary: ProgramSet) -> None:
        if not isinstance(vocabulary, ProgramSet):
            raise TypeError("vocabulary must be a ProgramSet")
        self.vocabulary = vocabulary

    def fit(self, vocabulary: ProgramSet | None = None, X: object = None, **kwargs: object):
        """Fit optional method parameters; vocabulary weights are never changed."""
        if vocabulary is not None and vocabulary is not self.vocabulary:
            raise StateError("a projector cannot replace its frozen vocabulary")
        return self

    def transform(self, X: ad.AnnData, **kwargs: object) -> StateResult:
        raise NotImplementedError

    def fit_transform(self, X: ad.AnnData, vocabulary: ProgramSet, **kwargs: object) -> StateResult:
        """Fit optional parameters and project one input."""
        self.fit(vocabulary=vocabulary, X=X, **kwargs)
        return self.transform(X, **kwargs)

    def get_params(self) -> dict[str, object]:
        """Return serializable projector parameters."""
        return {}

    def diagnostics(self) -> ProjectorSpec:
        """Return declared capabilities."""
        return self.spec


def aligned_matrix(adata: ad.AnnData, vocabulary: ProgramSet, *, layer: str | None = None):
    """Validate and align an AnnData matrix without densifying sparse input."""
    if not isinstance(adata, ad.AnnData):
        raise TypeError("adata must be an anndata.AnnData object")
    if not adata.var_names.is_unique:
        raise StateError("adata.var_names must be unique")
    matrix = get_matrix(adata, layer=layer)
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise StateError("selected expression matrix must be finite and numeric")
    if (values < 0).any():
        raise StateError("selected expression matrix must be nonnegative")
    lookup = {str(name): i for i, name in enumerate(adata.var_names)}
    present = tuple(name for name in vocabulary.feature_names if name in lookup)
    if not present:
        raise StateError("none of the vocabulary features are present in adata.var_names")
    indices = np.asarray([lookup[name] for name in present], dtype=np.int64)
    vocabulary_lookup = {name: i for i, name in enumerate(vocabulary.feature_names)}
    weight_indices = np.asarray(
        [vocabulary_lookup[name] for name in present], dtype=np.int64
    )
    return (
        matrix[:, indices],
        vocabulary.weights[:, weight_indices].T,
        present,
        tuple(name for name in vocabulary.feature_names if name not in lookup),
        tuple(str(name) for name in adata.var_names if name not in vocabulary.feature_names),
    )


def row_values(matrix: object, row: int) -> np.ndarray:
    """Return one selected expression row as a one-dimensional float array."""
    values = matrix[row]
    if sparse.issparse(values):
        values = values.toarray()
    return np.asarray(values, dtype=np.float64).ravel()


def make_result(
    adata: ad.AnnData,
    matrix: object,
    basis: np.ndarray,
    usages: np.ndarray,
    vocabulary: ProgramSet,
    present: tuple[str, ...],
    missing: tuple[str, ...],
    extra: tuple[str, ...],
    *,
    sample_id: str,
    projector_name: str,
    parameters: Mapping[str, object],
    reconstructed: np.ndarray | None = None,
) -> StateResult:
    reconstructed_values = usages @ basis.T
    if sparse.issparse(matrix):
        observed_squared = np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
        reconstructed_squared = np.sum(reconstructed_values * reconstructed_values, axis=1)
        cross = np.asarray(matrix.multiply(reconstructed_values).sum(axis=1)).ravel()
        errors = np.sqrt(np.maximum(observed_squared + reconstructed_squared - 2 * cross, 0.0))
        norms = np.sqrt(observed_squared)
        coverage = np.asarray(matrix.getnnz(axis=1)).ravel() / len(present)
    else:
        observed = np.asarray(matrix, dtype=np.float64)
        residual = observed - reconstructed_values
        errors = np.linalg.norm(residual, axis=1)
        norms = np.linalg.norm(observed, axis=1)
        coverage = np.count_nonzero(observed, axis=1) / len(present)
    relative = np.divide(errors, norms, out=np.zeros_like(errors), where=norms > 0)
    relative[(norms == 0) & (errors > 0)] = np.inf
    totals = usages.sum(axis=1)
    states = np.divide(
        usages, totals[:, None], out=np.zeros_like(usages), where=totals[:, None] > 0
    )
    return StateResult(
        usages=np.asarray(usages, dtype=np.float64), normalized_states=states,
        cell_names=tuple(map(str, adata.obs_names)),
        sample_id=str(sample_id), vocabulary=vocabulary,
        reconstructed_features=reconstructed if reconstructed is not None else reconstructed_values,
        projection_error=errors,
        relative_error=relative,
        feature_coverage=len(present) / vocabulary.n_features,
        observed_feature_coverage=coverage, uncertainty=None, projector_name=projector_name,
        projector_version="0.1", parameters=dict(parameters), vocabulary_id=vocabulary.sample_id,
        missing_features=missing, extra_features=extra,
    )
