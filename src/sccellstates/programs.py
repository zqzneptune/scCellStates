"""Independent within-sample gene-program estimation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import anndata as ad
import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF

from sccellstates.io import Matrix, get_matrix, validate_anndata


class ProgramError(ValueError):
    """Raised when gene-program estimation inputs or outputs are invalid."""


@dataclass(frozen=True)
class ProgramSet:
    """Gene weights learned independently from one biological sample.

    Parameters
    ----------
    sample_id
        Biological sample used to estimate these programs.
    feature_names
        Unique feature identifiers corresponding to columns of ``weights``.
    weights
        Nonnegative array with shape ``(n_programs, n_features)``.
    n_cells
        Number of cells used for estimation.
    estimator
        Short name of the estimator that produced the programs.
    """

    sample_id: str
    feature_names: tuple[str, ...]
    weights: np.ndarray
    n_cells: int
    estimator: str
    parameters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sample_id = str(self.sample_id).strip()
        feature_names = tuple(str(name) for name in self.feature_names)
        weights = np.array(self.weights, dtype=np.float64, copy=True)
        if not sample_id:
            raise ProgramError("sample_id must be a non-empty string")
        if self.n_cells < 1:
            raise ProgramError("n_cells must be at least 1")
        if not self.estimator:
            raise ProgramError("estimator must be a non-empty string")
        if not isinstance(self.parameters, Mapping):
            raise ProgramError("parameters must be a mapping")
        if weights.ndim != 2 or weights.shape[0] == 0 or weights.shape[1] == 0:
            raise ProgramError("weights must be a non-empty programs-by-features matrix")
        if weights.shape[1] != len(feature_names):
            raise ProgramError("feature_names length must match the weights feature axis")
        if len(set(feature_names)) != len(feature_names):
            raise ProgramError("feature_names must be unique")
        if not np.isfinite(weights).all():
            raise ProgramError("program weights must be finite")
        if (weights < 0).any():
            raise ProgramError("program weights must be nonnegative")
        if (weights.sum(axis=1) == 0).any():
            raise ProgramError("every program must have at least one positive weight")
        weights.setflags(write=False)
        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))

    @property
    def n_programs(self) -> int:
        """Number of programs in this sample."""
        return self.weights.shape[0]

    @property
    def n_features(self) -> int:
        """Number of genes represented by each program."""
        return self.weights.shape[1]


@dataclass(frozen=True)
class ProgramStabilityFit:
    """Consensus and diagnostics from repeated fits of one biological sample."""

    programs: ProgramSet
    run_ids: tuple[str, ...]
    anchor_run_id: str
    matched_similarities: np.ndarray
    retained_anchor_indices: np.ndarray
    min_similarity: float

    @property
    def requested_programs(self) -> int:
        """Number of programs in every input run."""
        return self.matched_similarities.shape[1]

    @property
    def retained_fraction(self) -> float:
        """Fraction of anchor programs stable in every other run."""
        return self.programs.n_programs / self.requested_programs


@runtime_checkable
class ProgramEstimator(Protocol):
    """Contract for an estimator that receives exactly one biological sample."""

    def fit(
        self,
        matrix: Matrix,
        *,
        feature_names: Sequence[str],
        sample_id: str,
    ) -> ProgramSet:
        """Estimate programs from one cells-by-features matrix."""
        ...


@dataclass(frozen=True)
class NMFProgramEstimator:
    """Deterministic nonnegative matrix factorization for one sample.

    Input values must already contain the caller's explicitly chosen
    preprocessing. This estimator never normalizes, scales, or selects genes.
    """

    n_programs: int
    random_state: int
    max_iter: int = 500
    tol: float = 1e-4
    alpha_W: float = 0.0
    alpha_H: float | str = "same"
    l1_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.n_programs < 1:
            raise ValueError("n_programs must be at least 1")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if self.tol <= 0:
            raise ValueError("tol must be positive")
        if self.alpha_W < 0:
            raise ValueError("alpha_W must be nonnegative")
        if isinstance(self.alpha_H, float | int) and self.alpha_H < 0:
            raise ValueError("alpha_H must be nonnegative or 'same'")
        if self.alpha_H != "same" and not isinstance(self.alpha_H, float | int):
            raise ValueError("alpha_H must be nonnegative or 'same'")
        if not 0 <= self.l1_ratio <= 1:
            raise ValueError("l1_ratio must be between 0 and 1")

    def fit(
        self,
        matrix: Matrix,
        *,
        feature_names: Sequence[str],
        sample_id: str,
    ) -> ProgramSet:
        """Fit NMF and return L1-normalized gene weights."""
        if matrix.ndim != 2:
            raise ProgramError("matrix must be two-dimensional")
        n_cells, n_features = matrix.shape
        if n_cells < 2:
            raise ProgramError("at least two cells are required to estimate programs")
        if self.n_programs > min(n_cells, n_features):
            raise ProgramError("n_programs cannot exceed cells or features")
        names = tuple(str(name) for name in feature_names)
        if len(names) != n_features:
            raise ProgramError("feature_names length must match the matrix feature axis")
        if len(set(names)) != len(names):
            raise ProgramError("feature_names must be unique")
        values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
        if not np.issubdtype(values.dtype, np.number):
            raise ProgramError("matrix must be numeric")
        if not np.isfinite(values).all():
            raise ProgramError("matrix contains NaN or infinite values")
        if (values < 0).any():
            raise ProgramError("NMF requires nonnegative values")

        model = NMF(
            n_components=self.n_programs,
            init="nndsvda",
            solver="cd",
            tol=self.tol,
            max_iter=self.max_iter,
            random_state=self.random_state,
            alpha_W=self.alpha_W,
            alpha_H=self.alpha_H,
            l1_ratio=self.l1_ratio,
            shuffle=False,
        )
        model.fit(matrix)
        weights = np.asarray(model.components_, dtype=np.float64)
        totals = weights.sum(axis=1, keepdims=True)
        if (totals == 0).any():
            raise ProgramError("NMF produced an empty program")
        weights /= totals
        return ProgramSet(
            sample_id=sample_id,
            feature_names=names,
            weights=weights,
            n_cells=n_cells,
            estimator="nmf",
            parameters={
                "n_programs": self.n_programs,
                "random_state": self.random_state,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "alpha_W": self.alpha_W,
                "alpha_H": self.alpha_H,
                "l1_ratio": self.l1_ratio,
            },
        )


EstimatorFactory = Callable[[str], ProgramEstimator]


def stabilize_programs(
    program_sets: tuple[ProgramSet, ...],
    *,
    sample_id: str,
    min_similarity: float = 0.5,
) -> ProgramStabilityFit:
    """Build a median consensus of programs reproducible across repeated fits.

    The first run is the deterministic anchor. Every later run is aligned to it
    by optimal one-to-one rank-correlation matching. An anchor program is kept
    only when its matched similarity reaches ``min_similarity`` in every run.
    Repeated fits are computational replicates of one biological sample and do
    not count as independent evidence of cross-sample recurrence.
    """
    if len(program_sets) < 2:
        raise ProgramError("at least two repeated program fits are required")
    if not -1 <= min_similarity <= 1:
        raise ValueError("min_similarity must be between -1 and 1")
    normalized_sample_id = str(sample_id).strip()
    if not normalized_sample_id:
        raise ProgramError("sample_id must be a non-empty string")
    if any(not isinstance(programs, ProgramSet) for programs in program_sets):
        raise TypeError("program_sets must contain only ProgramSet objects")

    anchor = program_sets[0]
    run_ids = tuple(programs.sample_id for programs in program_sets)
    if len(set(run_ids)) != len(run_ids):
        raise ProgramError("repeated program fit IDs must be unique")
    for programs in program_sets[1:]:
        if programs.feature_names != anchor.feature_names:
            raise ProgramError("repeated program fits must use the same feature order")
        if programs.n_programs != anchor.n_programs:
            raise ProgramError("repeated program fits must have the same number of programs")
        if programs.n_cells != anchor.n_cells:
            raise ProgramError("repeated program fits must use the same number of cells")

    # Local import avoids a module cycle: recurrence depends on ProgramSet.
    from sccellstates.recurrence import program_similarity

    similarities_by_run = np.empty((len(program_sets) - 1, anchor.n_programs), dtype=np.float64)
    matched_indices = np.empty((len(program_sets) - 1, anchor.n_programs), dtype=np.int64)
    for run_index, query in enumerate(program_sets[1:]):
        similarities = program_similarity(anchor, query)
        rows, columns = linear_sum_assignment(similarities, maximize=True)
        matched_indices[run_index, rows] = columns
        similarities_by_run[run_index, rows] = similarities[rows, columns]

    retained_indices = np.flatnonzero(np.all(similarities_by_run >= min_similarity, axis=0))
    if len(retained_indices) == 0:
        raise ProgramError("no program met the repeated-fit stability threshold")

    consensus_rows = []
    for anchor_index in retained_indices:
        rows = [anchor.weights[anchor_index]]
        rows.extend(
            program_sets[run_index + 1].weights[matched_indices[run_index, anchor_index]]
            for run_index in range(len(program_sets) - 1)
        )
        consensus = np.median(np.stack(rows), axis=0)
        total = consensus.sum()
        if total <= 0:
            raise ProgramError("stability consensus produced an empty program")
        consensus_rows.append(consensus / total)

    stable_programs = ProgramSet(
        sample_id=normalized_sample_id,
        feature_names=anchor.feature_names,
        weights=np.stack(consensus_rows),
        n_cells=anchor.n_cells,
        estimator=f"restart_consensus_{anchor.estimator}",
        parameters={
            "n_runs": len(program_sets),
            "requested_programs": anchor.n_programs,
            "min_similarity": min_similarity,
        },
    )
    similarities_by_run.setflags(write=False)
    retained_indices.setflags(write=False)
    return ProgramStabilityFit(
        programs=stable_programs,
        run_ids=run_ids,
        anchor_run_id=anchor.sample_id,
        matched_similarities=similarities_by_run,
        retained_anchor_indices=retained_indices,
        min_similarity=min_similarity,
    )


def fit_programs_by_sample(
    adata: ad.AnnData,
    *,
    sample_key: str,
    estimator_factory: EstimatorFactory,
    layer: str | None = None,
    sample_ids: Sequence[str] | None = None,
) -> tuple[ProgramSet, ...]:
    """Fit a fresh estimator independently within each selected sample.

    ``estimator_factory`` is called once per sample, preventing fitted state
    from being shared across biological samples. If ``sample_ids`` is supplied,
    all other samples are excluded before any estimator sees the data.
    """
    validate_anndata(adata, sample_key=sample_key, layer=layer)
    normalized_samples = adata.obs[sample_key].astype("string").str.strip()
    available = set(normalized_samples.tolist())
    if sample_ids is None:
        selected = tuple(sorted(available))
    else:
        selected = tuple(str(sample_id).strip() for sample_id in sample_ids)
        if not selected or any(not sample_id for sample_id in selected):
            raise ProgramError("sample_ids must contain non-empty identifiers")
        if len(set(selected)) != len(selected):
            raise ProgramError("sample_ids must be unique")
        missing = sorted(set(selected) - available)
        if missing:
            raise ProgramError(f"sample_ids are not present in adata.obs: {missing}")

    matrix = get_matrix(adata, layer=layer)
    feature_names = tuple(str(name) for name in adata.var_names)
    fitted: list[ProgramSet] = []
    for sample_id in selected:
        mask = normalized_samples.eq(sample_id).to_numpy()
        estimator = estimator_factory(sample_id)
        if not isinstance(estimator, ProgramEstimator):
            raise TypeError("estimator_factory must return a ProgramEstimator")
        result = estimator.fit(
            matrix[mask, :],
            feature_names=feature_names,
            sample_id=sample_id,
        )
        if result.sample_id != sample_id:
            raise ProgramError("estimator returned a ProgramSet for the wrong sample")
        if result.feature_names != feature_names:
            raise ProgramError("estimator changed or reordered the feature axis")
        if result.n_cells != int(mask.sum()):
            raise ProgramError("estimator reported an incorrect cell count")
        fitted.append(result)
    return tuple(fitted)
