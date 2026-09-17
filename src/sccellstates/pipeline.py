"""Configuration-driven, leakage-safe orchestration for scCellStates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from sccellstates.io import get_matrix, validate_anndata
from sccellstates.metrics import (
    reconstruction_by_partition,
    sample_bootstrap,
    technical_associations,
)
from sccellstates.programs import NMFProgramEstimator, fit_programs_by_sample
from sccellstates.recurrence import VocabularyFit, build_recurrent_vocabulary
from sccellstates.state import (
    CountResidualProgramScorer,
    DirectProgramState,
    ExpressionPCA,
    GeneStandardizedProgramScorer,
    LinearProgramState,
    MatchedControlProgramScorer,
    NonlinearProgramState,
    ProgramScorer,
    SampleSplit,
    TechnicalResidualProgramScorer,
    make_sample_split,
    sample_partition_labels,
    store_program_activities,
    store_state_coordinates,
)


class PipelineError(ValueError):
    """Raised when a pipeline configuration or fit is invalid."""


class MatrixPreprocessor(Protocol):
    """Train-fitted matrix transformer accepted by :func:`fit_pipeline`."""

    def fit(self, matrix: object, train_mask: np.ndarray) -> MatrixPreprocessor:
        """Fit preprocessing parameters using training cells only."""

    def transform(self, matrix: object) -> object:
        """Transform a matrix without updating fitted parameters."""


@dataclass
class LibrarySizeLog1p:
    """Normalize each cell by its selected-gene depth and apply ``log1p``."""

    target_sum: float = 10_000.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.target_sum) or self.target_sum <= 0:
            raise ValueError("target_sum must be positive and finite")

    def fit(self, matrix: object, train_mask: np.ndarray) -> LibrarySizeLog1p:
        if matrix.shape[0] != len(train_mask) or not np.asarray(train_mask).any():
            raise PipelineError("train_mask must match the matrix and select cells")
        self.training_n_cells_ = int(np.asarray(train_mask).sum())
        return self

    def transform(self, matrix: object) -> object:
        if not hasattr(self, "training_n_cells_"):
            raise PipelineError("LibrarySizeLog1p must be fitted before transform")
        if sparse.issparse(matrix):
            values = matrix.tocsr().astype(np.float64, copy=True)
            depths = np.asarray(values.sum(axis=1)).ravel()
            row_ids = np.repeat(np.arange(values.shape[0]), np.diff(values.indptr))
            values.data = np.log1p(
                values.data * self.target_sum / np.maximum(depths, 1e-12)[row_ids]
            )
            return values
        values = np.asarray(matrix, dtype=np.float64)
        depths = values.sum(axis=1, keepdims=True)
        return np.log1p(values * self.target_sum / np.maximum(depths, 1e-12))


@dataclass(frozen=True)
class PipelineConfig:
    """All choices needed for one reproducible discovery run."""

    sample_key: str
    test_samples: tuple[str, ...]
    validation_samples: tuple[str, ...] = ()
    layer: str | None = None
    preprocessing: str = "identity"
    n_programs: int = 8
    min_samples: int = 2
    min_similarity: float = 0.3
    n_permutations: int = 1_000
    score_method: str = "direct"
    state_components: int = 2
    state_methods: tuple[str, ...] = ("direct", "linear", "nonlinear", "pca")
    max_dense_elements: int = 10_000_000
    random_state: int = 0

    def __post_init__(self) -> None:
        if self.n_programs < 1 or self.state_components < 1:
            raise ValueError("n_programs and state_components must be positive")
        if self.min_samples < 2:
            raise ValueError("min_samples must be at least 2")
        if self.preprocessing not in {"identity", "library_size_log1p"}:
            raise ValueError("preprocessing must be 'identity' or 'library_size_log1p'")
        allowed_scores = {
            "direct",
            "gene_standardized",
            "count_residual",
            "technical_residual",
            "matched_control",
        }
        if self.score_method not in allowed_scores:
            raise ValueError(f"score_method must be one of {sorted(allowed_scores)}")
        allowed_states = {"direct", "linear", "nonlinear", "pca"}
        if not self.state_methods or not set(self.state_methods) <= allowed_states:
            raise ValueError(f"state_methods must use only {sorted(allowed_states)}")


@dataclass(frozen=True)
class PipelineResult:
    """Outputs from one pipeline fit, including candidate diagnostics."""

    adata: ad.AnnData
    split: SampleSplit
    vocabulary_fit: VocabularyFit
    activities: np.ndarray
    coordinates: Mapping[str, np.ndarray]
    reports: Mapping[str, pd.DataFrame]
    provenance: Mapping[str, object]


def _serializable(value: object) -> object:
    """Convert provenance values to AnnData-compatible scalar/list objects."""
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_serializable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _make_preprocessor(name: str) -> MatrixPreprocessor | None:
    return LibrarySizeLog1p() if name == "library_size_log1p" else None


def _annotated_copy(
    adata: ad.AnnData,
    *,
    layer: str | None,
    preprocessor: MatrixPreprocessor | None,
    train_mask: np.ndarray,
) -> ad.AnnData:
    result = adata.copy()
    matrix = get_matrix(result, layer=layer)
    if preprocessor is not None:
        preprocessor.fit(matrix, train_mask)
        matrix = preprocessor.transform(matrix)
    result.X = matrix
    return result


def _scorer(vocabulary: object, config: PipelineConfig, adata: ad.AnnData, split: SampleSplit):
    if config.score_method == "direct":
        return ProgramScorer(vocabulary)
    classes = {
        "gene_standardized": GeneStandardizedProgramScorer,
        "count_residual": CountResidualProgramScorer,
        "technical_residual": TechnicalResidualProgramScorer,
    }
    if config.score_method == "matched_control":
        scorer = MatchedControlProgramScorer(
            vocabulary,
            n_top_genes=min(50, vocabulary.n_features - 1),
            n_bins=min(25, vocabulary.n_features),
            control_size=50,
            random_state=config.random_state,
        )
    else:
        scorer = classes[config.score_method](vocabulary)
    return scorer.fit(adata, sample_key=config.sample_key, split=split)


def fit_pipeline(adata: ad.AnnData, config: PipelineConfig) -> PipelineResult:
    """Fit a complete candidate workflow without using held-out samples.

    The input is copied. Discovery, preprocessing, vocabulary construction,
    and state fitting use only ``split.train``; validation and test cells are
    transformed by the frozen objects. The returned candidate is not a
    biological validation or promotion decision.
    """
    validate_anndata(adata, sample_key=config.sample_key, layer=config.layer, min_samples=2)
    split = make_sample_split(
        adata,
        sample_key=config.sample_key,
        test_samples=config.test_samples,
        validation_samples=config.validation_samples,
    )
    labels = adata.obs[config.sample_key].astype("string").str.strip().to_numpy()
    train_mask = np.isin(labels, split.train)
    working = _annotated_copy(
        adata,
        layer=config.layer,
        preprocessor=_make_preprocessor(config.preprocessing),
        train_mask=train_mask,
    )
    programs = fit_programs_by_sample(
        working,
        sample_key=config.sample_key,
        sample_ids=split.train,
        # Every sample uses the same seed. Deriving one from the sample ID, or
        # from its position, would make renaming or reordering the samples
        # change the fitted programs, which would couple storage to analysis.
        estimator_factory=lambda _sample: NMFProgramEstimator(
            n_programs=config.n_programs,
            random_state=config.random_state,
        ),
    )
    vocabulary_fit = build_recurrent_vocabulary(
        programs,
        min_samples=min(config.min_samples, len(programs)),
        min_similarity=config.min_similarity,
        n_permutations=config.n_permutations,
        random_state=config.random_state,
    )
    scorer = _scorer(vocabulary_fit.vocabulary, config, working, split)
    activities = scorer.transform(working)
    coordinates: dict[str, np.ndarray] = {}
    reports: dict[str, pd.DataFrame] = {}
    partitions = sample_partition_labels(labels, split)

    for method in config.state_methods:
        if method == "direct":
            model = DirectProgramState()
            model.fit(activities, sample_ids=labels, split=split)
        elif method == "linear":
            model = LinearProgramState(config.state_components, random_state=config.random_state)
            model.fit(activities, sample_ids=labels, split=split)
        elif method == "nonlinear":
            model = NonlinearProgramState(config.state_components, random_state=config.random_state)
            model.fit(activities, sample_ids=labels, split=split)
        else:
            model = ExpressionPCA(
                config.state_components,
                random_state=config.random_state,
                max_dense_elements=config.max_dense_elements,
            )
            model.fit(working, sample_key=config.sample_key, split=split)
        values = model.transform(activities) if method != "pca" else model.transform(working)
        coordinates[method] = values
        if method != "pca":
            reconstructed = model.inverse_transform(values)
            reports[f"{method}_reconstruction"] = reconstruction_by_partition(
                activities, reconstructed, partitions
            )
        else:
            reports["pca_reconstruction"] = reconstruction_by_partition(
                np.asarray(get_matrix(working), dtype=float)
                if not sparse.issparse(get_matrix(working))
                else get_matrix(working).toarray(),
                model.inverse_transform(values),
                partitions,
            )
        reports[f"{method}_bootstrap"] = sample_bootstrap(
            values, labels, random_state=config.random_state
        )

    reports["technical_associations"] = technical_associations(
        coordinates["direct"], {"library_size": np.asarray(get_matrix(working).sum(axis=1)).ravel()}
    )
    store_program_activities(working, activities, overwrite=True)
    store_state_coordinates(
        working,
        coordinates["direct"],
        method="pipeline_candidate_direct",
        training_sample_ids=split.train,
        parameters={"score_method": config.score_method},
        overwrite=True,
    )
    provenance = {
        "schema_version": "1.0",
        "workflow": "candidate_pipeline",
        "validated": False,
        "sample_key": config.sample_key,
        "layer": config.layer,
        "training_sample_ids": list(split.train),
        "validation_sample_ids": list(split.validation),
        "test_sample_ids": list(split.test),
        "parameters": _serializable(config.__dict__),
    }
    existing = dict(working.uns.get("sccellstates", {}))
    existing["pipeline"] = provenance
    working.uns["sccellstates"] = existing
    return PipelineResult(
        working, split, vocabulary_fit, activities, coordinates, reports, provenance
    )


def tune_pipeline(adata: ad.AnnData, candidates: Sequence[PipelineConfig]) -> PipelineResult:
    """Select a candidate using validation-sample reconstruction only.

    Every candidate is independently fitted with its own training-only
    workflow. The test samples are never used for selection. The returned
    result records the selected index and validation scores in provenance.
    """
    configs = tuple(candidates)
    if not configs:
        raise PipelineError("at least one pipeline candidate is required")
    if any(not config.validation_samples for config in configs):
        raise PipelineError("every tuning candidate must declare validation_samples")
    first = configs[0]
    if any(
        config.sample_key != first.sample_key or config.test_samples != first.test_samples
        for config in configs[1:]
    ):
        raise PipelineError("tuning candidates must share sample_key and test_samples")

    results = tuple(fit_pipeline(adata, config) for config in configs)
    scores = []
    for result in results:
        report = result.reports["direct_reconstruction"]
        validation = report.loc[report["partition"] == "validation", "mean_squared_error"]
        if validation.empty:
            raise PipelineError("validation samples must contain cells")
        scores.append(float(validation.iloc[0]))
    selected_index = int(np.argmin(scores))
    selected = results[selected_index]
    tuning_report = pd.DataFrame(
        {"candidate": np.arange(len(scores)), "validation_mean_squared_error": scores}
    )
    provenance = dict(selected.provenance)
    provenance["tuning"] = {
        "selection_metric": "direct_reconstruction.validation.mean_squared_error",
        "selected_candidate": selected_index,
        "candidate_scores": scores,
    }
    reports = dict(selected.reports)
    reports["tuning"] = tuning_report
    return replace(selected, reports=reports, provenance=provenance)
