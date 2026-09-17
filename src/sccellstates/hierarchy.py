"""Lineage-aware hierarchical program vocabularies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import anndata as ad
import numpy as np

from sccellstates.io import InputError, validate_anndata
from sccellstates.programs import EstimatorFactory, ProgramSet, fit_programs_by_sample
from sccellstates.recurrence import VocabularyFit, build_recurrent_vocabulary


class HierarchyError(ValueError):
    """Raised when lineage-aware vocabulary construction cannot proceed."""


@dataclass(frozen=True)
class HierarchicalVocabularyFit:
    """Lineage-specific vocabularies and optional shared program evidence."""

    lineage_vocabularies: Mapping[str, VocabularyFit]
    shared_vocabulary: VocabularyFit | None
    training_sample_ids: tuple[str, ...]
    lineage_key: str
    sample_key: str
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        lineages = {str(lineage).strip(): fit for lineage, fit in self.lineage_vocabularies.items()}
        if not lineages or any(not lineage for lineage in lineages):
            raise HierarchyError("lineage_vocabularies must contain named lineages")
        samples = tuple(str(sample).strip() for sample in self.training_sample_ids)
        if not samples or any(not sample for sample in samples):
            raise HierarchyError("training_sample_ids must be non-empty")
        if len(set(samples)) != len(samples):
            raise HierarchyError("training_sample_ids must be unique")
        if any(not isinstance(fit, VocabularyFit) for fit in lineages.values()):
            raise TypeError("lineage_vocabularies must contain VocabularyFit objects")
        object.__setattr__(self, "lineage_vocabularies", MappingProxyType(lineages))
        object.__setattr__(self, "training_sample_ids", samples)
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))

    @property
    def lineages(self) -> tuple[str, ...]:
        """Sorted lineage identifiers represented by the fit."""
        return tuple(sorted(self.lineage_vocabularies))


def _lineage_labels(adata: ad.AnnData, lineage_key: str) -> np.ndarray:
    if lineage_key not in adata.obs:
        raise InputError(f"lineage key {lineage_key!r} is not present in adata.obs")
    values = adata.obs[lineage_key]
    if values.isna().any():
        raise InputError(f"adata.obs[{lineage_key!r}] contains missing lineage identifiers")
    labels = values.astype("string").str.strip().to_numpy()
    if (labels == "").any():
        raise InputError(f"adata.obs[{lineage_key!r}] contains empty lineage identifiers")
    return labels


def fit_hierarchical_vocabularies(
    adata: ad.AnnData,
    *,
    sample_key: str,
    lineage_key: str,
    training_sample_ids: Sequence[str],
    estimator_factory: EstimatorFactory,
    min_samples: int = 2,
    min_similarity: float = 0.3,
    n_permutations: int = 1_000,
    random_state: int,
    layer: str | None = None,
    include_shared: bool = True,
) -> HierarchicalVocabularyFit:
    """Fit recurrent vocabularies within lineages using training samples only.

    Programs are first estimated independently for each lineage and biological
    sample. A vocabulary is then built separately within each lineage. When
    ``include_shared`` is true, the lineage vocabularies are treated as the
    units for a second, explicit recurrence analysis to identify programs
    shared across lineages. The lineage metadata is a grouping variable, not a
    state target; condition and outcome columns are never inspected.
    """
    validate_anndata(adata, sample_key=sample_key, layer=layer, min_samples=2)
    _lineage_labels(adata, lineage_key)
    samples = tuple(str(value).strip() for value in training_sample_ids)
    if not samples or any(not sample for sample in samples):
        raise HierarchyError("training_sample_ids must contain non-empty identifiers")
    if len(set(samples)) != len(samples):
        raise HierarchyError("training_sample_ids must be unique")
    if min_samples < 2:
        raise ValueError("min_samples must be at least 2")

    work = adata.copy()
    lineage_labels = _lineage_labels(work, lineage_key)
    sample_labels = work.obs[sample_key].astype("string").str.strip().to_numpy()
    selected = np.isin(sample_labels, samples)
    if not selected.any():
        raise HierarchyError("training_sample_ids select no cells")
    work.obs["_sccs_lineage_sample"] = np.where(
        selected,
        np.asarray(
            [
                f"{lineage}::{sample}"
                for lineage, sample in zip(lineage_labels, sample_labels, strict=True)
            ]
        ),
        "_sccs_excluded",
    )

    lineages = tuple(sorted(set(lineage_labels[selected])))
    lineage_vocabularies: dict[str, VocabularyFit] = {}
    lineage_programs: dict[str, ProgramSet] = {}
    for lineage in lineages:
        group_ids = tuple(
            f"{lineage}::{sample}"
            for sample in samples
            if np.any((lineage_labels == lineage) & (sample_labels == sample))
        )
        if len(group_ids) < min_samples:
            raise HierarchyError(
                f"lineage {lineage!r} has {len(group_ids)} training samples; "
                f"at least {min_samples} are required"
            )
        programs = fit_programs_by_sample(
            work,
            sample_key="_sccs_lineage_sample",
            sample_ids=group_ids,
            estimator_factory=estimator_factory,
            layer=layer,
        )
        fit = build_recurrent_vocabulary(
            programs,
            min_samples=min_samples,
            min_similarity=min_similarity,
            n_permutations=n_permutations,
            random_state=random_state,
        )
        lineage_vocabularies[lineage] = fit
        lineage_programs[lineage] = ProgramSet(
            sample_id=lineage,
            feature_names=fit.vocabulary.feature_names,
            weights=fit.vocabulary.weights,
            n_cells=fit.vocabulary.n_cells,
            estimator="lineage_consensus",
            parameters={"source_training_samples": list(group_ids)},
        )

    shared = None
    if include_shared:
        if len(lineage_programs) < 2:
            raise HierarchyError("at least two lineages are required for shared vocabulary")
        shared = build_recurrent_vocabulary(
            tuple(lineage_programs.values()),
            min_samples=min(2, len(lineage_programs)),
            min_similarity=min_similarity,
            n_permutations=n_permutations,
            random_state=random_state,
        )
    return HierarchicalVocabularyFit(
        lineage_vocabularies=lineage_vocabularies,
        shared_vocabulary=shared,
        training_sample_ids=samples,
        lineage_key=lineage_key,
        sample_key=sample_key,
        parameters={
            "min_samples": min_samples,
            "min_similarity": min_similarity,
            "n_permutations": n_permutations,
            "random_state": random_state,
            "include_shared": include_shared,
        },
    )
