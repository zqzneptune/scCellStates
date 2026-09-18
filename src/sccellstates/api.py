"""User-facing single-sample program discovery workflows."""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np

from sccellstates.dataset import filter_anndata_cells, select_training_genes
from sccellstates.io import APIError, get_matrix, load_program_result, validate_anndata
from sccellstates.pipeline import LibrarySizeLog1p
from sccellstates.programs import (
    NMFProgramEstimator,
    ProgramSet,
    ProgramStabilityFit,
    stabilize_programs,
)
from sccellstates.projection import ProjectionBenchmark, available_projectors, make_projector
from sccellstates.recurrence import (
    FeatureOverlap,
    ProgramVocabulary,
    RecurrenceResult,
    VocabularyFit,
    build_recurrent_vocabulary,
    build_vocabulary,
    compute_recurrence,
    feature_overlap,
)
from sccellstates.samples import (
    SampleCollection,
    read_source,
    samples_from_atlas,
    samples_from_sources,
)
from sccellstates.state import ProgramScorer

type Input = str | Path | ad.AnnData

# Forked children inherit the parent's memory, so the large objects a fit works
# on are published here rather than handed to the pool as task arguments.
# Pickling a task argument would serialize the matrix, and an AnnData view
# serializes its entire parent: a 3,000-cell view of a 12,000-cell atlas
# pickles to 2.6 MB, the same as the atlas itself. Only a task index crosses
# the process boundary.
_PARALLEL_STATE: dict[str, object] = {}


def _published(key: str) -> Any:
    """Read state published for the current task."""
    try:
        return _PARALLEL_STATE[key]
    except KeyError as error:  # pragma: no cover - defensive
        raise APIError(f"parallel state {key!r} was not published") from error


@contextmanager
def _published_state(**state: object) -> Iterator[None]:
    global _PARALLEL_STATE
    previous = _PARALLEL_STATE
    _PARALLEL_STATE = state
    try:
        yield
    finally:
        _PARALLEL_STATE = previous


def _fork_context() -> multiprocessing.context.BaseContext:
    try:
        return multiprocessing.get_context("fork")
    except ValueError as error:  # pragma: no cover - platform dependent
        raise APIError(
            "n_jobs greater than 1 requires the 'fork' multiprocessing start method, "
            "which this platform does not provide. Fit with n_jobs=1 and distribute "
            "samples as separate jobs instead."
        ) from error


def _map_tasks(
    function: Callable[[int], Any],
    n_items: int,
    n_jobs: int,
    **state: object,
) -> list[Any]:
    """Apply ``function`` to ``0..n_items-1``, returning results in that order.

    The serial and parallel routes run the same callable over the same published
    state, so they cannot disagree about a result. Results are collected with
    ``map``, which preserves index order: the order repeated fits are combined
    in is part of the stability consensus and is not free to change.
    """
    if n_jobs < 1:
        raise ValueError("n_jobs must be at least 1")
    workers = min(n_jobs, n_items)
    with _published_state(**state):
        if workers <= 1:
            return [function(index) for index in range(n_items)]
        context = _fork_context()
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            return list(pool.map(function, range(n_items)))


def _fit_repeat(index: int) -> ProgramSet:
    """Fit one repeated estimate of the published sample."""
    sample_id = _published("sample_id")
    return NMFProgramEstimator(
        n_programs=_published("n_programs"),
        random_state=_published("random_state") + index,
        max_iter=_published("max_iter"),
        tol=_published("tol"),
    ).fit(
        _published("matrix"),
        feature_names=_published("feature_names"),
        sample_id=f"{sample_id}::run_{index}",
    )


def _fit_collection_entry(index: int) -> ProgramResult:
    """Fit one sample of the published cohort, on its own cells only."""
    collection: SampleCollection = _published("collection")
    entry = collection.samples[index]
    return fit(
        entry.source,
        modality=_published("modality"),
        method=_published("method"),
        sample_id=entry.sample_id,
        random_state=_published("random_state"),
        **_published("fit_kwargs"),
    )


@dataclass(frozen=True)
class ProgramResult:
    """Programs and cell-level usages discovered from one RNA sample."""

    programs: np.ndarray
    usages: np.ndarray
    cell_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    sample_id: str
    selected_K: int
    stability: ProgramStabilityFit | None
    provenance: Mapping[str, object]

    @property
    def loadings(self) -> np.ndarray:
        """Return the program-by-gene loading matrix."""
        return self.programs

    def to_program_set(self) -> ProgramSet:
        """Represent these candidate programs as a within-sample ``ProgramSet``."""
        return ProgramSet(
            sample_id=self.sample_id,
            feature_names=self.feature_names,
            weights=self.programs,
            n_cells=len(self.cell_names),
            estimator=str(self.provenance.get("method", "unknown")),
            parameters=self.provenance,
        )

    def __post_init__(self) -> None:
        programs = np.array(self.programs, dtype=np.float64, copy=True)
        usages = np.array(self.usages, dtype=np.float64, copy=True)
        if programs.ndim != 2 or programs.shape[0] == 0 or programs.shape[1] == 0:
            raise APIError("programs must be a non-empty programs-by-genes matrix")
        if usages.ndim != 2:
            raise APIError("usages must be a two-dimensional cells-by-programs matrix")
        if usages.shape != (len(self.cell_names), programs.shape[0]):
            raise APIError("usages shape must match the program and cell axes")
        if len(self.feature_names) != programs.shape[1]:
            raise APIError("feature_names length must match the program gene axis")
        if not self.sample_id.strip():
            raise APIError("sample_id must be a non-empty string")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise APIError("feature_names must be unique")
        if not np.isfinite(programs).all() or (programs < 0).any():
            raise APIError("programs must be finite and non-negative")
        if not np.isfinite(usages).all():
            raise APIError("usages must be finite")
        if self.stability is not None and not isinstance(self.stability, ProgramStabilityFit):
            raise APIError("stability must be a ProgramStabilityFit or None")
        programs.setflags(write=False)
        usages.setflags(write=False)
        object.__setattr__(self, "programs", programs)
        object.__setattr__(self, "usages", usages)
        object.__setattr__(self, "cell_names", tuple(map(str, self.cell_names)))
        object.__setattr__(self, "feature_names", tuple(map(str, self.feature_names)))
        object.__setattr__(self, "sample_id", str(self.sample_id).strip())
        if self.selected_K != programs.shape[0]:
            raise APIError("selected_K must match the returned number of programs")


@dataclass(frozen=True)
class ProjectionResult:
    """Cell usages obtained by applying a fixed recurrent vocabulary."""

    usages: np.ndarray
    cell_names: tuple[str, ...]
    vocabulary: ProgramSet
    provenance: Mapping[str, object]
    states: np.ndarray | None = None
    projection_error: np.ndarray | None = None
    relative_reconstruction_error: np.ndarray | None = None
    feature_coverage: float | None = None
    observed_feature_coverage: np.ndarray | None = None
    missing_features: tuple[str, ...] = ()
    extra_features: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    vocabulary_id: str | None = None


@dataclass(frozen=True)
class ProgramCollection:
    """Candidate programs fitted independently within each biological sample.

    Each entry is the untouched output of one :func:`fit` call, so the programs
    were estimated from that sample's cells alone, using that sample's own gene
    selection and preprocessing. Nothing here establishes that any program
    recurs: that requires :func:`sccellstates.recurrence.compute_recurrence`.
    """

    samples: tuple[ProgramResult, ...]
    program_sets: tuple[ProgramSet, ...] = field(init=False)

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        if not samples:
            raise APIError("a ProgramCollection must contain at least one sample")
        if any(not isinstance(result, ProgramResult) for result in samples):
            raise TypeError("samples must contain only ProgramResult objects")
        sample_ids = [result.sample_id for result in samples]
        if len(set(sample_ids)) != len(sample_ids):
            raise APIError("ProgramCollection sample IDs must be unique")
        object.__setattr__(self, "samples", samples)
        # Built once here rather than exposed as a property: ProgramSet copies
        # its weight matrix on construction, so a property would re-copy every
        # sample's full program matrix on every access.
        object.__setattr__(
            self, "program_sets", tuple(result.to_program_set() for result in samples)
        )

    @property
    def sample_ids(self) -> tuple[str, ...]:
        """Identifiers of the fitted biological samples, in fitting order."""
        return tuple(result.sample_id for result in self.samples)

    @property
    def n_samples(self) -> int:
        """Number of independently fitted biological samples."""
        return len(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)


@dataclass(frozen=True)
class CohortResult:
    """Programs, recurrence evidence, and a vocabulary from a cohort of samples.

    ``vocabulary`` is ``None`` when no program recurred in enough independent
    samples. That is a reportable outcome, not an error, because an absent
    recurrent program is a scientific result about the cohort.
    """

    programs: ProgramCollection
    recurrence: RecurrenceResult
    vocabulary: ProgramVocabulary | None
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.programs, ProgramCollection):
            raise TypeError("programs must be a ProgramCollection")
        if not isinstance(self.recurrence, RecurrenceResult):
            raise TypeError("recurrence must be a RecurrenceResult")
        if self.vocabulary is not None and not isinstance(self.vocabulary, ProgramVocabulary):
            raise TypeError("vocabulary must be a ProgramVocabulary or None")


def _read_input(source: Input) -> ad.AnnData:
    if isinstance(source, ad.AnnData):
        return source.copy()
    path = Path(source)
    return read_source(path)


def _validate_workflow(modality: str, method: str) -> None:
    if modality != "rna":
        raise APIError("only modality='rna' is currently implemented")
    if method not in {"nmf", "cnmf"}:
        raise APIError("method must be 'nmf' or 'cnmf'")


def _guard_single_sample(source: Input, sample_key: str) -> None:
    """Refuse to treat a container holding several samples as a single sample.

    Discovering programs by pooling donors would produce a decomposition that
    looks plausible but cannot support any claim about recurrence across
    samples, so the mistake is reported rather than silently accepted.
    """
    key = str(sample_key).strip()
    if not key:
        raise APIError("sample_key must be a non-empty string")
    adata = read_source(source)
    if key not in adata.obs:
        raise APIError(f"sample key {key!r} is not present in adata.obs")
    labels = adata.obs[key].astype("string").str.strip()
    n_samples = int(labels[labels.notna() & labels.ne("")].nunique())
    if n_samples > 1:
        raise APIError(
            f"fit() discovers programs in exactly one biological sample, but "
            f"adata.obs[{key!r}] holds {n_samples} samples. Fitting them together would "
            "pool cells across samples, so the resulting programs could not be used as "
            "evidence of cross-sample recurrence. Use fit_atlas() for one container "
            "holding several samples, or fit_samples() for several containers."
        )


def _prepare_single_sample(
    source: Input,
    *,
    sample_id: str,
    layer: str | None,
    preprocessing: str,
    min_counts_per_cell: float,
    min_genes_per_cell: int,
    n_top_genes: int | None,
    min_cells_per_gene: int,
    remove_mitochondrial: bool,
    remove_ribosomal: bool,
    exclude_genes: Sequence[str],
) -> ad.AnnData:
    result = _read_input(source)
    result.obs["_sccs_sample_id"] = sample_id
    validate_anndata(result, sample_key="_sccs_sample_id", layer=layer)
    result = filter_anndata_cells(
        result,
        layer=layer,
        min_counts_per_cell=min_counts_per_cell,
        min_genes_per_cell=min_genes_per_cell,
    )
    result = select_training_genes(
        result,
        sample_key="_sccs_sample_id",
        training_sample_ids=(sample_id,),
        layer=layer,
        n_top_genes=n_top_genes,
        min_cells_per_gene=min_cells_per_gene,
        remove_mitochondrial=remove_mitochondrial,
        remove_ribosomal=remove_ribosomal,
        exclude_genes=exclude_genes,
    )
    matrix = get_matrix(result, layer=layer)
    if preprocessing == "identity":
        transformed = matrix
    elif preprocessing == "library_size_log1p":
        preprocessor = LibrarySizeLog1p()
        transformed = preprocessor.fit(matrix, np.ones(result.n_obs, dtype=bool)).transform(matrix)
    else:
        raise APIError("preprocessing must be 'identity' or 'library_size_log1p'")
    result.X = transformed
    return result


def fit(
    source: Input,
    *,
    modality: str = "rna",
    method: str = "nmf",
    sample_id: str = "sample",
    sample_key: str | None = None,
    layer: str | None = None,
    n_programs: int = 8,
    n_repeats: int = 3,
    max_iter: int = 500,
    tol: float = 1e-4,
    stability_threshold: float = 0.5,
    preprocessing: str = "library_size_log1p",
    min_counts_per_cell: float = 0,
    min_genes_per_cell: int = 0,
    n_top_genes: int | None = None,
    min_cells_per_gene: int = 0,
    remove_mitochondrial: bool = False,
    remove_ribosomal: bool = False,
    exclude_genes: Sequence[str] = (),
    random_state: int = 0,
    n_jobs: int = 1,
) -> ProgramResult:
    """Discover programs and cell usages from one RNA sample.

    The returned programs are candidates discovered in this sample alone. They
    are not stable meta-programs, because nothing here establishes that they
    recur in any other sample. Use :func:`fit_atlas` or :func:`fit_samples` to
    fit several samples independently and evaluate recurrence across them.

    ``n_repeats`` controls within-sample algorithmic stability via a repeated
    fit consensus. Repeated fits are computational replicates of one sample and
    never count as independent evidence of biological recurrence.

    ``cnmf`` is currently a deterministic repeated-NMF consensus workflow;
    the public method name leaves room for a fuller cNMF implementation
    without changing the single-sample result contract.

    Parameters
    ----------
    sample_key
        Optional ``.obs`` column naming biological samples. Supplying it
        asserts that the input holds one sample: if the column contains more
        than one, :class:`APIError` is raised pointing at the cohort entry
        points. Omitting it performs no such check.
    n_jobs
        Worker processes used for the ``n_repeats`` estimates. The default of 1
        runs them serially in this process. Repeated estimates of one sample are
        independent, so a larger value only shortens wall-clock time: the
        estimates are combined in the same order either way, and the returned
        arrays do not depend on it. Requires the ``fork`` start method.
    """
    _validate_workflow(modality, method)
    sample_id = str(sample_id).strip()
    if not sample_id:
        raise APIError("sample_id must be a non-empty string")
    if sample_key is not None:
        _guard_single_sample(source, sample_key)
    if n_programs < 1 or n_repeats < 1:
        raise ValueError("n_programs and n_repeats must be positive")
    prepared = _prepare_single_sample(
        source,
        sample_id=sample_id,
        layer=layer,
        preprocessing=preprocessing,
        min_counts_per_cell=min_counts_per_cell,
        min_genes_per_cell=min_genes_per_cell,
        n_top_genes=n_top_genes,
        min_cells_per_gene=min_cells_per_gene,
        remove_mitochondrial=remove_mitochondrial,
        remove_ribosomal=remove_ribosomal,
        exclude_genes=exclude_genes,
    )
    matrix = get_matrix(prepared)
    feature_names = tuple(map(str, prepared.var_names))
    runs = tuple(
        _map_tasks(
            _fit_repeat,
            n_repeats,
            n_jobs,
            matrix=matrix,
            feature_names=feature_names,
            sample_id=sample_id,
            n_programs=n_programs,
            random_state=random_state,
            max_iter=max_iter,
            tol=tol,
        )
    )
    stability = None
    programs = runs[0]
    if n_repeats > 1:
        stability = stabilize_programs(
            runs, sample_id=sample_id, min_similarity=stability_threshold
        )
        programs = stability.programs
    usages = ProgramScorer(programs).transform(prepared)
    provenance = {
        "schema_version": "1.0",
        "workflow": "single_sample_program_discovery",
        "modality": modality,
        "method": method,
        "sample_id": sample_id,
        "layer": layer,
        "preprocessing": preprocessing,
        "n_programs_requested": n_programs,
        "n_repeats": n_repeats,
        "random_state": random_state,
        "max_iter": max_iter,
        "tol": tol,
        "fit_diagnostics": {str(index): dict(run.parameters) for index, run in enumerate(runs)},
        "effective_K": programs.n_programs,
        "selected_K": programs.n_programs,
        "input": dict(prepared.uns.get("sccellstates_input", {})),
    }
    return ProgramResult(
        programs=programs.weights,
        usages=usages,
        cell_names=tuple(map(str, prepared.obs_names)),
        feature_names=programs.feature_names,
        sample_id=sample_id,
        selected_K=programs.n_programs,
        stability=stability,
        provenance=provenance,
    )


def _layer_of(fit_kwargs: Mapping[str, object]) -> str | None:
    layer = fit_kwargs.get("layer")
    return None if layer is None else str(layer)


def _require_shared_features(programs: ProgramCollection) -> FeatureOverlap:
    """Report shared feature coverage, failing when comparison is impossible.

    Every sample selects its own genes, so each has its own feature axis. The
    intersection is what recurrence can be measured on, and per-sample
    ``n_top_genes`` can legitimately shrink it to nothing. That is a statement
    about the gene selection rather than a bug, and it is never repaired by
    quietly selecting genes across the whole cohort, which would weaken the
    independence that makes the comparison meaningful.
    """
    overlap = feature_overlap(programs.program_sets)
    if not overlap.is_comparable:
        counts = ", ".join(
            f"{sample_id}={n_features}"
            for sample_id, n_features in zip(
                overlap.sample_ids, overlap.n_features_by_sample, strict=True
            )
        )
        retained = ", ".join(
            f"{sample_id}={retention:.0%}"
            for sample_id, retention in zip(
                overlap.sample_ids, overlap.retention_by_sample, strict=True
            )
        )
        raise APIError(
            "programs cannot be compared because the samples share "
            f"{overlap.n_shared} feature(s) after per-sample gene selection. "
            f"Per-sample feature counts: {counts}. Retained in the shared axis: {retained}. "
            "Smallest pairwise overlap: "
            f"{overlap.minimum_pairwise_shared}. Each sample selects genes independently, so "
            "n_top_genes applies per sample; raise n_top_genes, lower min_cells_per_gene, or "
            "disable n_top_genes to keep a comparable feature axis."
        )
    return overlap


_REDUCE_RESERVED = frozenset(
    {
        "schema_version",
        "modality",
        "method",
        "min_samples",
        "min_similarity",
        "n_permutations",
        "max_redundancy",
        "random_state",
        "reference_sample_id",
        "n_vocabulary_programs",
        "fit_independence",
        "pooled_fit",
        "feature_overlap",
    }
)


def _require_recurrence_cohort(n_samples: int, min_samples: int) -> None:
    """Require a cohort size and support threshold that recurrence can run on."""
    if n_samples < 2:
        raise APIError(
            "cross-sample recurrence needs at least two biological samples, but the input "
            f"resolved to {n_samples}. Use fit() to discover candidate programs "
            "in a single sample."
        )
    if not 2 <= min_samples <= n_samples:
        raise APIError(
            f"min_samples must be between 2 and the number of samples "
            f"({n_samples}); got {min_samples}"
        )


def _reduce(
    programs: ProgramCollection,
    *,
    modality: str,
    method: str,
    min_samples: int,
    min_similarity: float,
    n_permutations: int,
    max_redundancy: float | None,
    random_state: int,
    provenance_extra: Mapping[str, object],
) -> CohortResult:
    """Match independently fitted programs and select a recurrent vocabulary.

    This is the only implementation of cohort-level recurrence, and it receives
    nothing but a :class:`ProgramCollection`. It therefore cannot depend on how
    the samples were fitted, named, ordered, or stored, which is what makes
    :func:`fit_atlas`, :func:`fit_samples`, and :func:`aggregate` the same
    computation rather than three that must be kept in agreement.

    The keyword set here is the equivalence contract: the entry points above
    expose ``min_samples``, ``min_similarity``, ``n_permutations``,
    ``max_redundancy``, and ``random_state`` with identical names and defaults.
    Everything else about a fit is already baked into the programs it produced
    and is deliberately absent from :func:`aggregate`.
    """
    collisions = sorted(_REDUCE_RESERVED & set(provenance_extra))
    if collisions:
        raise APIError(f"provenance_extra must not set reserved keys: {collisions}")
    _require_recurrence_cohort(programs.n_samples, min_samples)
    overlap = _require_shared_features(programs)
    recurrence = compute_recurrence(
        programs.program_sets,
        min_similarity=min_similarity,
        n_permutations=n_permutations,
        random_state=random_state,
    )
    vocabulary = build_vocabulary(
        recurrence, min_samples=min_samples, max_redundancy=max_redundancy
    )
    # Computed keys are written last, after the caller's provenance, so a caller
    # can never assert the no-pooling claim on its own behalf.
    provenance: dict[str, object] = dict(provenance_extra)
    provenance.update(
        {
            "schema_version": "1.0",
            "modality": modality,
            "method": method,
            "min_samples": min_samples,
            "min_similarity": min_similarity,
            "n_permutations": n_permutations,
            "max_redundancy": max_redundancy,
            "random_state": random_state,
            "reference_sample_id": recurrence.reference_sample_id,
            "n_vocabulary_programs": None if vocabulary is None else vocabulary.n_programs,
            "fit_independence": "per_sample",
            "pooled_fit": False,
            "feature_overlap": overlap.as_record(),
        }
    )
    return CohortResult(
        programs=programs,
        recurrence=recurrence,
        vocabulary=vocabulary,
        provenance=provenance,
    )


def _fit_collection(
    collection: SampleCollection,
    *,
    modality: str,
    method: str,
    min_samples: int,
    min_similarity: float,
    n_permutations: int,
    max_redundancy: float | None,
    random_state: int,
    n_jobs: int,
    fit_kwargs: Mapping[str, object],
) -> CohortResult:
    """Fit every sample independently, then quantify recurrence across them.

    The single-sample estimator is applied once per biological sample and never
    sees more than one sample's cells, which is the property that makes the
    later recurrence analysis meaningful.
    """
    # Checked before any fitting so an unusable cohort fails without paying for
    # a single decomposition.
    _require_recurrence_cohort(collection.n_samples, min_samples)
    # Every sample is fitted with the same seed. Deriving a seed from a sample
    # ID or its position would make renaming or reordering inputs change the
    # programs, which is exactly the coupling between storage and analysis that
    # this design exists to remove.
    # Each sample is fitted from its own cells only, whether the fits run here
    # or in a child process. The per-sample fits deliberately receive no
    # n_jobs: this argument parallelizes samples, and letting it also fan out
    # inside every sample would multiply the two.
    results = tuple(
        _map_tasks(
            _fit_collection_entry,
            collection.n_samples,
            n_jobs,
            collection=collection,
            modality=modality,
            method=method,
            random_state=random_state,
            fit_kwargs=dict(fit_kwargs),
        )
    )
    return _reduce(
        ProgramCollection(samples=results),
        modality=modality,
        method=method,
        min_samples=min_samples,
        min_similarity=min_similarity,
        n_permutations=n_permutations,
        max_redundancy=max_redundancy,
        random_state=random_state,
        provenance_extra={
            "workflow": "cohort_program_discovery",
            "storage": collection.storage,
            "sample_key": collection.sample_key,
            "sample_ids": list(collection.sample_ids),
            "n_samples": collection.n_samples,
            "fit_parameters": dict(fit_kwargs),
        },
    )


def fit_atlas(
    source: Input,
    *,
    sample_key: str,
    modality: str = "rna",
    method: str = "nmf",
    min_samples: int = 2,
    min_similarity: float = 0.3,
    n_permutations: int = 1_000,
    max_redundancy: float | None = None,
    random_state: int = 0,
    n_jobs: int = 1,
    **fit_kwargs: object,
) -> CohortResult:
    """Discover programs independently within each sample of one container.

    The container is a storage detail. The ``sample_key`` column defines the
    biological sample boundary, and that boundary is what recurrence is
    evaluated across. A pooled fit over every cell in the container is never
    performed: each sample is fitted by :func:`fit` on its own cells, with its
    own gene selection and its own preprocessing.

    Parameters
    ----------
    source
        One ``AnnData`` object or one ``.h5ad``, ``.h5``, or ``.zarr`` store
        holding several biological samples.
    sample_key
        ``.obs`` column that identifies independent biological samples.
    min_samples
        Independent samples that must support a program for it to qualify for
        the returned vocabulary.
    min_similarity
        Minimum rank similarity for matching a program across samples.
    n_permutations
        Random one-to-one assignments used for the matching null.
    max_redundancy
        Optional ceiling on the maximum absolute similarity allowed between two
        programs in the vocabulary. ``None`` reports redundancy without
        dropping anything.
    random_state
        Seed for every per-sample fit and for the matching null. The same seed
        is used for every sample so that renaming or reordering inputs cannot
        change the programs.
    n_jobs
        Worker processes used to fit the samples. The default of 1 fits them
        serially in this process. Samples are independent, so a larger value
        only shortens wall-clock time; the fits are collected in sample order
        either way. This is not forwarded to the per-sample fits, which stay
        serial, so the two levels cannot multiply. Requires ``fork``.
    **fit_kwargs
        Remaining keyword arguments forwarded to :func:`fit` for every sample,
        for example ``layer``, ``n_programs``, ``n_repeats``, and
        ``preprocessing``.

    Returns
    -------
    CohortResult
        Candidate programs per sample, recurrence evidence, and the qualifying
        vocabulary, which is ``None`` when nothing recurs.

    Raises
    ------
    APIError
        If ``sample_key`` is missing or identifies fewer than two samples, if
        ``sample_id`` is passed, or if the modality or method is unsupported.
    """
    _validate_workflow(modality, method)
    if "sample_id" in fit_kwargs:
        raise APIError("fit_atlas() names samples from sample_key; remove sample_id")
    collection = samples_from_atlas(source, sample_key=sample_key, layer=_layer_of(fit_kwargs))
    return _fit_collection(
        collection,
        modality=modality,
        method=method,
        min_samples=min_samples,
        min_similarity=min_similarity,
        n_permutations=n_permutations,
        max_redundancy=max_redundancy,
        random_state=random_state,
        n_jobs=n_jobs,
        fit_kwargs=fit_kwargs,
    )


def fit_samples(
    sources: Mapping[str, Input] | Sequence[Input],
    *,
    sample_key: str | None = None,
    modality: str = "rna",
    method: str = "nmf",
    min_samples: int = 2,
    min_similarity: float = 0.3,
    n_permutations: int = 1_000,
    max_redundancy: float | None = None,
    random_state: int = 0,
    n_jobs: int = 1,
    **fit_kwargs: object,
) -> CohortResult:
    """Discover programs independently within each of several containers.

    A mapping names each sample explicitly through its keys. A sequence treats
    each container as one biological sample, named from its file stem, unless
    ``sample_key`` is given, in which case each container may contribute
    several samples named by that ``.obs`` column.

    Each resolved sample is fitted by :func:`fit` on its own cells. Containers
    are never concatenated into a single matrix.

    Parameters
    ----------
    sources
        Mapping of sample ID to source, or sequence of sources.
    sample_key
        Optional ``.obs`` column naming biological samples. Cannot be combined
        with a mapping, whose keys would conflict with the column.
    random_state
        Seed for every per-sample fit and for the matching null. The same seed
        is used for every sample so that renaming or reordering inputs cannot
        change the programs.
    n_jobs
        Worker processes used to fit the samples. The default of 1 fits them
        serially in this process. Samples are independent, so a larger value
        only shortens wall-clock time; the fits are collected in sample order
        either way. This is not forwarded to the per-sample fits, which stay
        serial, so the two levels cannot multiply. Requires ``fork``.
    **fit_kwargs
        Remaining keyword arguments forwarded to :func:`fit` for every sample.

    Returns
    -------
    CohortResult
        Candidate programs per sample, recurrence evidence, and the qualifying
        vocabulary, which is ``None`` when nothing recurs.
    """
    _validate_workflow(modality, method)
    if "sample_id" in fit_kwargs:
        raise APIError("fit_samples() names samples from the inputs; remove sample_id")
    collection = samples_from_sources(sources, sample_key=sample_key, layer=_layer_of(fit_kwargs))
    return _fit_collection(
        collection,
        modality=modality,
        method=method,
        min_samples=min_samples,
        min_similarity=min_similarity,
        n_permutations=n_permutations,
        max_redundancy=max_redundancy,
        random_state=random_state,
        n_jobs=n_jobs,
        fit_kwargs=fit_kwargs,
    )


type ProgramResultSource = str | Path | ProgramResult

# Fit settings that every aggregated result must agree on. Selected features are
# deliberately absent: each sample selects its own genes by design, and
# recurrence is measured on the shared axis instead.
_PROGRAM_RESULT_AGREEMENT = (
    ("modality", "modality"),
    ("method", "method"),
    ("preprocessing", "preprocessing"),
    ("n_repeats", "n_repeats"),
)


def _load_one_result(source: object) -> ProgramResult:
    """Load one program result from a path or pass through a loaded result."""
    if isinstance(source, ProgramResult):
        return source
    if isinstance(source, str | Path):
        path = Path(source)
        if path.is_dir():
            raise APIError(
                f"expected one program result but {path} is a directory; pass a results "
                "directory on its own rather than inside a sequence"
            )
        return load_program_result(path)
    raise TypeError(
        f"program results must be paths or ProgramResult objects, not {type(source).__name__}"
    )


def _results_from_directory(directory: Path) -> tuple[ProgramResult, ...]:
    """Load every program result in one directory.

    Discovery is a non-recursive scan, so a nested ``vocabulary/`` directory
    written beside the results is not mistaken for one of them.
    """
    paths = sorted(directory.glob("*.h5ad"))
    if not paths:
        raise APIError(f"no .h5ad program results found in {directory}")
    return tuple(load_program_result(path) for path in paths)


def _read_program_results(
    results: ProgramResultSource | Sequence[ProgramResultSource] | Mapping[str, object],
) -> tuple[ProgramResult, ...]:
    """Resolve every supported input form into loaded program results."""
    if isinstance(results, ProgramResult):
        return (results,)
    if isinstance(results, Mapping):
        if not results:
            raise APIError("aggregate() needs at least one program result")
        loaded: list[ProgramResult] = []
        for key, source in results.items():
            expected = str(key).strip()
            if not expected:
                raise APIError("mapping keys must be non-empty sample IDs")
            result = _load_one_result(source)
            if result.sample_id != expected:
                raise APIError(
                    f"mapping key {expected!r} does not match the sample ID "
                    f"{result.sample_id!r} recorded in that program result"
                )
            loaded.append(result)
        return tuple(loaded)
    if isinstance(results, str | Path):
        path = Path(results)
        if path.is_dir():
            return _results_from_directory(path)
        if path.is_file():
            return (load_program_result(path),)
        raise APIError(
            f"aggregate() expects a directory of program results or a program result "
            f"file, but {path} is neither"
        )
    if isinstance(results, Sequence):
        if not results:
            raise APIError("aggregate() needs at least one program result")
        return tuple(_load_one_result(source) for source in results)
    raise TypeError(
        "results must be a directory path, a program result path, a sequence of either, "
        f"a mapping of sample ID to either, or a ProgramResult, not {type(results).__name__}"
    )


def _require_compatible_results(results: tuple[ProgramResult, ...]) -> tuple[str, str]:
    """Require results whose fit settings agree, and report the shared workflow.

    Aggregation combines fits that were run independently, so nothing guarantees
    they are comparable. Settings that change what a program means are required
    to match, which keeps ``min_samples`` counting samples of equal evidential
    quality. Selected features are not required to match: each sample selects
    its own genes, and recurrence is measured on the shared axis.
    """
    for key, label in _PROGRAM_RESULT_AGREEMENT:
        observed: dict[str, list[str]] = {}
        for result in results:
            observed.setdefault(str(result.provenance.get(key)), []).append(result.sample_id)
        if len(observed) > 1:
            detail = "; ".join(
                f"{value}={sorted(sample_ids)}" for value, sample_ids in sorted(observed.items())
            )
            raise APIError(
                f"program results must share the same {label} to be aggregated, but "
                f"these disagree: {detail}"
            )
    modality = str(results[0].provenance.get("modality", "")).strip()
    method = str(results[0].provenance.get("method", "")).strip()
    _validate_workflow(modality, method)
    return modality, method


def aggregate(
    results: ProgramResultSource
    | Sequence[ProgramResultSource]
    | Mapping[str, ProgramResultSource],
    *,
    min_samples: int = 2,
    min_similarity: float = 0.3,
    n_permutations: int = 1_000,
    max_redundancy: float | None = None,
    random_state: int = 0,
) -> CohortResult:
    """Combine independently fitted samples into recurrence evidence.

    Each input is the untouched result of one :func:`fit` call, so the programs
    were estimated from that sample's cells alone. This function brings those
    compact results together and performs the same reduction that
    :func:`fit_atlas` and :func:`fit_samples` perform, which makes fitting each
    sample separately and aggregating equivalent to fitting one container in a
    single call. The samples never need to be reachable at the same time, so
    they can be fitted as independent jobs on separate machines.

    Recurrence is computed from the stored program weights and feature axes. The
    per-sample usages are carried in each result for downstream use but are not
    needed here, so an atlas never has to be loaded in full.

    Parameters
    ----------
    results
        A directory holding ``.h5ad`` program results from
        :func:`~sccellstates.save_program_result`, one result file, a sequence
        of either, a mapping of sample ID to either, or already-loaded
        :class:`ProgramResult` objects. Sample IDs always come from the metadata
        inside each result, never from its file name. A mapping key is checked
        against that metadata rather than used to name the sample.
    min_samples
        Independent samples that must support a program for it to qualify for
        the returned vocabulary.
    min_similarity
        Minimum rank similarity for matching a program across samples.
    n_permutations
        Random one-to-one assignments used for the matching null.
    max_redundancy
        Optional ceiling on the maximum absolute similarity allowed between two
        programs in the vocabulary. ``None`` reports redundancy without
        dropping anything.
    random_state
        Seed for the matching null. Results are deterministic for a fixed seed.

    Returns
    -------
    CohortResult
        Candidate programs per sample, recurrence evidence, and the qualifying
        vocabulary, which is ``None`` when nothing recurs.

    Raises
    ------
    APIError
        If no results are found, sample IDs repeat, a mapping key disagrees with
        the stored sample ID, the results disagree on modality, method,
        preprocessing, or repeated-fit count, or fewer than two samples remain.
    InputError
        If a file is not a readable program result. Both derive from
        :class:`~sccellstates.InputError`, so one ``except`` catches them all.
    TypeError
        If an input is not a supported type.
    """
    loaded = _read_program_results(results)
    sample_ids = [result.sample_id for result in loaded]
    duplicates = sorted({name for name in sample_ids if sample_ids.count(name) > 1})
    if duplicates:
        raise APIError(f"program result sample IDs must be unique, but these repeat: {duplicates}")
    modality, method = _require_compatible_results(loaded)
    # Sorted by sample ID so the returned collection and its provenance do not
    # depend on directory listing order or on the order results were passed in.
    ordered = tuple(sorted(loaded, key=lambda result: result.sample_id))
    return _reduce(
        ProgramCollection(samples=ordered),
        modality=modality,
        method=method,
        min_samples=min_samples,
        min_similarity=min_similarity,
        n_permutations=n_permutations,
        max_redundancy=max_redundancy,
        random_state=random_state,
        provenance_extra={
            "workflow": "cohort_program_aggregation",
            "storage": "program_results",
            "sample_ids": [result.sample_id for result in ordered],
            "n_samples": len(ordered),
            "n_repeats": ordered[0].provenance.get("n_repeats"),
            "preprocessing": ordered[0].provenance.get("preprocessing"),
            "selected_K_by_sample": {result.sample_id: result.selected_K for result in ordered},
        },
    )


def find_recurrent_programs(
    results: ProgramCollection | Sequence[ProgramResult],
    *,
    min_samples: int = 2,
    min_similarity: float = 0.3,
    n_permutations: int = 1_000,
    random_state: int = 0,
) -> VocabularyFit:
    """Match independently discovered programs and build a recurrent vocabulary.

    This bundles recurrence evidence and the vocabulary into one
    :class:`~sccellstates.recurrence.VocabularyFit` and raises
    :class:`~sccellstates.recurrence.RecurrenceError` when no program recurs.
    Use :func:`~sccellstates.recurrence.compute_recurrence` followed by
    :func:`~sccellstates.recurrence.build_vocabulary` to obtain the evidence
    even when nothing recurs, which is what :func:`fit_atlas` and
    :func:`fit_samples` do.
    """
    programs = results if isinstance(results, ProgramCollection) else ProgramCollection(results)
    return build_recurrent_vocabulary(
        programs.program_sets,
        min_samples=min_samples,
        min_similarity=min_similarity,
        n_permutations=n_permutations,
        random_state=random_state,
    )


def _as_program_set(vocabulary: object) -> ProgramSet:
    """Resolve any vocabulary-bearing result to a frozen ``ProgramSet``."""
    if isinstance(vocabulary, ProgramSet):
        return vocabulary
    if isinstance(vocabulary, VocabularyFit):
        return vocabulary.vocabulary
    if isinstance(vocabulary, ProgramVocabulary):
        return vocabulary.programs
    if isinstance(vocabulary, CohortResult):
        if vocabulary.vocabulary is None:
            raise APIError(
                "this cohort result has no vocabulary because no program recurred in enough "
                "independent samples; there is nothing to project onto"
            )
        return vocabulary.vocabulary.programs
    raise APIError(
        "vocabulary must be a ProgramSet, VocabularyFit, ProgramVocabulary, or CohortResult"
    )


def project(
    source: Input,
    vocabulary: VocabularyFit | ProgramVocabulary | CohortResult | ProgramSet,
    *,
    modality: str = "rna",
    layer: str | None = None,
    preprocessing: str = "identity",
    method: str = "direct",
    sample_id: str = "projection",
    **projector_kwargs: object,
) -> ProjectionResult:
    """Project cells onto a fixed recurrent or sample-specific vocabulary.

    ``vocabulary`` accepts a bare ``ProgramSet``, a ``VocabularyFit`` from the
    cohort pipeline, a ``ProgramVocabulary`` from
    :func:`~sccellstates.recurrence.build_vocabulary`, or a :class:`CohortResult`
    straight from :func:`fit_atlas` or :func:`fit_samples`. A cohort result
    whose vocabulary is ``None`` is rejected rather than silently projecting
    onto nothing.
    """
    _validate_workflow(modality, "nmf")
    if method == "direct" and projector_kwargs:
        raise APIError("projector options are not valid for method='direct'")
    programs = _as_program_set(vocabulary)
    prepared = _prepare_single_sample(
        source,
        sample_id=sample_id,
        layer=layer,
        preprocessing=preprocessing,
        min_counts_per_cell=0,
        min_genes_per_cell=0,
        n_top_genes=None,
        min_cells_per_gene=0,
        remove_mitochondrial=False,
        remove_ribosomal=False,
        exclude_genes=(),
    )
    if method == "direct":
        usages = ProgramScorer(programs).transform(prepared)
        return ProjectionResult(
            usages=usages,
            cell_names=tuple(map(str, prepared.obs_names)),
            vocabulary=programs,
            provenance={
                "workflow": "fixed_vocabulary_projection",
                "method": method,
                "layer": layer,
            },
        )
    if method == "nnls":
        expected_preprocessing = programs.parameters.get("preprocessing")
        if expected_preprocessing is not None and preprocessing != expected_preprocessing:
            raise APIError(
                f"projection preprocessing {preprocessing!r} is incompatible with the "
                f"vocabulary scale {expected_preprocessing!r}"
            )
        projector = make_projector("nnls", programs, **projector_kwargs)
        return projector.transform(prepared, layer=None, sample_id=sample_id)
    if method not in available_projectors():
        raise APIError(
            f"unknown project method {method!r}; available: direct, "
            f"{', '.join(available_projectors())}"
        )
    try:
        projector = make_projector(method, programs, **projector_kwargs)
        if method == "neural":
            # Convenience path for one-input comparisons.  Scientific
            # benchmarking should fit this projector on discovery samples and
            # call transform on held-out samples explicitly.
            projector.fit(X=prepared, layer=None)
    except (TypeError, ValueError) as error:
        raise APIError(str(error)) from error
    return projector.transform(prepared, layer=None, sample_id=sample_id)


def compare_projectors(
    samples: Sequence[Input] | Mapping[str, Input],
    vocabulary: VocabularyFit | ProgramVocabulary | CohortResult | ProgramSet,
    *,
    methods: Sequence[str],
    modality: str = "rna",
    layer: str | None = None,
    preprocessing: str = "identity",
    projector_options: Mapping[str, Mapping[str, object]] | None = None,
    **projector_kwargs: object,
) -> ProjectionBenchmark:
    """Apply registered projectors to identical samples and vocabulary.

    This helper deliberately performs no model selection. It returns the raw
    comparable results so evaluation policy can be applied using sample-level
    splits and training-only tuning outside this convenience function.
    """
    names = tuple(str(method) for method in methods)
    if not names:
        raise APIError("methods must contain at least one projector")
    if isinstance(samples, Mapping):
        ordered_samples = tuple((name, samples[name]) for name in sorted(samples))
    else:
        ordered_samples = tuple((None, sample) for sample in samples)
    if not ordered_samples:
        raise APIError("samples must contain at least one input")
    options = (
        {}
        if projector_options is None
        else {str(name): dict(values) for name, values in projector_options.items()}
    )
    results = {
        method: tuple(
            project(
                sample,
                vocabulary,
                modality=modality,
                layer=layer,
                preprocessing=preprocessing,
                method=method,
                sample_id=sample_id or "projection",
                **projector_kwargs,
                **options.get(method, {}),
            )
            for sample_id, sample in ordered_samples
        )
        for method in names
    }
    return ProjectionBenchmark(results=results, methods=names)
