"""Cross-sample matching and recurrent gene-program vocabularies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

from sccellstates.programs import ProgramError, ProgramSet


class RecurrenceError(ValueError):
    """Raised when recurrent-program construction cannot proceed."""


@dataclass(frozen=True)
class MatchResult:
    """One-to-one optimal program matching and its assignment null."""

    reference_sample_id: str
    query_sample_id: str
    reference_indices: np.ndarray
    query_indices: np.ndarray
    similarities: np.ndarray
    similarity_matrix: np.ndarray
    null_scores: np.ndarray
    p_value: float

    @property
    def optimal_mean(self) -> float:
        """Mean similarity of the optimal one-to-one assignment."""
        return float(self.similarities.mean())

    @property
    def null_mean(self) -> float:
        """Mean similarity expected under random one-to-one assignment."""
        return float(self.null_scores.mean())

    @property
    def null_adjusted_mean(self) -> float:
        """Difference between optimal and mean random assignment scores."""
        return self.optimal_mean - self.null_mean


@dataclass(frozen=True)
class RedundancySummary:
    """Absolute within-vocabulary rank correlations excluding the diagonal."""

    sample_id: str
    mean_absolute_similarity: float
    max_absolute_similarity: float


@dataclass(frozen=True)
class FeatureOverlap:
    """How much feature axis the independently fitted samples actually share.

    Programs are estimated within each sample, so every sample selects its own
    genes and the samples need not share a feature axis. Recurrence can only be
    measured on the intersection, so this reports how much of each sample
    survives into the comparison. A large shortfall is a statement about the
    per-sample gene selection, not a failure of the data, and it is reported
    rather than silently repaired by selecting genes across the whole cohort.
    """

    sample_ids: tuple[str, ...]
    n_features_by_sample: tuple[int, ...]
    shared_features: tuple[str, ...]
    pairwise_shared: np.ndarray

    @property
    def n_samples(self) -> int:
        """Number of samples compared."""
        return len(self.sample_ids)

    @property
    def n_shared(self) -> int:
        """Features present in every sample, which recurrence is measured on."""
        return len(self.shared_features)

    @property
    def minimum_pairwise_shared(self) -> int:
        """Smallest number of features shared by any two samples."""
        if self.n_samples < 2:
            return self.n_shared
        off_diagonal = self.pairwise_shared[~np.eye(self.n_samples, dtype=bool)]
        return int(off_diagonal.min())

    @property
    def retention_by_sample(self) -> tuple[float, ...]:
        """Fraction of each sample's own features retained in the comparison."""
        return tuple(
            self.n_shared / n_features if n_features else 0.0
            for n_features in self.n_features_by_sample
        )

    @property
    def is_comparable(self) -> bool:
        """Whether enough features are shared to compute rank correlation."""
        return self.n_shared >= 2

    def as_record(self) -> dict[str, object]:
        """Summarize for provenance, pairing each sample with its retention."""
        return {
            "n_shared_features": self.n_shared,
            "minimum_pairwise_shared": self.minimum_pairwise_shared,
            "n_features_by_sample": {
                sample_id: n_features
                for sample_id, n_features in zip(
                    self.sample_ids, self.n_features_by_sample, strict=True
                )
            },
            "retention_by_sample": {
                sample_id: retention
                for sample_id, retention in zip(
                    self.sample_ids, self.retention_by_sample, strict=True
                )
            },
        }


@dataclass(frozen=True)
class VocabularyMember:
    """Programs contributing to one recurrent consensus program."""

    vocabulary_index: int
    anchor_program_index: int
    sample_ids: tuple[str, ...]
    program_indices: tuple[int, ...]
    similarities_to_anchor: tuple[float, ...]

    @property
    def support_count(self) -> int:
        """Number of independent samples supporting the consensus."""
        return len(self.sample_ids)


@dataclass(frozen=True)
class VocabularyFit:
    """Recurrent vocabulary plus matching and degeneracy diagnostics."""

    vocabulary: ProgramSet
    training_sample_ids: tuple[str, ...]
    reference_sample_id: str
    members: tuple[VocabularyMember, ...]
    pairwise_matches: tuple[MatchResult, ...]
    sample_redundancy: tuple[RedundancySummary, ...]
    vocabulary_redundancy: RedundancySummary
    min_samples: int
    min_similarity: float
    n_permutations: int
    random_state: int


@dataclass(frozen=True)
class RecurrenceResult:
    """Cross-sample recurrence evidence that makes no membership decision.

    Produced by :func:`compute_recurrence`. It records how independently fitted
    programs align across biological samples, including the optimal one-to-one
    matches, their random-assignment nulls, and the within-set redundancy of
    each input sample. It deliberately carries no threshold decision, so a weak
    or absent recurrence is reported as data rather than raised as an error.
    """

    program_sets: tuple[ProgramSet, ...]
    sample_ids: tuple[str, ...]
    common_features: tuple[str, ...]
    overlap: FeatureOverlap
    reference_sample_id: str
    normalized_weights: Mapping[str, np.ndarray]
    assignments: Mapping[str, Mapping[int, tuple[int, float]]]
    pairwise_matches: tuple[MatchResult, ...]
    sample_redundancy: tuple[RedundancySummary, ...]
    min_similarity: float
    n_permutations: int
    random_state: int

    @property
    def n_samples(self) -> int:
        """Number of independently fitted biological samples compared."""
        return len(self.program_sets)


@dataclass(frozen=True)
class ProgramVocabulary:
    """Programs that qualified as members of a stable meta-program vocabulary.

    Produced by :func:`build_vocabulary` from a :class:`RecurrenceResult`. A
    member qualifies only when it is supported above ``min_similarity`` in at
    least ``min_samples`` independently fitted biological samples. This is a
    membership decision over programs, not a biological validation claim about
    a cell-state representation.
    """

    programs: ProgramSet
    members: tuple[VocabularyMember, ...]
    vocabulary_redundancy: RedundancySummary
    training_sample_ids: tuple[str, ...]
    reference_sample_id: str
    min_samples: int
    min_similarity: float
    max_redundancy: float | None = None
    dropped_anchor_indices: tuple[int, ...] = ()
    parameters: Mapping[str, object] = MappingProxyType({})

    def __post_init__(self) -> None:
        members = tuple(self.members)
        if len(members) != self.programs.n_programs:
            raise RecurrenceError("every vocabulary program must have exactly one member record")
        if any(member.vocabulary_index != index for index, member in enumerate(members)):
            raise RecurrenceError("member vocabulary_index values must be 0..n-1 in order")
        object.__setattr__(self, "members", members)

    @property
    def n_programs(self) -> int:
        """Number of programs that qualified for the vocabulary."""
        return self.programs.n_programs

    @property
    def support_counts(self) -> tuple[int, ...]:
        """Independent samples supporting each vocabulary program, in order."""
        return tuple(member.support_count for member in self.members)


def _readonly(array: np.ndarray, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _feature_indices(programs: ProgramSet, feature_names: tuple[str, ...]) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(programs.feature_names)}
    return np.fromiter((lookup[name] for name in feature_names), dtype=np.int64)


def _shared_features(a: ProgramSet, b: ProgramSet) -> tuple[str, ...]:
    available = set(b.feature_names)
    shared = tuple(name for name in a.feature_names if name in available)
    if len(shared) < 2:
        raise RecurrenceError("program sets must share at least two features")
    return shared


def _rank_rows(weights: np.ndarray) -> np.ndarray:
    ranked = rankdata(weights, method="average", axis=1)
    ranked -= ranked.mean(axis=1, keepdims=True)
    scales = np.sqrt(np.sum(ranked * ranked, axis=1, keepdims=True))
    scales[scales == 0] = 1.0
    return ranked / scales


def program_similarity(reference: ProgramSet, query: ProgramSet) -> np.ndarray:
    """Compute Spearman program similarity across shared named features."""
    shared = _shared_features(reference, query)
    reference_weights = reference.weights[:, _feature_indices(reference, shared)]
    query_weights = query.weights[:, _feature_indices(query, shared)]
    return _rank_rows(reference_weights) @ _rank_rows(query_weights).T


def _random_assignment_scores(
    similarities: np.ndarray,
    *,
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_pairs = min(similarities.shape)
    scores = np.empty(n_permutations, dtype=np.float64)
    for index in range(n_permutations):
        rows = rng.permutation(similarities.shape[0])[:n_pairs]
        columns = rng.permutation(similarities.shape[1])[:n_pairs]
        scores[index] = similarities[rows, columns].mean()
    return scores


def _thresholded_assignment(
    similarities: np.ndarray, *, min_similarity: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return one-to-one matches while allowing each reference to be unmatched."""
    n_reference, n_query = similarities.shape
    augmented = np.full(
        (n_reference, n_query + n_reference),
        fill_value=-np.inf,
        dtype=np.float64,
    )
    augmented[:, :n_query] = similarities
    augmented[np.arange(n_reference), n_query + np.arange(n_reference)] = min_similarity
    rows, columns = linear_sum_assignment(augmented, maximize=True)
    real = columns < n_query
    return rows[real], columns[real]


def match_programs(
    reference: ProgramSet,
    query: ProgramSet,
    *,
    n_permutations: int = 1000,
    random_state: int,
) -> MatchResult:
    """Optimally match programs and compare with random one-to-one assignments."""
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")
    similarities = program_similarity(reference, query)
    rows, columns = linear_sum_assignment(similarities, maximize=True)
    matched = similarities[rows, columns]
    null_scores = _random_assignment_scores(
        similarities,
        n_permutations=n_permutations,
        rng=np.random.default_rng(random_state),
    )
    p_value = float((1 + np.count_nonzero(null_scores >= matched.mean())) / (n_permutations + 1))
    return MatchResult(
        reference_sample_id=reference.sample_id,
        query_sample_id=query.sample_id,
        reference_indices=_readonly(rows, dtype=np.int64),
        query_indices=_readonly(columns, dtype=np.int64),
        similarities=_readonly(matched),
        similarity_matrix=_readonly(similarities),
        null_scores=_readonly(null_scores),
        p_value=p_value,
    )


def program_redundancy(programs: ProgramSet) -> RedundancySummary:
    """Summarize within-set similarity so duplicate programs are visible."""
    if programs.n_programs == 1:
        return RedundancySummary(programs.sample_id, 0.0, 0.0)
    similarities = np.abs(program_similarity(programs, programs))
    off_diagonal = similarities[~np.eye(programs.n_programs, dtype=bool)]
    return RedundancySummary(
        sample_id=programs.sample_id,
        mean_absolute_similarity=float(off_diagonal.mean()),
        max_absolute_similarity=float(off_diagonal.max()),
    )


def feature_overlap(program_sets: tuple[ProgramSet, ...]) -> FeatureOverlap:
    """Report the shared feature coverage across independently fitted samples.

    Use this before relying on a recurrence result to judge whether the
    per-sample gene selections left enough common ground for the comparison to
    mean anything.
    """
    ordered = _ordered_sets(program_sets)
    sample_ids = tuple(item.sample_id for item in ordered)
    available = [set(item.feature_names) for item in ordered]
    common = set.intersection(*available)
    shared = tuple(name for name in ordered[0].feature_names if name in common)
    pairwise = np.empty((len(ordered), len(ordered)), dtype=np.int64)
    for row in range(len(ordered)):
        for column in range(len(ordered)):
            pairwise[row, column] = len(available[row] & available[column])
    pairwise.setflags(write=False)
    return FeatureOverlap(
        sample_ids=sample_ids,
        n_features_by_sample=tuple(item.n_features for item in ordered),
        shared_features=shared,
        pairwise_shared=pairwise,
    )


def _common_features(program_sets: tuple[ProgramSet, ...]) -> tuple[str, ...]:
    overlap = feature_overlap(program_sets)
    if not overlap.is_comparable:
        counts = ", ".join(
            f"{sample_id}={n_features}"
            for sample_id, n_features in zip(
                overlap.sample_ids, overlap.n_features_by_sample, strict=True
            )
        )
        raise RecurrenceError(
            "program sets share fewer than two features, so rank correlation cannot be "
            f"computed; per-sample feature counts are {counts}, with "
            f"{overlap.n_shared} shared"
        )
    return overlap.shared_features


def _normalized_weights(programs: ProgramSet, features: tuple[str, ...]) -> np.ndarray:
    weights = programs.weights[:, _feature_indices(programs, features)]
    totals = weights.sum(axis=1, keepdims=True)
    if (totals == 0).any():
        raise ProgramError("a program has zero weight across the shared features")
    return weights / totals


def _ordered_sets(program_sets: tuple[ProgramSet, ...]) -> tuple[ProgramSet, ...]:
    if len(program_sets) < 2:
        raise RecurrenceError("at least two training samples are required")
    ordered = tuple(sorted(program_sets, key=lambda item: item.sample_id))
    sample_ids = tuple(item.sample_id for item in ordered)
    if len(set(sample_ids)) != len(sample_ids):
        raise RecurrenceError("training sample IDs must be unique")
    return ordered


def _validate_min_samples(min_samples: int, n_samples: int) -> None:
    if not 2 <= min_samples <= n_samples:
        raise ValueError("min_samples must be between 2 and the number of training samples")


def compute_recurrence(
    program_sets: tuple[ProgramSet, ...],
    *,
    min_similarity: float = 0.3,
    n_permutations: int = 1000,
    random_state: int,
) -> RecurrenceResult:
    """Quantify how programs recur across independently fitted samples.

    Every pair of samples is matched one-to-one by rank correlation and scored
    against a random-assignment null. The medoid sample, chosen by its mean
    null-adjusted match score, becomes the anchor that all other samples are
    threshold-matched against.

    This function reports evidence only. It applies no membership threshold, so
    weak or absent recurrence is returned as a :class:`RecurrenceResult` rather
    than raised. Use :func:`build_vocabulary` to select the programs that
    qualify for a shared vocabulary.

    Parameters
    ----------
    program_sets
        Programs fitted independently within each biological sample. Repeated
        fits of one sample are computational replicates and must not be passed
        here as if they were independent samples.
    min_similarity
        Minimum rank similarity for an anchor program to be matched to a
        program in another sample.
    n_permutations
        Number of random one-to-one assignments used for the null distribution.
    random_state
        Seed controlling the null permutations. Results are deterministic for a
        fixed seed.

    Raises
    ------
    RecurrenceError
        If fewer than two samples are supplied, sample IDs are not unique, or
        the samples share fewer than two features.
    ValueError
        If ``min_similarity`` is outside ``[-1, 1]`` or ``n_permutations`` is
        less than one.
    """
    ordered_sets = _ordered_sets(program_sets)
    if not -1 <= min_similarity <= 1:
        raise ValueError("min_similarity must be between -1 and 1")
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")

    sample_ids = tuple(item.sample_id for item in ordered_sets)
    overlap = feature_overlap(ordered_sets)
    if not overlap.is_comparable:
        counts = ", ".join(
            f"{sample_id}={n_features}"
            for sample_id, n_features in zip(
                overlap.sample_ids, overlap.n_features_by_sample, strict=True
            )
        )
        raise RecurrenceError(
            "program sets share fewer than two features, so rank correlation cannot be "
            f"computed; per-sample feature counts are {counts}, with "
            f"{overlap.n_shared} shared"
        )
    common_features = overlap.shared_features
    seed_rng = np.random.default_rng(random_state)
    pairwise: list[MatchResult] = []
    adjusted_by_sample: dict[str, list[float]] = {sample_id: [] for sample_id in sample_ids}
    for left_index, left in enumerate(ordered_sets[:-1]):
        for right in ordered_sets[left_index + 1 :]:
            match = match_programs(
                left,
                right,
                n_permutations=n_permutations,
                random_state=int(seed_rng.integers(0, np.iinfo(np.int32).max)),
            )
            pairwise.append(match)
            adjusted_by_sample[left.sample_id].append(match.null_adjusted_mean)
            adjusted_by_sample[right.sample_id].append(match.null_adjusted_mean)

    reference = min(
        ordered_sets,
        key=lambda item: (-float(np.mean(adjusted_by_sample[item.sample_id])), item.sample_id),
    )
    normalized = {
        item.sample_id: _normalized_weights(item, common_features) for item in ordered_sets
    }
    assignments: dict[str, dict[int, tuple[int, float]]] = {}
    for query in ordered_sets:
        if query.sample_id == reference.sample_id:
            continue
        similarities = program_similarity(reference, query)
        rows, columns = _thresholded_assignment(similarities, min_similarity=min_similarity)
        assignments[query.sample_id] = {
            int(row): (int(column), float(similarities[row, column]))
            for row, column in zip(rows, columns, strict=True)
        }

    return RecurrenceResult(
        program_sets=ordered_sets,
        sample_ids=sample_ids,
        common_features=common_features,
        overlap=overlap,
        reference_sample_id=reference.sample_id,
        normalized_weights=MappingProxyType(normalized),
        assignments=MappingProxyType(
            {sample_id: MappingProxyType(dict(items)) for sample_id, items in assignments.items()}
        ),
        pairwise_matches=tuple(pairwise),
        sample_redundancy=tuple(program_redundancy(item) for item in ordered_sets),
        min_similarity=min_similarity,
        n_permutations=n_permutations,
        random_state=random_state,
    )


def build_vocabulary(
    recurrence: RecurrenceResult,
    *,
    min_samples: int,
    max_redundancy: float | None = None,
) -> ProgramVocabulary | None:
    """Select the programs that qualify as stable, recurrent meta-programs.

    An anchor program qualifies when it is supported above the recurrence
    ``min_similarity`` in at least ``min_samples`` independently fitted
    biological samples. Qualifying programs are combined by the feature-wise
    median of their L1-normalized gene weights.

    Parameters
    ----------
    recurrence
        Evidence from :func:`compute_recurrence`.
    min_samples
        Number of independent samples that must support a program.
    max_redundancy
        Optional ceiling on the maximum absolute rank similarity allowed
        between two programs in the returned vocabulary. When supplied, the
        most redundant programs are dropped greedily in anchor order, which
        enforces the non-redundancy condition of a stable meta-program. The
        default ``None`` only reports redundancy and never drops a program.

    Returns
    -------
    ProgramVocabulary or None
        ``None`` when no program reaches ``min_samples`` independent samples,
        or when ``max_redundancy`` removes every candidate.
    """
    if not isinstance(recurrence, RecurrenceResult):
        raise TypeError("recurrence must be a RecurrenceResult")
    _validate_min_samples(min_samples, recurrence.n_samples)
    if max_redundancy is not None and not 0 <= max_redundancy <= 1:
        raise ValueError("max_redundancy must be between 0 and 1")

    reference = next(
        item for item in recurrence.program_sets if item.sample_id == recurrence.reference_sample_id
    )
    consensus_rows: list[np.ndarray] = []
    members: list[VocabularyMember] = []
    for anchor_index in range(reference.n_programs):
        sample_members = [reference.sample_id]
        program_indices = [anchor_index]
        matched_similarities = [1.0]
        weight_rows = [recurrence.normalized_weights[reference.sample_id][anchor_index]]
        for sample_id in recurrence.sample_ids:
            if sample_id == reference.sample_id:
                continue
            matched = recurrence.assignments[sample_id].get(anchor_index)
            if matched is None or matched[1] < recurrence.min_similarity:
                continue
            program_index, similarity = matched
            sample_members.append(sample_id)
            program_indices.append(program_index)
            matched_similarities.append(similarity)
            weight_rows.append(recurrence.normalized_weights[sample_id][program_index])
        if len(sample_members) < min_samples:
            continue
        consensus = np.median(np.stack(weight_rows), axis=0)
        total = consensus.sum()
        if total == 0:
            raise RecurrenceError("consensus program has zero total weight")
        consensus_rows.append(consensus / total)
        members.append(
            VocabularyMember(
                vocabulary_index=len(consensus_rows) - 1,
                anchor_program_index=anchor_index,
                sample_ids=tuple(sample_members),
                program_indices=tuple(program_indices),
                similarities_to_anchor=tuple(matched_similarities),
            )
        )

    dropped_anchor_indices: tuple[int, ...] = ()
    if max_redundancy is not None and consensus_rows:
        # Candidates are considered in anchor order, which is the only
        # deterministic order available: the anchor is the medoid reference
        # sample, so this never depends on how the samples were stored.
        ranked = _rank_rows(np.stack(consensus_rows))
        retained: list[int] = []
        for index in range(len(consensus_rows)):
            if retained:
                ceiling = float(np.abs(ranked[index] @ ranked[retained].T).max())
                if ceiling > max_redundancy:
                    continue
            retained.append(index)
        dropped_anchor_indices = tuple(
            members[index].anchor_program_index
            for index in range(len(consensus_rows))
            if index not in set(retained)
        )
        consensus_rows = [consensus_rows[index] for index in retained]
        members = [
            replace(members[index], vocabulary_index=position)
            for position, index in enumerate(retained)
        ]

    if not consensus_rows:
        return None

    vocabulary = ProgramSet(
        sample_id="recurrent_vocabulary",
        feature_names=recurrence.common_features,
        weights=np.stack(consensus_rows),
        n_cells=sum(item.n_cells for item in recurrence.program_sets),
        estimator="median_consensus",
        parameters={
            "min_samples": min_samples,
            "min_similarity": recurrence.min_similarity,
            "n_permutations": recurrence.n_permutations,
            "random_state": recurrence.random_state,
        },
    )
    return ProgramVocabulary(
        programs=vocabulary,
        members=tuple(members),
        vocabulary_redundancy=program_redundancy(vocabulary),
        training_sample_ids=recurrence.sample_ids,
        reference_sample_id=recurrence.reference_sample_id,
        min_samples=min_samples,
        min_similarity=recurrence.min_similarity,
        max_redundancy=max_redundancy,
        dropped_anchor_indices=dropped_anchor_indices,
        parameters={
            "min_samples": min_samples,
            "min_similarity": recurrence.min_similarity,
            "n_permutations": recurrence.n_permutations,
            "random_state": recurrence.random_state,
            "max_redundancy": max_redundancy,
        },
    )


def build_recurrent_vocabulary(
    program_sets: tuple[ProgramSet, ...],
    *,
    min_samples: int,
    min_similarity: float = 0.3,
    n_permutations: int = 1000,
    random_state: int,
) -> VocabularyFit:
    """Build a consensus vocabulary from training samples only.

    A medoid sample is chosen using its mean null-adjusted pairwise match score.
    Every other sample is matched one-to-one to this anchor. Anchor programs
    supported above ``min_similarity`` in at least ``min_samples`` independent
    samples are retained, and their L1-normalized gene weights are combined by
    the feature-wise median.

    This is a convenience composition of :func:`compute_recurrence` and
    :func:`build_vocabulary` that bundles the recurrence evidence with the
    resulting vocabulary into a single :class:`VocabularyFit`. Prefer the two
    steps directly when recurrence evidence is wanted even though no program
    qualifies, because this function raises :class:`RecurrenceError` in that
    case instead of returning evidence.
    """
    ordered_sets = _ordered_sets(program_sets)
    _validate_min_samples(min_samples, len(ordered_sets))
    recurrence = compute_recurrence(
        program_sets,
        min_similarity=min_similarity,
        n_permutations=n_permutations,
        random_state=random_state,
    )
    vocabulary = build_vocabulary(recurrence, min_samples=min_samples)
    if vocabulary is None:
        raise RecurrenceError("no program met the recurrence requirements")
    return VocabularyFit(
        vocabulary=vocabulary.programs,
        training_sample_ids=recurrence.sample_ids,
        reference_sample_id=recurrence.reference_sample_id,
        members=vocabulary.members,
        pairwise_matches=recurrence.pairwise_matches,
        sample_redundancy=recurrence.sample_redundancy,
        vocabulary_redundancy=vocabulary.vocabulary_redundancy,
        min_samples=min_samples,
        min_similarity=recurrence.min_similarity,
        n_permutations=recurrence.n_permutations,
        random_state=recurrence.random_state,
    )
