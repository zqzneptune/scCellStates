import anndata as ad
import numpy as np
import pandas as pd
import pytest

import sccellstates as sccs


def program_set(
    sample_id: str,
    weights: np.ndarray,
    *,
    features: tuple[str, ...] | None = None,
) -> sccs.ProgramSet:
    if features is None:
        features = tuple(f"g{i}" for i in range(weights.shape[1]))
    return sccs.ProgramSet(
        sample_id=sample_id,
        feature_names=features,
        weights=weights,
        n_cells=100,
        estimator="synthetic",
    )


def base_programs() -> np.ndarray:
    return np.array(
        [
            [9, 8, 7, 1, 2, 3, 1, 1, 1],
            [1, 2, 1, 9, 8, 7, 1, 3, 2],
            [2, 1, 3, 1, 1, 2, 9, 8, 7],
        ],
        dtype=float,
    )


def test_matching_aligns_program_and_feature_permutations() -> None:
    base = base_programs()
    reference = program_set("a", base)
    feature_order = np.array([8, 2, 5, 1, 7, 0, 6, 4, 3])
    program_order = np.array([2, 0, 1])
    query = program_set(
        "b",
        base[program_order][:, feature_order],
        features=tuple(f"g{i}" for i in feature_order),
    )
    result = sccs.match_programs(reference, query, n_permutations=200, random_state=4)
    np.testing.assert_allclose(result.similarities, 1.0)
    np.testing.assert_array_equal(result.query_indices, [1, 2, 0])
    assert result.optimal_mean > result.null_mean


def test_assignment_null_and_redundancy_expose_degenerate_programs() -> None:
    repeated = np.tile(np.arange(1, 10, dtype=float), (3, 1))
    left = program_set("a", repeated)
    right = program_set("b", repeated)
    result = sccs.match_programs(left, right, n_permutations=100, random_state=8)
    redundancy = sccs.program_redundancy(left)
    assert result.optimal_mean == pytest.approx(1.0)
    assert result.null_mean == pytest.approx(1.0)
    assert result.null_adjusted_mean == pytest.approx(0.0)
    assert redundancy.mean_absolute_similarity == pytest.approx(1.0)
    assert redundancy.max_absolute_similarity == pytest.approx(1.0)


def test_recurrent_vocabulary_recovers_shared_synthetic_programs() -> None:
    rng = np.random.default_rng(12)
    base = base_programs()
    sample_sets = []
    for index in range(4):
        shared = np.clip(base + rng.normal(0, 0.01, size=base.shape), 0.01, None)
        private = np.full((1, base.shape[1]), 0.1)
        private[0, index * 2] = 10.0
        weights = np.vstack([shared, private])
        weights = weights[rng.permutation(weights.shape[0])]
        sample_sets.append(program_set(f"d{index}", weights))

    fit = sccs.build_recurrent_vocabulary(
        tuple(sample_sets),
        min_samples=4,
        min_similarity=0.8,
        n_permutations=200,
        random_state=33,
    )
    assert fit.vocabulary.n_programs == 3
    assert all(member.support_count == 4 for member in fit.members)
    assert fit.training_sample_ids == ("d0", "d1", "d2", "d3")
    recovered = sccs.program_similarity(program_set("truth", base), fit.vocabulary)
    rows, columns = np.unravel_index(np.argsort(recovered, axis=None)[-3:], recovered.shape)
    assert len(set(rows)) == 3
    assert len(set(columns)) == 3
    assert recovered[rows, columns].min() > 0.95


def test_recurrent_vocabulary_and_null_are_deterministic() -> None:
    base = base_programs()
    sample_sets = tuple(program_set(f"d{i}", np.roll(base, i, axis=0)) for i in range(3))
    first = sccs.build_recurrent_vocabulary(
        sample_sets,
        min_samples=3,
        min_similarity=0.9,
        n_permutations=50,
        random_state=91,
    )
    second = sccs.build_recurrent_vocabulary(
        sample_sets,
        min_samples=3,
        min_similarity=0.9,
        n_permutations=50,
        random_state=91,
    )
    np.testing.assert_array_equal(first.vocabulary.weights, second.vocabulary.weights)
    for first_match, second_match in zip(
        first.pairwise_matches, second.pairwise_matches, strict=True
    ):
        np.testing.assert_array_equal(first_match.null_scores, second_match.null_scores)


def test_similarity_requires_shared_features() -> None:
    left = program_set("a", np.ones((2, 2)), features=("a", "b"))
    right = program_set("b", np.ones((2, 2)), features=("c", "d"))
    with pytest.raises(sccs.RecurrenceError, match="share at least two"):
        sccs.program_similarity(left, right)


def test_store_vocabulary_maps_features_and_preserves_provenance(tmp_path) -> None:
    base = base_programs()
    sample_sets = tuple(program_set(f"d{i}", np.roll(base, i, axis=0)) for i in range(3))
    fit = sccs.build_recurrent_vocabulary(
        sample_sets,
        min_samples=3,
        min_similarity=0.9,
        n_permutations=25,
        random_state=5,
    )
    var_names = ["unused", *reversed(fit.vocabulary.feature_names)]
    adata = ad.AnnData(
        X=np.ones((2, len(var_names))),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=var_names),
        uns={"sccellstates": {"existing": "kept"}},
    )
    sccs.store_program_vocabulary(adata, fit)
    stored = adata.varm["sccs_programs"]
    assert stored.shape == (len(var_names), 3)
    np.testing.assert_array_equal(stored[0], 0.0)
    for feature_index, name in enumerate(fit.vocabulary.feature_names):
        target_index = adata.var_names.get_loc(name)
        np.testing.assert_allclose(stored[target_index], fit.vocabulary.weights[:, feature_index])
    assert adata.uns["sccellstates"]["existing"] == "kept"
    assert adata.uns["sccellstates"]["program_vocabulary"]["training_sample_ids"] == [
        "d0",
        "d1",
        "d2",
    ]
    output_path = tmp_path / "programs.h5ad"
    adata.write_h5ad(output_path)
    restored = ad.read_h5ad(output_path)
    np.testing.assert_allclose(restored.varm["sccs_programs"], stored)
    assert restored.uns["sccellstates"]["program_vocabulary"]["reference_sample_id"] in {
        "d0",
        "d1",
        "d2",
    }
    with pytest.raises(sccs.InputError, match="already exists"):
        sccs.store_program_vocabulary(adata, fit)


def test_build_recurrent_vocabulary_composes_compute_and_build() -> None:
    """The legacy aggregate must stay exactly the composition of the two steps.

    Comparing the null scores as well as the weights pins the order in which
    the seeded generator is consumed, which a qualitative recovery test would
    not catch.
    """
    base = base_programs()
    sample_sets = tuple(program_set(f"d{i}", np.roll(base, i, axis=0)) for i in range(3))
    options = {"min_similarity": 0.9, "n_permutations": 40, "random_state": 91}

    fit = sccs.build_recurrent_vocabulary(sample_sets, min_samples=3, **options)
    recurrence = sccs.compute_recurrence(sample_sets, **options)
    vocabulary = sccs.build_vocabulary(recurrence, min_samples=3)

    assert vocabulary is not None
    np.testing.assert_array_equal(fit.vocabulary.weights, vocabulary.programs.weights)
    assert fit.reference_sample_id == recurrence.reference_sample_id
    assert fit.training_sample_ids == recurrence.sample_ids
    assert [member.program_indices for member in fit.members] == [
        member.program_indices for member in vocabulary.members
    ]
    for aggregated, evidence in zip(fit.pairwise_matches, recurrence.pairwise_matches, strict=True):
        assert aggregated.reference_sample_id == evidence.reference_sample_id
        assert aggregated.query_sample_id == evidence.query_sample_id
        np.testing.assert_array_equal(aggregated.null_scores, evidence.null_scores)


def test_compute_recurrence_reports_absent_recurrence_without_raising() -> None:
    features = tuple(f"g{i}" for i in range(6))
    sample_sets = []
    for index in range(3):
        weights = np.full((2, 6), 0.01)
        weights[0, 2 * index] = 10.0
        weights[1, 2 * index + 1] = 10.0
        sample_sets.append(program_set(f"d{index}", weights, features=features))

    recurrence = sccs.compute_recurrence(
        sample_sets, min_similarity=0.9, n_permutations=20, random_state=0
    )

    assert recurrence.n_samples == 3
    assert recurrence.reference_sample_id in recurrence.sample_ids
    assert len(recurrence.pairwise_matches) == 3
    assert sccs.build_vocabulary(recurrence, min_samples=2) is None
    with pytest.raises(sccs.RecurrenceError, match="no program met"):
        sccs.build_recurrent_vocabulary(
            sample_sets, min_samples=2, min_similarity=0.9, n_permutations=20, random_state=0
        )


def test_build_vocabulary_drops_redundant_programs_when_asked() -> None:
    base = base_programs()
    duplicated = np.vstack([base, base])
    sample_sets = tuple(program_set(f"d{i}", duplicated) for i in range(3))
    recurrence = sccs.compute_recurrence(
        sample_sets, min_similarity=0.9, n_permutations=10, random_state=0
    )

    unbounded = sccs.build_vocabulary(recurrence, min_samples=3)
    bounded = sccs.build_vocabulary(recurrence, min_samples=3, max_redundancy=0.9)

    assert unbounded is not None
    assert unbounded.max_redundancy is None
    assert unbounded.dropped_anchor_indices == ()
    assert unbounded.vocabulary_redundancy.max_absolute_similarity > 0.9

    assert bounded is not None
    assert bounded.n_programs < unbounded.n_programs
    assert bounded.vocabulary_redundancy.max_absolute_similarity <= 0.9
    assert bounded.dropped_anchor_indices
    assert [member.vocabulary_index for member in bounded.members] == list(
        range(bounded.n_programs)
    )


def test_build_vocabulary_rejects_thresholds_outside_their_range() -> None:
    base = base_programs()
    sample_sets = tuple(program_set(f"d{i}", np.roll(base, i, axis=0)) for i in range(3))
    recurrence = sccs.compute_recurrence(
        sample_sets, min_similarity=0.5, n_permutations=10, random_state=0
    )

    with pytest.raises(ValueError, match="max_redundancy must be between"):
        sccs.build_vocabulary(recurrence, min_samples=2, max_redundancy=1.5)
    with pytest.raises(ValueError, match="min_samples must be between"):
        sccs.build_vocabulary(recurrence, min_samples=5)
    with pytest.raises(TypeError, match="RecurrenceResult"):
        sccs.build_vocabulary(("not", "evidence"), min_samples=2)


def test_feature_overlap_reports_per_sample_retention() -> None:
    left = program_set("a", np.ones((2, 5)), features=("g0", "g1", "g2", "g3", "g4"))
    right = program_set("b", np.ones((2, 3)), features=("g2", "g3", "g4"))

    overlap = sccs.feature_overlap((left, right))

    assert overlap.sample_ids == ("a", "b")
    assert overlap.n_features_by_sample == (5, 3)
    assert overlap.shared_features == ("g2", "g3", "g4")
    assert overlap.n_shared == 3
    assert overlap.retention_by_sample == (0.6, 1.0)
    assert overlap.minimum_pairwise_shared == 3
    assert overlap.is_comparable
    assert overlap.as_record()["n_features_by_sample"] == {"a": 5, "b": 3}


def test_feature_overlap_reports_insufficient_common_ground() -> None:
    left = program_set("a", np.ones((2, 3)), features=("g0", "g1", "g2"))
    right = program_set("b", np.ones((2, 3)), features=("g3", "g4", "g5"))

    overlap = sccs.feature_overlap((left, right))

    assert overlap.n_shared == 0
    assert overlap.retention_by_sample == (0.0, 0.0)
    assert not overlap.is_comparable
    with pytest.raises(sccs.RecurrenceError, match="fewer than two features"):
        sccs.compute_recurrence((left, right), min_similarity=0.5, random_state=0)


def test_recurrence_result_carries_the_overlap_diagnostic() -> None:
    base = base_programs()
    sample_sets = tuple(program_set(f"d{i}", np.roll(base, i, axis=0)) for i in range(3))

    recurrence = sccs.compute_recurrence(
        sample_sets, min_similarity=0.5, n_permutations=20, random_state=0
    )

    assert recurrence.overlap is not None
    assert recurrence.overlap.n_shared == len(recurrence.common_features)
    assert recurrence.overlap.retention_by_sample == (1.0, 1.0, 1.0)


def test_recurrence_consumes_only_persisted_program_results(tmp_path) -> None:
    """A loaded ProgramSet must not change the recurrence it takes part in.

    Persisting is only useful if the stored weights reproduce the computation
    exactly, so this compares the null scores as well as the weights: the null
    scores pin the order the seeded generator is consumed in, which a comparison
    of the vocabulary alone would not catch.
    """
    base = base_programs()
    sample_sets = tuple(
        program_set(f"d{index}", np.roll(base, index, axis=0)) for index in range(3)
    )
    for program in sample_sets:
        sccs.save_program_result(
            sccs.ProgramResult(
                programs=program.weights,
                usages=np.ones((program.n_cells, program.n_programs)),
                cell_names=tuple(f"cell_{index}" for index in range(program.n_cells)),
                feature_names=program.feature_names,
                sample_id=program.sample_id,
                selected_K=program.n_programs,
                stability=None,
                provenance={
                    "modality": "rna",
                    "method": "nmf",
                    "preprocessing": "identity",
                    "n_repeats": 1,
                    "selected_K": program.n_programs,
                },
            ),
            tmp_path,
        )
    restored = tuple(
        sccs.load_program_result(path).to_program_set() for path in sorted(tmp_path.glob("*.h5ad"))
    )
    options = {"min_similarity": 0.3, "n_permutations": 20, "random_state": 5}

    in_memory = sccs.build_recurrent_vocabulary(sample_sets, min_samples=3, **options)
    persisted = sccs.build_recurrent_vocabulary(restored, min_samples=3, **options)

    np.testing.assert_array_equal(in_memory.vocabulary.weights, persisted.vocabulary.weights)
    assert in_memory.reference_sample_id == persisted.reference_sample_id
    for first, second in zip(in_memory.pairwise_matches, persisted.pairwise_matches, strict=True):
        np.testing.assert_array_equal(first.similarities, second.similarities)
        np.testing.assert_array_equal(first.null_scores, second.null_scores)
        assert first.p_value == second.p_value
