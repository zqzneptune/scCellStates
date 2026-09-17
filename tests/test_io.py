import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import sccellstates as sccs


def example_adata(*, sparse_x: bool = True) -> ad.AnnData:
    values = np.array([[1, 0, 2], [0, 3, 1], [2, 1, 0], [4, 0, 1]], dtype=np.float32)
    matrix = sparse.csr_matrix(values) if sparse_x else values
    adata = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(
            {"donor_id": ["d1", "d1", "d2", "d2"]},
            index=["c1", "c2", "c3", "c4"],
        ),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )
    adata.layers["counts"] = matrix.copy()
    return adata


@pytest.mark.parametrize("sparse_x", [False, True])
def test_validate_dense_and_sparse_anndata(sparse_x: bool) -> None:
    adata = example_adata(sparse_x=sparse_x)
    summary = sccs.validate_anndata(adata, sample_key="donor_id", layer="counts", min_samples=2)
    assert summary.n_obs == 4
    assert summary.n_vars == 3
    assert summary.n_samples == 2
    assert summary.is_sparse is sparse_x


def test_get_matrix_preserves_sparse_object() -> None:
    adata = example_adata()
    assert sccs.get_matrix(adata, layer="counts") is adata.layers["counts"]


def test_missing_sample_metadata_is_rejected() -> None:
    adata = example_adata()
    with pytest.raises(sccs.InputError, match="sample key"):
        sccs.validate_anndata(adata, sample_key="sample_id")


def test_missing_sample_value_is_rejected() -> None:
    adata = example_adata()
    adata.obs.loc["c1", "donor_id"] = None
    with pytest.raises(sccs.InputError, match="missing sample"):
        sccs.validate_anndata(adata, sample_key="donor_id")


def test_duplicate_gene_names_are_rejected() -> None:
    adata = example_adata()
    adata.var_names = ["g1", "g1", "g3"]
    with pytest.raises(sccs.InputError, match="var_names"):
        sccs.validate_anndata(adata, sample_key="donor_id")


def test_layer_selection_is_explicit() -> None:
    adata = example_adata()
    with pytest.raises(sccs.InputError, match="not present"):
        sccs.get_matrix(adata, layer="normalized")


def test_negative_values_are_rejected() -> None:
    adata = example_adata(sparse_x=False)
    adata.X[0, 0] = -1
    with pytest.raises(sccs.InputError, match="negative"):
        sccs.validate_anndata(adata, sample_key="donor_id")


def _frozen_program_sets() -> tuple[sccs.ProgramSet, ...]:
    base = np.array(
        [
            [9, 8, 7, 1, 2, 3, 1, 1, 1],
            [1, 2, 1, 9, 8, 7, 1, 3, 2],
            [2, 1, 3, 1, 1, 2, 9, 8, 7],
        ],
        dtype=float,
    )
    features = tuple(f"g{index}" for index in range(9))
    return tuple(
        sccs.ProgramSet(
            sample_id=f"d{index}",
            feature_names=features,
            weights=np.roll(base, index, axis=0),
            n_cells=50,
            estimator="synthetic",
        )
        for index in range(3)
    )


def _frozen_inputs() -> tuple[sccs.ProgramVocabulary, object]:
    program_sets = _frozen_program_sets()
    recurrence = sccs.compute_recurrence(
        program_sets, min_similarity=0.9, n_permutations=20, random_state=3
    )
    vocabulary = sccs.build_vocabulary(recurrence, min_samples=3)
    assert vocabulary is not None
    return vocabulary, recurrence


def test_frozen_vocabulary_round_trips(tmp_path) -> None:
    vocabulary, recurrence = _frozen_inputs()
    destination = sccs.save_program_vocabulary(
        vocabulary, tmp_path / "vocab", recurrence=recurrence
    )

    assert destination.is_dir()
    restored = sccs.load_program_vocabulary(destination)

    np.testing.assert_array_equal(restored.programs.weights, vocabulary.programs.weights)
    assert restored.programs.feature_names == vocabulary.programs.feature_names
    assert restored.programs.sample_id == vocabulary.programs.sample_id
    assert restored.programs.n_cells == vocabulary.programs.n_cells
    assert restored.training_sample_ids == vocabulary.training_sample_ids
    assert restored.reference_sample_id == vocabulary.reference_sample_id
    assert restored.min_samples == vocabulary.min_samples
    assert restored.min_similarity == vocabulary.min_similarity
    assert restored.max_redundancy == vocabulary.max_redundancy
    assert restored.dropped_anchor_indices == vocabulary.dropped_anchor_indices
    assert restored.support_counts == vocabulary.support_counts
    assert [
        (member.sample_ids, member.program_indices, member.similarities_to_anchor)
        for member in restored.members
    ] == [
        (member.sample_ids, member.program_indices, member.similarities_to_anchor)
        for member in vocabulary.members
    ]
    np.testing.assert_allclose(
        restored.vocabulary_redundancy.max_absolute_similarity,
        vocabulary.vocabulary_redundancy.max_absolute_similarity,
    )


def test_frozen_vocabulary_keeps_links_to_member_feature_axes(tmp_path) -> None:
    """The consensus axis and the per-sample axes are both recorded."""
    vocabulary, recurrence = _frozen_inputs()
    destination = sccs.save_program_vocabulary(
        vocabulary, tmp_path / "vocab", recurrence=recurrence
    )

    feature_map = json.loads((destination / "feature_map.json").read_text())
    assert feature_map["consensus_features"] == list(vocabulary.programs.feature_names)
    assert set(feature_map["sample_features"]) == set(vocabulary.training_sample_ids)
    assert feature_map["sample_features"]["d0"] == list(
        recurrence.program_sets[0].feature_names
    )
    # A consensus feature's position in each member axis is derivable, not stored.
    for sample_id, axis in feature_map["sample_features"].items():
        indices = [axis.index(name) for name in feature_map["consensus_features"]]
        assert all(0 <= index < len(axis) for index in indices), sample_id


def test_frozen_vocabulary_records_recurrence_evidence(tmp_path) -> None:
    vocabulary, recurrence = _frozen_inputs()
    destination = sccs.save_program_vocabulary(
        vocabulary, tmp_path / "vocab", recurrence=recurrence
    )

    evidence = sccs.load_recurrence_evidence(destination)
    assert len(evidence["pairs"]) == len(recurrence.pairwise_matches)
    first, original = evidence["pairs"][0], recurrence.pairwise_matches[0]
    assert first["reference_sample_id"] == original.reference_sample_id
    assert first["query_sample_id"] == original.query_sample_id
    assert first["similarities"] == pytest.approx(list(original.similarities))
    assert first["null_scores"] == pytest.approx(list(original.null_scores))
    assert first["p_value"] == pytest.approx(original.p_value)


def test_frozen_vocabulary_without_recurrence_still_loads(tmp_path) -> None:
    vocabulary, _ = _frozen_inputs()
    destination = sccs.save_program_vocabulary(vocabulary, tmp_path / "vocab")

    restored = sccs.load_program_vocabulary(destination)
    np.testing.assert_array_equal(restored.programs.weights, vocabulary.programs.weights)
    assert sccs.load_recurrence_evidence(destination) == {"pairs": []}


def test_frozen_vocabulary_is_byte_identical_for_identical_inputs(tmp_path) -> None:
    """A frozen artifact is reproducible, so no timestamp is recorded."""
    vocabulary, recurrence = _frozen_inputs()
    first = sccs.save_program_vocabulary(vocabulary, tmp_path / "a", recurrence=recurrence)
    second = sccs.save_program_vocabulary(vocabulary, tmp_path / "b", recurrence=recurrence)

    for name in sorted(path.name for path in first.iterdir()):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_saving_a_vocabulary_refuses_to_clobber(tmp_path) -> None:
    vocabulary, _ = _frozen_inputs()
    sccs.save_program_vocabulary(vocabulary, tmp_path / "vocab")

    with pytest.raises(sccs.InputError, match="already exists"):
        sccs.save_program_vocabulary(vocabulary, tmp_path / "vocab")
    sccs.save_program_vocabulary(vocabulary, tmp_path / "vocab", overwrite=True)


def test_saving_rejects_recurrence_from_other_samples(tmp_path) -> None:
    vocabulary, recurrence = _frozen_inputs()
    renamed = sccs.compute_recurrence(
        tuple(
            sccs.ProgramSet(
                sample_id=f"{program_set.sample_id}_x",
                feature_names=program_set.feature_names,
                weights=program_set.weights,
                n_cells=program_set.n_cells,
                estimator=program_set.estimator,
            )
            for program_set in recurrence.program_sets
        ),
        min_similarity=0.9,
        n_permutations=5,
        random_state=0,
    )

    with pytest.raises(sccs.InputError, match="different samples"):
        sccs.save_program_vocabulary(vocabulary, tmp_path / "vocab", recurrence=renamed)
    with pytest.raises(TypeError, match="ProgramVocabulary"):
        sccs.save_program_vocabulary("not a vocabulary", tmp_path / "other")


def test_loading_rejects_a_path_that_is_not_a_vocabulary(tmp_path) -> None:
    with pytest.raises(sccs.InputError, match="not a directory"):
        sccs.load_program_vocabulary(tmp_path / "absent")

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(sccs.InputError, match="metadata.json"):
        sccs.load_program_vocabulary(empty)


# --- portable per-sample program results ------------------------------------


def _program_result(*, sample_id: str = "donor_1", stability: bool = True) -> sccs.ProgramResult:
    """One sample's result on its own feature axis, with no expression matrix."""
    rng = np.random.default_rng(7)
    features = tuple(f"g{index}" for index in range(6))
    programs = rng.random((3, 6)) + 0.1
    programs /= programs.sum(axis=1, keepdims=True)
    program_sets = tuple(
        sccs.ProgramSet(
            sample_id=f"{sample_id}::run_{run}",
            feature_names=features,
            weights=programs * (1.0 + run * 0.01),
            n_cells=20,
            estimator="synthetic",
        )
        for run in range(2)
    )
    return sccs.ProgramResult(
        programs=programs,
        usages=rng.random((20, 3)),
        cell_names=tuple(f"cell_{index}" for index in range(20)),
        feature_names=features,
        sample_id=sample_id,
        selected_K=3,
        stability=sccs.stabilize_programs(program_sets, sample_id=sample_id)
        if stability
        else None,
        provenance={
            "schema_version": "1.0",
            "workflow": "single_sample_program_discovery",
            "modality": "rna",
            "method": "nmf",
            "layer": None,
            "preprocessing": "identity",
            "n_programs_requested": 3,
            "n_repeats": 2,
            "random_state": 0,
            "selected_K": 3,
        },
    )


def test_program_result_round_trips(tmp_path) -> None:
    original = _program_result()
    destination = sccs.save_program_result(original, tmp_path / "donor_1.h5ad")

    assert destination.is_file()
    restored = sccs.load_program_result(destination)

    assert restored.sample_id == original.sample_id
    assert restored.selected_K == original.selected_K
    assert restored.cell_names == original.cell_names
    assert restored.feature_names == original.feature_names
    np.testing.assert_array_equal(restored.programs, original.programs)
    np.testing.assert_array_equal(restored.usages, original.usages)
    assert sccs.to_jsonable(dict(restored.provenance)) == sccs.to_jsonable(
        dict(original.provenance)
    )


def test_program_result_stability_diagnostics_round_trip(tmp_path) -> None:
    original = _program_result()
    assert original.stability is not None
    restored = sccs.load_program_result(sccs.save_program_result(original, tmp_path / "a.h5ad"))

    assert isinstance(restored.stability, sccs.ProgramStabilityFit)
    assert restored.stability.run_ids == original.stability.run_ids
    assert restored.stability.anchor_run_id == original.stability.anchor_run_id
    assert restored.stability.min_similarity == original.stability.min_similarity
    assert restored.stability.programs.estimator == original.stability.programs.estimator
    assert dict(restored.stability.programs.parameters) == dict(
        original.stability.programs.parameters
    )
    np.testing.assert_array_equal(
        restored.stability.matched_similarities, original.stability.matched_similarities
    )
    np.testing.assert_array_equal(
        restored.stability.retained_anchor_indices,
        original.stability.retained_anchor_indices,
    )
    # The consensus weights are stored once and shared, never duplicated on disk.
    np.testing.assert_array_equal(restored.stability.programs.weights, restored.programs)


def test_program_result_without_stability_round_trips(tmp_path) -> None:
    original = _program_result(stability=False)
    restored = sccs.load_program_result(sccs.save_program_result(original, tmp_path / "a.h5ad"))

    assert restored.stability is None


def test_program_result_never_stores_the_expression_matrix(tmp_path) -> None:
    destination = sccs.save_program_result(_program_result(), tmp_path / "a.h5ad")

    backed = ad.read_h5ad(destination, backed="r")
    try:
        assert "X" not in backed.file
    finally:
        backed.file.close()
    # An artifact is an exchange unit, not an input: it holds no expression.
    with pytest.raises(sccs.InputError, match="empty"):
        sccs.get_matrix(ad.read_h5ad(destination))


def test_program_result_uses_the_scverse_locations(tmp_path) -> None:
    original = _program_result()
    destination = sccs.save_program_result(original, tmp_path / "a.h5ad")
    adata = ad.read_h5ad(destination)

    np.testing.assert_array_equal(adata.varm["sccs_programs"], original.programs.T)
    np.testing.assert_array_equal(adata.obsm["X_sccs_programs"], original.usages)
    assert tuple(map(str, adata.obs_names)) == original.cell_names
    assert tuple(map(str, adata.var_names)) == original.feature_names
    record = adata.uns["sccellstates"]["program_result"]
    assert record["artifact"] == "program_result"
    assert record["sample_id"] == original.sample_id
    assert record["selected_K"] == original.selected_K


def test_program_result_artifact_is_byte_identical_for_identical_inputs(tmp_path) -> None:
    """A stored result is reproducible, so no timestamp is recorded."""
    original = _program_result()
    first = sccs.save_program_result(original, tmp_path / "a.h5ad")
    second = sccs.save_program_result(original, tmp_path / "b.h5ad")

    assert first.read_bytes() == second.read_bytes()


def test_saving_to_a_directory_names_the_artifact_from_the_sample_id(tmp_path) -> None:
    destination = sccs.save_program_result(_program_result(sample_id="donor_9"), tmp_path)

    assert destination == tmp_path / "donor_9.h5ad"
    assert destination.is_file()


def test_saving_refuses_a_sample_id_that_cannot_name_a_file(tmp_path) -> None:
    with pytest.raises(sccs.InputError, match="explicit .h5ad path"):
        sccs.save_program_result(_program_result(sample_id="../escape"), tmp_path)


def test_saving_a_program_result_refuses_to_clobber(tmp_path) -> None:
    original = _program_result()
    sccs.save_program_result(original, tmp_path / "a.h5ad")

    with pytest.raises(sccs.InputError, match="already exists"):
        sccs.save_program_result(original, tmp_path / "a.h5ad")
    sccs.save_program_result(original, tmp_path / "a.h5ad", overwrite=True)


def test_saving_rejects_duplicate_cell_names(tmp_path) -> None:
    original = _program_result()
    duplicated = sccs.ProgramResult(
        programs=original.programs,
        usages=original.usages,
        cell_names=("cell_0",) * len(original.cell_names),
        feature_names=original.feature_names,
        sample_id=original.sample_id,
        selected_K=original.selected_K,
        stability=None,
        provenance=original.provenance,
    )

    with pytest.raises(sccs.InputError, match="unique"):
        sccs.save_program_result(duplicated, tmp_path / "a.h5ad")


def test_saving_rejects_a_program_with_no_positive_weight(tmp_path) -> None:
    original = _program_result()
    programs = np.array(original.programs, copy=True)
    programs[1] = 0.0
    empty = sccs.ProgramResult(
        programs=programs,
        usages=original.usages,
        cell_names=original.cell_names,
        feature_names=original.feature_names,
        sample_id=original.sample_id,
        selected_K=original.selected_K,
        stability=None,
        provenance=original.provenance,
    )

    with pytest.raises(sccs.InputError, match="empty programs at indices"):
        sccs.save_program_result(empty, tmp_path / "a.h5ad")


def test_saving_requires_a_method_and_modality_in_provenance(tmp_path) -> None:
    original = _program_result()
    stripped = sccs.ProgramResult(
        programs=original.programs,
        usages=original.usages,
        cell_names=original.cell_names,
        feature_names=original.feature_names,
        sample_id=original.sample_id,
        selected_K=original.selected_K,
        stability=None,
        provenance={"modality": "rna"},
    )

    with pytest.raises(sccs.InputError, match="'method'"):
        sccs.save_program_result(stripped, tmp_path / "a.h5ad")


def test_loading_rejects_an_artifact_without_program_result_metadata(tmp_path) -> None:
    plain = tmp_path / "plain.h5ad"
    example_adata().write_h5ad(plain)

    with pytest.raises(sccs.InputError, match="not a program result"):
        sccs.load_program_result(plain)


def test_loading_a_vocabulary_artifact_names_the_right_loader(tmp_path) -> None:
    """Both artifacts use .varm['sccs_programs'], so the error must disambiguate."""
    program_sets = _frozen_program_sets()
    adata = ad.AnnData(
        X=np.ones((4, program_sets[0].n_features)),
        obs=pd.DataFrame(index=[f"c{index}" for index in range(4)]),
        var=pd.DataFrame(index=list(program_sets[0].feature_names)),
    )
    fit = sccs.build_recurrent_vocabulary(
        program_sets, min_samples=3, min_similarity=0.9, n_permutations=20, random_state=3
    )
    sccs.store_program_vocabulary(adata, fit)
    destination = tmp_path / "vocabulary.h5ad"
    adata.write_h5ad(destination)

    with pytest.raises(sccs.InputError, match="load_program_vocabulary"):
        sccs.load_program_result(destination)


def test_loading_rejects_an_artifact_that_carries_a_matrix(tmp_path) -> None:
    destination = sccs.save_program_result(_program_result(), tmp_path / "a.h5ad")
    adata = ad.read_h5ad(destination)
    adata.X = np.ones((adata.n_obs, adata.n_vars))
    adata.write_h5ad(destination)

    with pytest.raises(sccs.InputError, match="expression matrix"):
        sccs.load_program_result(destination)


def test_loading_rejects_an_unsupported_schema_version(tmp_path) -> None:
    destination = sccs.save_program_result(_program_result(), tmp_path / "a.h5ad")
    adata = ad.read_h5ad(destination)
    adata.uns["sccellstates"]["program_result"]["schema_version"] = "99.0"
    adata.write_h5ad(destination)

    with pytest.raises(sccs.InputError, match="schema_version"):
        sccs.load_program_result(destination)


def test_loading_rejects_inconsistent_axes(tmp_path) -> None:
    destination = sccs.save_program_result(_program_result(), tmp_path / "a.h5ad")
    adata = ad.read_h5ad(destination)
    adata.varm["sccs_programs"] = np.zeros((adata.n_vars, 2))
    adata.write_h5ad(destination)

    with pytest.raises(sccs.InputError, match="selected_K"):
        sccs.load_program_result(destination)


def test_loading_rejects_duplicate_names(tmp_path) -> None:
    destination = sccs.save_program_result(_program_result(), tmp_path / "a.h5ad")
    adata = ad.read_h5ad(destination)
    adata.obs_names = ["cell_0"] * adata.n_obs
    adata.write_h5ad(destination)

    with pytest.raises(sccs.InputError, match="duplicate cell names"):
        sccs.load_program_result(destination)


def test_loading_rejects_a_missing_file(tmp_path) -> None:
    with pytest.raises(sccs.InputError, match="not a file"):
        sccs.load_program_result(tmp_path / "absent.h5ad")
