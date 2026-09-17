import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import sccellstates as sccs


def _sample(seed: int) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    values = rng.poisson(3, size=(12, 8)).astype(float)
    return ad.AnnData(
        X=sparse.csr_matrix(values),
        obs=pd.DataFrame(index=[f"cell_{seed}_{i}" for i in range(12)]),
        var=pd.DataFrame(index=[f"G{i}" for i in range(8)]),
    )


def _block_atlas(*, donor_b_scale: float = 1.0) -> ad.AnnData:
    """Two donors sharing a strong block and each owning a private block.

    ``g0:g4`` varies strongly in both donors, ``g4:g8`` varies only in donor_a,
    and ``g8:g12`` varies only in donor_b. Per-sample gene selection therefore
    gives each donor a different feature axis while the shared block keeps the
    donors comparable.
    """
    n_cells = 12
    shared = np.tile([1.0, 40.0], 6)
    moderate = np.tile([2.0, 4.0, 6.0, 8.0, 10.0, 2.0], 2)
    blocks = {
        "donor_a": {slice(0, 4): shared, slice(4, 8): moderate},
        "donor_b": {slice(0, 4): shared, slice(8, 12): moderate * donor_b_scale},
    }
    rows = []
    labels = []
    for donor, columns in blocks.items():
        values = np.full((n_cells, 12), 5.0)
        for index, column_values in columns.items():
            values[:, index] = column_values[:, None]
        rows.append(values)
        labels.extend([donor] * n_cells)
    return ad.AnnData(
        X=sparse.csr_matrix(np.vstack(rows)),
        obs=pd.DataFrame(
            {"donor_id": labels},
            index=[f"cell_{index}" for index in range(2 * n_cells)],
        ),
        var=pd.DataFrame(index=[f"g{index}" for index in range(12)]),
    )


# --- single-sample discovery -------------------------------------------------


def test_fit_is_single_sample_and_returns_programs_and_usages() -> None:
    result = sccs.fit(
        _sample(1),
        modality="rna",
        method="cnmf",
        sample_id="donor_01",
        n_programs=3,
        n_repeats=1,
        random_state=42,
    )

    assert result.sample_id == "donor_01"
    assert result.programs.shape == (3, 8)
    assert result.loadings is result.programs
    assert result.usages.shape == (12, 3)
    assert result.stability is None
    assert result.provenance["workflow"] == "single_sample_program_discovery"


def test_fit_rejects_a_sample_key_spanning_several_samples() -> None:
    atlas = _block_atlas()

    with pytest.raises(sccs.APIError, match="fit_atlas"):
        sccs.fit(atlas, sample_key="donor_id")

    with pytest.raises(sccs.APIError, match="fit_samples"):
        sccs.fit(atlas, sample_key="donor_id")


def test_fit_accepts_a_sample_key_that_names_one_sample() -> None:
    atlas = _block_atlas()
    single = atlas[atlas.obs["donor_id"] == "donor_a"].copy()

    result = sccs.fit(
        single,
        sample_key="donor_id",
        sample_id="only",
        n_programs=2,
        n_repeats=1,
        preprocessing="identity",
        random_state=0,
    )

    assert result.sample_id == "only"


def test_fit_rejects_an_unknown_sample_key() -> None:
    with pytest.raises(sccs.APIError, match="not present"):
        sccs.fit(_sample(1), sample_key="absent")


# --- atlas and multi-file discovery ------------------------------------------


def test_fit_atlas_fits_each_sample_on_its_own_feature_axis() -> None:
    """Per-sample gene selection is a logical consequence of per-sample fitting.

    A pooled implementation makes one selection for the whole container, so
    both donors would share a feature axis. Independent fitting gives each
    donor the genes that vary within that donor alone.
    """
    result = sccs.fit_atlas(
        _block_atlas(),
        sample_key="donor_id",
        n_top_genes=8,
        n_programs=2,
        n_repeats=1,
        preprocessing="identity",
        random_state=0,
    )

    donor_a, donor_b = result.programs.samples
    assert result.programs.sample_ids == ("donor_a", "donor_b")
    assert set(donor_a.feature_names) == {f"g{index}" for index in range(8)}
    assert set(donor_b.feature_names) == {f"g{index}" for index in (0, 1, 2, 3, 8, 9, 10, 11)}
    assert set(donor_a.feature_names) != set(donor_b.feature_names)


def test_fit_atlas_never_passes_a_pooled_matrix_to_the_estimator(monkeypatch) -> None:
    observed: list[tuple[int, int]] = []
    original = sccs.NMFProgramEstimator.fit

    def spy(self, matrix, *, feature_names, sample_id):
        observed.append(matrix.shape)
        return original(self, matrix, feature_names=feature_names, sample_id=sample_id)

    monkeypatch.setattr(sccs.NMFProgramEstimator, "fit", spy)
    atlas = _block_atlas()

    sccs.fit_atlas(
        atlas,
        sample_key="donor_id",
        n_programs=2,
        n_repeats=1,
        preprocessing="identity",
        random_state=0,
    )

    assert observed == [(12, 12), (12, 12)]
    assert all(rows < atlas.n_obs for rows, _ in observed)


def test_fit_atlas_and_per_sample_files_agree() -> None:
    """Storage layout must not change the programs."""
    atlas = _block_atlas()
    from_atlas = sccs.fit_atlas(
        atlas,
        sample_key="donor_id",
        n_programs=2,
        n_repeats=1,
        preprocessing="identity",
        random_state=0,
    )
    from_files = sccs.fit_samples(
        {
            "donor_a": atlas[atlas.obs["donor_id"] == "donor_a"].copy(),
            "donor_b": atlas[atlas.obs["donor_id"] == "donor_b"].copy(),
        },
        n_programs=2,
        n_repeats=1,
        preprocessing="identity",
        random_state=0,
    )

    for left, right in zip(from_atlas.programs, from_files.programs, strict=True):
        assert left.sample_id == right.sample_id
        np.testing.assert_array_equal(left.programs, right.programs)


def test_perturbing_one_sample_leaves_the_others_programs_unchanged() -> None:
    kwargs = {
        "sample_key": "donor_id",
        "n_programs": 2,
        "n_repeats": 1,
        "preprocessing": "identity",
        "random_state": 0,
    }
    first = sccs.fit_atlas(_block_atlas(), **kwargs)
    second = sccs.fit_atlas(_block_atlas(donor_b_scale=2.0), **kwargs)

    untouched, perturbed = first.programs.samples
    untouched_after, perturbed_after = second.programs.samples
    assert untouched.sample_id == untouched_after.sample_id == "donor_a"
    np.testing.assert_array_equal(untouched.programs, untouched_after.programs)
    assert not np.array_equal(perturbed.programs, perturbed_after.programs)


def test_fit_samples_reads_files_and_names_them_from_stems(tmp_path) -> None:
    atlas = _block_atlas()
    paths = []
    for donor in ("donor_a", "donor_b"):
        path = tmp_path / f"{donor}.h5ad"
        atlas[atlas.obs["donor_id"] == donor].copy().write_h5ad(path)
        paths.append(path)

    result = sccs.fit_samples(
        paths,
        n_programs=2,
        n_repeats=1,
        preprocessing="identity",
        random_state=0,
    )

    assert result.programs.sample_ids == ("donor_a", "donor_b")
    assert result.provenance["storage"] == "files"
    assert result.provenance["pooled_fit"] is False
    assert result.provenance["fit_independence"] == "per_sample"


def test_fit_samples_reads_several_samples_from_one_container(tmp_path) -> None:
    path = tmp_path / "cohort.h5ad"
    _block_atlas().write_h5ad(path)

    result = sccs.fit_samples(
        [path],
        sample_key="donor_id",
        n_programs=2,
        n_repeats=1,
        preprocessing="identity",
        random_state=0,
    )

    assert result.programs.sample_ids == ("donor_a", "donor_b")
    assert result.provenance["storage"] == "files"
    assert result.provenance["sample_key"] == "donor_id"


def test_fit_samples_rejects_ambiguous_inputs() -> None:
    atlas = _block_atlas()
    with pytest.raises(sccs.APIError, match="must be unique"):
        sccs.fit_samples(
            [_sample(1), _sample(2)],
            n_programs=2,
            n_repeats=1,
            preprocessing="identity",
        )
    with pytest.raises(sccs.APIError, match="sample_key cannot be combined"):
        sccs.fit_samples(
            {"a": atlas, "b": atlas},
            sample_key="donor_id",
            n_programs=2,
            n_repeats=1,
            preprocessing="identity",
        )
    with pytest.raises(sccs.APIError, match="remove sample_id"):
        sccs.fit_samples(
            {"a": atlas, "b": atlas},
            sample_id="nope",
            n_programs=2,
            n_repeats=1,
            preprocessing="identity",
        )


def test_fit_samples_requires_at_least_two_samples() -> None:
    with pytest.raises(sccs.APIError, match="at least two biological samples"):
        sccs.fit_samples([_sample(1)], n_programs=2, n_repeats=1, preprocessing="identity")


def test_cohort_workflows_are_deterministic() -> None:
    kwargs = {
        "sample_key": "donor_id",
        "n_programs": 2,
        "n_repeats": 1,
        "preprocessing": "identity",
        "random_state": 3,
    }
    first = sccs.fit_atlas(_block_atlas(), **kwargs)
    second = sccs.fit_atlas(_block_atlas(), **kwargs)

    assert first.recurrence.reference_sample_id == second.recurrence.reference_sample_id
    if first.vocabulary is None or second.vocabulary is None:
        assert first.vocabulary is second.vocabulary is None
    else:
        np.testing.assert_array_equal(
            first.vocabulary.programs.weights, second.vocabulary.programs.weights
        )


# --- recurrence and projection -----------------------------------------------


def test_recurrence_and_projection_are_explicit_higher_level_operations() -> None:
    results = sccs.ProgramCollection(
        tuple(
            sccs.fit(
                source,
                sample_id=sample_id,
                n_programs=2,
                n_repeats=1,
                preprocessing="identity",
                random_state=7,
            )
            for sample_id, source in (("a", _sample(1)), ("b", _sample(1)))
        )
    )
    vocabulary = sccs.find_recurrent_programs(
        results, min_samples=2, n_permutations=5, random_state=7
    )
    projected = sccs.project(_sample(3), vocabulary)

    assert projected.usages.shape[1] == vocabulary.vocabulary.n_programs
    assert projected.usages.shape[0] == 12
    assert projected.vocabulary is vocabulary.vocabulary


def test_project_accepts_a_vocabulary_or_a_cohort_result() -> None:
    cohort = sccs.fit_atlas(
        _block_atlas(),
        sample_key="donor_id",
        n_programs=2,
        n_repeats=1,
        preprocessing="identity",
        random_state=0,
    )
    atlas = _block_atlas()
    if cohort.vocabulary is None:
        with pytest.raises(sccs.APIError, match="no vocabulary"):
            sccs.project(atlas, cohort)
    else:
        direct = sccs.project(atlas, cohort.vocabulary)
        via_cohort = sccs.project(atlas, cohort)
        np.testing.assert_array_equal(direct.usages, via_cohort.usages)
        assert direct.usages.shape[1] == cohort.vocabulary.n_programs


def test_project_rejects_an_unknown_vocabulary_type() -> None:
    with pytest.raises(sccs.APIError, match="vocabulary must be"):
        sccs.project(_sample(1), "not a vocabulary")


def _disjoint_atlas() -> ad.AnnData:
    """Two donors whose informative genes do not overlap at all."""
    n_cells = 10
    varying = np.tile([1.0, 40.0], n_cells // 2)
    blocks = {"donor_a": (0, 3), "donor_b": (3, 6)}
    rows = []
    labels = []
    for donor, (low, high) in blocks.items():
        values = np.full((n_cells, 6), 5.0)
        values[:, low:high] = varying[:, None]
        rows.append(values)
        labels.extend([donor] * n_cells)
    return ad.AnnData(
        X=sparse.csr_matrix(np.vstack(rows)),
        obs=pd.DataFrame(
            {"donor_id": labels}, index=[f"cell_{index}" for index in range(2 * n_cells)]
        ),
        var=pd.DataFrame(index=[f"g{index}" for index in range(6)]),
    )


def test_fit_atlas_reports_insufficient_feature_overlap() -> None:
    """Disjoint per-sample gene selection is reported, never silently repaired."""
    with pytest.raises(sccs.APIError) as error:
        sccs.fit_atlas(
            _disjoint_atlas(),
            sample_key="donor_id",
            n_top_genes=3,
            n_programs=2,
            n_repeats=1,
            preprocessing="identity",
            random_state=0,
        )

    message = str(error.value)
    assert "share 0 feature" in message
    assert "donor_a=3" in message and "donor_b=3" in message
    assert "n_top_genes" in message


def test_cohort_provenance_records_feature_overlap() -> None:
    result = sccs.fit_atlas(
        _block_atlas(),
        sample_key="donor_id",
        n_top_genes=8,
        n_programs=2,
        n_repeats=1,
        preprocessing="identity",
        random_state=0,
    )

    overlap = result.provenance["feature_overlap"]
    assert overlap["n_shared_features"] == 4
    assert overlap["n_features_by_sample"] == {"donor_a": 8, "donor_b": 8}
    assert overlap["retention_by_sample"] == {"donor_a": 0.5, "donor_b": 0.5}
    assert overlap["minimum_pairwise_shared"] == 4
    assert result.recurrence.overlap.n_shared == 4


def test_fit_atlas_programs_do_not_depend_on_sample_names() -> None:
    atlas = _block_atlas()
    renamed = atlas.copy()
    # Renamed so that sort order is preserved; otherwise this would compare
    # donor_a against donor_b rather than each donor against itself.
    renamed.obs["donor_id"] = renamed.obs["donor_id"].map(
        {"donor_a": "donor_aa", "donor_b": "donor_bb"}
    )
    options = {
        "sample_key": "donor_id",
        "n_programs": 2,
        "n_repeats": 1,
        "preprocessing": "identity",
        "random_state": 0,
    }

    first = sccs.fit_atlas(atlas, **options)
    second = sccs.fit_atlas(renamed, **options)

    for left, right in zip(first.programs, second.programs, strict=True):
        np.testing.assert_array_equal(left.programs, right.programs)


# --- distributed aggregation -------------------------------------------------


def _aggregate_options() -> dict[str, object]:
    """The reduction settings every cohort entry point must share."""
    return {
        "min_samples": 2,
        "min_similarity": 0.3,
        "n_permutations": 25,
        "max_redundancy": None,
        "random_state": 0,
    }


def _saved_donors(atlas: ad.AnnData, directory, **fit_kwargs):
    """Fit each donor on its own and freeze the result, as separate jobs would."""
    results = []
    for donor in sorted(atlas.obs["donor_id"].unique()):
        subset = atlas[atlas.obs["donor_id"] == donor].copy()
        result = sccs.fit(subset, sample_id=donor, **fit_kwargs)
        sccs.save_program_result(result, directory)
        results.append(result)
    return results


def test_fit_atlas_equals_fit_then_save_then_aggregate(tmp_path) -> None:
    """Distributing samples must be the same computation as fitting them together."""
    atlas = _block_atlas()
    options = _aggregate_options()
    fit_kwargs = {"n_programs": 2, "n_repeats": 2, "preprocessing": "identity"}

    from_atlas = sccs.fit_atlas(atlas, sample_key="donor_id", **options, **fit_kwargs)
    _saved_donors(atlas, tmp_path / "results", **fit_kwargs)
    aggregated = sccs.aggregate(tmp_path / "results", **options)

    assert from_atlas.vocabulary is not None
    assert aggregated.vocabulary is not None
    np.testing.assert_array_equal(
        from_atlas.vocabulary.programs.weights, aggregated.vocabulary.programs.weights
    )
    assert (
        from_atlas.vocabulary.programs.feature_names
        == aggregated.vocabulary.programs.feature_names
    )
    assert from_atlas.vocabulary.support_counts == aggregated.vocabulary.support_counts
    assert [
        (member.sample_ids, member.program_indices, member.similarities_to_anchor)
        for member in from_atlas.vocabulary.members
    ] == [
        (member.sample_ids, member.program_indices, member.similarities_to_anchor)
        for member in aggregated.vocabulary.members
    ]
    assert (
        from_atlas.vocabulary.dropped_anchor_indices
        == aggregated.vocabulary.dropped_anchor_indices
    )

    left, right = from_atlas.recurrence, aggregated.recurrence
    assert left.reference_sample_id == right.reference_sample_id
    assert left.sample_ids == right.sample_ids
    assert left.common_features == right.common_features
    assert left.overlap.as_record() == right.overlap.as_record()
    assert left.assignments == right.assignments
    assert [
        (summary.sample_id, summary.max_absolute_similarity)
        for summary in left.sample_redundancy
    ] == [
        (summary.sample_id, summary.max_absolute_similarity)
        for summary in right.sample_redundancy
    ]
    # Null scores pin the order the seeded generator is consumed in, which
    # comparing weights alone would not catch.
    for first, second in zip(left.pairwise_matches, right.pairwise_matches, strict=True):
        assert first.reference_sample_id == second.reference_sample_id
        assert first.query_sample_id == second.query_sample_id
        np.testing.assert_array_equal(first.similarities, second.similarities)
        np.testing.assert_array_equal(first.null_scores, second.null_scores)
        assert first.p_value == second.p_value

    assert from_atlas.programs.sample_ids == aggregated.programs.sample_ids
    for first, second in zip(from_atlas.programs, aggregated.programs, strict=True):
        assert first.sample_id == second.sample_id
        assert first.feature_names == second.feature_names
        assert first.cell_names == second.cell_names
        assert first.selected_K == second.selected_K
        np.testing.assert_array_equal(first.programs, second.programs)
        np.testing.assert_array_equal(first.usages, second.usages)
        # The stability diagnostics survive the artifact too.
        assert first.stability is not None and second.stability is not None
        np.testing.assert_array_equal(
            first.stability.matched_similarities, second.stability.matched_similarities
        )

    # Provenance agrees on every scientific setting and differs only where the
    # workflow genuinely differs: how the samples reached the reduction.
    for key in ("min_samples", "min_similarity", "n_permutations", "max_redundancy"):
        assert from_atlas.provenance[key] == aggregated.provenance[key]
    assert from_atlas.provenance["random_state"] == aggregated.provenance["random_state"]
    assert from_atlas.provenance["pooled_fit"] is False
    assert aggregated.provenance["pooled_fit"] is False
    assert from_atlas.provenance["fit_independence"] == "per_sample"
    assert aggregated.provenance["fit_independence"] == "per_sample"
    assert from_atlas.provenance["workflow"] == "cohort_program_discovery"
    assert aggregated.provenance["workflow"] == "cohort_program_aggregation"
    assert "storage" not in aggregated.provenance or (
        aggregated.provenance["storage"] == "program_results"
    )


def test_aggregate_names_samples_from_artifact_metadata_not_file_names(tmp_path) -> None:
    """A result carries its own sample ID, so file names cannot rename it."""
    atlas = _block_atlas()
    directory = tmp_path / "results"
    results = _saved_donors(
        atlas, directory, n_programs=2, n_repeats=1, preprocessing="identity"
    )
    for result, scrambled in zip(results, ["zzz.h5ad", "aaa.h5ad"], strict=True):
        (directory / f"{result.sample_id}.h5ad").rename(directory / scrambled)

    aggregated = sccs.aggregate(directory, **_aggregate_options())

    assert aggregated.programs.sample_ids == ("donor_a", "donor_b")


def test_aggregate_accepts_a_directory_a_sequence_a_mapping_and_loaded_results(
    tmp_path,
) -> None:
    atlas = _block_atlas()
    directory = tmp_path / "results"
    results = _saved_donors(
        atlas, directory, n_programs=2, n_repeats=1, preprocessing="identity"
    )
    options = _aggregate_options()
    paths = sorted(directory.glob("*.h5ad"))

    from_directory = sccs.aggregate(directory, **options)
    from_sequence = sccs.aggregate(paths, **options)
    named = {result.sample_id: path for result, path in zip(results, paths, strict=True)}
    from_mapping = sccs.aggregate(named, **options)
    from_objects = sccs.aggregate(results, **options)

    assert from_directory.vocabulary is not None
    for other in (from_sequence, from_mapping, from_objects):
        assert other.vocabulary is not None
        np.testing.assert_array_equal(
            from_directory.vocabulary.programs.weights, other.vocabulary.programs.weights
        )
        assert (
            from_directory.recurrence.reference_sample_id
            == other.recurrence.reference_sample_id
        )


def test_aggregate_is_deterministic(tmp_path) -> None:
    atlas = _block_atlas()
    directory = tmp_path / "results"
    _saved_donors(atlas, directory, n_programs=2, n_repeats=1, preprocessing="identity")

    first = sccs.aggregate(directory, **_aggregate_options())
    second = sccs.aggregate(directory, **_aggregate_options())

    assert first.vocabulary is not None and second.vocabulary is not None
    np.testing.assert_array_equal(
        first.vocabulary.programs.weights, second.vocabulary.programs.weights
    )
    for left, right in zip(
        first.recurrence.pairwise_matches, second.recurrence.pairwise_matches, strict=True
    ):
        np.testing.assert_array_equal(left.null_scores, right.null_scores)


def test_aggregate_rejects_duplicate_sample_ids(tmp_path) -> None:
    atlas = _block_atlas()
    directory = tmp_path / "results"
    results = _saved_donors(
        atlas, directory, n_programs=2, n_repeats=1, preprocessing="identity"
    )
    # Same artifact under a second name is still the same sample.
    (directory / f"{results[0].sample_id}.h5ad").rename(directory / "copy.h5ad")
    sccs.save_program_result(results[0], directory / "again.h5ad")

    with pytest.raises(sccs.APIError, match="sample IDs must be unique"):
        sccs.aggregate(directory, **_aggregate_options())


def _incompatible(directory, results, field: str, value: object) -> None:
    """Rewrite one result so it disagrees on a single fit setting."""
    from pathlib import Path

    target = next(Path(directory).glob(f"{results[0].sample_id}.h5ad"))
    adata = ad.read_h5ad(target)
    adata.uns["sccellstates"]["program_result"]["provenance"][field] = value
    adata.write_h5ad(target)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("modality", "atac", "same modality"),
        ("method", "pca", "same method"),
        ("preprocessing", "identity", "same preprocessing"),
        ("n_repeats", 9, "same n_repeats"),
    ],
)
def test_aggregate_rejects_incompatible_results(
    tmp_path, key: str, value: object, message: str
) -> None:
    atlas = _block_atlas()
    directory = tmp_path / "results"
    fit_kwargs = {"n_programs": 2, "n_repeats": 1, "preprocessing": "library_size_log1p"}
    results = _saved_donors(atlas, directory, **fit_kwargs)
    _incompatible(directory, results, key, value)

    with pytest.raises(sccs.APIError, match=message):
        sccs.aggregate(directory, **_aggregate_options())


def test_aggregate_rejects_a_mapping_key_that_disagrees_with_the_artifact(tmp_path) -> None:
    atlas = _block_atlas()
    directory = tmp_path / "results"
    results = _saved_donors(
        atlas, directory, n_programs=2, n_repeats=1, preprocessing="identity"
    )

    with pytest.raises(sccs.APIError, match="does not match the sample ID"):
        sccs.aggregate({"not_the_sample": results[0]}, **_aggregate_options())


def test_aggregate_reports_insufficient_feature_overlap(tmp_path) -> None:
    """Disjoint per-sample gene selection is reported, never silently repaired."""
    atlas = _disjoint_atlas()
    directory = tmp_path / "results"
    _saved_donors(
        atlas,
        directory,
        n_top_genes=3,
        n_programs=2,
        n_repeats=1,
        preprocessing="identity",
    )

    with pytest.raises(sccs.APIError, match="share 0 feature"):
        sccs.aggregate(directory, **_aggregate_options())


def test_aggregate_requires_at_least_two_results(tmp_path) -> None:
    atlas = _block_atlas()
    directory = tmp_path / "results"
    results = _saved_donors(
        atlas, directory, n_programs=2, n_repeats=1, preprocessing="identity"
    )

    with pytest.raises(sccs.APIError, match="at least two biological samples"):
        sccs.aggregate(results[0], **_aggregate_options())


def test_aggregate_rejects_an_empty_directory(tmp_path) -> None:
    empty = tmp_path / "results"
    empty.mkdir()

    with pytest.raises(sccs.APIError, match="no .h5ad program results found"):
        sccs.aggregate(empty)


def test_aggregate_rejects_a_directory_inside_a_sequence(tmp_path) -> None:
    directory = tmp_path / "results"
    directory.mkdir()

    with pytest.raises(sccs.APIError, match="on its own"):
        sccs.aggregate([directory])


def test_aggregate_rejects_a_missing_path(tmp_path) -> None:
    with pytest.raises(sccs.APIError, match="is neither"):
        sccs.aggregate(tmp_path / "absent")


def test_aggregate_rejects_an_unknown_input_type() -> None:
    with pytest.raises(TypeError, match="results must be"):
        sccs.aggregate(42)


def test_aggregate_rejects_an_empty_sequence() -> None:
    with pytest.raises(sccs.APIError, match="at least one program result"):
        sccs.aggregate([])


def test_aggregate_and_the_cohort_entry_points_share_recurrence_defaults() -> None:
    """The shared defaults are the equivalence contract, so they must not drift."""
    import inspect

    shared = ("min_samples", "min_similarity", "n_permutations", "max_redundancy", "random_state")
    for name in shared:
        defaults = {
            entry.__name__: inspect.signature(entry).parameters[name].default
            for entry in (sccs.aggregate, sccs.fit_atlas, sccs.fit_samples)
        }
        assert len(set(defaults.values())) == 1, f"{name} defaults disagree: {defaults}"
