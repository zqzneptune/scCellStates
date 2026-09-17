import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import sccellstates as sccs


def example_adata(*, sparse_x: bool = True) -> ad.AnnData:
    values = np.array(
        [
            [3, 0, 1],
            [2, 1, 0],
            [0, 3, 1],
            [1, 2, 1],
            [4, 0, 2],
            [0, 4, 2],
        ],
        dtype=np.float64,
    )
    matrix = sparse.csr_matrix(values) if sparse_x else values
    return ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(
            {"donor": ["d1", "d1", "d2", "d2", "d3", "d3"]},
            index=[f"c{index}" for index in range(6)],
        ),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )


def vocabulary() -> sccs.ProgramSet:
    return sccs.ProgramSet(
        sample_id="consensus",
        feature_names=("g1", "g3"),
        weights=np.array([[0.75, 0.25], [0.2, 0.8]]),
        n_cells=4,
        estimator="consensus",
    )


def test_sample_split_is_exhaustive_and_cell_labels_are_sample_aware() -> None:
    adata = example_adata()
    split = sccs.make_sample_split(
        adata, sample_key="donor", validation_samples=["d2"], test_samples=["d3"]
    )
    assert split == sccs.SampleSplit(train=("d1",), validation=("d2",), test=("d3",))
    assert sccs.sample_partition_labels(adata.obs["donor"], split).tolist() == [
        "train",
        "train",
        "validation",
        "validation",
        "test",
        "test",
    ]


def test_overlapping_sample_split_is_rejected() -> None:
    with pytest.raises(sccs.StateError, match="disjoint"):
        sccs.SampleSplit(train=("d1",), validation=("d2",), test=("d2",))


@pytest.mark.parametrize("sparse_x", [False, True])
def test_program_scorer_aligns_features_and_preserves_sparse_inputs(sparse_x: bool) -> None:
    adata = example_adata(sparse_x=sparse_x)
    activities = sccs.ProgramScorer(vocabulary()).transform(adata)
    expected = np.asarray(adata.X.toarray() if sparse_x else adata.X)[:, [0, 2]] @ np.array(
        [[0.75, 0.2], [0.25, 0.8]]
    )
    np.testing.assert_allclose(activities, expected)
    if sparse_x:
        assert sparse.issparse(adata.X)


def test_program_scorer_rejects_missing_features() -> None:
    adata = example_adata()[:, ["g1", "g2"]].copy()
    with pytest.raises(sccs.StateError, match="absent"):
        sccs.ProgramScorer(vocabulary()).transform(adata)


@pytest.mark.parametrize("sparse_x", [False, True])
@pytest.mark.parametrize("mutation", ["negative", "nonfinite"])
def test_program_scorer_validates_transform_matrix(sparse_x: bool, mutation: str) -> None:
    adata = example_adata(sparse_x=sparse_x)
    if sparse_x:
        values = adata.X.toarray()
        values[0, 0] = -1 if mutation == "negative" else np.nan
        adata.X = sparse.csr_matrix(values)
    else:
        adata.X[0, 0] = -1 if mutation == "negative" else np.nan
    expected = "negative" if mutation == "negative" else "NaN"
    with pytest.raises(sccs.StateError, match=expected):
        sccs.ProgramScorer(vocabulary()).transform(adata)


@pytest.mark.parametrize("sparse_x", [False, True])
def test_gene_standardized_scorer_uses_training_statistics_only(sparse_x: bool) -> None:
    adata = example_adata(sparse_x=sparse_x)
    split = sccs.SampleSplit(train=("d1", "d2"), validation=(), test=("d3",))
    scorer = sccs.GeneStandardizedProgramScorer(vocabulary()).fit(
        adata, sample_key="donor", split=split
    )
    values = np.asarray(adata.X.toarray() if sparse_x else adata.X)[:, [0, 2]]
    means = values[:4].mean(axis=0)
    scales = values[:4].std(axis=0)
    expected = ((values - means) / scales) @ vocabulary().weights.T
    np.testing.assert_allclose(scorer.transform(adata), expected)

    changed = adata.copy()
    if sparse_x:
        changed_values = changed.X.tolil()
        changed_values[4:, :] = 10_000
        changed.X = changed_values.tocsr()
    else:
        changed.X[4:, :] = 10_000
    changed_scorer = sccs.GeneStandardizedProgramScorer(vocabulary()).fit(
        changed, sample_key="donor", split=split
    )
    np.testing.assert_allclose(scorer.means_, changed_scorer.means_)
    np.testing.assert_allclose(scorer.scales_, changed_scorer.scales_)
    if sparse_x:
        assert sparse.issparse(adata.X)


def test_gene_standardized_scorer_handles_constant_training_gene() -> None:
    adata = example_adata(sparse_x=False)
    adata.X[:4, 2] = 1.0
    split = sccs.SampleSplit(train=("d1", "d2"), validation=(), test=("d3",))
    scorer = sccs.GeneStandardizedProgramScorer(vocabulary()).fit(
        adata, sample_key="donor", split=split
    )
    assert scorer.scales_[1] == 1.0
    assert np.isfinite(scorer.transform(adata)).all()


@pytest.mark.parametrize("sparse_x", [False, True])
def test_count_residual_scorer_is_sparse_safe_and_training_fitted(sparse_x: bool) -> None:
    adata = example_adata(sparse_x=sparse_x)
    split = sccs.SampleSplit(train=("d1", "d2"), validation=(), test=("d3",))
    scorer = sccs.CountResidualProgramScorer(vocabulary()).fit(
        adata, sample_key="donor", split=split
    )
    values = np.asarray(adata.X.toarray() if sparse_x else adata.X)[:, [0, 2]]
    training = values[:4]
    rates = (training.sum(axis=0) + 1e-8) / (training.sum() + 2e-8)
    depths = values.sum(axis=1)
    weights = vocabulary().weights
    expected = (values @ (weights.T / np.sqrt(rates)[:, None])) / np.sqrt(depths[:, None])
    expected -= np.sqrt(depths[:, None]) * (weights @ np.sqrt(rates)[:, None]).T
    np.testing.assert_allclose(scorer.transform(adata), expected)
    np.testing.assert_allclose(scorer.rates_, rates)
    assert np.isfinite(scorer.transform(adata)).all()
    if sparse_x:
        assert sparse.issparse(adata.X)

    changed = adata.copy()
    if sparse_x:
        changed_values = changed.X.tolil()
        changed_values[4:, :] = 10_000
        changed.X = changed_values.tocsr()
    else:
        changed.X[4:, :] = 10_000
    changed_fit = sccs.CountResidualProgramScorer(vocabulary()).fit(
        changed, sample_key="donor", split=split
    )
    np.testing.assert_allclose(scorer.rates_, changed_fit.rates_)


def test_count_residual_scorer_handles_zero_depth_and_requires_fit() -> None:
    adata = example_adata(sparse_x=False)
    adata.X[0, [0, 2]] = 0
    split = sccs.SampleSplit(train=("d1", "d2"), validation=(), test=("d3",))
    scorer = sccs.CountResidualProgramScorer(vocabulary())
    with pytest.raises(sccs.StateError, match="fitted"):
        scorer.transform(adata)
    scorer.fit(adata, sample_key="donor", split=split)
    assert np.all(scorer.transform(adata)[0] == 0)


@pytest.mark.parametrize("sparse_x", [False, True])
def test_technical_residual_scorer_is_sparse_safe_and_training_fitted(sparse_x: bool) -> None:
    adata = example_adata(sparse_x=sparse_x)
    split = sccs.SampleSplit(train=("d1", "d2"), validation=(), test=("d3",))
    scorer = sccs.TechnicalResidualProgramScorer(vocabulary()).fit(
        adata, sample_key="donor", split=split
    )
    values = np.asarray(adata.X.toarray() if sparse_x else adata.X)[:, [0, 2]]
    training = values[:4]
    design = np.column_stack(
        (
            np.ones(training.shape[0]),
            np.log1p(training.sum(axis=1)),
            np.log1p(np.count_nonzero(training, axis=1)),
        )
    )
    coefficients = np.linalg.lstsq(
        design.T @ design,
        design.T @ np.log1p(training),
        rcond=None,
    )[0]
    full_design = np.column_stack(
        (
            np.ones(values.shape[0]),
            np.log1p(values.sum(axis=1)),
            np.log1p(np.count_nonzero(values, axis=1)),
        )
    )
    expected = (np.log1p(values) - full_design @ coefficients) @ vocabulary().weights.T
    np.testing.assert_allclose(scorer.transform(adata), expected)

    changed = adata.copy()
    if sparse_x:
        changed_values = changed.X.tolil()
        changed_values[4:, :] = 10_000
        changed.X = changed_values.tocsr()
    else:
        changed.X[4:, :] = 10_000
    changed_fit = sccs.TechnicalResidualProgramScorer(vocabulary()).fit(
        changed, sample_key="donor", split=split
    )
    np.testing.assert_allclose(scorer.coefficients_, changed_fit.coefficients_)
    if sparse_x:
        assert sparse.issparse(adata.X)


def test_technical_residual_scorer_requires_fit() -> None:
    with pytest.raises(sccs.StateError, match="fitted"):
        sccs.TechnicalResidualProgramScorer(vocabulary()).transform(example_adata())


@pytest.mark.parametrize("sparse_x", [False, True])
def test_matched_control_scorer_is_deterministic_and_training_fitted(sparse_x: bool) -> None:
    adata = example_adata(sparse_x=sparse_x)
    split = sccs.SampleSplit(train=("d1", "d2"), validation=(), test=("d3",))
    kwargs = {
        "n_top_genes": 1,
        "n_bins": 1,
        "control_size": 1,
        "random_state": 17,
    }
    first = sccs.MatchedControlProgramScorer(vocabulary(), **kwargs).fit(
        adata, sample_key="donor", split=split
    )
    second = sccs.MatchedControlProgramScorer(vocabulary(), **kwargs).fit(
        adata, sample_key="donor", split=split
    )
    assert first.control_feature_names_ == (("g3",), ("g1",))
    assert first.control_feature_names_ == second.control_feature_names_
    assert first.training_sample_ids_ == ("d1", "d2")
    values = np.asarray(adata.X.toarray() if sparse_x else adata.X)
    expected = np.column_stack((values[:, 0] - values[:, 2], values[:, 2] - values[:, 0]))
    np.testing.assert_allclose(first.transform(adata), expected)
    reordered = adata[:, ["g3", "g2", "g1"]].copy()
    np.testing.assert_allclose(first.transform(reordered), expected)

    changed = adata.copy()
    if sparse_x:
        changed_values = changed.X.tolil()
        changed_values[4:, :] = 10_000
        changed.X = changed_values.tocsr()
    else:
        changed.X[4:, :] = 10_000
    changed_fit = sccs.MatchedControlProgramScorer(vocabulary(), **kwargs).fit(
        changed, sample_key="donor", split=split
    )
    assert first.control_feature_names_ == changed_fit.control_feature_names_
    if sparse_x:
        assert sparse.issparse(adata.X)


def test_training_fitted_scorers_require_fit_and_validate_parameters() -> None:
    adata = example_adata()
    with pytest.raises(sccs.StateError, match="fitted"):
        sccs.GeneStandardizedProgramScorer(vocabulary()).transform(adata)
    with pytest.raises(sccs.StateError, match="fitted"):
        sccs.MatchedControlProgramScorer(
            vocabulary(), n_top_genes=1, n_bins=1, random_state=0
        ).transform(adata)
    with pytest.raises(ValueError, match="smaller"):
        sccs.MatchedControlProgramScorer(
            vocabulary(), n_top_genes=vocabulary().n_features, random_state=0
        )


def test_program_activity_storage_requires_explicit_overwrite() -> None:
    adata = example_adata()
    activities = sccs.ProgramScorer(vocabulary()).transform(adata)
    sccs.store_program_activities(adata, activities)
    with pytest.raises(sccs.StateError, match="already exists"):
        sccs.store_program_activities(adata, activities)


def test_expression_pca_uses_only_training_cells() -> None:
    adata = example_adata(sparse_x=False)
    split = sccs.SampleSplit(train=("d1", "d2"), validation=(), test=("d3",))
    first = sccs.ExpressionPCA(2, random_state=7).fit(adata, sample_key="donor", split=split)
    changed = adata.copy()
    changed.X[4:, :] = 10_000
    second = sccs.ExpressionPCA(2, random_state=7).fit(changed, sample_key="donor", split=split)
    np.testing.assert_allclose(first.model_.components_, second.model_.components_)
    np.testing.assert_allclose(first.model_.mean_, second.model_.mean_)


def test_expression_pca_requires_explicit_dense_budget() -> None:
    adata = example_adata()
    split = sccs.SampleSplit(train=("d1", "d2"), validation=(), test=("d3",))
    with pytest.raises(sccs.StateError, match="dense conversion"):
        sccs.ExpressionPCA(1, random_state=0, max_dense_elements=5).fit(
            adata, sample_key="donor", split=split
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: sccs.DirectProgramState(),
        lambda: sccs.LinearProgramState(1, random_state=11),
        lambda: sccs.NonlinearProgramState(1, random_state=11, max_iter=1_000),
    ],
)
def test_program_baselines_are_deterministic_and_do_not_fit_test_cells(factory) -> None:
    activities = np.array([[0.0, 1.0], [1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [8.0, 9.0], [9.0, 8.0]])
    samples = ["d1", "d1", "d2", "d2", "d3", "d3"]
    split = sccs.SampleSplit(train=("d1", "d2"), validation=(), test=("d3",))
    changed = activities.copy()
    changed[4:] = 1000
    first = factory().fit(activities, sample_ids=samples, split=split)
    second = factory().fit(changed, sample_ids=samples, split=split)
    np.testing.assert_allclose(first.transform(activities[:4]), second.transform(activities[:4]))
    np.testing.assert_allclose(
        first.transform(activities),
        factory().fit(activities, sample_ids=samples, split=split).transform(activities),
    )


def test_state_storage_records_training_provenance() -> None:
    adata = example_adata()
    coordinates = np.arange(12, dtype=float).reshape(6, 2)
    sccs.store_state_coordinates(
        adata,
        coordinates,
        method="linear_program_pca",
        training_sample_ids=["d1", "d2"],
        parameters={"random_state": 3},
    )
    np.testing.assert_array_equal(adata.obsm["X_sccs_state"], coordinates)
    state = adata.uns["sccellstates"]["state_representation"]
    assert state["training_sample_ids"] == ["d1", "d2"]
    assert state["method"] == "linear_program_pca"
