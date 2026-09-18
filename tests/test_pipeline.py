import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import sccellstates as sccs


def pipeline_adata(*, sparse_x: bool) -> ad.AnnData:
    rng = np.random.default_rng(4)
    values = rng.poisson(3, size=(32, 8)).astype(float)
    values[:, :2] += rng.poisson(5, size=(32, 2))
    matrix = sparse.csr_matrix(values) if sparse_x else values
    samples = np.repeat(["a", "b", "c", "held"], 8)
    return ad.AnnData(
        X=matrix,
        layers={"counts": matrix.copy()},
        obs=pd.DataFrame({"sample": samples}, index=[f"cell_{i}" for i in range(32)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(8)]),
    )


@pytest.mark.parametrize("sparse_x", [False, True])
def test_pipeline_is_end_to_end_and_does_not_mutate_input(sparse_x: bool) -> None:
    adata = pipeline_adata(sparse_x=sparse_x)
    original = adata.X.copy()
    config = sccs.PipelineConfig(
        sample_key="sample",
        validation_samples=("b",),
        test_samples=("held",),
        layer="counts",
        n_programs=2,
        state_components=1,
        state_methods=("direct", "linear", "pca"),
        n_permutations=20,
        random_state=12,
    )
    result = sccs.fit_pipeline(adata, config)

    assert result.split == sccs.SampleSplit(train=("a", "c"), validation=("b",), test=("held",))
    assert result.activities.shape[0] == 32
    assert result.activities.shape[1] == result.vocabulary_fit.vocabulary.n_programs
    assert set(result.coordinates) == {"direct", "linear", "pca"}
    assert result.adata is not adata
    assert "pipeline" in result.adata.uns["sccellstates"]
    assert result.provenance["validated"] is False
    if sparse.issparse(original):
        np.testing.assert_array_equal(adata.X.toarray(), original.toarray())
    else:
        np.testing.assert_array_equal(adata.X, original)


def test_pipeline_preprocessing_is_deterministic_and_heldout_safe() -> None:
    config = sccs.PipelineConfig(
        sample_key="sample",
        test_samples=("held",),
        validation_samples=("b",),
        preprocessing="library_size_log1p",
        n_programs=2,
        state_methods=("direct",),
        n_permutations=10,
        random_state=8,
    )
    first = sccs.fit_pipeline(pipeline_adata(sparse_x=True), config)
    changed = pipeline_adata(sparse_x=True)
    changed_values = changed.X.tolil()
    changed_values[24:, :] = 1_000_000
    changed.X = changed_values.tocsr()
    second = sccs.fit_pipeline(changed, config)
    np.testing.assert_allclose(
        first.vocabulary_fit.vocabulary.weights, second.vocabulary_fit.vocabulary.weights
    )
    np.testing.assert_allclose(first.activities[:16], second.activities[:16])
    assert first.vocabulary_fit.training_sample_ids == ("a", "c")


def test_tuning_selects_on_validation_and_records_selection() -> None:
    base = dict(
        sample_key="sample",
        test_samples=("held",),
        validation_samples=("b",),
        state_methods=("direct",),
        n_permutations=10,
    )
    result = sccs.tune_pipeline(
        pipeline_adata(sparse_x=False),
        [
            sccs.PipelineConfig(**base, n_programs=1, random_state=1),
            sccs.PipelineConfig(**base, n_programs=2, random_state=2),
        ],
    )
    assert result.reports["tuning"].shape == (2, 2)
    assert result.provenance["tuning"]["selected_candidate"] in {0, 1}
    scores = result.reports["tuning"]["validation_mean_squared_error"]
    assert float(scores.max() - scores.min()) > 1e-8
    assert result.provenance["tuning"]["selection_metric"] == (
        "program_reconstruction.validation.mean_squared_error"
    )


def test_pipeline_seeds_every_sample_identically(monkeypatch) -> None:
    """No sample's seed may depend on its name or its position.

    A name-derived seed makes renaming or reordering donors change their
    programs, which would couple how data are stored to what is discovered.
    """
    recorded: list[int] = []
    original = sccs.NMFProgramEstimator.__init__

    def spy(self, *, n_programs, random_state, **kwargs):
        recorded.append(random_state)
        original(self, n_programs=n_programs, random_state=random_state, **kwargs)

    monkeypatch.setattr(sccs.NMFProgramEstimator, "__init__", spy)
    sccs.fit_pipeline(
        pipeline_adata(sparse_x=False),
        sccs.PipelineConfig(
            sample_key="sample",
            test_samples=("held",),
            layer="counts",
            preprocessing="library_size_log1p",
            n_programs=3,
            n_permutations=20,
            state_methods=("direct",),
            random_state=5,
        ),
    )

    assert recorded
    assert set(recorded) == {5}


def test_pipeline_programs_do_not_change_when_samples_are_renamed() -> None:
    adata = pipeline_adata(sparse_x=False)
    renamed = adata.copy()
    renamed.obs["sample"] = renamed.obs["sample"].map(lambda name: f"{name}_x")

    def config(sample_key_test: str) -> sccs.PipelineConfig:
        return sccs.PipelineConfig(
            sample_key="sample",
            test_samples=(sample_key_test,),
            layer="counts",
            preprocessing="library_size_log1p",
            n_programs=3,
            n_permutations=20,
            state_methods=("direct",),
            random_state=5,
        )

    first = sccs.fit_pipeline(adata, config("held"))
    second = sccs.fit_pipeline(renamed, config("held_x"))

    np.testing.assert_array_equal(
        first.vocabulary_fit.vocabulary.weights,
        second.vocabulary_fit.vocabulary.weights,
    )
