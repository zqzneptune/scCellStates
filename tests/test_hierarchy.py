import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import sccellstates as sccs


def hierarchy_adata() -> ad.AnnData:
    rng = np.random.default_rng(21)
    values = rng.poisson(3, size=(48, 10)).astype(float)
    values[:, :3] += rng.poisson(6, size=(48, 3))
    return ad.AnnData(
        X=sparse.csr_matrix(values),
        obs=pd.DataFrame(
            {
                "sample": np.repeat(["s1", "s2", "s3", "held"], 12),
                "lineage": np.tile(np.repeat(["L1", "L2"], 6), 4),
                "condition": np.repeat(["control", "treated"], 24),
            },
            index=[f"c{i}" for i in range(48)],
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(10)]),
    )


def estimator(_group: str) -> sccs.NMFProgramEstimator:
    return sccs.NMFProgramEstimator(n_programs=2, random_state=7)


def test_hierarchical_fit_keeps_lineages_separate_and_builds_shared_vocab() -> None:
    fit = sccs.fit_hierarchical_vocabularies(
        hierarchy_adata(),
        sample_key="sample",
        lineage_key="lineage",
        training_sample_ids=("s1", "s2", "s3"),
        estimator_factory=estimator,
        n_permutations=10,
        random_state=3,
    )
    assert fit.lineages == ("L1", "L2")
    assert fit.shared_vocabulary is not None
    assert all(
        set(v.training_sample_ids) == {"L1::s1", "L1::s2", "L1::s3"}
        or set(v.training_sample_ids) == {"L2::s1", "L2::s2", "L2::s3"}
        for v in fit.lineage_vocabularies.values()
    )
    repeat = sccs.fit_hierarchical_vocabularies(
        hierarchy_adata(),
        sample_key="sample",
        lineage_key="lineage",
        training_sample_ids=("s1", "s2", "s3"),
        estimator_factory=estimator,
        n_permutations=10,
        random_state=3,
    )
    np.testing.assert_array_equal(
        fit.shared_vocabulary.vocabulary.weights,
        repeat.shared_vocabulary.vocabulary.weights,
    )


def test_hierarchical_fit_excludes_heldout_sample_from_program_estimation() -> None:
    first = sccs.fit_hierarchical_vocabularies(
        hierarchy_adata(),
        sample_key="sample",
        lineage_key="lineage",
        training_sample_ids=("s1", "s2", "s3"),
        estimator_factory=estimator,
        n_permutations=10,
        random_state=3,
    )
    changed = hierarchy_adata()
    changed_values = changed.X.tolil()
    changed_values[36:, :] = 1_000_000
    changed.X = changed_values.tocsr()
    second = sccs.fit_hierarchical_vocabularies(
        changed,
        sample_key="sample",
        lineage_key="lineage",
        training_sample_ids=("s1", "s2", "s3"),
        estimator_factory=estimator,
        n_permutations=10,
        random_state=3,
    )
    for lineage in first.lineages:
        np.testing.assert_allclose(
            first.lineage_vocabularies[lineage].vocabulary.weights,
            second.lineage_vocabularies[lineage].vocabulary.weights,
        )


def test_hierarchical_fit_requires_complete_lineage_metadata() -> None:
    data = hierarchy_adata()
    data.obs.loc["c0", "lineage"] = None
    with pytest.raises(sccs.InputError, match="missing lineage"):
        sccs.fit_hierarchical_vocabularies(
            data,
            sample_key="sample",
            lineage_key="lineage",
            training_sample_ids=("s1", "s2"),
            estimator_factory=estimator,
            n_permutations=5,
            random_state=1,
        )
