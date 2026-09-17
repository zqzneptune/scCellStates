import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import sccellstates as sccs


def nonnegative_matrix() -> np.ndarray:
    rng = np.random.default_rng(42)
    activities = rng.gamma(2.0, 1.0, size=(40, 3))
    weights = np.zeros((3, 12))
    weights[0, :4] = [4, 3, 2, 1]
    weights[1, 4:8] = [4, 3, 2, 1]
    weights[2, 8:] = [4, 3, 2, 1]
    return activities @ weights


@pytest.mark.parametrize("sparse_input", [False, True])
def test_nmf_is_deterministic_for_dense_and_sparse_inputs(sparse_input: bool) -> None:
    values = nonnegative_matrix()
    matrix = sparse.csr_matrix(values) if sparse_input else values
    estimator = sccs.NMFProgramEstimator(n_programs=3, random_state=17)
    first = estimator.fit(matrix, feature_names=[f"g{i}" for i in range(12)], sample_id="d1")
    second = estimator.fit(matrix, feature_names=[f"g{i}" for i in range(12)], sample_id="d1")
    np.testing.assert_allclose(first.weights, second.weights)
    np.testing.assert_allclose(first.weights.sum(axis=1), 1.0)
    assert not first.weights.flags.writeable


def test_nmf_rejects_negative_values() -> None:
    values = nonnegative_matrix()
    values[0, 0] = -1
    estimator = sccs.NMFProgramEstimator(n_programs=3, random_state=17)
    with pytest.raises(sccs.ProgramError, match="nonnegative"):
        estimator.fit(values, feature_names=[f"g{i}" for i in range(12)], sample_id="d1")


class MeanEstimator:
    def __init__(self, seen: list[tuple[str, int]]) -> None:
        self.seen = seen

    def fit(
        self,
        matrix: np.ndarray | sparse.spmatrix,
        *,
        feature_names: tuple[str, ...],
        sample_id: str,
    ) -> sccs.ProgramSet:
        self.seen.append((sample_id, matrix.shape[0]))
        weights = np.asarray(matrix.mean(axis=0)).reshape(1, -1)
        return sccs.ProgramSet(
            sample_id=sample_id,
            feature_names=tuple(feature_names),
            weights=weights,
            n_cells=matrix.shape[0],
            estimator="mean",
        )


def leakage_fixture(heldout_value: float) -> ad.AnnData:
    values = np.array(
        [
            [1, 2, 3],
            [2, 3, 4],
            [4, 2, 1],
            [3, 2, 1],
            [heldout_value, 1, 1],
            [heldout_value, 2, 1],
        ],
        dtype=float,
    )
    return ad.AnnData(
        X=values,
        obs=pd.DataFrame(
            {"donor_id": ["train_a", "train_a", "train_b", "train_b", "held", "held"]},
            index=[f"c{i}" for i in range(6)],
        ),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )


def test_explicit_sample_selection_keeps_heldout_cells_out_of_estimators() -> None:
    outputs = []
    for heldout_value in (10.0, 1_000_000.0):
        seen: list[tuple[str, int]] = []

        def estimator_factory(
            _sample_id: str, *, recorder: list[tuple[str, int]] = seen
        ) -> MeanEstimator:
            return MeanEstimator(recorder)

        fitted = sccs.fit_programs_by_sample(
            leakage_fixture(heldout_value),
            sample_key="donor_id",
            sample_ids=["train_a", "train_b"],
            estimator_factory=estimator_factory,
        )
        assert seen == [("train_a", 2), ("train_b", 2)]
        assert tuple(item.sample_id for item in fitted) == ("train_a", "train_b")
        outputs.append(np.stack([item.weights for item in fitted]))
    np.testing.assert_array_equal(outputs[0], outputs[1])


def test_program_set_rejects_duplicate_features() -> None:
    with pytest.raises(sccs.ProgramError, match="unique"):
        sccs.ProgramSet(
            sample_id="d1",
            feature_names=("g1", "g1"),
            weights=np.ones((2, 2)),
            n_cells=5,
            estimator="test",
        )


def program_run(sample_id: str, weights: np.ndarray) -> sccs.ProgramSet:
    return sccs.ProgramSet(
        sample_id=sample_id,
        feature_names=("g1", "g2", "g3", "g4"),
        weights=weights,
        n_cells=20,
        estimator="test",
    )


def test_stabilize_programs_matches_permuted_runs_and_drops_unstable_programs() -> None:
    anchor_weights = np.array(
        [
            [0.7, 0.2, 0.1, 0.0],
            [0.0, 0.1, 0.2, 0.7],
            [0.4, 0.3, 0.2, 0.1],
        ]
    )
    perturbed = np.array(
        [
            [0.0, 0.1, 0.2, 0.7],
            [0.69, 0.21, 0.1, 0.0],
            [0.1, 0.4, 0.2, 0.3],
        ]
    )
    fit = sccs.stabilize_programs(
        (
            program_run("run_0", anchor_weights),
            program_run("run_1", perturbed),
            program_run("run_2", perturbed),
        ),
        sample_id="donor_1",
        min_similarity=0.9,
    )

    assert fit.programs.sample_id == "donor_1"
    assert fit.programs.n_programs == 2
    assert fit.requested_programs == 3
    assert fit.retained_fraction == pytest.approx(2 / 3)
    np.testing.assert_array_equal(fit.retained_anchor_indices, [0, 1])
    assert not fit.matched_similarities.flags.writeable
    assert not fit.retained_anchor_indices.flags.writeable
    np.testing.assert_allclose(fit.programs.weights.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    ("runs", "message"),
    [
        ((program_run("run_0", np.eye(4)[:2]),), "at least two"),
        (
            (
                program_run("same", np.eye(4)[:2]),
                program_run("same", np.eye(4)[:2]),
            ),
            "IDs must be unique",
        ),
        (
            (
                program_run("run_0", np.eye(4)[:2]),
                program_run("run_1", np.eye(4)[:3]),
            ),
            "same number of programs",
        ),
    ],
)
def test_stabilize_programs_validates_repeated_fits(
    runs: tuple[sccs.ProgramSet, ...], message: str
) -> None:
    with pytest.raises(sccs.ProgramError, match=message):
        sccs.stabilize_programs(runs, sample_id="donor_1")


def test_stabilize_programs_rejects_when_no_program_is_stable() -> None:
    anchor = program_run("run_0", np.eye(4)[:2])
    reversed_ranks = program_run("run_1", np.array([[0.0, 0.0, 0.4, 0.6], [0.6, 0.4, 0.0, 0.0]]))
    with pytest.raises(sccs.ProgramError, match="no program"):
        sccs.stabilize_programs((anchor, reversed_ranks), sample_id="donor_1", min_similarity=1.0)
