import numpy as np
import pytest

import sccellstates as sccs


def test_reconstruction_is_reported_by_sample_partition() -> None:
    observed = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    reconstructed = observed.copy()
    reconstructed[2:] += 1
    result = sccs.reconstruction_by_partition(
        observed, reconstructed, ["train", "train", "test", "test"]
    ).set_index("partition")
    assert result.loc["train", "mean_squared_error"] == 0
    assert result.loc["test", "mean_squared_error"] == 1


def test_technical_associations_include_absolute_spearman_values() -> None:
    coordinates = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
    result = sccs.technical_associations(coordinates, {"library_size": [1, 2, 3, 4]})
    assert result["spearman_r"].tolist() == pytest.approx([1.0, -1.0])
    assert result["absolute_spearman_r"].tolist() == pytest.approx([1.0, 1.0])


def test_technical_associations_report_constant_covariate_as_missing() -> None:
    result = sccs.technical_associations(np.ones((3, 1)), {"constant": [1, 1, 1]})
    assert np.isnan(result.loc[0, "spearman_r"])


def test_sample_bootstrap_resamples_samples_not_cells() -> None:
    values = np.array([[0.0], [0.0], [10.0]])
    result = sccs.sample_bootstrap(
        values, ["small", "small", "large"], n_bootstrap=500, random_state=4
    )
    assert result.loc[0, "estimate"] == pytest.approx(5.0)
    assert result.loc[0, "n_samples"] == 2
    assert result.loc[0, "lower"] < 5 < result.loc[0, "upper"]


def test_sample_bootstrap_is_deterministic_and_rejects_missing_values() -> None:
    values = np.arange(8, dtype=float).reshape(4, 2)
    first = sccs.sample_bootstrap(values, ["a", "a", "b", "b"], random_state=9)
    second = sccs.sample_bootstrap(values, ["a", "a", "b", "b"], random_state=9)
    np.testing.assert_array_equal(first.to_numpy(), second.to_numpy())
    with pytest.raises(sccs.MetricsError, match="finite"):
        sccs.sample_bootstrap(np.array([1.0, np.nan]), ["a", "b"], random_state=0)
