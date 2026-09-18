import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import sccellstates as sccs


def _adata(*, counts=True, sparse_x=True):
    values = np.array([[8, 1, 0], [1, 7, 1], [4, 4, 2]], dtype=float)
    if not counts:
        values /= values.sum(axis=1, keepdims=True)
    return ad.AnnData(
        X=sparse.csr_matrix(values) if sparse_x else values,
        obs=pd.DataFrame(index=["c1", "c2", "c3"]),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )


def _vocabulary():
    return sccs.ProgramSet(
        sample_id="frozen", feature_names=("g1", "g2", "g3"),
        weights=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
        n_cells=3, estimator="test",
    )


@pytest.mark.parametrize("method", ["nnls", "regularized_nnls", "simplex", "poisson"])
def test_registered_projectors_share_result_contract(method):
    kwargs = {"l2": 0.01} if method == "regularized_nnls" else {}
    result = sccs.project(_adata(), _vocabulary(), method=method, **kwargs)
    assert isinstance(result, sccs.StateResult)
    assert result.usages.shape == (3, 3)
    assert result.normalized_states.shape == (3, 3)
    np.testing.assert_allclose(result.normalized_states.sum(axis=1), 1)
    assert np.all(result.usages >= 0)
    assert result.vocabulary.weights.flags.writeable is False


def test_projector_registry_is_explicit():
    assert sccs.available_projectors() == ("nnls", "regularized_nnls", "simplex", "poisson")


def test_poisson_rejects_normalized_input():
    with pytest.raises(sccs.StateError, match="integer-valued"):
        sccs.project(_adata(counts=False), _vocabulary(), method="poisson")


def test_regularized_and_simplex_support_dense_input():
    for method, kwargs in (("regularized_nnls", {"l1": 0.1}), ("simplex", {})):
        result = sccs.project(_adata(sparse_x=False), _vocabulary(), method=method, **kwargs)
        assert np.isfinite(result.projection_error).all()


def test_simplex_is_stable_for_large_count_magnitudes():
    values = np.array([[800_000, 100_000, 0], [1, 700_000, 100_000]], dtype=float)
    adata = ad.AnnData(X=sparse.csr_matrix(values), obs=pd.DataFrame(index=["a", "b"]),
                       var=pd.DataFrame(index=["g1", "g2", "g3"]))
    result = sccs.project(adata, _vocabulary(), method="simplex")
    np.testing.assert_allclose(result.normalized_states.sum(axis=1), 1)
    assert np.isfinite(result.normalized_states).all()


def test_poisson_large_sparse_input_uses_count_mixture_update():
    values = np.array([[800, 100, 0], [1, 700, 100]], dtype=float)
    adata = ad.AnnData(X=sparse.csr_matrix(values), obs=pd.DataFrame(index=["a", "b"]),
                       var=pd.DataFrame(index=["g1", "g2", "g3"]))
    result = sccs.project(adata, _vocabulary(), method="poisson", max_iter=100)
    np.testing.assert_allclose(result.usages.sum(axis=1), values.sum(axis=1), rtol=1e-5)
    np.testing.assert_allclose(result.normalized_states.sum(axis=1), 1)


def test_compare_projectors_accepts_method_specific_options():
    benchmark = sccs.compare_projectors(
        [_adata()], _vocabulary(),
        methods=["nnls", "regularized_nnls", "simplex", "poisson"],
        projector_options={"regularized_nnls": {"l2": 0.01}, "poisson": {"max_iter": 50}},
    )
    assert benchmark.methods == ("nnls", "regularized_nnls", "simplex", "poisson")
    assert all(len(results) == 1 for results in benchmark.results.values())


def test_projection_artifacts_round_trip(tmp_path):
    vocabulary = _vocabulary()
    projector = sccs.make_projector("nnls", vocabulary)
    result = sccs.project(_adata(), vocabulary, method="nnls", sample_id="held_out")
    benchmark = sccs.compare_projectors(
        {"held_out": _adata()}, vocabulary, methods=["nnls"]
    )

    projector_path = sccs.save_projector(projector, tmp_path / "projector")
    restored_projector = sccs.load_projector(projector_path)
    restored_result = restored_projector.transform(_adata(), sample_id="held_out")
    np.testing.assert_array_equal(restored_result.usages, result.usages)

    result_path = sccs.save_state_result(result, tmp_path / "state")
    loaded_result = sccs.load_state_result(result_path)
    np.testing.assert_array_equal(loaded_result.normalized_states, result.normalized_states)
    assert loaded_result.sample_id == "held_out"

    benchmark_path = sccs.save_projection_benchmark(benchmark, tmp_path / "benchmark")
    loaded_benchmark = sccs.load_projection_benchmark(benchmark_path)
    np.testing.assert_array_equal(
        loaded_benchmark.results["nnls"][0].usages,
        benchmark.results["nnls"][0].usages,
    )
