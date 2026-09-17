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
    summary = sccs.validate_anndata(
        adata, sample_key="donor_id", layer="counts", min_samples=2
    )
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
