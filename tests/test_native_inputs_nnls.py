import gzip

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite

import sccellstates as sccs


def _write_10x_h5(path, *, mixed=True):
    # genes x cells CSC: g1=[2, 0], antibody=[0, 9], g2=[1, 3]
    with h5py.File(path, "w") as handle:
        matrix = handle.create_group("matrix")
        matrix.create_dataset("data", data=np.array([2, 1, 9, 3], dtype=np.float32))
        matrix.create_dataset("indices", data=np.array([0, 2, 1, 2], dtype=np.int64))
        matrix.create_dataset("indptr", data=np.array([0, 2, 4], dtype=np.int64))
        matrix.create_dataset("shape", data=np.array([3, 2], dtype=np.int64))
        matrix.create_dataset("barcodes", data=np.array([b"c1", b"c2"]))
        features = matrix.create_group("features")
        features.create_dataset("id", data=np.array([b"g1", b"ab", b"g2"]))
        features.create_dataset("name", data=np.array([b"G1", b"AB", b"G2"]))
        if mixed:
            features.create_dataset(
                "feature_type",
                data=np.array([b"Gene Expression", b"Antibody Capture", b"Gene Expression"]),
            )


def test_native_10x_h5_is_detected_and_feature_types_are_filtered(tmp_path):
    path = tmp_path / "filtered_feature_bc_matrix.h5"
    _write_10x_h5(path)
    result = sccs.fit(path, n_programs=1, n_repeats=1, preprocessing="identity")
    assert result.feature_names == ("g1", "g2")
    assert sparse.issparse(sccs.samples_from_source(path).samples[0].source.X)


def test_h5ad_and_native_10x_are_distinguished(tmp_path):
    h5ad_path = tmp_path / "data.h5"
    ad.AnnData(
        X=sparse.csr_matrix([[1, 2], [3, 4]]),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    ).write_h5ad(h5ad_path)
    assert sccs.samples_from_source(h5ad_path).samples[0].source.shape == (2, 2)
    tenx_path = tmp_path / "raw_feature_bc_matrix.h5"
    _write_10x_h5(tenx_path)
    assert sccs.samples_from_source(tenx_path).samples[0].source.n_vars == 2


def test_legacy_and_compressed_mtx_directory(tmp_path):
    values = sparse.coo_matrix(np.array([[1, 0], [0, 2]], dtype=np.float32))
    for compressed in (False, True):
        directory = tmp_path / ("compressed" if compressed else "plain")
        directory.mkdir()
        suffix = ".gz" if compressed else ""
        matrix_path = directory / f"matrix.mtx{suffix}"
        if compressed:
            with gzip.open(matrix_path, "wb") as handle:
                mmwrite(handle, values)
            (directory / "genes.tsv.gz").write_bytes(gzip.compress(b"g1\tG1\n" b"g2\tG2\n"))
            (directory / "barcodes.tsv.gz").write_bytes(gzip.compress(b"c1\nc2\n"))
        else:
            mmwrite(matrix_path, values)
            (directory / "genes.txt").write_text("g1\tG1\ng2\tG2\n")
            (directory / "barcodes.txt").write_text("c1\nc2\n")
        loaded = sccs.samples_from_source(directory).samples[0].source
        assert loaded.shape == (2, 2)
        assert sparse.issparse(loaded.X)


def test_nnls_projection_recovers_normalized_mixtures_and_reports_alignment():
    vocabulary = sccs.ProgramSet(
        sample_id="v1",
        feature_names=("g1", "g2", "g3"),
        weights=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
        n_cells=2,
        estimator="synthetic",
        parameters={"preprocessing": "identity"},
    )
    adata = ad.AnnData(
        X=sparse.csr_matrix([[2, 1, 1, 99], [0, 3, 0, 88]], dtype=float),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["g3", "g1", "g2", "extra"]),
    )
    result = sccs.project(adata, vocabulary, method="nnls", preprocessing="identity")
    np.testing.assert_allclose(result.states, [[.25, .25, .5], [1, 0, 0]])
    np.testing.assert_allclose(result.usages.sum(axis=1), [4, 3])
    assert result.feature_coverage == 1
    assert result.extra_features == ("extra",)
    assert result.missing_features == ()
    assert np.all(result.usages >= 0)


def test_nnls_projection_missing_features_and_incompatible_scale():
    vocabulary = sccs.ProgramSet(
        sample_id="v1", feature_names=("g1", "g2"), weights=np.array([[1, 0], [0, 1]]),
        n_cells=1, estimator="synthetic", parameters={"preprocessing": "library_size_log1p"},
    )
    adata = ad.AnnData(X=sparse.csr_matrix([[2.0]]), var=pd.DataFrame(index=["g1"]))
    with np.testing.assert_raises_regex(sccs.APIError, "incompatible"):
        sccs.project(adata, vocabulary, method="nnls", preprocessing="identity")
    result = sccs.project(adata, vocabulary, method="nnls", preprocessing="library_size_log1p")
    assert result.missing_features == ("g2",)
    assert result.feature_coverage == .5
