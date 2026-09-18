"""Small, explicit loaders and QC helpers for local 10x inputs."""

from __future__ import annotations

import gzip
import re
import tarfile
from collections.abc import Iterable
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread

from sccellstates.io import InputError


class DatasetError(InputError):
    """Raised when a 10x dataset cannot be read or passes invalid QC options."""


def _sample_id(name: str) -> str:
    match = re.match(r"(?P<sample>GSM\d+)(?:_.*)?$", name)
    return match.group("sample") if match else Path(name).stem


def _read_text(handle: object) -> list[str]:
    return [line.decode().rstrip("\r\n") for line in handle]  # type: ignore[union-attr]


def _maybe_gzip(handle: object) -> object:
    """Wrap a binary handle in gzip only when its name indicates compression."""
    name = getattr(handle, "name", "")
    return gzip.GzipFile(fileobj=handle) if str(name).endswith(".gz") else handle


def _read_features(handle: object) -> tuple[list[str], list[str]]:
    rows = [line.split("\t") for line in _read_text(handle)]
    if not rows or len(rows[0]) < 2:
        raise DatasetError("10x features file is empty or malformed")
    feature_ids = [row[0] for row in rows]
    symbols = [row[1] for row in rows]
    if len(set(feature_ids)) != len(feature_ids):
        raise DatasetError("10x feature identifiers must be unique")
    return feature_ids, symbols


def _read_10x_h5(path: Path) -> ad.AnnData:
    """Read a Cell Ranger HDF5 matrix, retaining only Gene Expression."""
    try:
        with h5py.File(path, "r") as handle:
            if "matrix" not in handle or not isinstance(handle["matrix"], h5py.Group):
                raise DatasetError("HDF5 file is neither AnnData H5AD nor a 10x matrix")
            group = handle["matrix"]
            required = {"data", "indices", "indptr", "shape", "barcodes", "features"}
            if not required.issubset(group):
                missing = sorted(required - set(group))
                raise DatasetError(f"malformed 10x HDF5: missing /matrix entries {missing}")
            features = group["features"]
            if not isinstance(features, h5py.Group) or not {"id", "name"}.issubset(features):
                raise DatasetError("malformed 10x HDF5: /matrix/features is incomplete")
            feature_ids = [
                str(value, "utf-8") if isinstance(value, bytes) else str(value)
                for value in features["id"][:]
            ]
            symbols = [
                str(value, "utf-8") if isinstance(value, bytes) else str(value)
                for value in features["name"][:]
            ]
            if "feature_type" in features:
                kinds = [
                    str(value, "utf-8") if isinstance(value, bytes) else str(value)
                    for value in features["feature_type"][:]
                ]
                keep = np.asarray([kind == "Gene Expression" for kind in kinds])
            else:
                keep = np.ones(len(feature_ids), dtype=bool)
            if not keep.any():
                raise DatasetError("10x HDF5 contains no Gene Expression features")
            barcodes = [
                str(value, "utf-8") if isinstance(value, bytes) else str(value)
                for value in group["barcodes"][:]
            ]
            shape = tuple(int(value) for value in group["shape"][:])
            if len(shape) != 2 or shape != (len(feature_ids), len(barcodes)):
                raise DatasetError("malformed 10x HDF5: matrix dimensions do not match annotations")
            values = (
                sparse.csc_matrix(
                    (group["data"][:], group["indices"][:], group["indptr"][:]),
                    shape=shape,
                    dtype=np.float32,
                )
                .tocsr()
                .T[:, keep]
                .tocsr()
            )
    except OSError as error:
        raise DatasetError(f"cannot read HDF5 input {path}: {error}") from error
    if len(set(np.asarray(feature_ids)[keep])) != int(keep.sum()):
        raise DatasetError("10x Gene Expression feature identifiers must be unique")
    names = list(np.asarray(feature_ids)[keep])
    selected_symbols = list(np.asarray(symbols)[keep])
    obs_names = [str(name) for name in barcodes]
    return ad.AnnData(
        X=values,
        obs=pd.DataFrame({"sample_id": [path.stem] * len(obs_names)}, index=obs_names),
        var=pd.DataFrame(
            {"gene_id": names, "gene_symbol": selected_symbols},
            index=pd.Index(names, name="gene"),
        ),
        layers={"counts": values.copy()},
        uns={
            "sccellstates_input": {
                "format": "10x_hdf5",
                "counts_source": "matrix",
                "feature_selection": "Gene Expression",
            }
        },
    )


def read_rna_input(path: str | Path) -> ad.AnnData:
    """Read an H5AD, native 10x HDF5, or supported 10x MTX directory."""
    source = Path(path)
    if source.is_dir():
        return read_10x_samples(source)
    if not source.is_file():
        raise DatasetError(f"input does not exist: {source}")
    try:
        with h5py.File(source, "r") as handle:
            is_10x = "matrix" in handle and isinstance(handle["matrix"], h5py.Group)
    except OSError as error:
        raise DatasetError(f"unsupported or unreadable HDF5 input {source}") from error
    if is_10x:
        return _read_10x_h5(source)
    if source.suffix.lower() in {".h5ad", ".h5"}:
        try:
            return ad.read_h5ad(source)
        except (OSError, ValueError, KeyError) as error:
            raise DatasetError(
                f"HDF5 input {source} is neither a valid AnnData H5AD nor a valid 10x matrix"
            ) from error
    raise DatasetError(f"unsupported RNA input: {source}")


def _read_one_sample(
    *,
    sample_id: str,
    barcodes: object,
    features: object,
    matrix: object,
    min_counts_per_cell: float,
    min_genes_per_cell: int,
    max_cells: int | None,
) -> ad.AnnData:
    barcode_names = _read_text(barcodes)
    feature_ids, symbols = _read_features(features)
    values = mmread(matrix)
    values = sparse.csc_matrix(values, dtype=np.float32).tocsr().T
    if values.shape != (len(barcode_names), len(feature_ids)):
        raise DatasetError(f"10x matrix dimensions do not match sample {sample_id!r}")
    counts = np.asarray(values.sum(axis=1)).ravel()
    detected = np.asarray((values > 0).sum(axis=1)).ravel()
    keep = (counts >= min_counts_per_cell) & (detected >= min_genes_per_cell)
    if not keep.any():
        raise DatasetError(f"QC removed every cell from sample {sample_id!r}")
    if max_cells is not None and int(keep.sum()) > max_cells:
        selected = np.flatnonzero(keep)[:max_cells]
        keep = np.zeros_like(keep)
        keep[selected] = True
    # Feature IDs are the stable cross-sample axis; symbols are retained only
    # as annotations because symbol ordering and uniqueness can vary by file.
    names, original_symbols = feature_ids, symbols
    obs_names = [f"{sample_id}:{barcode_names[i]}" for i in np.flatnonzero(keep)]
    return ad.AnnData(
        X=values[keep],
        obs=pd.DataFrame({"sample_id": sample_id}, index=obs_names),
        var=pd.DataFrame(
            {"gene_id": feature_ids, "gene_symbol": original_symbols},
            index=pd.Index(names, name="gene"),
        ),
        layers={"counts": values[keep].copy()},
        uns={
            "sccellstates_input": {
                "format": "10x_mtx",
                "counts_source": "matrix.mtx",
                "feature_selection": "Gene Expression or legacy feature table",
            }
        },
    )


def _parts_from_directory(path: Path) -> dict[str, dict[str, Path]]:
    files = list(path.glob("*_barcodes.tsv.gz"))
    parts: dict[str, dict[str, Path]] = {}
    for barcode in files:
        prefix = barcode.name.removesuffix("_barcodes.tsv.gz")
        parts[prefix] = {
            "barcodes": barcode,
            "features": path / f"{prefix}_features.tsv.gz",
            "matrix": path / f"{prefix}_matrix.mtx.gz",
        }
    # Cell Ranger's standard single-sample directory has unprefixed files;
    # accept compressed and uncompressed, plus the legacy genes names.
    if not parts:

        def find(*names: str) -> Path | None:
            for name in names:
                candidate = path / name
                if candidate.exists():
                    return candidate
            return None

        barcodes = find("barcodes.tsv.gz", "barcodes.tsv", "barcodes.txt")
        features = find(
            "features.tsv.gz",
            "features.tsv",
            "genes.tsv.gz",
            "genes.tsv",
            "genes.txt.gz",
            "genes.txt",
        )
        matrix = find("matrix.mtx.gz", "matrix.mtx")
        if barcodes and features and matrix:
            parts["sample"] = {"barcodes": barcodes, "features": features, "matrix": matrix}
    return parts


def _parts_from_tar(path: Path) -> tuple[tarfile.TarFile, dict[str, dict[str, tarfile.TarInfo]]]:
    archive = tarfile.open(path, mode="r:*")
    parts: dict[str, dict[str, tarfile.TarInfo]] = {}
    for member in archive.getmembers():
        name = Path(member.name).name
        for suffix, key in (
            ("_barcodes.tsv.gz", "barcodes"),
            ("_features.tsv.gz", "features"),
            ("_matrix.mtx.gz", "matrix"),
        ):
            if name.endswith(suffix):
                prefix = name.removesuffix(suffix)
                parts.setdefault(prefix, {})[key] = member
                break
    return archive, parts


def read_10x_samples(
    path: str | Path,
    *,
    sample_ids: Iterable[str] | None = None,
    min_counts_per_cell: float = 0,
    min_genes_per_cell: int = 0,
    max_cells_per_sample: int | None = None,
) -> ad.AnnData:
    """Read one or more 10x samples from a directory or tar archive.

    Cell QC is applied independently within each sample. Gene selection is
    intentionally separate and should be performed on training samples only.
    """
    if min_counts_per_cell < 0 or min_genes_per_cell < 0:
        raise ValueError("cell QC thresholds must be non-negative")
    if max_cells_per_sample is not None and max_cells_per_sample < 1:
        raise ValueError("max_cells_per_sample must be positive")
    source = Path(path)
    requested = None if sample_ids is None else set(map(str, sample_ids))
    archive: tarfile.TarFile | None = None
    if source.is_dir():
        parts = _parts_from_directory(source)
    elif source.is_file() and tarfile.is_tarfile(source):
        archive, parts = _parts_from_tar(source)
    else:
        raise DatasetError(f"input must be a 10x directory or tar archive: {source}")
    try:
        selected = sorted(
            (prefix, files)
            for prefix, files in parts.items()
            if requested is None or _sample_id(prefix) in requested or prefix in requested
        )
        if not selected:
            raise DatasetError("no requested 10x samples were found")
        datasets: list[ad.AnnData] = []
        for prefix, files in selected:
            if set(files) != {"barcodes", "features", "matrix"}:
                raise DatasetError(f"incomplete 10x sample: {prefix!r}")
            sample_id = _sample_id(prefix)
            if archive is None:
                handles = {key: path.open("rb") for key, path in files.items()}
            else:
                handles = {key: archive.extractfile(member) for key, member in files.items()}
            if any(handle is None for handle in handles.values()):
                raise DatasetError(f"cannot read 10x sample: {prefix!r}")
            try:
                datasets.append(
                    _read_one_sample(
                        sample_id=sample_id,
                        barcodes=_maybe_gzip(handles["barcodes"]),
                        features=_maybe_gzip(handles["features"]),
                        matrix=_maybe_gzip(handles["matrix"]),
                        min_counts_per_cell=min_counts_per_cell,
                        min_genes_per_cell=min_genes_per_cell,
                        max_cells=max_cells_per_sample,
                    )
                )
            finally:
                for handle in handles.values():
                    if handle is not None:
                        handle.close()
        reference = datasets[0].var_names
        common = reference
        for dataset in datasets[1:]:
            common = common.intersection(dataset.var_names, sort=False)
        if len(common) == 0:
            raise DatasetError("10x samples have no shared feature identifiers")
        datasets = [dataset[:, common].copy() for dataset in datasets]
        return ad.concat(datasets, axis=0, join="inner", merge="same", index_unique=None)
    finally:
        if archive is not None:
            archive.close()


def select_training_genes(
    adata: ad.AnnData,
    *,
    sample_key: str,
    training_sample_ids: Iterable[str],
    layer: str | None = None,
    n_top_genes: int | None = None,
    min_cells_per_gene: int = 0,
    remove_mitochondrial: bool = False,
    remove_ribosomal: bool = False,
    exclude_genes: Iterable[str] = (),
) -> ad.AnnData:
    """Select variable genes using only declared training samples."""
    if n_top_genes is not None and n_top_genes < 1:
        raise ValueError("n_top_genes must be positive")
    if min_cells_per_gene < 0:
        raise ValueError("min_cells_per_gene must be non-negative")
    labels = adata.obs[sample_key].astype("string").to_numpy()
    train = np.isin(labels, list(training_sample_ids))
    if not train.any():
        raise DatasetError("training samples contain no cells")
    counts = adata.X if layer is None else adata.layers[layer]
    training_counts = counts[train]
    detected = np.asarray((training_counts > 0).sum(axis=0)).ravel()
    keep = detected >= min_cells_per_gene
    names = adata.var_names.astype(str)
    symbols = adata.var.get("gene_symbol", pd.Series("", index=adata.var_names)).astype(str)
    names_upper = names.str.upper()
    symbols_upper = symbols.str.upper()
    if remove_mitochondrial:
        keep &= ~np.asarray(names_upper.str.startswith("MT-"))
        keep &= ~symbols_upper.str.startswith("MT-").to_numpy()
    if remove_ribosomal:
        keep &= ~(
            np.asarray(names_upper.str.startswith(("RPS", "RPL")))
            | symbols_upper.str.startswith(("RPS", "RPL")).to_numpy()
        )
    excluded = {str(name).upper() for name in exclude_genes}
    if excluded:
        keep &= ~np.asarray(names_upper.isin(excluded))
        keep &= ~symbols_upper.isin(excluded).to_numpy()
    if n_top_genes is not None and int(keep.sum()) > n_top_genes:
        if sparse.issparse(training_counts):
            log_counts = training_counts.copy().astype(np.float64)
            log_counts.data = np.log1p(log_counts.data)
            means = np.asarray(log_counts.mean(axis=0)).ravel()
            means_sq = np.asarray(log_counts.power(2).mean(axis=0)).ravel()
        else:
            log_counts = np.log1p(np.asarray(training_counts, dtype=np.float64))
            means = log_counts.mean(axis=0)
            means_sq = np.square(log_counts).mean(axis=0)
        variance = np.maximum(means_sq - means**2, 0)
        eligible = np.flatnonzero(keep)
        chosen = eligible[np.argsort(variance[eligible], kind="stable")[-n_top_genes:]]
        keep = np.zeros(adata.n_vars, dtype=bool)
        keep[chosen] = True
    if not keep.any():
        raise DatasetError("gene QC removed every gene")
    return adata[:, keep].copy()


def filter_anndata_cells(
    adata: ad.AnnData,
    *,
    layer: str | None = None,
    min_counts_per_cell: float = 0,
    min_genes_per_cell: int = 0,
    max_cells_per_sample: int | None = None,
    sample_key: str | None = None,
) -> ad.AnnData:
    """Apply explicit cell QC to an existing AnnData object."""
    if min_counts_per_cell < 0 or min_genes_per_cell < 0:
        raise ValueError("cell QC thresholds must be non-negative")
    if max_cells_per_sample is not None and max_cells_per_sample < 1:
        raise ValueError("max_cells_per_sample must be positive")
    matrix = adata.X if layer is None else adata.layers[layer]
    counts = np.asarray(matrix.sum(axis=1)).ravel()
    detected = np.asarray((matrix > 0).sum(axis=1)).ravel()
    keep = (counts >= min_counts_per_cell) & (detected >= min_genes_per_cell)
    if sample_key is not None and max_cells_per_sample is not None:
        labels = adata.obs[sample_key].astype("string").to_numpy()
        for sample in pd.unique(labels):
            indices = np.flatnonzero(keep & (labels == sample))
            if len(indices) > max_cells_per_sample:
                keep[indices[max_cells_per_sample:]] = False
    if not keep.any():
        raise DatasetError("QC removed every cell")
    return adata[keep].copy()
