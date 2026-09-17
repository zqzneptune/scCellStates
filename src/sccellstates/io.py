"""AnnData-first input contracts for scCellStates."""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

if TYPE_CHECKING:
    from sccellstates.api import ProgramResult
    from sccellstates.programs import ProgramStabilityFit
    from sccellstates.recurrence import ProgramVocabulary, RecurrenceResult, VocabularyFit

Matrix = np.ndarray | sparse.spmatrix

_VOCABULARY_SCHEMA_VERSION = "1.0"
_PROGRAM_RESULT_SCHEMA_VERSION = "1.0"
_SUPPORTED_PROGRAM_RESULT_SCHEMAS = frozenset({_PROGRAM_RESULT_SCHEMA_VERSION})
_PROGRAM_RESULT_KEY = "program_result"
_H5AD_SUFFIXES = frozenset({".h5ad", ".h5"})


class InputError(ValueError):
    """Raised when an input violates the scCellStates data contract."""


class APIError(InputError):
    """Raised when a public workflow input is invalid.

    Every input-contract failure a public workflow can raise derives from
    :class:`InputError`, so ``except sccellstates.InputError`` catches bad paths,
    bad metadata, and bad workflow arguments alike.
    """


@dataclass(frozen=True)
class AnnDataSummary:
    """Validated dimensions and sample information for an AnnData object."""

    n_obs: int
    n_vars: int
    n_samples: int
    sample_key: str
    layer: str | None
    is_sparse: bool


def get_matrix(adata: ad.AnnData, *, layer: str | None = None) -> Matrix:
    """Return `.X` or an explicitly selected layer without copying or densifying."""
    if not isinstance(adata, ad.AnnData):
        raise TypeError("adata must be an anndata.AnnData object")
    if layer is None:
        matrix = adata.X
    else:
        try:
            matrix = adata.layers[layer]
        except KeyError:
            raise InputError(f"layer {layer!r} is not present in adata.layers") from None
    if matrix is None:
        location = "adata.X" if layer is None else f"adata.layers[{layer!r}]"
        raise InputError(f"{location} is empty")
    return matrix


def _validate_values(matrix: Matrix) -> None:
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if values.ndim != 1 and np.asarray(matrix).ndim != 2:
        raise InputError("the selected expression matrix must be two-dimensional")
    if not np.issubdtype(values.dtype, np.number):
        raise InputError("the selected expression matrix must be numeric")
    if not np.isfinite(values).all():
        raise InputError("the selected expression matrix contains NaN or infinite values")
    if (values < 0).any():
        raise InputError("the selected expression matrix contains negative values")


def validate_anndata(
    adata: ad.AnnData,
    *,
    sample_key: str,
    layer: str | None = None,
    min_samples: int = 1,
) -> AnnDataSummary:
    """Validate an AnnData object without changing it.

    Parameters
    ----------
    adata
        Cells-by-genes annotated matrix.
    sample_key
        Column in ``adata.obs`` identifying independent biological samples.
    layer
        Expression layer to validate. ``None`` selects ``adata.X``.
    min_samples
        Minimum number of distinct samples required by the caller. Use two or
        more for cross-sample recurrence analyses.
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError("adata must be an anndata.AnnData object")
    if min_samples < 1:
        raise ValueError("min_samples must be at least 1")
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise InputError("adata must contain at least one cell and one gene")
    if not adata.obs_names.is_unique:
        raise InputError("adata.obs_names must be unique")
    if not adata.var_names.is_unique:
        raise InputError("adata.var_names must be unique")
    if sample_key not in adata.obs:
        raise InputError(f"sample key {sample_key!r} is not present in adata.obs")

    samples = adata.obs[sample_key]
    if samples.isna().any():
        raise InputError(f"adata.obs[{sample_key!r}] contains missing sample identifiers")
    normalized = samples.astype("string").str.strip()
    if normalized.eq("").any():
        raise InputError(f"adata.obs[{sample_key!r}] contains empty sample identifiers")
    n_samples = int(normalized.nunique())
    if n_samples < min_samples:
        raise InputError(
            f"at least {min_samples} biological samples are required; found {n_samples}"
        )

    matrix = get_matrix(adata, layer=layer)
    if matrix.shape != adata.shape:
        raise InputError(
            f"selected matrix shape {matrix.shape} does not match AnnData shape {adata.shape}"
        )
    _validate_values(matrix)
    return AnnDataSummary(
        n_obs=adata.n_obs,
        n_vars=adata.n_vars,
        n_samples=n_samples,
        sample_key=sample_key,
        layer=layer,
        is_sparse=sparse.issparse(matrix),
    )


def store_program_vocabulary(
    adata: ad.AnnData,
    fit: VocabularyFit,
    *,
    overwrite: bool = False,
) -> None:
    """Store a recurrent vocabulary and provenance using scverse conventions.

    Genes not used to build the vocabulary receive zero weights. Existing
    ``.varm['sccs_programs']`` or program provenance is not replaced unless
    ``overwrite=True`` is explicit.
    """
    from sccellstates.recurrence import VocabularyFit

    if not isinstance(adata, ad.AnnData):
        raise TypeError("adata must be an anndata.AnnData object")
    if not isinstance(fit, VocabularyFit):
        raise TypeError("fit must be a VocabularyFit")
    if not adata.var_names.is_unique:
        raise InputError("adata.var_names must be unique")
    if "sccs_programs" in adata.varm and not overwrite:
        raise InputError("adata.varm['sccs_programs'] already exists")
    existing = adata.uns.get("sccellstates", {})
    if not isinstance(existing, dict):
        raise InputError("adata.uns['sccellstates'] must be a dictionary")
    if "program_vocabulary" in existing and not overwrite:
        raise InputError("adata.uns['sccellstates']['program_vocabulary'] already exists")

    feature_lookup = {str(name): index for index, name in enumerate(adata.var_names)}
    missing = sorted(set(fit.vocabulary.feature_names) - feature_lookup.keys())
    if missing:
        preview = missing[:5]
        raise InputError(f"vocabulary features are absent from adata.var_names: {preview}")
    weights = np.zeros((adata.n_vars, fit.vocabulary.n_programs), dtype=np.float64)
    indices = np.fromiter(
        (feature_lookup[name] for name in fit.vocabulary.feature_names), dtype=np.int64
    )
    weights[indices, :] = fit.vocabulary.weights.T

    pairwise = {
        f"{match.reference_sample_id}__{match.query_sample_id}": {
            "reference_sample_id": match.reference_sample_id,
            "query_sample_id": match.query_sample_id,
            "optimal_mean": match.optimal_mean,
            "null_mean": match.null_mean,
            "null_adjusted_mean": match.null_adjusted_mean,
            "assignment_p_value": match.p_value,
        }
        for match in fit.pairwise_matches
    }
    members = {
        str(member.vocabulary_index): {
            "vocabulary_index": member.vocabulary_index,
            "anchor_program_index": member.anchor_program_index,
            "sample_ids": np.asarray(member.sample_ids, dtype=str),
            "program_indices": np.asarray(member.program_indices, dtype=np.int64),
            "similarities_to_anchor": np.asarray(member.similarities_to_anchor, dtype=np.float64),
        }
        for member in fit.members
    }
    sample_redundancy = {
        summary.sample_id: {
            "sample_id": summary.sample_id,
            "mean_absolute_similarity": summary.mean_absolute_similarity,
            "max_absolute_similarity": summary.max_absolute_similarity,
        }
        for summary in fit.sample_redundancy
    }
    provenance = dict(existing)
    provenance.setdefault("schema_version", "1.0")
    provenance["program_vocabulary"] = {
        "estimator": fit.vocabulary.estimator,
        "parameters": dict(fit.vocabulary.parameters),
        "training_sample_ids": list(fit.training_sample_ids),
        "reference_sample_id": fit.reference_sample_id,
        "n_programs": fit.vocabulary.n_programs,
        "n_features": fit.vocabulary.n_features,
        "members": members,
        "pairwise_matches": pairwise,
        "sample_redundancy": sample_redundancy,
        "vocabulary_redundancy": {
            "mean_absolute_similarity": (fit.vocabulary_redundancy.mean_absolute_similarity),
            "max_absolute_similarity": fit.vocabulary_redundancy.max_absolute_similarity,
        },
    }
    adata.varm["sccs_programs"] = weights
    adata.uns["sccellstates"] = provenance


def to_jsonable(value: object) -> Any:
    """Convert provenance values to JSON-serializable scalars and containers.

    Arrays are handled as well as scalars, because a value stored in ``.uns``
    and read back from HDF5 arrives as an ``ndarray`` even when it was written
    as a list.
    """
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=False) + "\n")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise InputError(f"required vocabulary file is missing: {path.name}") from None
    except json.JSONDecodeError as error:
        raise InputError(f"vocabulary file is not valid JSON: {path.name}") from error


def save_program_vocabulary(
    vocabulary: ProgramVocabulary,
    path: str | Path,
    *,
    recurrence: RecurrenceResult | None = None,
    overwrite: bool = False,
) -> Path:
    """Freeze a recurrent vocabulary to a self-contained directory.

    The vocabulary is written on its own consensus feature axis, which is the
    intersection of the per-sample axes the programs were fitted on. Because
    independently fitted samples need not share a feature axis, links back to
    the contributing programs are kept alongside: ``members.json`` records which
    sample and program index supports each vocabulary program, and
    ``feature_map.json`` records each sample's own feature axis together with
    the consensus axis. A consensus feature's position within any sample's axis
    is therefore recoverable, but is not duplicated on disk.

    This is the format to use when a vocabulary must be frozen and reloaded.
    :func:`store_program_vocabulary` writes into an ``AnnData`` object and
    cannot represent several per-sample gene axes at once.

    Parameters
    ----------
    vocabulary
        Recurrent vocabulary to freeze.
    path
        Destination directory, created if absent.
    recurrence
        Optional evidence from the same cohort. When supplied, pairwise
        matching results and their assignment nulls are stored for audit, and
        must describe the same samples as ``vocabulary``.
    overwrite
        Replace an existing directory. Without it, an existing path is an error.

    Returns
    -------
    pathlib.Path
        The directory that was written.

    Raises
    ------
    InputError
        If the path exists without ``overwrite``, or ``recurrence`` describes
        different samples than ``vocabulary``.
    TypeError
        If ``vocabulary`` or ``recurrence`` is the wrong type.
    """
    from sccellstates.recurrence import ProgramVocabulary, RecurrenceResult

    if not isinstance(vocabulary, ProgramVocabulary):
        raise TypeError("vocabulary must be a ProgramVocabulary")
    if recurrence is not None and not isinstance(recurrence, RecurrenceResult):
        raise TypeError("recurrence must be a RecurrenceResult or None")
    if recurrence is not None and (
        recurrence.sample_ids != vocabulary.training_sample_ids
        or recurrence.reference_sample_id != vocabulary.reference_sample_id
    ):
        raise InputError("recurrence describes different samples than the vocabulary")

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise InputError(f"vocabulary directory already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    programs = vocabulary.programs
    np.savez(
        destination / "programs.npz",
        weights=np.asarray(programs.weights, dtype=np.float64),
        feature_names=np.asarray(programs.feature_names, dtype=np.str_),
    )
    _write_json(
        destination / "metadata.json",
        {
            "schema_version": _VOCABULARY_SCHEMA_VERSION,
            "artifact": "program_vocabulary",
            "sample_id": programs.sample_id,
            "estimator": programs.estimator,
            "n_programs": programs.n_programs,
            "n_features": programs.n_features,
            "n_cells": programs.n_cells,
            "training_sample_ids": list(vocabulary.training_sample_ids),
            "reference_sample_id": vocabulary.reference_sample_id,
            "min_samples": vocabulary.min_samples,
            "min_similarity": vocabulary.min_similarity,
            "max_redundancy": vocabulary.max_redundancy,
            "dropped_anchor_indices": list(vocabulary.dropped_anchor_indices),
            "support_counts": list(vocabulary.support_counts),
            "vocabulary_redundancy": {
                "mean_absolute_similarity": (
                    vocabulary.vocabulary_redundancy.mean_absolute_similarity
                ),
                "max_absolute_similarity": (
                    vocabulary.vocabulary_redundancy.max_absolute_similarity
                ),
            },
        },
    )
    _write_json(
        destination / "members.json",
        {
            "members": [
                {
                    "vocabulary_index": member.vocabulary_index,
                    "anchor_program_index": member.anchor_program_index,
                    "sample_ids": list(member.sample_ids),
                    "program_indices": list(member.program_indices),
                    "similarities_to_anchor": list(member.similarities_to_anchor),
                }
                for member in vocabulary.members
            ]
        },
    )

    feature_map: dict[str, object] = {
        "consensus_features": list(vocabulary.programs.feature_names),
        "sample_features": {},
    }
    pair_records: list[dict[str, str]] = []
    evidence: dict[str, np.ndarray] = {}
    if recurrence is not None:
        feature_map["sample_features"] = {
            program_set.sample_id: list(program_set.feature_names)
            for program_set in recurrence.program_sets
        }
        for index, match in enumerate(recurrence.pairwise_matches):
            pair_records.append(
                {
                    "reference_sample_id": match.reference_sample_id,
                    "query_sample_id": match.query_sample_id,
                }
            )
            evidence[f"pair_{index}_reference_indices"] = match.reference_indices
            evidence[f"pair_{index}_query_indices"] = match.query_indices
            evidence[f"pair_{index}_similarities"] = match.similarities
            evidence[f"pair_{index}_similarity_matrix"] = match.similarity_matrix
            evidence[f"pair_{index}_null_scores"] = match.null_scores
            evidence[f"pair_{index}_p_value"] = np.asarray(match.p_value)
    _write_json(destination / "feature_map.json", feature_map)
    np.savez(destination / "recurrence.npz", **evidence)
    _write_json(
        destination / "provenance.json",
        {
            "schema_version": _VOCABULARY_SCHEMA_VERSION,
            "has_recurrence_evidence": recurrence is not None,
            "parameters": dict(vocabulary.parameters),
            "recurrence_pairs": pair_records,
            "note": (
                "No timestamp is recorded so that identical inputs produce "
                "byte-identical artifacts."
            ),
        },
    )
    return destination


def load_program_vocabulary(path: str | Path) -> ProgramVocabulary:
    """Load a vocabulary frozen by :func:`save_program_vocabulary`.

    The vocabulary-level redundancy summary is recomputed from the stored
    weights rather than read back, so it can never drift from the programs.

    Raises
    ------
    InputError
        If a required file is missing or is not valid JSON.
    """
    from sccellstates.programs import ProgramSet
    from sccellstates.recurrence import (
        ProgramVocabulary,
        VocabularyMember,
        program_redundancy,
    )

    root = Path(path)
    if not root.is_dir():
        raise InputError(f"vocabulary path is not a directory: {root}")
    metadata = _read_json(root / "metadata.json")
    provenance = _read_json(root / "provenance.json")
    members_payload = _read_json(root / "members.json")
    try:
        arrays = np.load(root / "programs.npz", allow_pickle=False)
        weights = arrays["weights"]
        feature_names = tuple(map(str, arrays["feature_names"]))
    except FileNotFoundError:
        raise InputError("required vocabulary file is missing: programs.npz") from None

    programs = ProgramSet(
        sample_id=str(metadata["sample_id"]),
        feature_names=feature_names,
        weights=weights,
        n_cells=int(metadata["n_cells"]),
        estimator=str(metadata["estimator"]),
        parameters=dict(provenance.get("parameters", {})),
    )
    members = tuple(
        VocabularyMember(
            vocabulary_index=int(entry["vocabulary_index"]),
            anchor_program_index=int(entry["anchor_program_index"]),
            sample_ids=tuple(map(str, entry["sample_ids"])),
            program_indices=tuple(int(index) for index in entry["program_indices"]),
            similarities_to_anchor=tuple(
                float(value) for value in entry["similarities_to_anchor"]
            ),
        )
        for entry in members_payload["members"]
    )
    return ProgramVocabulary(
        programs=programs,
        members=members,
        vocabulary_redundancy=program_redundancy(programs),
        training_sample_ids=tuple(map(str, metadata["training_sample_ids"])),
        reference_sample_id=str(metadata["reference_sample_id"]),
        min_samples=int(metadata["min_samples"]),
        min_similarity=float(metadata["min_similarity"]),
        max_redundancy=metadata["max_redundancy"],
        dropped_anchor_indices=tuple(int(i) for i in metadata["dropped_anchor_indices"]),
        parameters=dict(provenance.get("parameters", {})),
    )


def _program_result_envelope(adata: ad.AnnData, *, name: str) -> Mapping[str, Any]:
    """Return the stored program-result record, or explain what the file holds."""
    existing = adata.uns.get("sccellstates")
    if not isinstance(existing, Mapping):
        raise InputError(
            f"{name} carries no sccellstates metadata, so it is not a program result"
        )
    record = existing.get(_PROGRAM_RESULT_KEY)
    if record is None:
        if "program_vocabulary" in existing:
            raise InputError(
                f"{name} holds a recurrent vocabulary, not a per-sample program result; "
                "load it with load_program_vocabulary()"
            )
        raise InputError(f"{name} carries no {_PROGRAM_RESULT_KEY!r} metadata")
    if not isinstance(record, Mapping):
        raise InputError(f"{name} has a malformed {_PROGRAM_RESULT_KEY!r} record")
    if record.get("artifact") != _PROGRAM_RESULT_KEY:
        raise InputError(
            f"{name} is not a program result artifact: expected "
            f"artifact={_PROGRAM_RESULT_KEY!r}, found {record.get('artifact')!r}"
        )
    version = record.get("schema_version")
    if version not in _SUPPORTED_PROGRAM_RESULT_SCHEMAS:
        supported = ", ".join(sorted(_SUPPORTED_PROGRAM_RESULT_SCHEMAS))
        raise InputError(
            f"{name} has unsupported program result schema_version {version!r}; "
            f"this version reads {supported}"
        )
    return record


def _load_stability(
    record: object,
    *,
    name: str,
    programs: np.ndarray,
    feature_names: tuple[str, ...],
    sample_id: str,
    n_cells: int,
) -> ProgramStabilityFit | None:
    """Rebuild repeated-fit diagnostics, reusing the already-loaded weights.

    The consensus weights are stored once, in ``.varm``, and are bit-identical
    to the result's own programs, so the consensus ``ProgramSet`` is rebuilt on
    the loaded array rather than carrying a second copy on disk.
    """
    from sccellstates.programs import ProgramError, ProgramSet, ProgramStabilityFit

    if record is None:
        return None
    if not isinstance(record, Mapping):
        raise InputError(f"{name} has malformed stability diagnostics")
    try:
        consensus = ProgramSet(
            sample_id=sample_id,
            feature_names=feature_names,
            weights=programs,
            n_cells=n_cells,
            estimator=str(record["estimator"]),
            parameters=dict(record["parameters"]),
        )
        return ProgramStabilityFit(
            programs=consensus,
            run_ids=tuple(map(str, record["run_ids"])),
            anchor_run_id=str(record["anchor_run_id"]),
            matched_similarities=np.asarray(record["matched_similarities"], dtype=np.float64),
            retained_anchor_indices=np.asarray(
                record["retained_anchor_indices"], dtype=np.int64
            ),
            min_similarity=float(record["min_similarity"]),
        )
    except KeyError as error:
        raise InputError(f"{name} stability diagnostics are missing {error.args[0]!r}") from None
    except ProgramError as error:
        raise InputError(f"{name} stability diagnostics are inconsistent: {error}") from error


def save_program_result(
    result: ProgramResult,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Freeze one sample's programs and usages to a self-contained H5AD artifact.

    ``obs_names`` and ``var_names`` carry the cell and feature axes,
    ``.varm['sccs_programs']`` holds the programs-by-gene weights, and
    ``.obsm['X_sccs_programs']`` holds the cells-by-programs usages, following
    the package's scverse conventions. ``.uns['sccellstates']['program_result']``
    records the schema version, the sample ID, the fit provenance, and the
    repeated-fit stability diagnostics, so a loaded result is interchangeable
    with the one that was fitted.

    The expression matrix is never stored. This artifact is a compact result,
    not a copy of the data, and it is an exchange unit for :func:`aggregate`.
    A consequence is that it is deliberately not a valid input to
    :func:`sccellstates.fit` or :func:`sccellstates.project`, which both need
    expression values.

    ``path`` may name a directory, in which case the artifact is written to
    ``<sample_id>.h5ad`` inside it, which is what lets every sample of a cohort
    be written into one results directory.

    No timestamp and no software version is recorded, so identical inputs
    produce byte-identical artifacts within one environment.

    Parameters
    ----------
    result
        Programs and usages discovered in exactly one biological sample.
    path
        Destination ``.h5ad`` file, or a directory to name one from
        ``result.sample_id``.
    overwrite
        Replace an existing artifact. Without it, an existing path is an error.

    Returns
    -------
    pathlib.Path
        The file that was written.

    Raises
    ------
    TypeError
        If ``result`` is not a :class:`~sccellstates.ProgramResult`.
    APIError
        If cell names are not unique, a program has no positive weight, the
        provenance lacks a method or modality, or a directory destination
        cannot be derived from ``sample_id``.
    InputError
        If the destination exists without ``overwrite``.
    """
    from sccellstates.api import ProgramResult

    if not isinstance(result, ProgramResult):
        raise TypeError("result must be a ProgramResult")
    destination = _program_result_destination(result.sample_id, path)
    if destination.exists() and not overwrite:
        raise InputError(f"program result already exists: {destination}")
    # Created here, as save_program_vocabulary does, so a job can write straight
    # into a fresh results directory without a separate mkdir step.
    destination.parent.mkdir(parents=True, exist_ok=True)
    if len(set(result.cell_names)) != len(result.cell_names):
        raise InputError("program result cell names must be unique to be stored as an H5AD")
    empty = np.flatnonzero(result.programs.sum(axis=1) <= 0)
    if empty.size:
        raise InputError(
            "program result has empty programs at indices "
            f"{empty.tolist()}; every program needs at least one positive weight"
        )
    for key in ("method", "modality"):
        if not str(result.provenance.get(key, "")).strip():
            raise InputError(f"program result provenance must record a non-empty {key!r}")

    # An explicit object dtype is required, not cosmetic: a string-dtype index
    # is encoded differently depending on the pandas string-inference option, so
    # only the object dtype yields the same bytes on every machine.
    artifact = ad.AnnData(
        X=None,
        obs=pd.DataFrame(index=pd.Index(result.cell_names, dtype="object")),
        var=pd.DataFrame(index=pd.Index(result.feature_names, dtype="object")),
    )
    artifact.varm["sccs_programs"] = np.ascontiguousarray(result.programs.T)
    artifact.obsm["X_sccs_programs"] = np.ascontiguousarray(result.usages)
    artifact.uns["sccellstates"] = {
        "schema_version": _PROGRAM_RESULT_SCHEMA_VERSION,
        _PROGRAM_RESULT_KEY: {
            "schema_version": _PROGRAM_RESULT_SCHEMA_VERSION,
            "artifact": _PROGRAM_RESULT_KEY,
            "sample_id": result.sample_id,
            "selected_K": result.selected_K,
            "provenance": to_jsonable(dict(result.provenance)),
            "stability": _stability_record(result.stability),
        },
    }
    artifact.write_h5ad(destination)
    return destination


def _program_result_destination(sample_id: str, path: str | Path) -> Path:
    """Resolve a file or directory destination, refusing an escaping sample ID."""
    destination = Path(path)
    if destination.suffix.lower() in _H5AD_SUFFIXES:
        return destination
    if destination.exists() and not destination.is_dir():
        raise InputError(f"program result path is not a directory: {destination}")
    name = str(sample_id).strip()
    if not name or name in {".", ".."} or set(name) & {"/", "\\", "\x00"}:
        raise InputError(
            f"sample ID {sample_id!r} cannot name an artifact file; pass an explicit "
            ".h5ad path instead"
        )
    return destination / f"{name}.h5ad"


def _stability_record(stability: ProgramStabilityFit | None) -> dict[str, object] | None:
    """Render repeated-fit diagnostics as storable primitives."""
    if stability is None:
        return None
    return {
        "estimator": stability.programs.estimator,
        "parameters": to_jsonable(dict(stability.programs.parameters)),
        "run_ids": list(stability.run_ids),
        "anchor_run_id": stability.anchor_run_id,
        "matched_similarities": np.asarray(stability.matched_similarities, dtype=np.float64),
        "retained_anchor_indices": np.asarray(
            stability.retained_anchor_indices, dtype=np.int64
        ),
        "min_similarity": float(stability.min_similarity),
    }


def load_program_result(path: str | Path) -> ProgramResult:
    """Load a per-sample result frozen by :func:`save_program_result`.

    The cell and feature names come from ``obs_names`` and ``var_names``, and
    the returned result is interchangeable with the one that was fitted: its
    programs, usages, stability diagnostics, and provenance all round-trip.

    Raises
    ------
    InputError
        If the file is missing, unreadable, carries an expression matrix, is not
        a program result artifact, or fails the result's own invariants.
    """
    from sccellstates.api import ProgramResult

    source = Path(path)
    if not source.is_file():
        raise InputError(f"program result is not a file: {source}")
    name = source.name
    try:
        # anndata only warns about duplicate names, and this loader rejects them
        # outright below, so its warning would be a duplicate of our own error.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Observation names are not unique")
            warnings.filterwarnings("ignore", message="Variable names are not unique")
            adata = ad.read_h5ad(source)
    except (OSError, ValueError) as error:
        raise InputError(f"{name} is not a readable H5AD file: {error}") from error
    # Identify the artifact before judging its contents, so that a vocabulary
    # stored in an AnnData with its own expression matrix is reported as the
    # vocabulary it is rather than as a malformed program result.
    record = _program_result_envelope(adata, name=name)
    if adata.X is not None:
        raise InputError(
            f"{name} carries an expression matrix, so it is not a program result artifact"
        )
    if not adata.obs_names.is_unique:
        raise InputError(f"{name} has duplicate cell names")
    if not adata.var_names.is_unique:
        raise InputError(f"{name} has duplicate feature names")
    for key in (("varm", "sccs_programs"), ("obsm", "X_sccs_programs")):
        if key[1] not in getattr(adata, key[0]):
            raise InputError(f"{name} is missing {key[0]}[{key[1]!r}]")

    # Stored as features-by-programs in varm, matching the scverse convention.
    stored_weights = np.asarray(adata.varm["sccs_programs"], dtype=np.float64)
    usages = np.asarray(adata.obsm["X_sccs_programs"], dtype=np.float64)
    selected_K = int(record["selected_K"])
    if stored_weights.shape != (adata.n_vars, selected_K) or usages.shape != (
        adata.n_obs,
        selected_K,
    ):
        raise InputError(
            f"{name} declares selected_K={selected_K} but stores a "
            f"{stored_weights.shape} weight matrix and a {usages.shape} usage matrix, "
            "so they do not match its feature, cell, and program axes"
        )
    programs = stored_weights.T
    provenance = record["provenance"]
    if not isinstance(provenance, Mapping):
        raise InputError(f"{name} has malformed program result provenance")
    if "selected_K" in provenance and int(provenance["selected_K"]) != selected_K:
        raise InputError(f"{name} selected_K disagrees with its provenance")

    cell_names = tuple(map(str, adata.obs_names))
    feature_names = tuple(map(str, adata.var_names))
    # AnnData versions that omit null values from HDF5 mappings can drop the
    # explicit ``stability: None`` and ``layer: None`` fields.  Both are valid
    # states of the public result contract, so restore those defaults while
    # loading rather than making portable artifacts version-dependent.
    stability_record = record.get("stability")
    provenance = dict(provenance)
    provenance.setdefault("layer", None)
    try:
        return ProgramResult(
            programs=programs,
            usages=usages,
            cell_names=cell_names,
            feature_names=feature_names,
            sample_id=str(record["sample_id"]),
            selected_K=selected_K,
            stability=_load_stability(
                stability_record,
                name=name,
                programs=programs,
                feature_names=feature_names,
                sample_id=str(record["sample_id"]),
                n_cells=len(cell_names),
            ),
            provenance=dict(provenance),
        )
    except APIError as error:
        raise InputError(f"{name} is not a valid program result: {error}") from error
    except KeyError as error:
        raise InputError(f"{name} is missing {error.args[0]!r}") from None


def load_recurrence_evidence(path: str | Path) -> Mapping[str, Any]:
    """Read the audit evidence stored beside a frozen vocabulary.

    Returned as plain, JSON-serializable data rather than a ``RecurrenceResult``:
    rebuilding the live object would require every contributing sample's full
    program matrix, which is deliberately not duplicated here.
    """
    root = Path(path)
    metadata_files = _read_json(root / "provenance.json")
    pairs = metadata_files.get("recurrence_pairs") or []
    if not pairs:
        return {"pairs": []}
    try:
        arrays = np.load(root / "recurrence.npz", allow_pickle=False)
    except FileNotFoundError:
        raise InputError("required vocabulary file is missing: recurrence.npz") from None
    records = []
    for index, pair in enumerate(pairs):
        records.append(
            {
                "reference_sample_id": pair["reference_sample_id"],
                "query_sample_id": pair["query_sample_id"],
                "reference_indices": arrays[f"pair_{index}_reference_indices"].tolist(),
                "query_indices": arrays[f"pair_{index}_query_indices"].tolist(),
                "similarities": arrays[f"pair_{index}_similarities"].tolist(),
                "similarity_matrix": arrays[f"pair_{index}_similarity_matrix"].tolist(),
                "null_scores": arrays[f"pair_{index}_null_scores"].tolist(),
                "p_value": float(arrays[f"pair_{index}_p_value"]),
            }
        )
    return {"pairs": records}
