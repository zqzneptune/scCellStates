"""Resolve storage layouts into explicit biological sample boundaries.

Single-cell datasets arrive in several physical arrangements: one file holding
one sample, one atlas file holding many donors, or many files each holding one
donor. Those arrangements are a storage detail. The scientific unit is the
biological sample, and every input mode in this package is normalized here into
the same :class:`SampleCollection` so that later stages never need to know how
the data were stored.

Resolving a collection does not read matrices into memory beyond what the
:mod:`anndata` slicing requires, and it never copies cells. Each sample is
materialized only when it is fitted.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import pandas as pd

from sccellstates.io import APIError, validate_anndata

type Source = str | Path | ad.AnnData

_H5AD_SUFFIXES = frozenset({".h5ad", ".h5"})
_STORAGE_KINDS = frozenset({"single", "atlas", "files"})


class SampleError(APIError):
    """Raised when inputs cannot be resolved into biological samples.

    A subclass of :class:`sccellstates.io.APIError`, so a caller that catches
    the public workflow error catches sample-resolution failures too.
    """


@dataclass(frozen=True)
class SampleEntry:
    """One biological sample and the container it was resolved from.

    ``source`` is either a path or an ``AnnData`` object, which may be a view
    onto a larger container. Callers that mutate must copy first; the fitting
    entry points already do.
    """

    sample_id: str
    source: Source

    def __post_init__(self) -> None:
        sample_id = str(self.sample_id).strip()
        if not sample_id:
            raise SampleError("sample_id must be a non-empty string")
        object.__setattr__(self, "sample_id", sample_id)


@dataclass(frozen=True)
class SampleCollection:
    """Biological samples resolved from one or more storage containers.

    The ``storage`` field records how the samples happened to be stored. It is
    provenance only and must never change how the samples are analysed: a
    collection of donors sliced from one atlas file and a collection of donors
    read from separate files carry identical meaning and are fitted
    independently either way.

    Installing this collection is explicitly *not* a claim that its samples are
    comparable, recurrent, or biologically related. Establishing recurrence is
    the job of :func:`sccellstates.recurrence.compute_recurrence`.
    """

    samples: tuple[SampleEntry, ...]
    storage: str
    sample_key: str | None = None

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        if not samples:
            raise SampleError("a SampleCollection must contain at least one sample")
        if self.storage not in _STORAGE_KINDS:
            raise SampleError(f"storage must be one of {sorted(_STORAGE_KINDS)}")
        sample_ids = [entry.sample_id for entry in samples]
        duplicates = sorted({name for name in sample_ids if sample_ids.count(name) > 1})
        if duplicates:
            raise SampleError(
                f"sample IDs must be unique across all inputs, but these repeat: {duplicates}. "
                "Samples are named from an explicit sample_key, from mapping keys, or from file "
                "stems; rename the inputs or pass a mapping of sample_id to source."
            )
        object.__setattr__(self, "samples", samples)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        """Identifiers of the biological samples, in resolution order."""
        return tuple(entry.sample_id for entry in self.samples)

    @property
    def n_samples(self) -> int:
        """Number of independent biological samples."""
        return len(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[SampleEntry]:
        return iter(self.samples)


def read_source(source: Source) -> ad.AnnData:
    """Load a path or pass through an ``AnnData`` object, without copying.

    ``.zarr`` stores are read through the optional ``zarr`` dependency. Zarr is
    not a core requirement, so install it explicitly or load the store yourself
    and pass the resulting ``AnnData`` object.
    """
    if isinstance(source, ad.AnnData):
        return source
    path = Path(source)
    suffix = path.suffix.lower()
    if suffix in _H5AD_SUFFIXES:
        return ad.read_h5ad(path)
    if suffix == ".zarr":
        try:
            return ad.read_zarr(path)
        except ImportError as error:
            raise SampleError(
                "reading a .zarr store requires the optional 'zarr' dependency; install it "
                "or load the store and pass an AnnData object instead"
            ) from error
    raise SampleError("input must be an AnnData object or an .h5ad, .h5, or .zarr store")


def _split_by_sample_key(
    source: Source,
    *,
    sample_key: str,
    layer: str | None,
) -> tuple[SampleEntry, ...]:
    adata = read_source(source)
    validate_anndata(adata, sample_key=sample_key, layer=layer, min_samples=1)
    labels = adata.obs[sample_key].astype("string").str.strip()
    entries = []
    for value in sorted(pd.unique(labels.dropna())):
        sample_id = str(value)
        mask = labels.eq(sample_id).to_numpy()
        entries.append(SampleEntry(sample_id=sample_id, source=adata[mask]))
    return tuple(entries)


def samples_from_source(source: Source) -> SampleCollection:
    """Resolve one container that holds exactly one biological sample."""
    return SampleCollection(
        samples=(SampleEntry(sample_id="sample", source=read_source(source)),),
        storage="single",
        sample_key=None,
    )


def samples_from_atlas(
    source: Source,
    *,
    sample_key: str,
    layer: str | None = None,
) -> SampleCollection:
    """Resolve one container into one entry per biological sample.

    ``sample_key`` is required and must name a column in ``.obs``. The
    container is sliced, not pooled: each returned entry is a view holding only
    the cells of one sample.
    """
    key = str(sample_key).strip()
    if not key:
        raise SampleError("sample_key must be a non-empty string")
    adata = read_source(source)
    validate_anndata(adata, sample_key=key, layer=layer, min_samples=2)
    labels = adata.obs[key].astype("string").str.strip()
    entries = []
    for value in sorted(pd.unique(labels.dropna())):
        sample_id = str(value)
        mask = labels.eq(sample_id).to_numpy()
        entries.append(SampleEntry(sample_id=sample_id, source=adata[mask]))
    return SampleCollection(samples=tuple(entries), storage="atlas", sample_key=key)


def samples_from_sources(
    sources: Mapping[str, Source] | Sequence[Source],
    *,
    sample_key: str | None = None,
    layer: str | None = None,
) -> SampleCollection:
    """Resolve several containers into biological samples.

    A mapping names each sample explicitly through its keys. A sequence names
    each sample from the file stem, or from ``sample_key`` when one is given,
    in which case a single container may contribute several samples.

    Parameters
    ----------
    sources
        Mapping of sample ID to source, or sequence of sources.
    sample_key
        Optional ``.obs`` column naming biological samples inside each
        container. When supplied, sample IDs come from the data rather than
        from file names, and a mapping may not be used because its keys would
        conflict with the column.
    layer
        Expression layer used to validate each container. ``None`` selects
        ``adata.X``.

    Raises
    ------
    SampleError
        If the sequence is empty, a container yields no samples, ``sample_key``
        is combined with a mapping, or two inputs resolve to the same sample ID.
    """
    key = None if sample_key is None else str(sample_key).strip()
    if sample_key is not None and not key:
        raise SampleError("sample_key must be a non-empty string")

    if isinstance(sources, Mapping):
        if key is not None:
            raise SampleError(
                "sample_key cannot be combined with a mapping of sample IDs to sources; "
                "pass a sequence of sources when samples are named by an .obs column"
            )
        if not sources:
            raise SampleError("at least one input source is required")
        entries = []
        for sample_id, source in sources.items():
            name = str(sample_id).strip()
            if not name:
                raise SampleError("mapping keys must be non-empty sample IDs")
            entries.append(SampleEntry(sample_id=name, source=read_source(source)))
        return SampleCollection(samples=tuple(entries), storage="files", sample_key=None)

    items = tuple(sources)
    if not items:
        raise SampleError("at least one input source is required")
    if key is not None:
        entries = []
        for source in items:
            entries.extend(_split_by_sample_key(source, sample_key=key, layer=layer))
        if not entries:
            raise SampleError("the inputs contain no biological samples")
        return SampleCollection(samples=tuple(entries), storage="files", sample_key=key)

    entries = []
    for source in items:
        if isinstance(source, ad.AnnData):
            name = "sample"
        else:
            name = Path(source).stem
        if not name:
            raise SampleError(f"cannot derive a sample ID from input {source!r}")
        entries.append(SampleEntry(sample_id=name, source=read_source(source)))
    return SampleCollection(samples=tuple(entries), storage="files", sample_key=None)
