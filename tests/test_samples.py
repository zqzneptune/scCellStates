import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import sccellstates as sccs


def _atlas() -> ad.AnnData:
    values = np.arange(36, dtype=float).reshape(12, 3) + 1.0
    return ad.AnnData(
        X=sparse.csr_matrix(values),
        obs=pd.DataFrame(
            {"donor_id": ["d2", "d2", "d1", "d1"] * 3},
            index=[f"cell_{index}" for index in range(12)],
        ),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )


def _single(seed: int) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    values = rng.poisson(3, size=(6, 4)).astype(float)
    return ad.AnnData(
        X=sparse.csr_matrix(values),
        obs=pd.DataFrame(index=[f"c{seed}_{i}" for i in range(6)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(4)]),
    )


def test_atlas_splits_into_one_entry_per_biological_sample() -> None:
    collection = sccs.samples_from_atlas(_atlas(), sample_key="donor_id")

    assert collection.sample_ids == ("d1", "d2")
    assert collection.n_samples == 2
    assert collection.storage == "atlas"
    assert collection.sample_key == "donor_id"

    cells = [set(entry.source.obs_names) for entry in collection]
    assert len(cells[0] & cells[1]) == 0
    assert all(len(entry.source) == 6 for entry in collection)


def test_atlas_requires_a_sample_key_present_in_obs() -> None:
    with pytest.raises(sccs.InputError, match="sample key"):
        sccs.samples_from_atlas(_atlas(), sample_key="absent")
    with pytest.raises(sccs.SampleError, match="non-empty"):
        sccs.samples_from_atlas(_atlas(), sample_key="  ")


def test_atlas_requires_at_least_two_biological_samples() -> None:
    atlas = _atlas()
    single = atlas[atlas.obs["donor_id"] == "d1"].copy()
    with pytest.raises(sccs.InputError, match="at least 2 biological samples"):
        sccs.samples_from_atlas(single, sample_key="donor_id")


def test_a_single_container_resolves_to_one_sample() -> None:
    collection = sccs.samples_from_source(_single(1))

    assert collection.n_samples == 1
    assert collection.sample_ids == ("sample",)
    assert collection.storage == "single"
    assert collection.sample_key is None


def test_sources_are_named_from_mapping_keys_then_file_stems(tmp_path) -> None:
    path = tmp_path / "donor_07.h5ad"
    _single(1).write_h5ad(path)

    from_stem = sccs.samples_from_sources([path])
    assert from_stem.sample_ids == ("donor_07",)
    assert from_stem.storage == "files"

    from_key = sccs.samples_from_sources({"explicit": path})
    assert from_key.sample_ids == ("explicit",)
    assert from_key.storage == "files"


def test_one_container_can_contribute_several_samples() -> None:
    collection = sccs.samples_from_sources([_atlas()], sample_key="donor_id")

    assert collection.sample_ids == ("d1", "d2")
    assert collection.storage == "files"
    assert collection.sample_key == "donor_id"


def test_duplicate_sample_ids_are_rejected(tmp_path) -> None:
    """Two same-named files from different directories must not silently merge."""
    first = tmp_path / "cohort_a" / "donor.h5ad"
    second = tmp_path / "cohort_b" / "donor.h5ad"
    first.parent.mkdir()
    second.parent.mkdir()
    _single(1).write_h5ad(first)
    _single(2).write_h5ad(second)

    with pytest.raises(sccs.SampleError, match="must be unique"):
        sccs.samples_from_sources([first, second])

    with pytest.raises(sccs.SampleError, match="must be unique"):
        sccs.samples_from_sources([_single(1), _single(2)])


def test_mapping_keys_and_sample_key_cannot_be_combined() -> None:
    with pytest.raises(sccs.SampleError, match="cannot be combined"):
        sccs.samples_from_sources({"a": _atlas()}, sample_key="donor_id")


def test_empty_inputs_are_rejected() -> None:
    with pytest.raises(sccs.SampleError, match="at least one input"):
        sccs.samples_from_sources([])
    with pytest.raises(sccs.SampleError, match="at least one input"):
        sccs.samples_from_sources({})


def test_unsupported_paths_are_rejected(tmp_path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("not an anndata store\n")
    with pytest.raises(sccs.SampleError, match="AnnData object"):
        sccs.samples_from_source(path)


def test_resolution_does_not_copy_the_container() -> None:
    """Samples are windows onto the container, so an atlas is held once."""
    atlas = _atlas()
    collection = sccs.samples_from_atlas(atlas, sample_key="donor_id")

    assert all(entry.source is not atlas for entry in collection)
    assert sum(entry.source.n_obs for entry in collection) == atlas.n_obs
