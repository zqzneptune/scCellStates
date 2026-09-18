import json

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

import sccellstates as sccs
from sccellstates.cli import main


def test_run_accepts_atlas_h5ad_and_writes_dedicated_results(tmp_path) -> None:
    values = np.arange(48, dtype=float).reshape(8, 6) + 1
    atlas = ad.AnnData(
        X=sparse.csr_matrix(values),
        obs=pd.DataFrame(
            {"donor": ["a", "a", "b", "b", "c", "c", "held", "held"]},
            index=[f"cell_{i}" for i in range(8)],
        ),
        var=pd.DataFrame(index=["MT-G0", "G1", "G2", "RPS3", "G4", "G5"]),
    )
    input_path = tmp_path / "atlas.h5ad"
    output_path = tmp_path / "results" / "run_a"
    atlas.write_h5ad(input_path)

    assert (
        main(
            [
                "run",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--sample-key",
                "donor",
                "--validation-samples",
                "b",
                "--test-samples",
                "held",
                "--n-top-genes",
                "4",
                "--n-programs",
                "2",
                "--n-permutations",
                "5",
                "--state-methods",
                "direct",
                "--remove-mitochondrial",
                "--remove-ribosomal",
            ]
        )
        == 0
    )
    assert (output_path / "candidate.h5ad").exists()
    assert (output_path / "run.json").exists()


def _cohort_atlas() -> ad.AnnData:
    rng = np.random.default_rng(0)
    n_cells = 10
    shared = rng.integers(1, 30, size=(n_cells, 4))
    rows = []
    labels = []
    for donor, (low, high) in {"d1": (4, 7), "d2": (7, 10), "d3": (9, 12)}.items():
        values = np.full((n_cells, 12), 5.0)
        values[:, 0:4] = shared
        values[:, low:high] = rng.integers(1, 25, size=(n_cells, high - low))
        rows.append(values)
        labels.extend([donor] * n_cells)
    return ad.AnnData(
        X=sparse.csr_matrix(np.vstack(rows)),
        obs=pd.DataFrame({"donor": labels}, index=[f"cell_{i}" for i in range(n_cells * 3)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(12)]),
    )


def test_fit_writes_programs_for_one_sample(tmp_path) -> None:
    input_path = tmp_path / "sample.h5ad"
    output_path = tmp_path / "results" / "single"
    _cohort_atlas().write_h5ad(input_path)

    assert (
        main(
            [
                "fit",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--sample-id",
                "donor_x",
                "--n-programs",
                "2",
                "--n-repeats",
                "1",
                "--preprocessing",
                "identity",
            ]
        )
        == 0
    )
    assert (output_path / "donor_x.h5ad").exists()
    # The summary is named per sample so independent jobs can share a directory.
    assert (output_path / "donor_x.json").exists()
    # The artifact is the portable program result, not a copy of the input.
    loaded = sccs.load_program_result(output_path / "donor_x.h5ad")
    assert loaded.sample_id == "donor_x"
    assert loaded.selected_K == 2


def test_fit_fits_an_atlas_per_sample_and_records_no_pooling(tmp_path) -> None:
    input_path = tmp_path / "atlas.h5ad"
    output_path = tmp_path / "results" / "atlas"
    _cohort_atlas().write_h5ad(input_path)

    assert (
        main(
            [
                "fit",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--sample-key",
                "donor",
                "--n-programs",
                "3",
                "--n-repeats",
                "1",
                "--preprocessing",
                "identity",
                "--n-permutations",
                "20",
            ]
        )
        == 0
    )

    summary = json.loads((output_path / "run.json").read_text())
    assert summary["provenance"]["pooled_fit"] is False
    assert summary["provenance"]["fit_independence"] == "per_sample"
    assert summary["provenance"]["sample_ids"] == ["d1", "d2", "d3"]
    assert [entry["sample_id"] for entry in summary["samples"]] == ["d1", "d2", "d3"]
    assert all(entry["n_cells"] < 30 for entry in summary["samples"])
    assert summary["n_pairwise_matches"] == 3


def test_fit_reads_several_files_as_separate_samples(tmp_path) -> None:
    atlas = _cohort_atlas()
    paths = []
    for donor in ("d1", "d2"):
        path = tmp_path / f"{donor}.h5ad"
        atlas[atlas.obs["donor"] == donor].copy().write_h5ad(path)
        paths.append(path)
    output_path = tmp_path / "results" / "files"

    assert (
        main(
            [
                "fit",
                "--input",
                *(str(path) for path in paths),
                "--output",
                str(output_path),
                "--n-programs",
                "2",
                "--n-repeats",
                "1",
                "--preprocessing",
                "identity",
                "--n-permutations",
                "20",
            ]
        )
        == 0
    )

    summary = json.loads((output_path / "run.json").read_text())
    assert summary["provenance"]["storage"] == "files"
    assert summary["provenance"]["sample_ids"] == ["d1", "d2"]


def _write_donor_files(atlas: ad.AnnData, directory) -> list:
    """Split one atlas into one H5AD per donor, as a distributed run would."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for donor in sorted(atlas.obs["donor"].unique()):
        path = directory / f"{donor}.h5ad"
        atlas[atlas.obs["donor"] == donor].copy().write_h5ad(path)
        paths.append(path)
    return paths


def test_fit_writes_portable_results_for_every_donor_of_an_atlas(tmp_path) -> None:
    input_path = tmp_path / "atlas.h5ad"
    output_path = tmp_path / "results" / "atlas"
    _cohort_atlas().write_h5ad(input_path)

    assert (
        main(
            [
                "fit",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--sample-key",
                "donor",
                "--n-programs",
                "2",
                "--n-repeats",
                "1",
                "--preprocessing",
                "identity",
                "--n-permutations",
                "5",
            ]
        )
        == 0
    )
    # A local cohort run leaves the same per-sample artifacts a distributed one does.
    for donor in ("d1", "d2", "d3"):
        artifact = output_path / "samples" / f"{donor}.h5ad"
        assert artifact.exists()
        assert sccs.load_program_result(artifact).sample_id == donor


def test_aggregate_combines_independently_fitted_samples(tmp_path) -> None:
    donor_dir = tmp_path / "donors"
    results = tmp_path / "results"
    model = tmp_path / "model"
    for path in _write_donor_files(_cohort_atlas(), donor_dir):
        assert (
            main(
                [
                    "fit",
                    "--input",
                    str(path),
                    "--output",
                    str(results),
                    "--sample-id",
                    path.stem,
                    "--n-programs",
                    "2",
                    "--n-repeats",
                    "1",
                    "--preprocessing",
                    "identity",
                ]
            )
            == 0
        )

    assert (
        main(
            [
                "aggregate",
                "--input",
                str(results),
                "--output",
                str(model),
                "--min-samples",
                "2",
                "--n-permutations",
                "5",
            ]
        )
        == 0
    )

    assert (model / "run.json").exists()
    summary = json.loads((model / "run.json").read_text())
    assert summary["provenance"]["workflow"] == "cohort_program_aggregation"
    assert summary["provenance"]["pooled_fit"] is False
    assert summary["provenance"]["sample_ids"] == ["d1", "d2", "d3"]
    # Aggregating re-emits the per-sample artifacts it consumed.
    assert (model / "samples" / "d1.h5ad").exists()
