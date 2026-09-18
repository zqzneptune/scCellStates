"""Command-line entry points for reproducible candidate pipeline runs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from sccellstates import __version__
from sccellstates.api import aggregate as aggregate_results
from sccellstates.api import compare_projectors, fit_atlas, fit_samples, project
from sccellstates.api import fit as fit_single_sample
from sccellstates.dataset import (
    filter_anndata_cells,
    select_training_genes,
)
from sccellstates.io import (
    load_program_vocabulary,
    save_program_result,
    save_program_vocabulary,
    save_projection_benchmark,
    save_state_result,
    store_program_vocabulary,
    to_jsonable,
)
from sccellstates.pipeline import PipelineConfig, fit_pipeline


def _csv_values(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _jsonable(value: object) -> object:
    return to_jsonable(value)


def _run(args: argparse.Namespace) -> int:
    sources = [Path(path) for path in args.input]
    if any(source.suffix.lower() not in {".h5ad", ".h5"} for source in sources):
        raise ValueError("--input accepts one or more H5AD files")
    datasets = [ad.read_h5ad(source) for source in sources]
    sample_key = args.sample_key
    if any(sample_key not in dataset.obs for dataset in datasets):
        raise ValueError(f"sample key {sample_key!r} is not present in every H5AD obs")
    if len(datasets) == 1:
        adata = datasets[0]
    else:
        common = datasets[0].var_names
        for dataset in datasets[1:]:
            common = common.intersection(dataset.var_names, sort=False)
        if len(common) == 0:
            raise ValueError("input H5AD files have no shared genes")
        adata = ad.concat([dataset[:, common] for dataset in datasets], join="inner")
    layer = args.layer
    adata = filter_anndata_cells(
        adata,
        layer=layer,
        sample_key=sample_key,
        min_counts_per_cell=args.min_counts_per_cell,
        min_genes_per_cell=args.min_genes_per_cell,
        max_cells_per_sample=args.max_cells_per_sample,
    )
    test_samples = _csv_values(args.test_samples)
    validation_samples = _csv_values(args.validation_samples)
    all_samples = tuple(map(str, adata.obs[sample_key].unique()))
    excluded = set(test_samples) | set(validation_samples)
    training_samples = tuple(sample for sample in all_samples if sample not in excluded)
    adata = select_training_genes(
        adata,
        sample_key=sample_key,
        training_sample_ids=training_samples,
        n_top_genes=args.n_top_genes,
        min_cells_per_gene=args.min_cells_per_gene,
        layer=layer,
        remove_mitochondrial=args.remove_mitochondrial,
        remove_ribosomal=args.remove_ribosomal,
        exclude_genes=_csv_values(args.exclude_genes),
    )
    config = PipelineConfig(
        sample_key=sample_key,
        test_samples=test_samples,
        validation_samples=validation_samples,
        layer=layer,
        preprocessing=args.preprocessing,
        n_programs=args.n_programs,
        min_samples=args.min_samples,
        min_similarity=args.min_similarity,
        n_permutations=args.n_permutations,
        score_method=args.score_method,
        state_components=args.state_components,
        state_methods=_csv_values(args.state_methods),
        max_dense_elements=args.max_dense_elements,
        random_state=args.seed,
    )
    result = fit_pipeline(adata, config)
    store_program_vocabulary(result.adata, result.vocabulary_fit, overwrite=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=args.overwrite)
    result.adata.write_h5ad(output / "candidate.h5ad")
    for name, report in result.reports.items():
        if isinstance(report, pd.DataFrame):
            report.to_csv(output / f"{name}.csv", index=False)
    metadata = {
        "input": [str(source.resolve()) for source in sources],
        "n_obs": result.adata.n_obs,
        "n_vars": result.adata.n_vars,
        "samples": all_samples,
        "qc": {
            "min_counts_per_cell": args.min_counts_per_cell,
            "min_genes_per_cell": args.min_genes_per_cell,
            "min_cells_per_gene": args.min_cells_per_gene,
            "max_cells_per_sample": args.max_cells_per_sample,
            "n_top_genes": args.n_top_genes,
            "remove_mitochondrial": args.remove_mitochondrial,
            "remove_ribosomal": args.remove_ribosomal,
            "exclude_genes": _csv_values(args.exclude_genes),
        },
        "provenance": result.provenance,
        "output": str(output.resolve()),
    }
    (output / "run.json").write_text(json.dumps(_jsonable(metadata), indent=2) + "\n")
    print(f"wrote {output / 'candidate.h5ad'}")
    return 0


def _cohort_summary(result: object) -> dict[str, object]:
    """Summarize a cohort fit for ``run.json`` without duplicating matrices."""
    vocabulary = result.vocabulary
    return {
        "provenance": result.provenance,
        "samples": [
            {
                "sample_id": entry.sample_id,
                "n_cells": len(entry.cell_names),
                "n_features": len(entry.feature_names),
                "n_programs": entry.selected_K,
            }
            for entry in result.programs
        ],
        "reference_sample_id": result.recurrence.reference_sample_id,
        "n_pairwise_matches": len(result.recurrence.pairwise_matches),
        "vocabulary": None
        if vocabulary is None
        else {
            "n_programs": vocabulary.n_programs,
            "support_counts": list(vocabulary.support_counts),
            "max_redundancy": vocabulary.max_redundancy,
            "dropped_anchor_indices": list(vocabulary.dropped_anchor_indices),
            "mean_absolute_similarity": (vocabulary.vocabulary_redundancy.mean_absolute_similarity),
            "max_absolute_similarity": vocabulary.vocabulary_redundancy.max_absolute_similarity,
        },
    }


def _write_cohort(result: object, output: Path, *, overwrite: bool) -> None:
    """Write a cohort summary and every per-sample result.

    Each sample's portable result is written under ``samples/`` as well, so a
    local cohort run leaves behind exactly the artifacts a distributed run
    would, and can be re-aggregated or inspected donor by donor.
    """
    output.mkdir(parents=True, exist_ok=overwrite)
    (output / "run.json").write_text(
        json.dumps(_jsonable(_cohort_summary(result)), indent=2) + "\n"
    )
    samples_dir = output / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    for program_result in result.programs:
        save_program_result(program_result, samples_dir, overwrite=True)
    message = f"wrote {output / 'run.json'}"
    message += f"\nwrote {len(result.programs)} per-sample result(s) to {samples_dir}"
    if result.vocabulary is None:
        message += "\nno program recurred in enough independent samples"
    else:
        destination = save_program_vocabulary(
            result.vocabulary,
            output / "vocabulary",
            recurrence=result.recurrence,
            overwrite=True,
        )
        message += f"\nwrote {destination}"
        message += f" with {result.vocabulary.n_programs} recurrent program(s)"
    print(message)


def _fit_options(args: argparse.Namespace) -> dict[str, object]:
    """Fit settings shared by every input shape the ``fit`` command accepts.

    ``fit``, ``fit_atlas``, and ``fit_samples`` all accept these names, so one
    mapping forwards them to whichever entry point the inputs select.
    """
    return {
        "max_iter": args.max_iter,
        "tol": args.tol,
        "stability_threshold": args.stability_threshold,
        "min_counts_per_cell": args.min_counts_per_cell,
        "min_genes_per_cell": args.min_genes_per_cell,
        "min_cells_per_gene": args.min_cells_per_gene,
        "n_top_genes": args.n_top_genes,
        "remove_mitochondrial": args.remove_mitochondrial,
        "remove_ribosomal": args.remove_ribosomal,
        "exclude_genes": _csv_values(args.exclude_genes),
        "n_jobs": args.n_jobs,
    }


def _fit(args: argparse.Namespace) -> int:
    """Run program discovery for one sample, one atlas, or several containers."""
    sources = [Path(path) for path in args.input]
    options = _fit_options(args)
    if args.sample_key is None and len(sources) > 1:
        if args.sample_id is not None:
            raise ValueError("--sample-id applies to a single input; drop it or use --sample-key")
        shared = {
            "modality": "rna",
            "method": args.method,
            "layer": args.layer,
            "n_programs": args.n_programs,
            "n_repeats": args.n_repeats,
            "preprocessing": args.preprocessing,
            "min_samples": args.min_samples,
            "min_similarity": args.min_similarity,
            "n_permutations": args.n_permutations,
            "random_state": args.seed,
            **options,
        }
        result = fit_samples(sources, **shared)
        _write_cohort(result, Path(args.output), overwrite=args.overwrite)
        return 0

    if args.sample_key is not None:
        shared = {
            "sample_key": args.sample_key,
            "modality": "rna",
            "method": args.method,
            "layer": args.layer,
            "n_programs": args.n_programs,
            "n_repeats": args.n_repeats,
            "preprocessing": args.preprocessing,
            "min_samples": args.min_samples,
            "min_similarity": args.min_similarity,
            "n_permutations": args.n_permutations,
            "random_state": args.seed,
            **options,
        }
        if len(sources) == 1:
            result = fit_atlas(sources[0], **shared)
        else:
            result = fit_samples(sources, **shared)
        _write_cohort(result, Path(args.output), overwrite=args.overwrite)
        return 0

    if args.sample_id is None:
        raise ValueError("--sample-id is required when --sample-key is not given")
    result = fit_single_sample(
        args.input[0],
        modality="rna",
        method=args.method,
        sample_id=args.sample_id,
        layer=args.layer,
        n_programs=args.n_programs,
        n_repeats=args.n_repeats,
        preprocessing=args.preprocessing,
        random_state=args.seed,
        **options,
    )
    output = Path(args.output)
    # Independent jobs for different samples share one results directory, so
    # writing into an existing directory is the normal case rather than a
    # clobber. Only re-running the *same* sample needs --overwrite.
    output.mkdir(parents=True, exist_ok=True)
    artifact = save_program_result(result, output, overwrite=args.overwrite)
    # Named per sample so a shared directory accumulates one summary per donor
    # instead of every run overwriting a single run.json.
    (output / f"{result.sample_id}.json").write_text(
        json.dumps(_jsonable(result.provenance), indent=2) + "\n"
    )
    print(f"wrote {artifact}")
    return 0


def _aggregate(args: argparse.Namespace) -> int:
    """Combine independently fitted per-sample results into a vocabulary."""
    result = aggregate_results(
        Path(args.input),
        min_samples=args.min_samples,
        min_similarity=args.min_similarity,
        n_permutations=args.n_permutations,
        max_redundancy=args.max_redundancy,
        random_state=args.seed,
    )
    _write_cohort(result, Path(args.output), overwrite=args.overwrite)
    return 0


def _projector_options(args: argparse.Namespace, method: str) -> dict[str, object]:
    """Translate CLI projector flags into method-specific constructor options."""
    if method == "regularized_nnls":
        return {"l1": args.l1, "l2": args.l2}
    if method == "poisson":
        return {"max_iter": args.max_iter, "tol": args.tol}
    return {}


def _project(args: argparse.Namespace) -> int:
    vocabulary = load_program_vocabulary(args.vocabulary)
    method = args.method
    result = project(
        Path(args.input),
        vocabulary,
        layer=args.layer,
        preprocessing=args.preprocessing,
        method=method,
        sample_id=args.sample_id or Path(args.input).stem,
        **_projector_options(args, method),
    )
    destination = save_state_result(result, args.output, overwrite=args.overwrite)
    print(f"wrote {destination}")
    return 0


def _benchmark_table(benchmark: object) -> pd.DataFrame:
    """Render stable method-comparable fields for a TSV report."""
    rows = []
    for method in benchmark.methods:
        for result in benchmark.results[method]:
            rows.append(
                {
                    "method": method,
                    "sample": result.sample_id,
                    "projection_error": float(result.projection_error.mean()),
                    "state_sparsity": float(np.mean(result.normalized_states == 0)),
                    "feature_coverage": result.feature_coverage,
                    "success": True,
                }
            )
    return pd.DataFrame(rows)


def _compare(args: argparse.Namespace) -> int:
    vocabulary = load_program_vocabulary(args.vocabulary)
    methods = _csv_values(args.methods)
    options = {method: _projector_options(args, method) for method in methods}
    samples = {Path(path).stem: Path(path) for path in args.input}
    benchmark = compare_projectors(
        samples,
        vocabulary,
        methods=methods,
        layer=args.layer,
        preprocessing=args.preprocessing,
        projector_options=options,
    )
    destination = save_projection_benchmark(benchmark, args.output, overwrite=args.overwrite)
    _benchmark_table(benchmark).to_csv(destination / "comparison.tsv", sep="\t", index=False)
    print(f"wrote {destination}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sccellstates")
    # Read from the package rather than a literal: a copy here silently goes
    # stale the moment the version is bumped.
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")
    run = subparsers.add_parser("run", help="run a candidate pipeline on H5AD input")
    run.add_argument("--input", required=True, nargs="+", help="one or more H5AD files")
    run.add_argument("--output", required=True, help="dedicated output directory")
    run.add_argument("--sample-key", required=True, help="H5AD obs column for biological samples")
    run.add_argument("--layer", help="H5AD expression layer; default: X")
    run.add_argument("--validation-samples", default="", help="comma-separated sample IDs")
    run.add_argument("--test-samples", required=True, help="comma-separated sample IDs")
    run.add_argument("--min-counts-per-cell", type=float, default=0)
    run.add_argument("--min-genes-per-cell", type=int, default=0)
    run.add_argument("--min-cells-per-gene", type=int, default=3)
    run.add_argument("--remove-mitochondrial", action="store_true")
    run.add_argument("--remove-ribosomal", action="store_true")
    run.add_argument("--exclude-genes", default="", help="comma-separated gene IDs or symbols")
    run.add_argument("--max-cells-per-sample", type=int)
    run.add_argument("--n-top-genes", type=int, default=2_000)
    run.add_argument(
        "--preprocessing", choices=("identity", "library_size_log1p"), default="library_size_log1p"
    )
    run.add_argument("--n-programs", type=int, default=8)
    run.add_argument("--min-samples", type=int, default=2)
    run.add_argument("--min-similarity", type=float, default=0.3)
    run.add_argument("--n-permutations", type=int, default=100)
    run.add_argument("--score-method", default="direct")
    run.add_argument("--state-components", type=int, default=2)
    run.add_argument("--state-methods", default="direct,linear,pca")
    run.add_argument("--max-dense-elements", type=int, default=10_000_000)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--overwrite", action="store_true")
    run.set_defaults(handler=_run)
    fit = subparsers.add_parser(
        "fit",
        help="discover programs in one sample, one atlas, or several H5AD files",
        description=(
            "Discover programs independently within each biological sample, then "
            "evaluate recurrence across them. One input without --sample-key is a "
            "single sample; one input with --sample-key is an atlas whose donors are "
            "fitted separately; several inputs are several samples."
        ),
    )
    fit.add_argument("--input", required=True, nargs="+", help="one or more H5AD files")
    fit.add_argument("--output", required=True, help="dedicated output directory")
    fit.add_argument(
        "--sample-id",
        help="identifier for a single sample; required unless --sample-key is given",
    )
    fit.add_argument(
        "--sample-key",
        help="H5AD obs column naming biological samples; fits each one independently",
    )
    fit.add_argument("--method", choices=("cnmf", "nmf"), default="cnmf")
    fit.add_argument("--layer", help="H5AD expression layer; default: X")
    fit.add_argument("--n-programs", type=int, default=8)
    fit.add_argument("--n-repeats", type=int, default=3)
    fit.add_argument(
        "--preprocessing",
        choices=("identity", "library_size_log1p"),
        default="library_size_log1p",
    )
    fit.add_argument("--min-samples", type=int, default=2, help="samples supporting a program")
    fit.add_argument("--min-similarity", type=float, default=0.3)
    fit.add_argument("--n-permutations", type=int, default=1_000)
    # Fit settings. Defaults mirror fit() exactly, so adding these flags cannot
    # change what an existing invocation fits.
    fit.add_argument("--max-iter", type=int, default=500)
    fit.add_argument("--tol", type=float, default=1e-4)
    fit.add_argument("--stability-threshold", type=float, default=0.5)
    fit.add_argument("--min-counts-per-cell", type=float, default=0)
    fit.add_argument("--min-genes-per-cell", type=int, default=0)
    fit.add_argument("--min-cells-per-gene", type=int, default=0)
    fit.add_argument(
        "--n-top-genes",
        type=int,
        default=None,
        help="keep this many most-variable genes; default: every gene",
    )
    fit.add_argument(
        "--remove-mitochondrial",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="drop mitochondrial genes (--no-remove-mitochondrial to keep them)",
    )
    fit.add_argument(
        "--remove-ribosomal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="drop ribosomal genes (--no-remove-ribosomal to keep them)",
    )
    fit.add_argument("--exclude-genes", default="", help="comma-separated gene IDs or symbols")
    fit.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="worker processes for repeated and per-sample fits; 1 disables",
    )
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--overwrite", action="store_true")
    fit.set_defaults(handler=_fit)
    aggregate = subparsers.add_parser(
        "aggregate",
        help="combine independently fitted per-sample results",
        description=(
            "Load per-sample program results written by 'fit' and evaluate "
            "recurrence across them. The same reduction runs here as in a single "
            "'fit' over an atlas, so fitting each sample on its own machine and "
            "aggregating is equivalent to fitting them together. Samples are "
            "named from the metadata inside each result, never from file names."
        ),
    )
    aggregate.add_argument(
        "--input",
        required=True,
        help="directory of .h5ad program results, or one result file",
    )
    aggregate.add_argument("--output", required=True, help="dedicated output directory")
    aggregate.add_argument(
        "--min-samples", type=int, default=2, help="samples supporting a program"
    )
    aggregate.add_argument("--min-similarity", type=float, default=0.3)
    aggregate.add_argument("--n-permutations", type=int, default=1_000)
    aggregate.add_argument(
        "--max-redundancy",
        type=float,
        help="ceiling on similarity between two vocabulary programs; default: report only",
    )
    aggregate.add_argument("--seed", type=int, default=0)
    aggregate.add_argument("--overwrite", action="store_true")
    aggregate.set_defaults(handler=_aggregate)
    project_parser = subparsers.add_parser(
        "project", help="project one expression matrix onto a frozen vocabulary"
    )
    project_parser.add_argument("--input", required=True, help="AnnData or native 10x RNA input")
    project_parser.add_argument("--vocabulary", required=True, help="saved vocabulary directory")
    project_parser.add_argument("--output", required=True, help="state-result output directory")
    project_parser.add_argument("--sample-id")
    project_parser.add_argument(
        "--method", choices=("nnls", "regularized_nnls", "simplex", "poisson"), default="nnls"
    )
    project_parser.add_argument("--layer")
    project_parser.add_argument(
        "--preprocessing", choices=("identity", "library_size_log1p"), default="identity"
    )
    project_parser.add_argument("--l1", type=float, default=0.0)
    project_parser.add_argument("--l2", type=float, default=0.01)
    project_parser.add_argument("--max-iter", type=int, default=1_000)
    project_parser.add_argument("--tol", type=float, default=1e-8)
    project_parser.add_argument("--overwrite", action="store_true")
    project_parser.set_defaults(handler=_project)
    compare = subparsers.add_parser(
        "compare", help="compare fixed-vocabulary projectors on expression matrices"
    )
    compare.add_argument(
        "--input", required=True, nargs="+", help="AnnData or native 10x RNA inputs"
    )
    compare.add_argument("--vocabulary", required=True, help="saved vocabulary directory")
    compare.add_argument("--output", required=True, help="benchmark output directory")
    compare.add_argument("--methods", default="nnls,regularized_nnls,simplex,poisson")
    compare.add_argument("--layer")
    compare.add_argument(
        "--preprocessing", choices=("identity", "library_size_log1p"), default="identity"
    )
    compare.add_argument("--l1", type=float, default=0.0)
    compare.add_argument("--l2", type=float, default=0.01)
    compare.add_argument("--max-iter", type=int, default=1_000)
    compare.add_argument("--tol", type=float, default=1e-8)
    compare.add_argument("--overwrite", action="store_true")
    compare.set_defaults(handler=_compare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the package command line interface."""
    parser = _parser()
    args = parser.parse_args(argv)
    if hasattr(args, "handler"):
        return args.handler(args)
    parser.print_help()
    return 0
