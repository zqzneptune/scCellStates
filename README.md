# scCellStates

[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

`sccellstates` is a Python package for discovering reproducible, continuous
cell-state representations across biological samples in single-cell genomics
datasets.

Cell states are represented as continuous configurations of biological
programs rather than discrete labels. Program discovery is performed within
samples before recurrence is evaluated across samples, helping distinguish
shared biological structure from donor- or batch-specific effects.

## Installation

Install the package with pip:

```bash
python -m pip install sccellstates
```

Python 3.12 or later is required.

Optional integrations can be installed as extras:

```bash
python -m pip install "sccellstates[scanpy]"
python -m pip install "sccellstates[multimodal]"
```

## AnnData input

AnnData is the canonical single-modality container. Observations are cells,
variables are genes, and an `.obs` column identifies independent biological
samples.

```python
import sccellstates as sccs

summary = sccs.validate_anndata(
    adata,
    sample_key="donor_id",
    layer="counts",
    min_samples=2,
)

counts = sccs.get_matrix(adata, layer="counts")
print(summary.n_samples)
```

Validation is non-mutating and checks:

- non-empty cells and genes;
- unique observation and variable names;
- complete, non-empty biological sample identifiers;
- numeric, finite, non-negative expression values;
- the requested minimum number of biological samples; and
- explicit matrix-layer selection.

Sparse matrices remain sparse. The package does not silently normalize data,
infer that `.raw` contains counts, alter names, or overwrite `.X`.

## Program discovery: one sample, an atlas, or many files

Programs are always fitted independently within biological samples before
recurrence is evaluated across them. How the data happen to be stored does not
change that. One file holding one donor, one atlas file holding a hundred
donors, and a hundred files each holding one donor are the same analysis.

### One sample

```python
result = sccs.fit(
    "sample.h5ad",
    modality="rna",
    method="cnmf",
    sample_id="donor_01",
)

result.loadings       # program x gene array
result.usages         # cell x program array
result.stability      # repeated-fit diagnostics, when requested
result.selected_K     # number of returned programs
```

The result holds **candidate programs**. One sample cannot establish that a
program recurs in anything else, so `fit()` never returns a recurrent
vocabulary.

Passing `sample_key` asserts that the input holds exactly one sample. If the
column names more than one, `fit()` raises and points at the cohort entry
points rather than pooling donors into a single decomposition.

### One atlas file with many donors

```python
result = sccs.fit_atlas(
    "atlas.h5ad",
    sample_key="donor_id",
    layer="counts",
    modality="rna",
    method="cnmf",
    n_programs=8,
    min_samples=3,
)

result.programs     # ProgramCollection: one entry per donor
result.recurrence   # RecurrenceResult: pairwise matches and assignment nulls
result.vocabulary   # ProgramVocabulary, or None when nothing recurred
```

`sample_key` defines the biological sample boundary, and recurrence is
evaluated across that boundary. The atlas is a storage container only. Each
donor is fitted on its own cells with its own gene selection and its own
preprocessing, and donors are never concatenated into a pooled fit. The same
input always returns `pooled_fit: false` in its provenance.

### Many files

```python
result = sccs.fit_samples(
    ["donor_01.h5ad", "donor_02.h5ad", "donor_03.h5ad"],
    modality="rna",
    method="cnmf",
)
```

A mapping names samples explicitly through its keys; a sequence names them from
file stems. A `sample_key` lets one file contribute several samples, so a
directory of multi-donor files works the same way:

```python
result = sccs.fit_samples(files, sample_key="donor_id")
```

Sample identifiers must be unique across all inputs, so two files named
`donor_01.h5ad` from different directories are reported rather than merged.

### Shared feature coverage

Each sample selects its own genes, so independently fitted samples need not
share a feature axis. Recurrence is measured on the intersection, and every
cohort result records how much of each sample survived into it:

```python
cohort.provenance["feature_overlap"]
# {"n_shared_features": 1843,
#  "minimum_pairwise_shared": 1790,
#  "n_features_by_sample": {"donor_1": 2000, ...},
#  "retention_by_sample": {"donor_1": 0.92, ...}}
```

Read this before trusting a recurrence result. If the shared axis shrinks far
below the per-sample gene counts, the comparison is being made on a small
fraction of each donor's programs. When the intersection falls below two genes
comparison is impossible, and the call raises with those counts rather than
quietly falling back to selecting genes across the whole cohort, which would
weaken the independence the comparison depends on.

All three modes resolve to the same internal representation and the same two
steps: `compute_recurrence()` measures how programs recur across samples, then
`build_vocabulary()` selects those that qualify. Call those directly when you
want the recurrence evidence even though nothing qualifies, because
`find_recurrent_programs()` composes them and raises when nothing recurs.

`project()` scores cells against a frozen vocabulary and accepts a `ProgramSet`,
a `VocabularyFit`, a `ProgramVocabulary`, or a `CohortResult` directly. The
lower-level estimator and pipeline APIs remain available for advanced
workflows.

## Two meanings of stability

The package keeps apart two claims that are easy to conflate.

**Within-sample algorithmic stability** asks whether the decomposition of one
sample is reproducible: across random seeds, NMF initializations, subsampled
cells, or the choice of rank. `fit()` reports it through `n_repeats` and
`result.stability`, which retains only programs reproducible across repeated
fits of that sample. Repeated fits are computational replicates of one sample
and never count as independent evidence of biology.

**Cross-sample biological recurrence** asks whether programs discovered
separately in different samples represent the same process. `compute_recurrence()`
reports it as optimal one-to-one matches, random-assignment nulls, and
per-sample redundancy.

Neither is sufficient alone. A stable meta-program vocabulary requires
within-sample stability, cross-sample recurrence above the assignment null, and
non-redundancy, which is why `build_vocabulary()` takes `min_samples` and an
optional `max_redundancy`. Qualifying for a vocabulary is a threshold decision
about programs, not a biological validation claim about a state representation;
use `decide_promotion()` with held-out evidence for that.

## Choosing a biological population

Discover programs within a biologically coherent population. Proximal tubule
cells across many donors is the intended scope for tubule state programs. All
kidney cell types across those same donors is not: the dominant programs would
reflect epithelial, endothelial, immune, and stromal identity rather than
within-cell-type state variation.

The package does not require a particular annotation method, but it does not
choose this boundary for you. Subset to the population you mean before calling
`fit_atlas()` or `fit_samples()`.

## Portable per-sample results

One sample's programs are a self-contained artifact, so a large atlas can be
analysed by fitting each donor independently, on separate machines if needed,
and combining the compact results afterwards:

```python
result = sccs.fit("donor_01.h5ad", sample_id="donor_01")
sccs.save_program_result(result, "results/")       # results/donor_01.h5ad

# ... the same command runs for every donor, in any order, on any node ...

cohort = sccs.aggregate("results/")                 # same reduction as fit_atlas
cohort.vocabulary                                   # recurrent programs, or None
```

The artifact is an `AnnData` on that sample's own feature axis: programs in
`.varm["sccs_programs"]`, usages in `.obsm["X_sccs_programs"]`, and the fit
provenance and repeated-fit diagnostics in
`.uns["sccellstates"]["program_result"]`. It carries no expression matrix, so it
is a compact result rather than a copy of the data, and it is deliberately not a
valid input to `fit()` or `project()`, which both need expression values.

`aggregate()` accepts a results directory, one result file, a sequence of
either, or a mapping of sample ID to either. **Sample IDs always come from the
metadata inside each result, never from file names**, so an artifact can be
renamed, moved, or copied between machines without changing the analysis. A
mapping key is checked against that metadata rather than used to name a sample.

Because `aggregate()` performs the same reduction as `fit_atlas()` and
`fit_samples()`, distributing the samples is the same computation as fitting
them together. Results must agree on modality, method, preprocessing, and the
number of repeated fits, so the `min_samples` support threshold always counts
samples of equal evidential quality. They need **not** share a feature axis:
each sample selects its own genes by design, and recurrence is measured on the
features they do share.

Identical inputs produce byte-identical artifacts within one environment, so a
result can be checksummed. Nothing timestamped and no software version is
recorded; the artifact's `schema_version` is the only version gate, because it
is the only one that changes what the stored fields mean.

## Freezing a recurrent vocabulary

A recurrent vocabulary can be written to a self-contained directory and
reloaded later, which is what makes one discovery run reusable for projecting
new samples.

```python
cohort = sccs.fit_atlas("atlas.h5ad", sample_key="donor_id", layer="counts")
sccs.save_program_vocabulary(
    cohort.vocabulary,
    "results/vocabulary",
    recurrence=cohort.recurrence,
)

frozen = sccs.load_program_vocabulary("results/vocabulary")
usages = sccs.project("held_out.h5ad", frozen)
```

The vocabulary is stored on its own consensus feature axis, the intersection of
the per-sample axes its programs were fitted on. Because those axes differ, the
links back are stored too: `members.json` records which sample and program
index supports each vocabulary program, and `feature_map.json` records each
sample's own axis alongside the consensus axis. Pairwise matching and its
assignment nulls go to `recurrence.npz` for audit.

Identical inputs produce byte-identical artifacts, so nothing timestamped is
written and a frozen vocabulary can be diffed or checksummed.

This is separate from `store_program_vocabulary()`, which writes into an
`AnnData` object's `.varm` and `.uns` and therefore cannot represent several
per-sample gene axes at once.

## Recurrent program vocabulary

Phase 1 supports deterministic NMF program discovery independently within each
biological sample, rank-based one-to-one program matching, random-assignment
nulls, redundancy diagnostics, and training-sample-only consensus vocabularies.
Input expression must already contain the preprocessing chosen by the caller;
the estimator does not normalize or select genes implicitly.

```python
import sccellstates as sccs

training_donors = ["donor_1", "donor_2", "donor_3"]
programs = sccs.fit_programs_by_sample(
    adata,
    sample_key="donor_id",
    layer="log_normalized",
    sample_ids=training_donors,
    estimator_factory=lambda _sample: sccs.NMFProgramEstimator(
        n_programs=10,
        random_state=42,
    ),
)

fit = sccs.build_recurrent_vocabulary(
    programs,
    min_samples=2,
    min_similarity=0.3,
    n_permutations=1_000,
    random_state=42,
)
sccs.store_program_vocabulary(adata, fit)
```

Repeated fits of one sample can be aligned before cross-sample recurrence:

```python
stable = sccs.stabilize_programs(
    tuple(repeated_program_fits),
    sample_id="donor_1",
    min_similarity=0.5,
)
donor_programs = stable.programs
```

This retains only anchor programs that match in every computational repeat.
Repeated fits improve within-sample stability but never count as independent
biological samples when constructing a recurrent vocabulary.

## Runnable candidate pipeline

`fit_pipeline()` connects sample-aware splitting, train-fitted preprocessing,
within-sample NMF, recurrent vocabulary construction, frozen program scoring,
baseline representations, sample-level uncertainty, and provenance. It returns
a copied and annotated `AnnData`; the input object is not modified.

```python
config = sccs.PipelineConfig(
    sample_key="donor_id",
    validation_samples=("donor_4",),
    test_samples=("donor_5",),
    layer="counts",
    preprocessing="library_size_log1p",
    n_programs=10,
    random_state=42,
)
result = sccs.fit_pipeline(adata, config)
candidate = result.coordinates["direct"]
```

Only training samples are used to fit preprocessing parameters, discover
programs, construct the recurrent vocabulary, and fit state representations.
Validation and test samples are transformed with those frozen objects. Pipeline
outputs are candidates for evaluation, not claims of biological validation or
promotion.

## Lineage-aware vocabularies

For datasets containing multiple lineages, hierarchical vocabulary fitting can
keep lineage-specific recurrence separate while optionally identifying programs
shared across lineages. Lineage identifiers are grouping metadata, not state
targets, and training samples must be declared explicitly.

```python
hierarchy = sccs.fit_hierarchical_vocabularies(
    adata,
    sample_key="donor_id",
    lineage_key="lineage",
    training_sample_ids=("donor_1", "donor_2", "donor_3"),
    estimator_factory=lambda _group: sccs.NMFProgramEstimator(
        n_programs=10, random_state=42
    ),
)
lineage_fit = hierarchy.lineage_vocabularies["proximal_tubule"]
```

The optional shared vocabulary is evidence of recurrent programs across the
lineage-specific vocabularies; it does not establish biological equivalence or
replace held-out evaluation.

The stored gene weights use `.varm["sccs_programs"]`; matching, null,
recurrence, and redundancy provenance is recorded under
`.uns["sccellstates"]["program_vocabulary"]`. Held-out samples must not be
included in `sample_ids` or in preprocessing fitted before this workflow.

## Program scoring and baseline representations

A fixed recurrent vocabulary can score cells without refitting on validation or
test samples. Sample partitions are defined by biological sample identifiers,
and baseline estimators fit only cells belonging to the training samples.

```python
split = sccs.make_sample_split(
    adata,
    sample_key="donor_id",
    validation_samples=["donor_4"],
    test_samples=["donor_5"],
)

activities = sccs.ProgramScorer(fit.vocabulary).transform(
    adata,
    layer="log_normalized",
)
sccs.store_program_activities(adata, activities)

n_state_components = min(5, fit.vocabulary.n_programs)
model = sccs.LinearProgramState(
    n_components=n_state_components,
    random_state=42,
).fit(
    activities,
    sample_ids=adata.obs["donor_id"],
    split=split,
)
coordinates = model.transform(activities)
sccs.store_state_coordinates(
    adata,
    coordinates,
    method="linear_program_pca",
    training_sample_ids=split.train,
    parameters={"n_components": n_state_components, "random_state": 42},
)
```

`GeneStandardizedProgramScorer` and `MatchedControlProgramScorer` provide
training-fitted alternatives when raw weighted averages retain expression-depth
effects. The former fits gene means and variances; the latter selects
expression-matched control genes. Both fit only explicitly designated training
samples, transform held-out data without refitting, and preserve sparse input
matrices.

`CountResidualProgramScorer` is an additional count-scale diagnostic. It fits
gene rates on training samples and uses each cell's selected-count total as an
explicit observation offset when calculating program residuals.

`TechnicalResidualProgramScorer` is an additional nuisance-residual diagnostic.
It fits per-gene regressions on training cells using an intercept, selected-gene
library depth, and detected-gene count, then scores held-out cells from the
resulting residuals.

The package also provides standardized direct program activities, expression
PCA, and a single-bottleneck MLP as explicit comparison baselines. Expression
PCA requires a caller-selected dense-conversion limit and raises before a
sparse matrix exceeds it. Reconstruction metrics are reported separately by
train, validation, and test partition; technical associations are available as
coordinate-wise Spearman correlations.

Program activities use `.obsm["X_sccs_programs"]`. A representation selected by
the caller uses `.obsm["X_sccs_state"]` with fit provenance under
`.uns["sccellstates"]["state_representation"]`. No baseline is selected
automatically.

## Command line

The primary command discovers programs in one H5AD sample:

```bash
python -m sccellstates fit \
  --input sample.h5ad \
  --output results/sample_01 \
  --sample-id donor_01 \
  --method cnmf \
  --n-programs 8 \
  --seed 42
```

This writes `results/sample_01/donor_01.h5ad`, a portable program result with
program loadings in `.varm["sccs_programs"]` and usages in
`.obsm["X_sccs_programs"]`, plus a `donor_01.json` summary. Several independent
jobs may write into one shared `--output` directory: writing a *different*
sample there is normal and needs no `--overwrite`, which is only required to
replace the same sample again.

To distribute a large atlas, run that command once per donor, on as many nodes
as you like, then combine the results:

```bash
python -m sccellstates aggregate \
  --input results/ \
  --output model/ \
  --min-samples 3 \
  --seed 42
```

This loads every `.h5ad` in the directory, evaluates recurrence across them, and
writes `model/run.json`, the per-sample results under `model/samples/`, and a
frozen `model/vocabulary/` when programs recur. Donors are named from the
metadata inside each result, so the file names are irrelevant. No scheduler is
required or assumed: SLURM, PBS, cloud batch, and a plain shell loop all work,
because the package only ever sees one sample at a time.

The same command accepts an atlas, or several files, when `--sample-key` names
the biological sample column:

```bash
python -m sccellstates fit \
  --input atlas.h5ad \
  --output results/atlas_01 \
  --sample-key donor_id \
  --layer counts \
  --min-samples 3 \
  --n-programs 8 \
  --seed 42
```

Each donor is fitted independently and `run.json` records the recurrence
summary, the shared feature coverage, and `pooled_fit: false`. Each donor's
portable result is also written to `samples/`, so a local run leaves behind the
same artifacts a distributed one does. When programs recur, the vocabulary is
frozen to a `vocabulary/` directory that `load_program_vocabulary()` can read
back.

For the existing end-to-end candidate workflow, the package also provides a
reproducible H5AD cohort command. Use one atlas file whose
`.obs` contains an explicit donor/sample column, or pass multiple H5AD files
with the same gene identifiers. QC, training-only gene selection, program
discovery, state fitting, and result writing are all controlled by the package:

```bash
python -m sccellstates run \
  --input atlas.h5ad \
  --output results/run_a \
  --sample-key donor_id \
  --validation-samples donor_4 \
  --test-samples donor_5 \
  --min-counts-per-cell 500 \
  --min-genes-per-cell 200 \
  --min-cells-per-gene 3 \
  --n-top-genes 2000 \
  --preprocessing library_size_log1p \
  --n-programs 8 \
  --seed 42
```

Each output directory contains `candidate.h5ad`, CSV diagnostic reports, and
`run.json` with the split, parameters, QC settings, and provenance. The CLI
does not infer donor identity from filenames and does not use validation or
test samples to fit preprocessing, gene selection, or the vocabulary.

## Development

Create an isolated environment and run the test suite:

```bash
uv sync --all-groups
.venv/bin/python -m pytest
.venv/bin/ruff check .
```

## License

Copyright © 2026 Dr. Qingzhou Zhang. Released under the [MIT License](LICENSE).
