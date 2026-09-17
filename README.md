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

## Development

Create an isolated environment and run the test suite:

```bash
uv sync --all-groups
.venv/bin/python -m pytest
.venv/bin/ruff check .
```

## License

Copyright © 2026 Dr. Qingzhou Zhang. Released under the [MIT License](LICENSE).
