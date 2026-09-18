"""Nonnegative fixed-vocabulary projectors."""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from sccellstates.projection.base import (
    ProjectorSpec,
    StateProjector,
    aligned_matrix,
    make_result,
    row_values,
)


class NNLSProjector(StateProjector):
    """Project each cell independently with non-negative least squares."""

    spec = ProjectorSpec("nnls")

    def __init__(self, vocabulary, *, error_warning_threshold: float = 1.0):
        super().__init__(vocabulary)
        if error_warning_threshold <= 0 or not np.isfinite(error_warning_threshold):
            raise ValueError("error_warning_threshold must be positive and finite")
        self.error_warning_threshold = float(error_warning_threshold)

    def transform(self, adata, *, layer=None, sample_id="projection"):
        matrix, basis, present, missing, extra = aligned_matrix(adata, self.vocabulary, layer=layer)
        usages = np.zeros((adata.n_obs, self.vocabulary.n_programs))
        for row in range(adata.n_obs):
            values = row_values(matrix, row)
            usages[row], _ = nnls(basis, values)
        result = make_result(
            adata,
            matrix,
            basis,
            usages,
            self.vocabulary,
            present,
            missing,
            extra,
            sample_id=sample_id,
            projector_name=self.spec.name,
            parameters=self.get_params(),
        )
        warnings = list(result.warnings)
        if missing:
            warnings.append(f"{len(missing)} vocabulary features are missing")
        if np.any(result.relative_error > self.error_warning_threshold):
            warnings.append("some cells have high relative reconstruction error")
        return result.__class__(**{**result.__dict__, "warnings": tuple(warnings)})

    def get_params(self):
        return {"error_warning_threshold": self.error_warning_threshold}
