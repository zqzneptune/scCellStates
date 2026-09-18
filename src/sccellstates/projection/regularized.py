"""Regularized nonnegative projection."""

# Closures intentionally capture one cell's fixed objective data.
# ruff: noqa: B023, E501

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from sccellstates.projection.base import (
    ProjectorSpec,
    StateProjector,
    aligned_matrix,
    make_result,
    row_values,
)


class RegularizedNNLSProjector(StateProjector):
    """Nonnegative projection with optional L1 and L2 usage penalties."""

    spec = ProjectorSpec("regularized_nnls")

    def __init__(self, vocabulary, *, l1: float = 0.0, l2: float = 0.0):
        super().__init__(vocabulary)
        if min(l1, l2) < 0 or not np.isfinite(l1 + l2):
            raise ValueError("l1 and l2 must be finite and nonnegative")
        if l1 == 0 and l2 == 0:
            raise ValueError("at least one regularization penalty must be positive")
        self.l1, self.l2 = float(l1), float(l2)

    def transform(self, adata, *, layer=None, sample_id="projection"):
        matrix, basis, present, missing, extra = aligned_matrix(adata, self.vocabulary, layer=layer)
        usages = np.zeros((adata.n_obs, self.vocabulary.n_programs))
        for row in range(adata.n_obs):
            values = row_values(matrix, row)
            def objective(h):
                return 0.5 * np.sum((values - basis @ h) ** 2) + self.l1 * np.sum(h) + 0.5 * self.l2 * np.sum(h * h)
            def gradient(h):
                return basis.T @ (basis @ h - values) + self.l1 + self.l2 * h
            fit = minimize(objective, np.zeros(self.vocabulary.n_programs), jac=gradient,
                           bounds=[(0, None)] * self.vocabulary.n_programs, method="L-BFGS-B")
            if not fit.success:
                raise RuntimeError(f"regularized projection failed for cell {row}: {fit.message}")
            usages[row] = fit.x
        return make_result(adata, matrix, basis, usages, self.vocabulary, present, missing, extra,
                           sample_id=sample_id, projector_name=self.spec.name,
                           parameters=self.get_params())

    def get_params(self):
        return {"l1": self.l1, "l2": self.l2}
