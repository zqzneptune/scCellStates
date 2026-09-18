"""Simplex-constrained state composition projection."""

# Closures intentionally capture one cell's fixed objective data.
# ruff: noqa: B023, E501

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from sccellstates.projection.base import (
    ProjectionError,
    ProjectorSpec,
    StateProjector,
    aligned_matrix,
    make_result,
    row_values,
)


class SimplexProjector(StateProjector):
    """Infer state composition directly under nonnegativity and sum-to-one."""

    spec = ProjectorSpec("simplex")

    def transform(self, adata, *, layer=None, sample_id="projection"):
        matrix, basis, present, missing, extra = aligned_matrix(adata, self.vocabulary, layer=layer)
        k = self.vocabulary.n_programs
        initial = np.full(k, 1.0 / k)
        usages = np.zeros((adata.n_obs, k))
        constraints = {"type": "eq", "fun": lambda h: np.sum(h) - 1.0, "jac": lambda h: np.ones(k)}
        for row in range(adata.n_obs):
            values = row_values(matrix, row)
            # A simplex represents composition, so optimize against the cell's
            # observed composition rather than raw library-size magnitudes.
            # Scaling also keeps SLSQP's finite tolerances meaningful for
            # count matrices with large depths.
            total = float(values.sum())
            target = values / total if total > 0 else np.zeros_like(values)

            def objective(h):
                return 0.5 * np.sum((target - basis @ h) ** 2)

            def gradient(h):
                return basis.T @ (basis @ h - target)

            fit = minimize(
                objective,
                initial,
                jac=gradient,
                bounds=[(0, 1)] * k,
                constraints=constraints,
                method="SLSQP",
                options={"ftol": 1e-12, "maxiter": 500},
            )
            if not fit.success:
                raise ProjectionError(f"simplex projection failed for cell {row}: {fit.message}")
            usages[row] = fit.x
        return make_result(
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

    def get_params(self):
        return {}
