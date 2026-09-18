"""Poisson likelihood projection for raw count matrices."""

# Closures intentionally capture one cell's fixed objective data.
# ruff: noqa: B023, E501

from __future__ import annotations

import numpy as np

from sccellstates.projection.base import (
    ProjectionError,
    ProjectorSpec,
    StateError,
    StateProjector,
    aligned_matrix,
    make_result,
)


class PoissonProjector(StateProjector):
    """Infer usages by maximizing a fixed-basis Poisson count likelihood."""

    spec = ProjectorSpec("poisson", requires_raw_counts=True)

    def __init__(self, vocabulary, *, max_iter: int = 1_000, tol: float = 1e-8):
        super().__init__(vocabulary)
        if max_iter < 1 or tol <= 0 or not np.isfinite(tol):
            raise ProjectionError("max_iter must be positive and tol must be finite and positive")
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def transform(self, adata, *, layer=None, sample_id="projection"):
        matrix, basis, present, missing, extra = aligned_matrix(adata, self.vocabulary, layer=layer)
        values = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        if not np.equal(values, np.floor(values)).all():
            raise StateError("poisson projection requires integer-valued raw counts")
        # Normalize each program loading to a count-rate distribution. The
        # Poisson likelihood then reduces to a mixture-composition problem;
        # multiplicative EM solves all cells together and preserves sparse
        # input until this selected vocabulary-sized matrix is materialized.
        basis_scale = basis / np.maximum(basis.sum(axis=0, keepdims=True), 1e-12)
        depths = values.sum(axis=1)
        proportions = np.full(
            (adata.n_obs, self.vocabulary.n_programs),
            1.0 / self.vocabulary.n_programs,
            dtype=np.float64,
        )
        active = depths > 0
        for _ in range(self.max_iter):
            means = np.maximum(proportions @ basis_scale.T, 1e-12)
            updated = proportions * ((values / means) @ basis_scale)
            row_totals = updated.sum(axis=1, keepdims=True)
            updated[active] /= np.maximum(row_totals[active], 1e-12)
            updated[~active] = proportions[~active]
            if np.max(np.abs(updated - proportions)) <= self.tol:
                proportions = updated
                break
            proportions = updated
        usages = proportions * depths[:, None]
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
        return {"max_iter": self.max_iter, "tol": self.tol}
