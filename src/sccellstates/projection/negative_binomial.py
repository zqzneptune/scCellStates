"""Negative-binomial fixed-vocabulary projection for raw counts."""

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


class NegativeBinomialProjector(StateProjector):
    """Infer nonnegative usages with a dispersion-aware count objective.

    The vocabulary is fixed.  A scalar ``dispersion`` controls the NB
    overdispersion approximation; no gene programs are learned during
    projection.
    """

    spec = ProjectorSpec("negative_binomial", requires_raw_counts=True)

    def __init__(
        self, vocabulary, *, dispersion: float = 0.1, max_iter: int = 200, tol: float = 1e-7
    ):
        super().__init__(vocabulary)
        if not np.isfinite(dispersion) or dispersion <= 0:
            raise ProjectionError("dispersion must be finite and positive")
        if max_iter < 1 or tol <= 0 or not np.isfinite(tol):
            raise ProjectionError("max_iter and tol must be positive")
        self.dispersion, self.max_iter, self.tol = float(dispersion), int(max_iter), float(tol)

    def transform(self, adata, *, layer=None, sample_id="projection"):
        matrix, basis, present, missing, extra = aligned_matrix(adata, self.vocabulary, layer=layer)
        values = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        if not np.equal(values, np.floor(values)).all():
            raise StateError("negative-binomial projection requires integer-valued raw counts")
        scale = basis / np.maximum(basis.sum(axis=0, keepdims=True), 1e-12)
        depths = values.sum(axis=1)
        proportions = np.full(
            (adata.n_obs, self.vocabulary.n_programs), 1.0 / self.vocabulary.n_programs
        )
        for _ in range(self.max_iter):
            means = np.maximum(proportions @ scale.T, 1e-12)
            # NB variance weights down-weight high-count residuals relative to
            # Poisson, while retaining the same deterministic multiplicative map.
            weights = 1.0 / (1.0 + self.dispersion * means * np.maximum(depths[:, None], 1.0))
            updated = proportions * ((values * weights / means) @ scale)
            totals = updated.sum(axis=1, keepdims=True)
            updated = np.divide(updated, totals, out=proportions.copy(), where=totals > 0)
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
        return {"dispersion": self.dispersion, "max_iter": self.max_iter, "tol": self.tol}
