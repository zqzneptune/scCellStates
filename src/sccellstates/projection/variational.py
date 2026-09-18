"""Deterministic uncertainty-aware fixed-vocabulary projection."""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from sccellstates.projection.base import (
    ProjectionError,
    ProjectorSpec,
    StateProjector,
    aligned_matrix,
    make_result,
)


class VariationalProjector(StateProjector):
    """Approximate posterior usages with deterministic count perturbations."""

    spec = ProjectorSpec("variational", requires_raw_counts=True, supports_uncertainty=True)

    def __init__(
        self, vocabulary, *, n_samples: int = 8, random_state: int = 0, noise_scale: float = 0.05
    ):
        super().__init__(vocabulary)
        if n_samples < 2 or noise_scale < 0:
            raise ProjectionError("n_samples must be at least 2 and noise_scale nonnegative")
        self.n_samples = int(n_samples)
        self.random_state = int(random_state)
        self.noise_scale = float(noise_scale)

    def transform(self, adata, *, layer=None, sample_id="projection"):
        matrix, basis, present, missing, extra = aligned_matrix(adata, self.vocabulary, layer=layer)
        values = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        rng = np.random.default_rng(self.random_state)
        draws = []
        for _ in range(self.n_samples):
            noisy = np.maximum(
                values + rng.normal(0, self.noise_scale * np.sqrt(values + 1), values.shape),
                0,
            )
            draw = np.vstack([nnls(basis, row)[0] for row in noisy])
            draws.append(draw)
        usages = np.mean(draws, axis=0)
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
        return result.__class__(**{**result.__dict__, "uncertainty": np.std(draws, axis=0)})

    def get_params(self):
        return {
            "n_samples": self.n_samples,
            "random_state": self.random_state,
            "noise_scale": self.noise_scale,
        }
