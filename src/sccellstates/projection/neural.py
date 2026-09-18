"""Small deterministic neural amortized fixed-vocabulary projector."""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls
from sklearn.neural_network import MLPRegressor

from sccellstates.projection.base import (
    ProjectionError,
    ProjectorSpec,
    StateError,
    StateProjector,
    aligned_matrix,
    make_result,
)


class NeuralProjector(StateProjector):
    """Learn a small feature-to-usage map while keeping vocabulary weights fixed."""

    spec = ProjectorSpec("neural", requires_training=True, deterministic=True)

    def __init__(
        self,
        vocabulary,
        *,
        hidden_layer_sizes=(32,),
        max_iter: int = 200,
        random_state: int = 0,
        alpha: float = 1e-4,
    ):
        super().__init__(vocabulary)
        if max_iter < 1 or alpha < 0:
            raise ProjectionError("max_iter must be positive and alpha nonnegative")
        self.hidden_layer_sizes = tuple(int(x) for x in hidden_layer_sizes)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.alpha = float(alpha)
        self._model = None

    def fit(self, vocabulary=None, X=None, **kwargs):
        super().fit(vocabulary=vocabulary, X=X, **kwargs)
        if X is None:
            raise StateError("neural projection requires training data passed to fit")
        matrix, basis, *_ = aligned_matrix(X, self.vocabulary, layer=kwargs.get("layer"))
        values = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        targets = np.vstack([nnls(basis, row)[0] for row in values])
        scale = np.maximum(values.sum(axis=1, keepdims=True), 1.0)
        model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
            shuffle=False,
        )
        model.fit(values / scale, targets / scale)
        self._model = model
        return self

    def transform(self, adata, *, layer=None, sample_id="projection"):
        if self._model is None:
            raise StateError("neural projector must be fitted before transform")
        matrix, basis, present, missing, extra = aligned_matrix(adata, self.vocabulary, layer=layer)
        values = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        scale = np.maximum(values.sum(axis=1, keepdims=True), 1.0)
        usages = np.maximum(self._model.predict(values / scale), 0.0) * scale
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
        return {
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "alpha": self.alpha,
        }

    def get_state(self):
        """Return fitted neural weights for portable persistence."""
        if self._model is None:
            return {}
        return {
            **{f"coefs_{i}": value for i, value in enumerate(self._model.coefs_)},
            **{f"intercepts_{i}": value for i, value in enumerate(self._model.intercepts_)},
        }

    def set_state(self, state):
        """Restore fitted neural weights written by :func:`save_projector`."""
        coefs = [state[key] for key in sorted(state) if key.startswith("coefs_")]
        intercepts = [state[key] for key in sorted(state) if key.startswith("intercepts_")]
        if not coefs or len(coefs) != len(intercepts):
            raise StateError("malformed persisted neural projector weights")
        model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
            shuffle=False,
        )
        model.fit(np.zeros((1, coefs[0].shape[0])), np.zeros((1, intercepts[-1].shape[0])))
        model.coefs_ = [np.asarray(value, dtype=np.float64) for value in coefs]
        model.intercepts_ = [np.asarray(value, dtype=np.float64) for value in intercepts]
        self._model = model
        return self
