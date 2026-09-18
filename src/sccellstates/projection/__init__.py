"""Fixed-vocabulary state inference methods."""

from sccellstates.projection.base import (
    ProjectionBenchmark,
    ProjectionError,
    ProjectorSpec,
    StateProjector,
    StateResult,
)
from sccellstates.projection.negative_binomial import NegativeBinomialProjector
from sccellstates.projection.neural import NeuralProjector
from sccellstates.projection.nnls import NNLSProjector
from sccellstates.projection.poisson import PoissonProjector
from sccellstates.projection.registry import available_projectors, make_projector
from sccellstates.projection.regularized import RegularizedNNLSProjector
from sccellstates.projection.simplex import SimplexProjector
from sccellstates.projection.variational import VariationalProjector

__all__ = [
    "NNLSProjector",
    "PoissonProjector",
    "NegativeBinomialProjector",
    "NeuralProjector",
    "ProjectionBenchmark",
    "ProjectionError",
    "ProjectorSpec",
    "RegularizedNNLSProjector",
    "SimplexProjector",
    "VariationalProjector",
    "StateProjector",
    "StateResult",
    "available_projectors",
    "make_projector",
]
