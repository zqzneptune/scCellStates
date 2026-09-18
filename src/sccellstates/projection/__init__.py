"""Fixed-vocabulary state inference methods."""

from sccellstates.projection.base import (
    ProjectionBenchmark,
    ProjectorSpec,
    StateProjector,
    StateResult,
)
from sccellstates.projection.nnls import NNLSProjector
from sccellstates.projection.poisson import PoissonProjector
from sccellstates.projection.registry import available_projectors, make_projector
from sccellstates.projection.regularized import RegularizedNNLSProjector
from sccellstates.projection.simplex import SimplexProjector

__all__ = [
    "NNLSProjector", "PoissonProjector", "ProjectionBenchmark", "ProjectorSpec",
    "RegularizedNNLSProjector",
    "SimplexProjector", "StateProjector", "StateResult", "available_projectors", "make_projector",
]
