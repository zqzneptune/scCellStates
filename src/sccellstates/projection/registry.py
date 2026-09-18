"""Small built-in registry for Phase-2 projectors."""

from __future__ import annotations

from sccellstates.projection.base import ProjectionError
from sccellstates.projection.negative_binomial import NegativeBinomialProjector
from sccellstates.projection.neural import NeuralProjector
from sccellstates.projection.nnls import NNLSProjector
from sccellstates.projection.poisson import PoissonProjector
from sccellstates.projection.regularized import RegularizedNNLSProjector
from sccellstates.projection.simplex import SimplexProjector
from sccellstates.projection.variational import VariationalProjector

PROJECTORS = {
    "nnls": NNLSProjector,
    "regularized_nnls": RegularizedNNLSProjector,
    "simplex": SimplexProjector,
    "poisson": PoissonProjector,
    "negative_binomial": NegativeBinomialProjector,
    "neural": NeuralProjector,
    "variational": VariationalProjector,
}


def available_projectors() -> tuple[str, ...]:
    """Return names of implemented fixed-vocabulary projectors."""
    return tuple(PROJECTORS)


def make_projector(name: str, vocabulary, **kwargs):
    """Construct a registered projector or report an actionable error."""
    try:
        cls = PROJECTORS[name]
    except KeyError as error:
        raise ProjectionError(
            f"unknown projector {name!r}; available: {', '.join(PROJECTORS)}"
        ) from error
    return cls(vocabulary, **kwargs)
