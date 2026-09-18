"""Small built-in registry for Phase-2 projectors."""

from __future__ import annotations

from sccellstates.projection.nnls import NNLSProjector
from sccellstates.projection.poisson import PoissonProjector
from sccellstates.projection.regularized import RegularizedNNLSProjector
from sccellstates.projection.simplex import SimplexProjector

PROJECTORS = {
    "nnls": NNLSProjector,
    "regularized_nnls": RegularizedNNLSProjector,
    "simplex": SimplexProjector,
    "poisson": PoissonProjector,
}


def available_projectors() -> tuple[str, ...]:
    """Return names of implemented fixed-vocabulary projectors."""
    return tuple(PROJECTORS)


def make_projector(name: str, vocabulary, **kwargs):
    """Construct a registered projector or report an actionable error."""
    try:
        cls = PROJECTORS[name]
    except KeyError as error:
        raise ValueError(
            f"unknown projector {name!r}; available: {', '.join(PROJECTORS)}"
        ) from error
    return cls(vocabulary, **kwargs)
