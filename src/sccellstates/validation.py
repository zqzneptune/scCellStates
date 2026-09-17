"""Evidence contracts for held-out candidate validation and promotion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


class ValidationError(ValueError):
    """Raised when validation evidence is internally inconsistent."""


Decision = Literal["promote", "retain_candidate", "insufficient_evidence"]


@dataclass(frozen=True)
class ValidationEvidence:
    """Explicit evidence fields required before promoting a candidate."""

    held_out_samples: tuple[str, ...]
    independent_study_evaluated: bool = False
    transfer_passed: bool = False
    baselines_compared: bool = False
    technical_controls_passed: bool = False
    uncertainty_reported: bool = False
    recurrence_reported: bool = False
    redundancy_checked: bool = False
    interpretation_complete: bool = False

    def __post_init__(self) -> None:
        samples = tuple(str(value).strip() for value in self.held_out_samples)
        if not samples or any(not value for value in samples):
            raise ValidationError("held_out_samples must contain non-empty identifiers")
        if len(set(samples)) != len(samples):
            raise ValidationError("held_out_samples must be unique")
        object.__setattr__(self, "held_out_samples", samples)


@dataclass(frozen=True)
class PromotionDecision:
    """Auditable result of applying the M4 promotion gate."""

    decision: Decision
    reasons: tuple[str, ...]


def decide_promotion(evidence: ValidationEvidence) -> PromotionDecision:
    """Return a conservative promotion decision from explicit evidence.

    Missing evidence produces ``insufficient_evidence``. A completed
    evaluation with a failed transfer or technical gate retains the candidate
    but does not promote it. This function never interprets condition or
    outcome labels and makes no dynamical claims.
    """
    if not isinstance(evidence, ValidationEvidence):
        raise TypeError("evidence must be a ValidationEvidence object")
    required = {
        "independent study": evidence.independent_study_evaluated,
        "held-out transfer": evidence.transfer_passed,
        "baseline comparison": evidence.baselines_compared,
        "technical controls": evidence.technical_controls_passed,
        "sample-level uncertainty": evidence.uncertainty_reported,
        "recurrence diagnostics": evidence.recurrence_reported,
        "redundancy diagnostics": evidence.redundancy_checked,
        "post-fit interpretation": evidence.interpretation_complete,
    }
    missing = tuple(name for name, passed in required.items() if not passed)
    if not evidence.independent_study_evaluated:
        return PromotionDecision("insufficient_evidence", missing)
    if missing:
        return PromotionDecision("retain_candidate", missing)
    return PromotionDecision("promote", ())
