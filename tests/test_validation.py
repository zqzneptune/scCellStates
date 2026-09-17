import pytest

import sccellstates as sccs


def test_promotion_requires_independent_evidence() -> None:
    evidence = sccs.ValidationEvidence(held_out_samples=("donor_held_out",))
    decision = sccs.decide_promotion(evidence)
    assert decision.decision == "insufficient_evidence"
    assert "independent study" in decision.reasons


def test_complete_evidence_can_promote() -> None:
    evidence = sccs.ValidationEvidence(
        held_out_samples=("donor_held_out", "study_2"),
        independent_study_evaluated=True,
        transfer_passed=True,
        baselines_compared=True,
        technical_controls_passed=True,
        uncertainty_reported=True,
        recurrence_reported=True,
        redundancy_checked=True,
        interpretation_complete=True,
    )
    assert sccs.decide_promotion(evidence) == sccs.PromotionDecision("promote", ())


def test_failed_transfer_retains_candidate_after_evaluation() -> None:
    evidence = sccs.ValidationEvidence(
        held_out_samples=("donor_held_out",), independent_study_evaluated=True
    )
    assert sccs.decide_promotion(evidence).decision == "retain_candidate"


def test_validation_evidence_rejects_duplicate_samples() -> None:
    with pytest.raises(sccs.ValidationError, match="unique"):
        sccs.ValidationEvidence(held_out_samples=("d1", "d1"))
