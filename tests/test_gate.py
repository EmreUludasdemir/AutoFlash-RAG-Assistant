from config import RERANK_GATE


def decide(best_score: float) -> str:
    return "abstained" if best_score < RERANK_GATE else "answered"


def test_score_at_or_above_gate_answers():
    assert decide(RERANK_GATE) == "answered"
    assert decide(RERANK_GATE + 3.6) == "answered"


def test_score_below_gate_abstains():
    assert decide(RERANK_GATE - 0.01) == "abstained"
    assert decide(-2.99) == "abstained"
