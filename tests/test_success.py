from gridiron.coverage.success import GATE8, score_higher_is_better


def test_locked_gates_are_the_contract():
    assert GATE8["accuracy_lift"] == 0.15
    assert GATE8["man_zone"] == 0.82
    assert GATE8["macro_f1"] == 0.40


def test_score_higher_is_better_hits_eight_at_the_gate():
    assert score_higher_is_better(0.15, 0.15, 0.25) == 8
    assert score_higher_is_better(0.25, 0.15, 0.25) == 10
    assert score_higher_is_better(0.14, 0.15, 0.25) <= 7
    assert score_higher_is_better(0.0, 0.15, 0.25) == 1
