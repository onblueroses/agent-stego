import math

import pytest

from experiments.token_detection import compute_auc, find_rate_at_threshold


def test_compute_auc_perfect_ranking() -> None:
    assert compute_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_compute_auc_ties_are_order_independent() -> None:
    labels = [0, 1, 0, 1]
    tied_scores = [0.5, 0.5, 0.5, 0.5]

    assert compute_auc(labels, tied_scores) == 0.5
    assert compute_auc(list(reversed(labels)), tied_scores) == 0.5


@pytest.mark.parametrize(
    ("labels", "scores"),
    [
        ([0, 0], [0.1, 0.2]),
        ([1, 1], [0.8, 0.9]),
        ([0, 1], [0.1, math.nan]),
        ([False, 1], [0.1, 0.9]),
        ([0], [0.1, 0.2]),
    ],
)
def test_compute_auc_rejects_invalid_samples(labels, scores) -> None:
    with pytest.raises(ValueError):
        compute_auc(labels, scores)


def test_threshold_rate_does_not_split_ties() -> None:
    labels = [1, 0, 1, 0]
    scores = [0.9, 0.9, 0.8, 0.1]

    assert find_rate_at_threshold(labels, scores, 0.0) == 0.0
    assert find_rate_at_threshold(labels, scores, 0.5) == 1.0


@pytest.mark.parametrize("target", [-0.1, 1.1, math.inf, math.nan])
def test_threshold_rate_rejects_invalid_target(target) -> None:
    with pytest.raises(ValueError):
        find_rate_at_threshold([0, 1], [0.1, 0.9], target)
