"""Regression tests for leakage-resistant probe experiment design."""

import pytest

from src.token_stego.experiment_design import (
    assert_no_cross_split_duplicates,
    grouped_train_test_indices,
    select_unique_prompts,
    split_manifest,
)


def test_grouped_split_keeps_repeated_groups_together() -> None:
    groups = ["a", "a", "b", "b", "c", "c", "d", "d"]
    train_idx, test_idx = grouped_train_test_indices(groups, seed=7)

    train_groups = {groups[i] for i in train_idx}
    test_groups = {groups[i] for i in test_idx}
    assert train_groups.isdisjoint(test_groups)
    assert train_idx
    assert test_idx


def test_duplicate_guard_rejects_cross_split_observation() -> None:
    samples = [("prompt", (1, 2)), ("prompt", (1, 2))]
    with pytest.raises(ValueError, match="leakage"):
        assert_no_cross_split_duplicates(samples, [0], [1])


def test_default_probe_prompts_are_unique_and_manifest_has_no_overlap() -> None:
    prompts = select_unique_prompts(20)
    train_idx, test_idx = grouped_train_test_indices(prompts, seed=42)
    manifest = split_manifest(prompts, train_idx, test_idx, seed=42)

    assert len(prompts) == len(set(prompts))
    assert manifest["group_overlap"] == []
    assert manifest["train_count"] + manifest["test_count"] == 20


def test_probe_request_cannot_cycle_past_declared_prompt_groups() -> None:
    with pytest.raises(ValueError, match="unique prompt groups"):
        select_unique_prompts(25)


def test_split_controls_reject_implicit_or_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="sample count"):
        select_unique_prompts(8.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        grouped_train_test_indices(["a", "b"], train_fraction=float("nan"))
    with pytest.raises(ValueError, match="seed"):
        grouped_train_test_indices(["a", "b"], seed=True)
