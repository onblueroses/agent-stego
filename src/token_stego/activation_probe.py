"""Activation probe for detecting token-likelihood steganography.

Trains a simple MLP to classify "normal inference" vs "stego inference" from
captured hidden states. A result is evidence only with group-disjoint data,
uncertainty, a same-rig positive control, and a stated detection limit.
"""

import math
import random
from collections.abc import Hashable, Sequence

import torch
from torch import Tensor, nn

from .latent import LatentSnapshot

# These defaults preserve the historical probe architecture and optimizer for
# replay. They are baselines, not claims that the values are optimal.
DEFAULT_PROBE_HIDDEN_DIM = 128
DEFAULT_TRAINING_EPOCHS = 100
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_REPLAY_SEED = 42
# The audit protocol requires a two-sided 95% interval. Two thousand resamples
# preserve that protocol's resolution; power still depends on independent groups.
DEFAULT_BOOTSTRAP_RESAMPLES = 2000
CONFIDENCE_LEVEL = 0.95


class ActivationProbe(nn.Module):
    """2-layer MLP probe for stego vs normal classification."""

    def __init__(
        self, input_dim: int, hidden_dim: int = DEFAULT_PROBE_HIDDEN_DIM
    ) -> None:
        super().__init__()
        if (
            isinstance(input_dim, bool)
            or not isinstance(input_dim, int)
            or input_dim < 1
        ):
            raise ValueError("Probe input dimension must be a positive integer")
        if (
            isinstance(hidden_dim, bool)
            or not isinstance(hidden_dim, int)
            or hidden_dim < 1
        ):
            raise ValueError("Probe hidden dimension must be a positive integer")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


def snapshot_to_features(snap: LatentSnapshot, layer: int | None = None) -> Tensor:
    """Extract a fixed-size feature vector from a LatentSnapshot.

    When layer is specified, returns mean-pooled hidden states from that
    single layer. When layer is None, concatenates mean-pooled vectors
    from ALL captured layers (sorted by layer index), producing a vector
    of size num_layers * hidden_dim.
    """
    if not snap.hidden_states:
        raise ValueError("Snapshot has no hidden states")

    if layer is not None:
        hidden = snap.hidden_states[layer].float()
        if hidden.dim() == 3:
            hidden = hidden[0]
        return hidden.mean(dim=0)

    # Concatenate mean-pooled features from all captured layers
    layer_features = []
    for layer_idx in sorted(snap.hidden_states.keys()):
        hidden = snap.hidden_states[layer_idx].float()
        if hidden.dim() == 3:
            hidden = hidden[0]
        layer_features.append(hidden.mean(dim=0))
    return torch.cat(layer_features, dim=0)


def train_probe(
    normal_snaps: list[LatentSnapshot],
    stego_snaps: list[LatentSnapshot],
    layer: int | None = None,
    epochs: int = DEFAULT_TRAINING_EPOCHS,
    lr: float = DEFAULT_LEARNING_RATE,
    seed: int = DEFAULT_REPLAY_SEED,
) -> ActivationProbe:
    """Train a probe to distinguish normal from stego inference.

    Args:
        normal_snaps: Snapshots from normal (non-stego) inference.
        stego_snaps: Snapshots from stego inference.
        layer: Which layer's hidden states to use. None = last captured.
        epochs: Training epochs.
        lr: Learning rate.

    Returns:
        Trained ActivationProbe.
    """
    if not normal_snaps or not stego_snaps:
        raise ValueError("Probe training requires both normal and stego snapshots")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("Probe training epochs must be a positive integer")
    if (
        not isinstance(lr, (int, float))
        or isinstance(lr, bool)
        or not math.isfinite(lr)
        or lr <= 0
    ):
        raise ValueError("Probe learning rate must be finite and positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Probe replay seed must be an integer")

    # Build feature matrix and labels
    features = []
    labels = []
    for snap in normal_snaps:
        features.append(snapshot_to_features(snap, layer))
        labels.append(0.0)
    for snap in stego_snaps:
        features.append(snapshot_to_features(snap, layer))
        labels.append(1.0)

    x = torch.stack(features)
    y = torch.tensor(labels)

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        probe = ActivationProbe(input_dim=x.shape[1])
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    probe.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = probe(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

    probe.eval()
    return probe


def evaluate_probe(
    probe: ActivationProbe,
    snapshots: list[LatentSnapshot],
    labels: list[int],
    layer: int | None = None,
    seed: int = DEFAULT_REPLAY_SEED,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    group_ids: Sequence[Hashable] | None = None,
) -> dict:
    """Evaluate a probe and bootstrap AUC over groups when supplied."""
    if not snapshots:
        raise ValueError("Probe evaluation requires at least one snapshot")
    if len(snapshots) != len(labels):
        raise ValueError("Probe snapshots and labels must have equal length")
    if any(isinstance(label, bool) or label not in (0, 1) for label in labels):
        raise ValueError("Probe labels must be binary values 0 or 1")
    if set(labels) != {0, 1}:
        raise ValueError("Probe AUC requires both negative and positive labels")
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples < 2
    ):
        raise ValueError("Probe bootstrap requires at least two resamples")
    if group_ids is not None and len(group_ids) != len(labels):
        raise ValueError("Probe group IDs and labels must have equal length")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Probe bootstrap seed must be an integer")
    features = [snapshot_to_features(snap, layer) for snap in snapshots]
    x = torch.stack(features)

    with torch.no_grad():
        logits = probe(x)
        scores = torch.sigmoid(logits).tolist()

    auc = _binary_auc(scores, labels)

    bootstrap_aucs = _bootstrap_aucs(
        scores,
        labels,
        group_ids=group_ids,
        seed=seed,
        samples=bootstrap_samples,
    )
    bootstrap_aucs.sort()
    tail_probability = (1 - CONFIDENCE_LEVEL) / 2
    lower_index = int(tail_probability * (len(bootstrap_aucs) - 1))
    upper_index = int((1 - tail_probability) * (len(bootstrap_aucs) - 1))
    auc_ci = [bootstrap_aucs[lower_index], bootstrap_aucs[upper_index]]

    # 0.5 is the fixed sigmoid decision boundary for this equal-cost classifier.
    preds = [1 if s > 0.5 else 0 for s in scores]
    accuracy = sum(p == lab for p, lab in zip(preds, labels)) / len(labels)

    return {
        "auc": auc,
        "auc_ci_95": auc_ci,
        "accuracy": accuracy,
        "scores": scores,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_unit": "group" if group_ids is not None else "observation",
    }


def _bootstrap_aucs(
    scores: list[float],
    labels: list[int],
    *,
    group_ids: Sequence[Hashable] | None,
    seed: int,
    samples: int,
) -> list[float]:
    rng = random.Random(seed)
    aucs: list[float] = []
    if group_ids is None:
        positive_scores = [score for score, label in zip(scores, labels) if label == 1]
        negative_scores = [score for score, label in zip(scores, labels) if label == 0]
        if len(positive_scores) < 2 or len(negative_scores) < 2:
            raise ValueError(
                "Observation bootstrap requires two examples from each class"
            )
        for _ in range(samples):
            sampled_positive = rng.choices(positive_scores, k=len(positive_scores))
            sampled_negative = rng.choices(negative_scores, k=len(negative_scores))
            aucs.append(
                _binary_auc(
                    sampled_negative + sampled_positive,
                    [0] * len(sampled_negative) + [1] * len(sampled_positive),
                )
            )
        return aucs

    unique_groups = list(dict.fromkeys(group_ids))
    if len(unique_groups) < 2:
        raise ValueError("Group bootstrap requires at least two independent groups")
    indices_by_group = {
        group: [index for index, value in enumerate(group_ids) if value == group]
        for group in unique_groups
    }
    for group, indices in indices_by_group.items():
        if {labels[index] for index in indices} != {0, 1}:
            raise ValueError(
                f"Probe group {group!r} must contain both classes for paired bootstrap"
            )
    for _ in range(samples):
        sampled_scores: list[float] = []
        sampled_labels: list[int] = []
        for group in rng.choices(unique_groups, k=len(unique_groups)):
            for index in indices_by_group[group]:
                sampled_scores.append(scores[index])
                sampled_labels.append(labels[index])
        aucs.append(_binary_auc(sampled_scores, sampled_labels))
    return aucs


def _binary_auc(scores: list[float], labels: list[int]) -> float:
    """Compute tie-aware binary ROC AUC without external dependencies."""
    pairs = sorted(zip(scores, labels), key=lambda p: -p[0])
    tp = fp = 0
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos

    if total_pos == 0 or total_neg == 0:
        raise ValueError("Binary AUC requires both negative and positive labels")

    auc = 0.0
    prev_fpr = prev_tpr = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            if pairs[j][1] == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        tpr = tp / total_pos
        fpr = fp / total_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_fpr, prev_tpr = fpr, tpr
        i = j
    return auc
