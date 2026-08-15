"""Tombstone for the invalidated detectability-scaling experiment."""

RETIRED_DETECTABILITY_SCALING_ERROR = (
    "detectability_scaling is retired: its controls leaked sequence length and "
    "its repeated prompts crossed evaluation groups. Use coupled_probe.py with "
    "the current group-disjoint experiment contract instead."
)


def main() -> None:
    """Reject execution before importing model libraries or creating artifacts."""
    raise SystemExit(RETIRED_DETECTABILITY_SCALING_ERROR)


if __name__ == "__main__":
    main()
