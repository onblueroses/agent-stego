"""Tombstone for the retired conversation-detection experiment."""

RETIRED_CONVERSATION_DETECTION_ERROR = (
    "conversation_detection is retired: its probe had no preregistered power "
    "or positive control and converted arbitrary AUC cutoffs into categorical "
    "verdicts. Rebuild it with grouped tasks and the current experiment contract."
)


def main() -> None:
    """Reject execution before importing model libraries or creating artifacts."""
    raise SystemExit(RETIRED_CONVERSATION_DETECTION_ERROR)


if __name__ == "__main__":
    main()
