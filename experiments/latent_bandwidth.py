"""Tombstone for the retired latent-bandwidth experiment."""

RETIRED_LATENT_BANDWIDTH_ERROR = (
    "latent_bandwidth is retired: repeated prompt groups crossed the probe's "
    "train/test boundary, and its fixed AUC cutoffs had no power or decision "
    "basis. Rebuild it with src.token_stego.experiment_design before rerunning."
)


def main() -> None:
    """Reject execution before importing model libraries or creating artifacts."""
    raise SystemExit(RETIRED_LATENT_BANDWIDTH_ERROR)


if __name__ == "__main__":
    main()
