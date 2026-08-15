"""Safety checks for experiment payloads supplied through the environment."""

import os

SYNTHETIC_CONFIRMATION_ENV = "SYNTHETIC_PAYLOAD_CONFIRMED"


def require_synthetic_override(name: str) -> None:
    """Require an explicit acknowledgement before accepting payload-like input."""
    if name in os.environ and os.environ.get(SYNTHETIC_CONFIRMATION_ENV) != "yes":
        raise RuntimeError(
            f"Set {SYNTHETIC_CONFIRMATION_ENV}=yes before using a {name} override"
        )


def synthetic_text(default: str, *, name: str = "SECRET") -> str:
    """Return a synthetic text fixture, guarding any environment override."""
    require_synthetic_override(name)
    return os.environ.get(name, default)
