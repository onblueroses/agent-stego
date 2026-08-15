"""Fail-closed tests for explicit remote-egress authorization."""

from datetime import UTC, datetime, timedelta

import pytest

from src.remote_policy import (
    RemoteEgressError,
    RemoteEgressPolicy,
    require_remote_egress,
)

NOW = datetime(2026, 8, 14, tzinfo=UTC)
OPENROUTER_DESTINATION = "https://openrouter.ai/api/v1"


def _policy(**overrides: object) -> RemoteEgressPolicy:
    values: dict[str, object] = {
        "approval_record_id": "approval-001",
        "provider": "openrouter",
        "destination": OPENROUTER_DESTINATION,
        "action": "send synthetic experiment prompts",
        # One hour is a deterministic test window around NOW.
        "expires_at": NOW + timedelta(hours=1),
        "synthetic_payload_confirmed": True,
    }
    values.update(overrides)
    return RemoteEgressPolicy(**values)  # type: ignore[arg-type]


def test_remote_egress_requires_explicit_capability() -> None:
    with pytest.raises(RemoteEgressError, match="in-process policy"):
        require_remote_egress(
            policy=None,
            provider="openrouter",
            destination=OPENROUTER_DESTINATION,
            action="send synthetic experiment prompts",
            now=NOW,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"approval_record_id": "bad id"}, "record id"),
        ({"provider": "OpenRouter"}, "lowercase"),
        ({"action": ""}, "action"),
        ({"expires_at": datetime(2026, 8, 14)}, "timezone-aware"),
        ({"synthetic_payload_confirmed": False}, "synthetic"),
    ],
)
def test_policy_rejects_ambiguous_scope(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(RemoteEgressError, match=message):
        _policy(**overrides)


@pytest.mark.parametrize(
    "destination",
    (
        "http://openrouter.ai/api/v1",
        "https://user:secret@openrouter.ai/api/v1",
        "https://openrouter.ai:8443/api/v1",
        "https://openrouter.ai/api/v1?query=1",
        "https://openrouter.ai/api/v1#fragment",
        "https://OpenRouter.ai/api/v1",
        "https://127.0.0.1/api/v1",
    ),
)
def test_remote_destinations_reject_unsafe_forms(destination: str) -> None:
    with pytest.raises(RemoteEgressError, match="remote"):
        _policy(destination=destination)


def test_exact_authorization_returns_auditable_binding() -> None:
    authorization = require_remote_egress(
        policy=_policy(),
        provider="openrouter",
        destination=OPENROUTER_DESTINATION,
        action="send synthetic experiment prompts",
        now=NOW,
    )
    assert authorization.approval_record_id == "approval-001"
    assert authorization.host == "openrouter.ai"
    assert authorization.destination == OPENROUTER_DESTINATION


def test_expired_or_mismatched_policy_fails_closed() -> None:
    with pytest.raises(RemoteEgressError, match="expired"):
        require_remote_egress(
            policy=_policy(expires_at=NOW),
            provider="openrouter",
            destination=OPENROUTER_DESTINATION,
            action="send synthetic experiment prompts",
            now=NOW,
        )
    with pytest.raises(RemoteEgressError, match="action"):
        require_remote_egress(
            policy=_policy(),
            provider="openrouter",
            destination=OPENROUTER_DESTINATION,
            action="send production records",
            now=NOW,
        )
