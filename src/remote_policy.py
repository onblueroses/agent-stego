"""Fail-closed authorization for repository-initiated remote inference."""

import ipaddress
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlsplit

_IDENTIFIER_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_APPROVAL_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*\Z")
# DNS label and full-name limits come from RFC 1035 and RFC 1034.
_HOST_PATTERN = re.compile(
    r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z"
)


class RemoteEgressError(RuntimeError):
    """A remote action lacks an exact, explicit authorization."""


@dataclass(frozen=True)
class RemoteEgressPolicy:
    """An explicit, expiry-bounded capability for one synthetic remote action."""

    approval_record_id: str
    provider: str
    destination: str
    action: str
    expires_at: datetime
    synthetic_payload_confirmed: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.approval_record_id, str)
            or _APPROVAL_ID_PATTERN.fullmatch(self.approval_record_id) is None
        ):
            raise RemoteEgressError("approval record id is not canonical")
        _validate_provider(self.provider)
        _https_hostname(self.destination)
        _validate_action(self.action)
        if (
            not isinstance(self.expires_at, datetime)
            or self.expires_at.tzinfo is None
            or self.expires_at.utcoffset() is None
        ):
            raise RemoteEgressError("remote policy expiry must be timezone-aware")
        if self.synthetic_payload_confirmed is not True:
            raise RemoteEgressError(
                "remote inference policy must explicitly confirm synthetic payloads"
            )


@dataclass(frozen=True)
class RemoteAuthorization:
    """The exact capability binding revalidated at a network boundary."""

    approval_record_id: str
    provider: str
    destination: str
    host: str
    action: str
    expires_at: datetime


def require_remote_egress(
    *,
    policy: RemoteEgressPolicy | None,
    provider: str,
    destination: str,
    action: str,
    now: datetime | None = None,
) -> RemoteAuthorization:
    """Revalidate an explicit capability at one declared network boundary."""
    if not isinstance(policy, RemoteEgressPolicy):
        raise RemoteEgressError(
            "remote egress requires an explicit in-process policy capability"
        )
    checked_provider = _validate_provider(provider)
    checked_action = _validate_action(action)
    host = _https_hostname(destination)
    current_time = datetime.now(UTC) if now is None else now
    if (
        not isinstance(current_time, datetime)
        or current_time.tzinfo is None
        or current_time.utcoffset() is None
    ):
        raise RemoteEgressError("policy validation time must be timezone-aware")
    if current_time >= policy.expires_at:
        raise RemoteEgressError("remote egress policy has expired")
    if checked_provider != policy.provider:
        raise RemoteEgressError("remote provider is outside the scoped policy")
    if destination != policy.destination:
        raise RemoteEgressError("remote destination does not match the scoped policy")
    if checked_action != policy.action:
        raise RemoteEgressError("remote action does not match the scoped policy")
    return RemoteAuthorization(
        approval_record_id=policy.approval_record_id,
        provider=checked_provider,
        destination=destination,
        host=host,
        action=checked_action,
        expires_at=policy.expires_at,
    )


def _validate_provider(value: object) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise RemoteEgressError(
            "remote provider must use the canonical lowercase identifier grammar"
        )
    return value


def _validate_action(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(ord(character) < 32 for character in value)
    ):
        raise RemoteEgressError("remote action description is invalid")
    return value


def _https_hostname(destination: object) -> str:
    if not isinstance(destination, str):
        raise RemoteEgressError("remote destination must be an HTTPS URL")
    try:
        parsed = urlsplit(destination)
        port = parsed.port
    except ValueError as exc:
        raise RemoteEgressError("remote destination URL is malformed") from exc
    if (
        parsed.scheme != "https"
        or parsed.netloc != parsed.netloc.lower()
        or parsed.username is not None
        or parsed.password is not None
        or parsed.hostname is None
        or port not in (None, 443)  # HTTPS default from RFC 9110 service mapping.
        or parsed.fragment
        or parsed.query
    ):
        raise RemoteEgressError(
            "remote destination must use HTTPS without credentials, query, "
            "nonstandard ports, or fragments"
        )
    return _validate_hostname(parsed.hostname)


def _validate_hostname(value: str) -> str:
    try:
        ipaddress.ip_address(value)
    except ValueError:
        pass
    else:
        raise RemoteEgressError("remote host must be an exact DNS name, not an IP")
    if (
        len(value) > 253  # RFC 1034 full DNS name limit without a trailing dot.
        or _HOST_PATTERN.fullmatch(value) is None
        or value.startswith(".")
        or value.endswith(".")
        or "*" in value
    ):
        raise RemoteEgressError(
            "remote host must be one exact canonical lowercase DNS name"
        )
    return value
