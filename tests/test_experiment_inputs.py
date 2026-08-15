import pytest

from src.experiment_inputs import synthetic_text


def test_synthetic_text_uses_default_without_override(monkeypatch) -> None:
    monkeypatch.delenv("SECRET", raising=False)
    monkeypatch.delenv("SYNTHETIC_PAYLOAD_CONFIRMED", raising=False)

    assert synthetic_text("HELLO") == "HELLO"


def test_synthetic_text_rejects_unconfirmed_override(monkeypatch) -> None:
    monkeypatch.setenv("SECRET", "NOT-A-CREDENTIAL")
    monkeypatch.delenv("SYNTHETIC_PAYLOAD_CONFIRMED", raising=False)

    with pytest.raises(RuntimeError, match="SYNTHETIC_PAYLOAD_CONFIRMED=yes"):
        synthetic_text("HELLO")


def test_synthetic_text_accepts_confirmed_override(monkeypatch) -> None:
    monkeypatch.setenv("SECRET", "SYNTHETIC-FIXTURE")
    monkeypatch.setenv("SYNTHETIC_PAYLOAD_CONFIRMED", "yes")

    assert synthetic_text("HELLO") == "SYNTHETIC-FIXTURE"
