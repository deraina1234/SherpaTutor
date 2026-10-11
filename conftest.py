# Root conftest so `pytest` puts the repo root on sys.path and tests can import contracts.
import pytest


@pytest.fixture(autouse=True)
def no_live_llm_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never call Gemini: hints.py loads .env.local on import, so unset its keys.

    Tests that need a key pass one explicitly (and mock the client).
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
