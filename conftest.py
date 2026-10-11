# Root conftest so `pytest` puts the repo root on sys.path and tests can import contracts.
import pytest


@pytest.fixture(autouse=True)
def no_live_llm_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never call Gemini, Nemotron, or a SLIM node: hints.py loads .env.local on import, so unset their settings.

    Tests that need a key pass one explicitly (and mock the client).
    """
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "NEMOTRON_BASE_URL", "NEMOTRON_MODEL", "NEMOTRON_API_KEY", "NEMOTRON_SYSTEM", "NEMOTRON_ENABLE_THINKING",
                "SLIM_ENDPOINT", "SLIM_SHARED_SECRET", "SLIM_CHANNEL", "SLIM_OBSERVERS", "SLIM_APP_NAME"):
        monkeypatch.delenv(key, raising=False)
