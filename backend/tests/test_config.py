"""Tests for repository-root model configuration and client setup."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

import app.main as main_module
from app.config import ModelConfigError, ModelSettings, load_model_settings
from app.drafting import create_anthropic_client
from app.main import create_app
from app.storage import SceneStorage


def test_environment_values_override_repository_dotenv(tmp_path: Path) -> None:
    """Process environment values take precedence over dotenv values."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            (
                'BASE_URL="https://file.example/anthropic"',
                'API_KEY="file-secret"',
                'MODEL="file-model"',
            )
        ),
        encoding="utf-8",
    )

    settings = load_model_settings(
        env_file,
        {
            "BASE_URL": " https://environment.example/root ",
            "API_KEY": " environment-secret ",
            "MODEL": " environment-model ",
        },
    )

    assert settings == ModelSettings(
        base_url="https://environment.example/root",
        api_key="environment-secret",
        model="environment-model",
    )


@pytest.mark.parametrize(
    ("environment", "expected_names"),
    [
        ({}, "BASE_URL, API_KEY, MODEL"),
        (
            {
                "BASE_URL": "not a URL",
                "API_KEY": "super-secret",
                "MODEL": " ",
            },
            "BASE_URL, MODEL",
        ),
    ],
)
def test_invalid_configuration_reports_only_variable_names(
    tmp_path: Path,
    environment: dict[str, str],
    expected_names: str,
) -> None:
    """Missing, blank, and invalid values never appear in the error."""
    with pytest.raises(ModelConfigError) as caught:
        load_model_settings(tmp_path / "missing.env", environment)

    assert str(caught.value) == (
        f"Invalid model configuration: {expected_names}"
    )
    assert "super-secret" not in str(caught.value)
    assert "not a URL" not in str(caught.value)


def test_fastapi_startup_fails_when_model_configuration_is_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production startup surfaces a sanitized configuration failure."""

    def fail_settings() -> ModelSettings:
        raise ModelConfigError("Invalid model configuration: API_KEY")

    monkeypatch.setattr(main_module, "load_model_settings", fail_settings)
    application = create_app(SceneStorage(tmp_path / "scenes"))

    async def start_application() -> None:
        async with application.router.lifespan_context(application):
            pass

    with pytest.raises(
        ModelConfigError,
        match="Invalid model configuration: API_KEY",
    ):
        asyncio.run(start_application())


def test_anthropic_client_uses_exact_root_timeout_and_no_retries() -> None:
    """The SDK client receives the configured transport options."""
    captured: dict[str, Any] = {}
    sentinel = object()

    def fake_anthropic(**kwargs: Any) -> object:
        captured.update(kwargs)
        return sentinel

    settings = ModelSettings(
        base_url="https://gateway.example/anthropic",
        api_key="secret-key",
        model="claude-haiku-4-5",
    )

    client = create_anthropic_client(settings, fake_anthropic)

    assert client is sentinel
    assert captured == {
        "api_key": "secret-key",
        "base_url": "https://gateway.example/anthropic",
        "timeout": 60.0,
        "max_retries": 0,
    }
