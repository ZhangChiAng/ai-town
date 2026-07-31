"""Tests for dual model configuration and client lifecycle."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

import app.main as main_module
from app.config import (
    ConfiguredModels,
    ModelConfigError,
    ModelSettings,
    load_model_settings,
)
from app.drafting import create_anthropic_client, create_responses_client
from app.main import create_app
from app.storage import SceneStorage


def _valid_environment() -> dict[str, str]:
    """Return one complete, protocol-consistent process environment."""
    return {
        "ANTHROPIC_BASE_URL": "https://anthropic.example/root",
        "ANTHROPIC_API_KEY": "anthropic-secret",
        "ANTHROPIC_MODEL": "anthropic/claude-test",
        "RESPONSES_BASE_URL": "https://responses.example/v1",
        "RESPONSES_API_KEY": "responses-secret",
        "RESPONSES_MODEL": "gpt-test",
    }


def _configured_models() -> ConfiguredModels:
    """Build settings without reading the developer's real environment."""
    return ConfiguredModels(
        anthropic=ModelSettings(
            base_url="https://anthropic.example/root",
            api_key="anthropic-secret",
            model="anthropic/claude-test",
        ),
        responses=ModelSettings(
            base_url="https://responses.example/v1",
            api_key="responses-secret",
            model="gpt-test",
        ),
    )


def test_environment_values_override_repository_dotenv(tmp_path: Path) -> None:
    """Each process value takes precedence over its dotenv counterpart."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            f'{name}="file-{value}"'
            for name, value in _valid_environment().items()
        ),
        encoding="utf-8",
    )

    settings = load_model_settings(env_file, _valid_environment())

    assert settings == _configured_models()


def test_old_single_model_variables_no_longer_configure_the_app(
    tmp_path: Path,
) -> None:
    """The removed three-variable configuration is fully ignored."""
    environment = {
        "BASE_URL": "https://legacy.example/root",
        "API_KEY": "legacy-secret",
        "MODEL": "claude-legacy",
    }

    with pytest.raises(ModelConfigError) as caught:
        load_model_settings(tmp_path / "missing.env", environment)

    assert str(caught.value) == (
        "Invalid model configuration: ANTHROPIC_BASE_URL, "
        "ANTHROPIC_API_KEY, ANTHROPIC_MODEL, RESPONSES_BASE_URL, "
        "RESPONSES_API_KEY, RESPONSES_MODEL"
    )
    assert "legacy" not in str(caught.value)


@pytest.mark.parametrize(
    ("updates", "expected_names"),
    [
        ({"ANTHROPIC_API_KEY": " "}, "ANTHROPIC_API_KEY"),
        (
            {
                "ANTHROPIC_BASE_URL": "not a URL",
                "RESPONSES_BASE_URL": "ftp://responses.example",
            },
            "ANTHROPIC_BASE_URL, RESPONSES_BASE_URL",
        ),
        (
            {
                "ANTHROPIC_MODEL": "gpt-5",
                "RESPONSES_MODEL": "vendor/CLAUDE-test",
            },
            "ANTHROPIC_MODEL, RESPONSES_MODEL",
        ),
    ],
)
def test_invalid_configuration_reports_only_variable_names(
    tmp_path: Path,
    updates: dict[str, str],
    expected_names: str,
) -> None:
    """Missing, invalid, and protocol-conflicting values stay secret."""
    environment = {**_valid_environment(), **updates}

    with pytest.raises(ModelConfigError) as caught:
        load_model_settings(tmp_path / "missing.env", environment)

    assert str(caught.value) == (
        f"Invalid model configuration: {expected_names}"
    )
    assert "not a URL" not in str(caught.value)
    assert "vendor/CLAUDE-test" not in str(caught.value)


def test_fastapi_startup_fails_when_model_configuration_is_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production startup surfaces a sanitized configuration failure."""

    def fail_settings() -> ConfiguredModels:
        raise ModelConfigError("Invalid model configuration: ANTHROPIC_API_KEY")

    monkeypatch.setattr(main_module, "load_model_settings", fail_settings)
    application = create_app(SceneStorage(tmp_path / "scenes"))

    async def start_application() -> None:
        async with application.router.lifespan_context(application):
            pass

    with pytest.raises(
        ModelConfigError,
        match="Invalid model configuration: ANTHROPIC_API_KEY",
    ):
        asyncio.run(start_application())


@pytest.mark.parametrize(
    ("factory", "settings"),
    [
        (
            create_anthropic_client,
            ModelSettings(
                base_url="https://gateway.example/anthropic",
                api_key="anthropic-key",
                model="claude-haiku-4-5",
            ),
        ),
        (
            create_responses_client,
            ModelSettings(
                base_url="https://gateway.example/v1",
                api_key="responses-key",
                model="gpt-5",
            ),
        ),
    ],
)
def test_clients_use_exact_root_timeout_and_no_retries(
    factory: Any,
    settings: ModelSettings,
) -> None:
    """Both SDKs receive the configured transport options unchanged."""
    captured: dict[str, Any] = {}
    sentinel = object()

    def fake_client(**kwargs: Any) -> object:
        captured.update(kwargs)
        return sentinel

    client = factory(settings, fake_client)

    assert client is sentinel
    assert captured == {
        "api_key": settings.api_key,
        "base_url": settings.base_url,
        "timeout": 60.0,
        "max_retries": 0,
    }


def test_fastapi_initializes_both_services_and_closes_both_clients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One lifespan owns both protocol clients and concrete model services."""
    settings = _configured_models()
    events: list[str] = []

    class ClosingClient:
        """Record lifecycle events for one protocol client."""

        def __init__(self, protocol: str) -> None:
            self.protocol = protocol

        def close(self) -> None:
            events.append(f"close:{self.protocol}")

    anthropic_client = ClosingClient("anthropic")
    responses_client = ClosingClient("responses")
    monkeypatch.setattr(main_module, "load_model_settings", lambda: settings)
    monkeypatch.setattr(
        main_module,
        "create_anthropic_client",
        lambda received: (
            events.append(f"create:{received.model}") or anthropic_client
        ),
    )
    monkeypatch.setattr(
        main_module,
        "create_responses_client",
        lambda received: (
            events.append(f"create:{received.model}") or responses_client
        ),
    )
    application = create_app(SceneStorage(tmp_path / "scenes"))

    async def run_lifespan() -> None:
        async with application.router.lifespan_context(application):
            assert list(application.state.message_draft_services) == [
                settings.anthropic.model,
                settings.responses.model,
            ]
            assert [
                option.model_dump()
                for option in application.state.model_options
            ] == [
                {
                    "protocol": "anthropic",
                    "model": settings.anthropic.model,
                },
                {
                    "protocol": "responses",
                    "model": settings.responses.model,
                },
            ]

    asyncio.run(run_lifespan())

    assert events == [
        f"create:{settings.anthropic.model}",
        f"create:{settings.responses.model}",
        "close:responses",
        "close:anthropic",
    ]


def test_first_client_is_closed_when_second_client_initialization_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial startup cannot leak the already-created Anthropic client."""
    closed = 0

    class ClosingClient:
        """Count cleanup after partial initialization."""

        def close(self) -> None:
            nonlocal closed
            closed += 1

    monkeypatch.setattr(main_module, "load_model_settings", _configured_models)
    monkeypatch.setattr(
        main_module,
        "create_anthropic_client",
        lambda _settings: ClosingClient(),
    )

    def fail_responses(_settings: ModelSettings) -> object:
        raise RuntimeError("responses startup failed")

    monkeypatch.setattr(main_module, "create_responses_client", fail_responses)
    application = create_app(SceneStorage(tmp_path / "scenes"))

    async def run_lifespan() -> None:
        async with application.router.lifespan_context(application):
            pass

    with pytest.raises(RuntimeError, match="responses startup failed"):
        asyncio.run(run_lifespan())

    assert closed == 1
