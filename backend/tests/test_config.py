"""Tests for application model-backend startup and lifecycle wiring."""

import asyncio
from pathlib import Path

import pytest

import app.main as main_module
from app.main import create_app
from app.model_backends import ModelBackendSettings
from app.model_config import ModelSettings
from app.storage import SceneStorage
from tests.helpers import FakeBackend


def _settings(model: str, protocol: str = "shared") -> ModelSettings:
    """Build resolved settings without reading repository configuration."""
    return ModelSettings(
        model=model,
        protocol=protocol,
        base_url="https://example.test/api",
        api_key="test-secret",
    )


def test_lifespan_builds_toml_order_and_closes_backends_in_reverse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup owns an ordered registry and reverses it on shutdown."""
    configured = tuple(_settings(model) for model in ("one", "two", "three"))
    events: list[str] = []

    async def factory(settings: ModelBackendSettings) -> FakeBackend:
        events.append(f"create:{settings.model}")
        return FakeBackend([], model=settings.model, events=events)

    monkeypatch.setattr(main_module, "load_model_settings", lambda: configured)
    monkeypatch.setattr(main_module, "BACKEND_FACTORIES", {"shared": factory})
    application = create_app(SceneStorage(tmp_path / "scenes"))

    async def run_lifespan() -> None:
        async with application.router.lifespan_context(application):
            assert list(application.state.model_backends) == [
                "one",
                "two",
                "three",
            ]
            assert [
                option.model_dump()
                for option in application.state.model_options
            ] == [
                {"model": "one"},
                {"model": "two"},
                {"model": "three"},
            ]

    asyncio.run(run_lifespan())

    assert events == [
        "create:one",
        "create:two",
        "create:three",
        "close:three",
        "close:two",
        "close:one",
    ]


def test_injected_backend_bypasses_configuration_and_is_caller_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests can inject one backend without loading or closing it implicitly."""
    backend = FakeBackend([], model="injected")

    def fail_if_loaded() -> tuple[ModelSettings, ...]:
        raise AssertionError("configuration must not load")

    monkeypatch.setattr(main_module, "load_model_settings", fail_if_loaded)
    application = create_app(SceneStorage(tmp_path / "scenes"), backend)

    async def run_lifespan() -> None:
        async with application.router.lifespan_context(application):
            assert list(application.state.model_backends) == ["injected"]

    asyncio.run(run_lifespan())

    assert backend.close_calls == 0
