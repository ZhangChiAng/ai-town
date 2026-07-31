"""Tests for application model-backend startup and lifecycle wiring."""

import asyncio
from pathlib import Path

import pytest

import app.main as main_module
from app.main import create_app
from app.model_backends import (
    BackendRegistryError,
    ModelBackendSettings,
    ModelConversation,
    ModelGeneration,
    ModelUsage,
    PreparedModelRequest,
)
from app.model_config import ModelSettings
from app.storage import SceneStorage


class ClosingBackend:
    """Minimal backend that records application-owned lifecycle events."""

    def __init__(self, model: str, events: list[str]) -> None:
        """Bind one exact model and shared event sink."""
        self._model = model
        self._events = events
        self.close_calls = 0

    @property
    def model(self) -> str:
        """Return the configured model name."""
        return self._model

    def prepare(
        self,
        conversation: ModelConversation,
    ) -> PreparedModelRequest:
        """Provide an unused JSON-safe payload for structural compatibility."""
        return PreparedModelRequest(
            payload={"engine": self.model, "next": conversation.current_input}
        )

    def generate(
        self,
        prepared: PreparedModelRequest,
    ) -> ModelGeneration:
        """Provide an unused neutral result for structural compatibility."""
        del prepared
        return ModelGeneration(
            content="unused",
            reasoning=(),
            usage=ModelUsage(
                input_tokens=0,
                output_tokens=0,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
        )

    def close(self) -> None:
        """Record one resource-release call."""
        self.close_calls += 1
        self._events.append(f"close:{self.model}")


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
    """Startup owns an ordered registry for arbitrary model counts."""
    configured = tuple(_settings(model) for model in ("one", "two", "three"))
    events: list[str] = []

    def factory(settings: ModelBackendSettings) -> ClosingBackend:
        events.append(f"create:{settings.model}")
        return ClosingBackend(settings.model, events)

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


def test_lifespan_sanitizes_partial_factory_failure_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later startup failure closes earlier resources without data leaks."""
    configured = (_settings("first"), _settings("second"))
    events: list[str] = []
    secret = "provider failure containing test-secret"

    def factory(settings: ModelBackendSettings) -> ClosingBackend:
        events.append(f"create:{settings.model}")
        if settings.model == "second":
            raise RuntimeError(secret)
        return ClosingBackend(settings.model, events)

    monkeypatch.setattr(main_module, "load_model_settings", lambda: configured)
    monkeypatch.setattr(main_module, "BACKEND_FACTORIES", {"shared": factory})
    application = create_app(SceneStorage(tmp_path / "scenes"))

    async def start_application() -> None:
        async with application.router.lifespan_context(application):
            pass

    with pytest.raises(BackendRegistryError) as caught:
        asyncio.run(start_application())

    assert (
        str(caught.value) == "Invalid backend registry: backend factory failed"
    )
    assert secret not in str(caught.value)
    assert events == ["create:first", "create:second", "close:first"]


def test_injected_backend_bypasses_configuration_and_is_caller_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests can inject one backend without loading or closing it implicitly."""
    events: list[str] = []
    backend = ClosingBackend("injected", events)

    def fail_if_loaded() -> tuple[ModelSettings, ...]:
        raise AssertionError("configuration must not load")

    monkeypatch.setattr(main_module, "load_model_settings", fail_if_loaded)
    application = create_app(SceneStorage(tmp_path / "scenes"), backend)

    async def run_lifespan() -> None:
        async with application.router.lifespan_context(application):
            assert list(application.state.model_backends) == ["injected"]

    asyncio.run(run_lifespan())

    assert backend.close_calls == 0
    assert events == []
