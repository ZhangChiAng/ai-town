"""Tests for the ordered model backend registry."""

import asyncio

import pytest

from app.model_backends.contracts import (
    BackendFactory,
    ModelBackend,
    ModelBackendSettings,
    ModelConversation,
    ModelGeneration,
    ModelUsage,
)
from app.model_backends.registry import (
    BackendRegistryError,
    ModelBackendRegistry,
    create_model_backend_registry,
)
from app.model_config import ModelSettings


class RecordingBackend:
    """Minimal backend that records registry-owned lifecycle events."""

    def __init__(self, model: str, events: list[str]) -> None:
        """Remember exact identity and a shared event sink."""
        self._model = model
        self._events = events

    @property
    def model(self) -> str:
        """Return the configured model exactly."""
        return self._model

    async def generate(
        self,
        conversation: ModelConversation,
    ) -> ModelGeneration:
        """Provide the unused generation method without external I/O."""
        return ModelGeneration(
            content="output",
            reasoning=(),
            usage=ModelUsage(
                input_tokens=0,
                output_tokens=0,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
            request_snapshot={"input": conversation.current_input},
        )

    async def aclose(self) -> None:
        """Record lifecycle cleanup."""
        self._events.append(f"close:{self.model}")


def _settings(
    model: str,
    protocol: str = "shared_protocol",
) -> ModelSettings:
    """Build resolved settings without reading developer configuration."""
    return ModelSettings(
        model=model,
        protocol=protocol,
        base_url="https://example.test/api",
        api_key="test-secret",
    )


def _recording_factory(events: list[str]) -> BackendFactory:
    """Return a factory that records ordered backend creation."""

    async def create(settings: ModelBackendSettings) -> ModelBackend:
        events.append(f"create:{settings.model}")
        return RecordingBackend(settings.model, events)

    return create


def test_registry_preserves_order_with_shared_protocols() -> None:
    """Mapping views and lookup preserve exact TOML-derived ordering."""
    events: list[str] = []
    configured = (
        _settings("Vendor/Model"),
        _settings("vendor/model"),
        _settings("third/model", protocol="other_protocol"),
    )
    factories = {
        "shared_protocol": _recording_factory(events),
        "other_protocol": _recording_factory(events),
    }

    registry = asyncio.run(create_model_backend_registry(configured, factories))

    assert isinstance(registry, ModelBackendRegistry)
    assert registry.models == (
        "Vendor/Model",
        "vendor/model",
        "third/model",
    )
    assert list(registry) == list(registry.models)
    assert list(registry.keys()) == list(registry.models)
    assert [backend.model for backend in registry.values()] == list(
        registry.models
    )
    assert [model for model, _backend in registry.items()] == list(
        registry.models
    )
    assert registry["Vendor/Model"].model == "Vendor/Model"
    assert events == [
        "create:Vendor/Model",
        "create:vendor/model",
        "create:third/model",
    ]
    with pytest.raises(KeyError):
        _ = registry["VENDOR/MODEL"]


def test_unknown_protocol_is_rejected_before_any_factory_runs() -> None:
    """Factory keys are exact and validated before resources are created."""
    events: list[str] = []
    configured = (
        _settings("first"),
        _settings("second", protocol="unsupported-secret-value"),
    )

    with pytest.raises(
        BackendRegistryError, match="unknown protocol"
    ) as caught:
        asyncio.run(
            create_model_backend_registry(
                configured,
                {"shared_protocol": _recording_factory(events)},
            )
        )

    assert events == []
    assert "unsupported-secret-value" not in str(caught.value)


def test_duplicate_models_are_rejected_before_any_factory_runs() -> None:
    """Only exact duplicates collide; model names are not case-folded."""
    events: list[str] = []

    with pytest.raises(BackendRegistryError, match="duplicate model"):
        asyncio.run(
            create_model_backend_registry(
                (_settings("duplicate"), _settings("duplicate")),
                {"shared_protocol": _recording_factory(events)},
            )
        )

    assert events == []


def test_factory_failure_is_sanitized_after_reverse_cleanup() -> None:
    """A failed startup cleans up without exposing factory exception data."""
    events: list[str] = []
    secret = "factory-exception-secret"

    async def fail_on_third(
        settings: ModelBackendSettings,
    ) -> ModelBackend:
        events.append(f"create:{settings.model}")
        if settings.model == "third":
            raise RuntimeError(secret)
        return RecordingBackend(settings.model, events)

    configured = tuple(_settings(name) for name in ("first", "second", "third"))

    with pytest.raises(
        BackendRegistryError, match="backend factory failed"
    ) as caught:
        asyncio.run(
            create_model_backend_registry(
                configured,
                {"shared_protocol": fail_on_third},
            )
        )

    error = caught.value
    assert secret not in str(error)
    assert "third" not in str(error)
    assert "shared_protocol" not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None
    assert events == [
        "create:first",
        "create:second",
        "create:third",
        "close:second",
        "close:first",
    ]


def test_partial_failure_cleanup_attempts_every_backend() -> None:
    """A cleanup error cannot leak an earlier backend or mask startup."""
    events: list[str] = []

    class FailingCleanupBackend(RecordingBackend):
        """Backend that records cleanup before failing."""

        async def aclose(self) -> None:
            """Fail only after making the cleanup attempt observable."""
            await super().aclose()
            raise RuntimeError("cleanup secret")

    async def factory(settings: ModelBackendSettings) -> ModelBackend:
        events.append(f"create:{settings.model}")
        if settings.model == "third":
            raise RuntimeError("factory secret")
        if settings.model == "second":
            return FailingCleanupBackend(settings.model, events)
        return RecordingBackend(settings.model, events)

    with pytest.raises(
        BackendRegistryError, match="backend factory failed"
    ) as caught:
        asyncio.run(
            create_model_backend_registry(
                tuple(_settings(name) for name in ("first", "second", "third")),
                {"shared_protocol": factory},
            )
        )

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert events == [
        "create:first",
        "create:second",
        "create:third",
        "close:second",
        "close:first",
    ]


def test_invalid_factory_result_is_sanitized_and_reverse_cleaned() -> None:
    """Returned objects are validated inside the protected startup boundary."""
    events: list[str] = []
    secret = "secret-from-model-property"

    class ExplodingIdentityBackend(RecordingBackend):
        """Own a resource but fail while exposing its registry identity."""

        @property
        def model(self) -> str:
            """Simulate a malformed adapter property with sensitive detail."""
            raise RuntimeError(secret)

        async def aclose(self) -> None:
            """Record cleanup without reading the broken model property."""
            events.append("close:exploding")

    async def factory(settings: ModelBackendSettings) -> ModelBackend:
        events.append(f"create:{settings.model}")
        if settings.model == "second":
            return ExplodingIdentityBackend(settings.model, events)
        return RecordingBackend(settings.model, events)

    with pytest.raises(
        BackendRegistryError, match="backend factory failed"
    ) as caught:
        asyncio.run(
            create_model_backend_registry(
                (_settings("first"), _settings("second")),
                {"shared_protocol": factory},
            )
        )

    error = caught.value
    assert secret not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None
    assert events == [
        "create:first",
        "create:second",
        "close:exploding",
        "close:first",
    ]


def test_none_factory_result_closes_previously_created_backends() -> None:
    """A factory returning no backend is a sanitized partial failure."""
    events: list[str] = []

    async def factory(settings: ModelBackendSettings) -> ModelBackend:
        events.append(f"create:{settings.model}")
        if settings.model == "second":
            return None  # type: ignore[return-value]
        return RecordingBackend(settings.model, events)

    with pytest.raises(BackendRegistryError, match="backend factory failed"):
        asyncio.run(
            create_model_backend_registry(
                (_settings("first"), _settings("second")),
                {"shared_protocol": factory},
            )
        )

    assert events == ["create:first", "create:second", "close:first"]


def test_normal_close_is_reverse_order_and_idempotent() -> None:
    """The registry owns one reverse-order close pass on normal shutdown."""
    events: list[str] = []
    registry = asyncio.run(
        create_model_backend_registry(
            tuple(_settings(name) for name in ("first", "second", "third")),
            {"shared_protocol": _recording_factory(events)},
        )
    )

    asyncio.run(registry.aclose())
    asyncio.run(registry.aclose())

    assert events == [
        "create:first",
        "create:second",
        "create:third",
        "close:third",
        "close:second",
        "close:first",
    ]


def test_factory_model_mismatch_is_closed_and_rejected() -> None:
    """A factory cannot silently register a backend under a false identity."""
    events: list[str] = []

    async def mismatched_factory(
        _settings: ModelBackendSettings,
    ) -> ModelBackend:
        events.append("create:configured")
        return RecordingBackend("different", events)

    with pytest.raises(BackendRegistryError, match="wrong model"):
        asyncio.run(
            create_model_backend_registry(
                (_settings("configured"),),
                {"shared_protocol": mismatched_factory},
            )
        )

    assert events == ["create:configured", "close:different"]


def test_close_attempts_every_backend_when_one_close_fails() -> None:
    """One close error is re-raised only after remaining resources are freed."""
    events: list[str] = []

    class FailingCloseBackend(RecordingBackend):
        """Backend double whose close records and then fails."""

        async def aclose(self) -> None:
            """Record the attempted cleanup before raising."""
            await super().aclose()
            raise RuntimeError("close failed")

    async def failing_factory(
        settings: ModelBackendSettings,
    ) -> ModelBackend:
        return FailingCloseBackend(settings.model, events)

    factories: dict[str, BackendFactory] = {
        "normal": _recording_factory(events),
        "failing": failing_factory,
    }
    registry = asyncio.run(
        create_model_backend_registry(
            (
                _settings("first", protocol="normal"),
                _settings("second", protocol="failing"),
                _settings("third", protocol="normal"),
            ),
            factories,
        )
    )

    with pytest.raises(RuntimeError, match="close failed"):
        asyncio.run(registry.aclose())

    assert events[-3:] == ["close:third", "close:second", "close:first"]
    events_after_failure = list(events)
    asyncio.run(registry.aclose())
    assert events == events_after_failure
