"""Ordered registry and lifecycle owner for model backends."""

import logging
from collections.abc import Iterable, Iterator, Mapping
from typing import Protocol

from app.model_backends.contracts import (
    BackendFactory,
    ModelBackend,
    ModelBackendSettings,
)
from app.structured_logging import log_event

LOGGER = logging.getLogger(__name__)


class BackendRegistryError(RuntimeError):
    """Raised when backend registration is ambiguous or unsupported."""


class _RegistrySettings(ModelBackendSettings, Protocol):
    """Resolved settings plus the factory key used only by the registry."""

    @property
    def protocol(self) -> str:
        """Return the exact configured backend factory key."""


class ModelBackendRegistry(Mapping[str, ModelBackend]):
    """Read-only, insertion-ordered model-to-backend mapping."""

    def __init__(self, backends: Iterable[ModelBackend]) -> None:
        """Store backends in order and reject ambiguous model names."""
        registered: dict[str, ModelBackend] = {}
        for backend in backends:
            if backend.model in registered:
                raise BackendRegistryError(
                    "Invalid backend registry: duplicate model"
                )
            registered[backend.model] = backend
        self._backends = registered
        self._closed = False

    def __getitem__(self, model: str) -> ModelBackend:
        """Return the backend registered for an exact model name."""
        return self._backends[model]

    def __iter__(self) -> Iterator[str]:
        """Iterate model names in configuration order."""
        return iter(self._backends)

    def __len__(self) -> int:
        """Return the number of registered model names."""
        return len(self._backends)

    @property
    def models(self) -> tuple[str, ...]:
        """Return exact model names in configuration order."""
        return tuple(self._backends)

    async def aclose(self) -> None:
        """Close every owned backend once, in reverse creation order."""
        if self._closed:
            return
        self._closed = True

        first_error: BaseException | None = None
        for backend in reversed(tuple(self._backends.values())):
            try:
                await backend.aclose()
            except BaseException as error:
                # Continue cleanup so one broken backend cannot leak the rest.
                log_event(
                    LOGGER,
                    logging.ERROR,
                    "model.backend_close.failed",
                    "Model backend close failed.",
                    exception=error,
                    model=backend.model,
                )
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


async def create_model_backend_registry(
    settings: Iterable[_RegistrySettings],
    factories: Mapping[str, BackendFactory],
) -> ModelBackendRegistry:
    """Create backends in order and clean up after partial factory failure.

    Args:
        settings: Ordered resolved configurations with protocol factory keys.
        factories: Protocol keys mapped to backend factories.

    Returns:
        A lifecycle-owning ordered registry keyed by exact model name.

    Raises:
        BackendRegistryError: If model names repeat, a protocol has no factory,
            a factory fails, or a factory violates the configured model
            identity.
    """
    ordered_settings = tuple(settings)
    _validate_settings(ordered_settings, factories)

    backends: list[ModelBackend] = []
    factory_failed = False
    for model_settings in ordered_settings:
        backend: ModelBackend | None = None
        backend_model = ""
        try:
            backend = await factories[model_settings.protocol](model_settings)
            if not isinstance(backend, ModelBackend):
                raise TypeError("factory returned an invalid backend")
            backend_model = backend.model
        except BaseException as error:
            log_event(
                LOGGER,
                logging.ERROR,
                "model.backend_creation.failed",
                "Model backend creation failed.",
                exception=error,
                model=model_settings.model,
                protocol=model_settings.protocol,
            )
            cleanup_targets = [*backends]
            if backend is not None:
                cleanup_targets.append(backend)
            await _close_after_failed_creation(cleanup_targets)
            if not isinstance(error, Exception):
                raise
            factory_failed = True
            break

        assert backend is not None
        backends.append(backend)
        if backend_model != model_settings.model:
            await _close_after_failed_creation(backends)
            raise BackendRegistryError(
                "Invalid backend registry: factory returned wrong model"
            ) from None

    if factory_failed:
        # Raising outside the handler drops provider exceptions completely.
        raise BackendRegistryError(
            "Invalid backend registry: backend factory failed"
        ) from None

    registry_failed = False
    registry: ModelBackendRegistry | None = None
    try:
        registry = ModelBackendRegistry(backends)
    except BaseException as error:
        await _close_after_failed_creation(backends)
        if not isinstance(error, Exception):
            raise
        registry_failed = True

    if registry_failed:
        # A hostile or stateful backend identity must not leak startup data.
        raise BackendRegistryError(
            "Invalid backend registry: backend factory failed"
        ) from None
    assert registry is not None
    return registry


def _validate_settings(
    settings: tuple[_RegistrySettings, ...],
    factories: Mapping[str, BackendFactory],
) -> None:
    """Reject deterministic registration errors before creating resources."""
    seen_models: set[str] = set()
    for index, model_settings in enumerate(settings):
        if model_settings.model in seen_models:
            raise BackendRegistryError(
                f"Invalid backend registry at settings[{index}]: "
                "duplicate model"
            )
        seen_models.add(model_settings.model)
        if model_settings.protocol not in factories:
            raise BackendRegistryError(
                f"Invalid backend registry at settings[{index}].protocol: "
                "unknown protocol"
            )


async def _close_after_failed_creation(backends: list[ModelBackend]) -> None:
    """Best-effort reverse cleanup without masking the creation failure."""
    for backend in reversed(backends):
        try:
            await backend.aclose()
        except BaseException as error:
            # The active factory error is the actionable startup failure.
            log_event(
                LOGGER,
                logging.ERROR,
                "model.backend_cleanup.failed",
                "Model backend cleanup after creation failure failed.",
                exception=error,
                model=backend.model,
            )
            continue
