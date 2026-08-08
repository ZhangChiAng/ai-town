"""Protocol-neutral model backend ports and concrete adapters."""

from app.model_backends.anthropic_messages import (
    create_anthropic_messages_backend,
)
from app.model_backends.contracts import (
    BackendFactory,
    JsonObject,
    JsonValue,
    LoggedModelError,
    ModelBackend,
    ModelBackendSettings,
    ModelConversation,
    ModelGeneration,
    ModelReasoning,
    ModelTurn,
    ModelUsage,
    ReasoningType,
)
from app.model_backends.deepseek_responses import (
    create_deepseek_responses_backend,
)
from app.model_backends.minimax_responses import (
    create_minimax_responses_backend,
)
from app.model_backends.pydantic_ai_backend import (
    PydanticAIBackend,
    create_request_capture_client,
)
from app.model_backends.registry import (
    BackendRegistryError,
    ModelBackendRegistry,
    create_model_backend_registry,
)

__all__ = [
    "BackendFactory",
    "BackendRegistryError",
    "JsonObject",
    "JsonValue",
    "LoggedModelError",
    "ModelBackend",
    "ModelBackendRegistry",
    "ModelBackendSettings",
    "ModelConversation",
    "ModelGeneration",
    "ModelReasoning",
    "ModelTurn",
    "ModelUsage",
    "PydanticAIBackend",
    "ReasoningType",
    "create_anthropic_messages_backend",
    "create_deepseek_responses_backend",
    "create_minimax_responses_backend",
    "create_model_backend_registry",
    "create_request_capture_client",
]
