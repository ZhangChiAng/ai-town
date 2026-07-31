"""Protocol-neutral contracts for model backend adapters."""

import math
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

type JsonValue = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)
type JsonObject = dict[str, JsonValue]
type ReasoningType = Literal["thinking", "summary_text", "reasoning_text"]


def _require_non_blank(value: str, field_name: str) -> None:
    """Require meaningful text without changing its exact whitespace."""
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def _require_json_safe(value: JsonValue) -> None:
    """Reject values that JSON would coerce or serialize non-portably."""
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if math.isfinite(value):
            return
        raise ValueError("payload floats must be finite")
    if type(value) is list:
        for item in value:
            _require_json_safe(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("payload object keys must be strings")
            _require_json_safe(item)
        return
    raise ValueError("payload must contain only JSON-safe values")


@dataclass(frozen=True, slots=True)
class ModelTurn:
    """One confirmed protocol-neutral user/assistant exchange."""

    input: str
    output: str

    def __post_init__(self) -> None:
        """Validate text while preserving it verbatim."""
        _require_non_blank(self.input, "input")
        _require_non_blank(self.output, "output")


@dataclass(frozen=True, slots=True)
class ModelConversation:
    """Complete context visible to one model call."""

    system_prompt: str
    turns: tuple[ModelTurn, ...]
    current_input: str

    def __post_init__(self) -> None:
        """Require the two current-call text fields to be meaningful."""
        _require_non_blank(self.system_prompt, "system_prompt")
        _require_non_blank(self.current_input, "current_input")


@dataclass(frozen=True, slots=True)
class PreparedModelRequest:
    """Credential-free, JSON-safe request payload prepared for one backend."""

    payload: JsonObject

    def __post_init__(self) -> None:
        """Keep preview and generation payloads safely serializable."""
        _require_json_safe(self.payload)


@dataclass(frozen=True, slots=True)
class ModelReasoning:
    """One provider-approved, browser-only reasoning text block."""

    type: ReasoningType
    text: str

    def __post_init__(self) -> None:
        """Reject empty reasoning blocks without normalizing text."""
        _require_non_blank(self.text, "reasoning text")


@dataclass(frozen=True, slots=True)
class ModelUsage:
    """Protocol-neutral token accounting for one generation."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int

    def __post_init__(self) -> None:
        """Require exact non-negative integer counts."""
        for field_name in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ModelGeneration:
    """Validated protocol-neutral result of one upstream call."""

    content: str
    reasoning: tuple[ModelReasoning, ...]
    usage: ModelUsage

    def __post_init__(self) -> None:
        """Require visible model output without changing it."""
        _require_non_blank(self.content, "content")


@runtime_checkable
class ModelBackendSettings(Protocol):
    """Resolved settings surface consumed by backend factories."""

    @property
    def model(self) -> str:
        """Return the configured, case-sensitive model name."""

    @property
    def base_url(self) -> str:
        """Return the validated upstream API root."""

    @property
    def api_key(self) -> str:
        """Return the resolved secret used only to initialize a client."""


@runtime_checkable
class ModelBackend(Protocol):
    """Hexagonal port implemented by every concrete model adapter."""

    @property
    def model(self) -> str:
        """Return the exact configured model name."""

    def prepare(
        self,
        conversation: ModelConversation,
    ) -> PreparedModelRequest:
        """Purely prepare one credential-free provider payload."""

    def generate(
        self,
        prepared: PreparedModelRequest,
    ) -> ModelGeneration:
        """Call the upstream provider exactly once for the prepared payload."""

    def close(self) -> None:
        """Release resources owned by this backend."""


class BackendFactory(Protocol):
    """Callable that creates one backend from resolved model settings."""

    def __call__(
        self,
        settings: ModelBackendSettings,
        /,
    ) -> ModelBackend:
        """Create a backend for exactly one configured model."""
