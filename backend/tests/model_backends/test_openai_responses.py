"""Tests for the protocol-specific OpenAI Responses backend."""

from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from app.model_backends import (
    BackendFactory,
    ModelBackend,
    ModelConversation,
    ModelGeneration,
    ModelReasoning,
    ModelTurn,
    ModelUsage,
)
from app.model_backends.openai_responses import (
    OpenAIResponsesBackend,
    create_openai_responses_backend,
)
from tests.model_backends.contract import (
    BackendContractCase,
    BackendContractTests,
)

MODEL = "vendor/GPT-Case-Sensitive"
BASE_URL = "https://responses.example/v1"
SECRET = "responses-secret-never-preview"
CONVERSATION = ModelConversation(
    system_prompt="SYSTEM exact\nsecond line",
    turns=(
        ModelTurn(input="first user", output="first assistant"),
        ModelTurn(input="second user", output="second assistant"),
    ),
    current_input="current user\nexact",
)
EXPECTED_PAYLOAD = {
    "model": MODEL,
    "instructions": "SYSTEM exact\nsecond line",
    "input": [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "first user"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "input_text", "text": "first assistant"}],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "second user"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "input_text", "text": "second assistant"}],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "current user\nexact"}],
        },
    ],
    "max_output_tokens": 2048,
    "store": False,
}
EXPECTED_GENERATION = ModelGeneration(
    content="visible answer",
    reasoning=(
        ModelReasoning(type="summary_text", text="short summary"),
        ModelReasoning(type="reasoning_text", text="readable reasoning"),
    ),
    usage=ModelUsage(
        input_tokens=12,
        output_tokens=7,
        cache_creation_input_tokens=3,
        cache_read_input_tokens=5,
    ),
)


@dataclass(frozen=True, slots=True)
class FakeSettings:
    """Resolved settings passed to the adapter factory."""

    model: str = MODEL
    base_url: str = BASE_URL
    api_key: str = SECRET


class FakeResponsesResource:
    """Capture calls and return one configured provider result."""

    def __init__(self, response: Any) -> None:
        """Store the response or exception returned by create."""
        self.response = response
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        """Record exactly one request before returning its result."""
        self.requests.append(deepcopy(kwargs))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FakeClient:
    """Minimal OpenAI client double with lifecycle counters."""

    def __init__(self, response: Any) -> None:
        """Expose the fake Responses resource."""
        self.responses = FakeResponsesResource(response)
        self.close_calls = 0

    def close(self) -> None:
        """Record one close operation."""
        self.close_calls += 1


def valid_response() -> dict[str, Any]:
    """Return a strict completed response with reasoning and usage."""
    return {
        "status": "completed",
        "output": [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "short summary"}],
                "content": [
                    {
                        "type": "reasoning_text",
                        "text": "readable reasoning",
                    }
                ],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "visible answer"}],
            },
        ],
        "usage": {
            "input_tokens": 20,
            "output_tokens": 7,
            "total_tokens": 27,
            "input_tokens_details": {
                "cache_write_tokens": 3,
                "cached_tokens": 5,
            },
        },
    }


def build_contract_case() -> BackendContractCase:
    """Create one isolated Responses backend for the shared contract suite."""
    client = FakeClient(valid_response())
    backend = OpenAIResponsesBackend(MODEL, client)
    return BackendContractCase(
        backend=backend,
        expected_model=MODEL,
        conversation=CONVERSATION,
        expected_payload=EXPECTED_PAYLOAD,
        expected_generation=EXPECTED_GENERATION,
        upstream_call_count=lambda: len(client.responses.requests),
        close_call_count=lambda: client.close_calls,
        forbidden_payload_values=(SECRET, BASE_URL),
    )


class TestOpenAIResponsesBackendContract(BackendContractTests):
    """Apply the reusable backend contract to OpenAI Responses."""

    contract_case_factory = staticmethod(build_contract_case)


def generate_response(response: Any) -> ModelGeneration:
    """Generate once using a standard prepared request and fake client."""
    backend = OpenAIResponsesBackend(MODEL, FakeClient(response))
    return backend.generate(backend.prepare(CONVERSATION))


def test_prepare_preserves_full_history_without_anthropic_fields() -> None:
    """Preparation maps every turn verbatim into the Responses wire shape."""
    backend = OpenAIResponsesBackend(MODEL, FakeClient(valid_response()))

    payload = backend.prepare(CONVERSATION).payload

    assert payload == EXPECTED_PAYLOAD
    assert {"messages", "system", "cache_control"}.isdisjoint(payload)
    assert all(
        block["type"] == "input_text"
        for message in payload["input"]
        for block in message["content"]
    )


def test_factory_uses_exact_bounded_client_parameters() -> None:
    """The factory initializes one client without retries or hidden defaults."""
    received: list[dict[str, Any]] = []
    client = FakeClient(valid_response())

    def client_factory(**kwargs: Any) -> FakeClient:
        received.append(kwargs)
        return client

    factory: BackendFactory = create_openai_responses_backend
    backend = create_openai_responses_backend(
        FakeSettings(),
        client_factory=client_factory,
    )

    assert factory is create_openai_responses_backend
    assert isinstance(backend, ModelBackend)
    assert backend.model == MODEL
    assert received == [
        {
            "api_key": SECRET,
            "base_url": BASE_URL,
            "timeout": 60.0,
            "max_retries": 0,
        }
    ]
    assert client.responses.requests == []


def test_upstream_error_is_sanitized_without_retry() -> None:
    """Provider failures escape once through a credential-free error."""
    client = FakeClient(RuntimeError(f"body contains {SECRET}"))
    backend = OpenAIResponsesBackend(MODEL, client)

    with pytest.raises(RuntimeError) as raised:
        backend.generate(backend.prepare(CONVERSATION))

    assert str(raised.value) == "OpenAI Responses request failed"
    assert SECRET not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert len(client.responses.requests) == 1


def test_client_creation_error_is_sanitized() -> None:
    """Factory failures cannot expose the secret passed during setup."""

    def failing_client_factory(**_kwargs: Any) -> FakeClient:
        raise RuntimeError(f"initialization included {SECRET}")

    with pytest.raises(RuntimeError) as raised:
        create_openai_responses_backend(
            FakeSettings(),
            client_factory=failing_client_factory,
        )

    assert str(raised.value) == "OpenAI Responses client creation failed"
    assert SECRET not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_reasoning_exposes_only_non_blank_readable_blocks() -> None:
    """Empty readable fields are omitted while provider text is preserved."""
    response = valid_response()
    response["output"][0]["summary"].append(
        {"type": "summary_text", "text": " \n"}
    )
    response["output"][0]["content"].append(
        {"type": "reasoning_text", "text": "second thought"}
    )

    generation = generate_response(response)

    assert generation.reasoning == (
        ModelReasoning(type="summary_text", text="short summary"),
        ModelReasoning(type="reasoning_text", text="readable reasoning"),
        ModelReasoning(type="reasoning_text", text="second thought"),
    )


@pytest.mark.parametrize(
    ("section", "block"),
    [
        ("summary", {"type": "reasoning_text", "text": "wrong list"}),
        ("content", {"type": "summary_text", "text": "wrong list"}),
        ("content", {"type": "tool_result", "text": "unknown"}),
    ],
)
def test_unknown_or_misplaced_reasoning_blocks_are_rejected(
    section: str,
    block: dict[str, str],
) -> None:
    """Reasoning permits only each provider-defined readable block type."""
    response = valid_response()
    response["output"][0][section].append(block)

    with pytest.raises(ValueError, match="reasoning block"):
        generate_response(response)


def test_encrypted_reasoning_is_ignored_without_exposure() -> None:
    """Encrypted provider state is ignored while visible output stays valid."""
    response = valid_response()
    response["output"][0]["encrypted_content"] = "private-ciphertext"

    generation = generate_response(response)

    assert generation == EXPECTED_GENERATION
    assert "private-ciphertext" not in repr(generation)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda response: response.update(status="failed"),
        lambda response: response["output"].append(
            {"type": "tool_call", "name": "unknown"}
        ),
        lambda response: response["output"].append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "duplicate"}],
            }
        ),
        lambda response: response["output"][1].update(role="user"),
        lambda response: response["output"][1]["content"].append(
            {"type": "output_text", "text": "duplicate"}
        ),
        lambda response: response["output"][1]["content"][0].update(
            type="refusal"
        ),
        lambda response: response["output"][1]["content"][0].update(text="  "),
    ],
    ids=(
        "incomplete",
        "unknown-item",
        "duplicate-message",
        "non-assistant",
        "duplicate-output-text",
        "unknown-content-block",
        "blank-output",
    ),
)
def test_invalid_completed_response_shapes_raise_value_error(
    mutate: Any,
) -> None:
    """Malformed provider results fail locally instead of being normalized."""
    response = valid_response()
    mutate(response)

    with pytest.raises(ValueError):
        generate_response(response)


def test_usage_accepts_matching_compatible_locations() -> None:
    """Equal cache counts may appear in details and compatibility fields."""
    response = valid_response()
    response["usage"].update(cache_write_tokens=3, cached_tokens=5)

    generation = generate_response(response)

    assert generation.usage == EXPECTED_GENERATION.usage


def test_usage_accepts_top_level_compatibility_fields_without_details() -> None:
    """Compatibility endpoints may place cache counts on usage itself."""
    response = valid_response()
    response["usage"].pop("input_tokens_details")
    response["usage"].update(cache_write_tokens=3, cached_tokens=5)

    generation = generate_response(response)

    assert generation.usage == EXPECTED_GENERATION.usage


def test_usage_accepts_missing_optional_total_tokens() -> None:
    """Compatibility endpoints may omit the otherwise validated total."""
    response = valid_response()
    response["usage"].pop("total_tokens")

    generation = generate_response(response)

    assert generation.usage == EXPECTED_GENERATION.usage


@pytest.mark.parametrize(
    "mutate",
    [
        lambda usage: usage.update(cache_write_tokens=4),
        lambda usage: usage.update(total_tokens=28),
        lambda usage: usage.update(input_tokens=True),
        lambda usage: usage.update(output_tokens=-1, total_tokens=19),
        lambda usage: usage["input_tokens_details"].update(cached_tokens=21),
        lambda usage: usage.update(input_tokens_details=None),
        lambda usage: usage.update(input_tokens_details="invalid"),
    ],
    ids=(
        "conflicting-location",
        "inconsistent-total",
        "bool-count",
        "negative-count",
        "cache-over-total",
        "null-details",
        "invalid-details",
    ),
)
def test_invalid_usage_is_rejected(mutate: Any) -> None:
    """Usage counts must be exact, non-negative, and internally consistent."""
    response = valid_response()
    mutate(response["usage"])

    with pytest.raises(ValueError):
        generate_response(response)


@pytest.mark.parametrize("top_level_fallback", [None, 3])
def test_nullable_optional_cache_count_uses_zero_or_other_location(
    top_level_fallback: int | None,
) -> None:
    """A null optional detail is absent and may fall back to usage itself."""
    response = valid_response()
    response["usage"]["input_tokens_details"]["cache_write_tokens"] = None
    if top_level_fallback is not None:
        response["usage"]["cache_write_tokens"] = top_level_fallback

    generation = generate_response(response)

    assert generation.usage == ModelUsage(
        input_tokens=15 - (top_level_fallback or 0),
        output_tokens=7,
        cache_creation_input_tokens=top_level_fallback or 0,
        cache_read_input_tokens=5,
    )


def test_sdk_style_object_response_is_supported() -> None:
    """Parsing reads SDK model attributes as well as test-friendly mappings."""

    def to_object(value: Any) -> Any:
        if isinstance(value, dict):
            return SimpleNamespace(
                **{key: to_object(item) for key, item in value.items()}
            )
        if isinstance(value, list):
            return [to_object(item) for item in value]
        return value

    generation = generate_response(to_object(valid_response()))

    assert generation == EXPECTED_GENERATION
