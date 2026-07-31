"""Contract and adapter tests for Anthropic Messages model backends."""

import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from app.model_backends import (
    ModelBackend,
    ModelConversation,
    ModelGeneration,
    ModelReasoning,
    ModelTurn,
    ModelUsage,
)
from app.model_backends.anthropic_messages import (
    AnthropicMessagesBackend,
    create_anthropic_messages_backend,
)
from tests.model_backends.contract import (
    BackendContractCase,
    BackendContractTests,
)

MODEL = "anthropic/Claude-CaseSensitive"
BASE_URL = "https://anthropic.example/api"
API_KEY = "anthropic-secret-never-preview"
SYSTEM_PROMPT = "SYSTEM exact\n第二行"
CONVERSATION = ModelConversation(
    system_prompt=SYSTEM_PROMPT,
    turns=(
        ModelTurn(input="first user", output="first assistant"),
        ModelTurn(input="second user", output="second assistant"),
    ),
    current_input="current user\nexact",
)
CACHE_CONTROL = {"type": "ephemeral", "ttl": "5m"}
EXPECTED_PAYLOAD = {
    "model": MODEL,
    "max_tokens": 1024,
    "system": [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": CACHE_CONTROL,
        }
    ],
    "messages": [
        {
            "role": "user",
            "content": [{"type": "text", "text": "first user"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "first assistant"}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "second user"}],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "second assistant",
                    "cache_control": CACHE_CONTROL,
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "current user\nexact"}],
        },
    ],
}
EXPECTED_GENERATION = ModelGeneration(
    content="visible answer",
    reasoning=(ModelReasoning(type="thinking", text="safe thought"),),
    usage=ModelUsage(
        input_tokens=17,
        output_tokens=6,
        cache_creation_input_tokens=5,
        cache_read_input_tokens=9,
    ),
)


@dataclass(frozen=True, slots=True)
class FakeSettings:
    """Resolved settings for one isolated adapter test."""

    model: str = MODEL
    base_url: str = BASE_URL
    api_key: str = API_KEY


class FakeMessages:
    """Record Anthropic calls and return one configured response at a time."""

    def __init__(self, responses: list[Any]) -> None:
        """Keep FIFO responses and an exact request log."""
        self._responses = responses
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        """Record one call before returning or raising its response."""
        self.requests.append(deepcopy(kwargs))
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    """Expose the minimal Messages and lifecycle client surface."""

    def __init__(self, responses: list[Any]) -> None:
        """Initialize the fake with provider-shaped responses."""
        self.messages = FakeMessages(responses)
        self.close_calls = 0

    def close(self) -> None:
        """Record one lifecycle close."""
        self.close_calls += 1


class CapturingClientFactory:
    """Capture SDK initialization parameters without opening a network."""

    def __init__(self, client: FakeClient) -> None:
        """Store the client returned by the factory."""
        self.client = client
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> FakeClient:
        """Record exact constructor arguments and return the fake client."""
        self.calls.append(kwargs)
        return self.client


def anthropic_response(
    text: str = "visible answer",
    *,
    content: list[Any] | None = None,
    usage: Any | None = None,
) -> SimpleNamespace:
    """Build a provider-shaped Anthropic response."""
    return SimpleNamespace(
        content=(
            [
                SimpleNamespace(
                    type="thinking",
                    thinking="safe thought",
                    signature="private-signature",
                ),
                SimpleNamespace(
                    type="redacted_thinking",
                    data="private-redacted-data",
                ),
                SimpleNamespace(type="text", text=text),
            ]
            if content is None
            else content
        ),
        usage=(
            SimpleNamespace(
                input_tokens=17,
                output_tokens=6,
                cache_creation_input_tokens=5,
                cache_read_input_tokens=9,
            )
            if usage is None
            else usage
        ),
    )


def build_backend(
    responses: list[Any] | None = None,
) -> tuple[AnthropicMessagesBackend, FakeClient]:
    """Create one adapter with an isolated fake client."""
    client = FakeClient(
        [anthropic_response()] if responses is None else responses
    )
    backend = create_anthropic_messages_backend(
        FakeSettings(),
        CapturingClientFactory(client),
    )
    return backend, client


def build_contract_case() -> BackendContractCase:
    """Create an Anthropic implementation for the reusable contract suite."""
    backend, client = build_backend()
    return BackendContractCase(
        backend=backend,
        expected_model=MODEL,
        conversation=CONVERSATION,
        expected_payload=EXPECTED_PAYLOAD,
        expected_generation=EXPECTED_GENERATION,
        upstream_call_count=lambda: len(client.messages.requests),
        close_call_count=lambda: client.close_calls,
        forbidden_payload_values=(API_KEY, BASE_URL),
    )


class TestAnthropicMessagesBackendContract(BackendContractTests):
    """Apply every reusable backend check to the Anthropic adapter."""

    contract_case_factory = staticmethod(build_contract_case)


def test_backend_structurally_satisfies_frozen_port() -> None:
    """The concrete adapter implements the protocol without inheritance."""
    backend, _client = build_backend()

    assert isinstance(backend, ModelBackend)


def test_prepare_has_exact_history_order_and_two_cache_breakpoints() -> None:
    """Cache only the system prompt and latest confirmed assistant output."""
    backend, client = build_backend()

    payload = backend.prepare(CONVERSATION).payload

    assert payload == EXPECTED_PAYLOAD
    assert len(client.messages.requests) == 0
    serialized = json.dumps(payload)
    assert serialized.count('"cache_control"') == 2
    assert "cache_control" not in payload["messages"][-1]["content"][0]


def test_prepare_without_history_caches_only_system() -> None:
    """A first call leaves its current user block outside the prompt cache."""
    backend, _client = build_backend()
    conversation = ModelConversation(
        system_prompt="first system",
        turns=(),
        current_input="first user",
    )

    payload = backend.prepare(conversation).payload

    assert payload == {
        "model": MODEL,
        "max_tokens": 1024,
        "system": [
            {
                "type": "text",
                "text": "first system",
                "cache_control": CACHE_CONTROL,
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "first user"}],
            }
        ],
    }


def test_generate_passes_prepared_payload_once_without_mutation() -> None:
    """Generation delegates the exact prepared payload in one SDK call."""
    backend, client = build_backend()
    prepared = backend.prepare(CONVERSATION)

    result = backend.generate(prepared)

    assert result == EXPECTED_GENERATION
    assert client.messages.requests == [EXPECTED_PAYLOAD]
    assert prepared.payload == EXPECTED_PAYLOAD


def test_generate_exposes_only_readable_thinking() -> None:
    """Ignore signatures, redacted data, and blank thinking blocks."""
    content = [
        {
            "type": "thinking",
            "thinking": "first safe thought",
            "signature": "signature-secret",
        },
        {
            "type": "thinking",
            "thinking": "  ",
            "signature": "blank-signature-secret",
        },
        {
            "type": "redacted_thinking",
            "data": "redacted-secret",
        },
        {"type": "text", "text": "  visible text  "},
    ]
    backend, _client = build_backend([anthropic_response(content=content)])

    result = backend.generate(backend.prepare(CONVERSATION))

    assert result.content == "visible text"
    assert result.reasoning == (
        ModelReasoning(type="thinking", text="first safe thought"),
    )
    serialized = repr(result)
    assert "signature-secret" not in serialized
    assert "blank-signature-secret" not in serialized
    assert "redacted-secret" not in serialized


def test_generate_maps_usage_and_defaults_absent_cache_counts() -> None:
    """Map strict token counts while treating absent cache fields as zero."""
    usage = SimpleNamespace(input_tokens=23, output_tokens=8)
    backend, _client = build_backend([anthropic_response(usage=usage)])

    result = backend.generate(backend.prepare(CONVERSATION))

    assert result.usage == ModelUsage(
        input_tokens=23,
        output_tokens=8,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(content=[], usage=SimpleNamespace()),
        anthropic_response(
            content=[
                {"type": "text", "text": "one"},
                {"type": "text", "text": "two"},
            ]
        ),
        anthropic_response(
            content=[
                {"type": "tool_use", "name": "unsafe"},
                {"type": "text", "text": "visible"},
            ]
        ),
        anthropic_response(
            content=[
                {"type": [], "text": "invalid type shape"},
                {"type": "text", "text": "visible"},
            ]
        ),
        anthropic_response(content=[{"type": "text", "text": "  "}]),
        anthropic_response(
            content=[
                {"type": "thinking", "thinking": 12},
                {"type": "text", "text": "visible"},
            ]
        ),
        anthropic_response(
            usage=SimpleNamespace(
                input_tokens=True,
                output_tokens=1,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            )
        ),
        anthropic_response(
            usage=SimpleNamespace(
                input_tokens=1,
                output_tokens=1,
                cache_creation_input_tokens=-1,
                cache_read_input_tokens=0,
            )
        ),
    ],
)
def test_generate_rejects_unknown_or_unsafe_response_shapes(
    response: Any,
) -> None:
    """Unsafe provider output fails after exactly one call without retry."""
    backend, client = build_backend([response])

    with pytest.raises(ValueError, match="Anthropic"):
        backend.generate(backend.prepare(CONVERSATION))

    assert len(client.messages.requests) == 1


def test_upstream_exception_is_sanitized_and_not_retried() -> None:
    """Provider exceptions cannot disclose their body or credentials."""
    provider_detail = f"provider body with {API_KEY}"
    backend, client = build_backend([OSError(provider_detail)])

    with pytest.raises(RuntimeError) as exc_info:
        backend.generate(backend.prepare(CONVERSATION))

    assert str(exc_info.value) == "Anthropic Messages request failed"
    assert provider_detail not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert len(client.messages.requests) == 1


def test_factory_uses_exact_bounded_sdk_parameters() -> None:
    """Factory passes credentials only to the client constructor."""
    client = FakeClient([anthropic_response()])
    client_factory = CapturingClientFactory(client)

    backend = create_anthropic_messages_backend(
        FakeSettings(),
        client_factory,
    )

    assert backend.model == MODEL
    assert client_factory.calls == [
        {
            "api_key": API_KEY,
            "base_url": BASE_URL,
            "timeout": 60.0,
            "max_retries": 0,
        }
    ]
    assert API_KEY not in json.dumps(backend.prepare(CONVERSATION).payload)


def test_factory_exception_is_sanitized() -> None:
    """Client-construction failures cannot expose settings or secrets."""

    def failing_factory(**_kwargs: Any) -> FakeClient:
        raise OSError(f"cannot connect using {API_KEY}")

    with pytest.raises(RuntimeError) as exc_info:
        create_anthropic_messages_backend(FakeSettings(), failing_factory)

    assert str(exc_info.value) == "Anthropic Messages client creation failed"
    assert API_KEY not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_generation_result_contains_no_private_provider_fields() -> None:
    """The neutral result has only content, readable reasoning, and usage."""
    backend, _client = build_backend()

    result = backend.generate(backend.prepare(CONVERSATION))

    assert asdict(result) == {
        "content": "visible answer",
        "reasoning": ({"type": "thinking", "text": "safe thought"},),
        "usage": {
            "input_tokens": 17,
            "output_tokens": 6,
            "cache_creation_input_tokens": 5,
            "cache_read_input_tokens": 9,
        },
    }
