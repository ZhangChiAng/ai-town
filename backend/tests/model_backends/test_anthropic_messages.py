"""Captured-wire tests for the Anthropic Pydantic AI adapter."""

import asyncio
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from app.model_backends import (
    JsonObject,
    ModelBackend,
    ModelConversation,
    ModelGeneration,
    ModelReasoning,
    ModelTurn,
    ModelUsage,
)
from app.model_backends import anthropic_messages as adapter_module
from app.model_backends.anthropic_messages import (
    create_anthropic_messages_backend,
)
from tests.model_backends.contract import (
    BackendContractCase,
    BackendContractTests,
)

MODEL = "claude-sonnet-4-5"
BASE_URL = "https://anthropic.example"
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
CACHE_CONTROL: JsonObject = {"type": "ephemeral", "ttl": "5m"}
EXPECTED_PAYLOAD: JsonObject = {
    "max_tokens": 1024,
    "messages": [
        {
            "role": "user",
            "content": [{"text": "first user", "type": "text"}],
        },
        {
            "role": "assistant",
            "content": [{"text": "first assistant", "type": "text"}],
        },
        {
            "role": "user",
            "content": [{"text": "second user", "type": "text"}],
        },
        {
            "role": "assistant",
            "content": [{"text": "second assistant", "type": "text"}],
        },
        {
            "role": "user",
            "content": [
                {
                    "text": "current user\nexact",
                    "type": "text",
                    "cache_control": CACHE_CONTROL,
                }
            ],
        },
    ],
    "model": MODEL,
    "stream": False,
    "system": [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": CACHE_CONTROL,
        }
    ],
}


@dataclass(frozen=True, slots=True)
class FakeSettings:
    """Resolved settings for one isolated adapter test."""

    model: str = MODEL
    base_url: str = BASE_URL
    api_key: str = API_KEY


class RecordingTransport(httpx.MockTransport):
    """Record decoded request bodies and lifecycle calls."""

    def __init__(
        self,
        response_body: JsonObject,
        *,
        status_code: int = 200,
    ) -> None:
        """Configure one repeatable JSON response."""
        self.response_body = response_body
        self.status_code = status_code
        self.requests: list[httpx.Request] = []
        self.request_bodies: list[JsonObject] = []
        self.close_calls = 0
        super().__init__(self._handle_request)

    async def _handle_request(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        """Record the completed wire body and return configured JSON."""
        value = json.loads(await request.aread())
        assert type(value) is dict
        self.requests.append(request)
        self.request_bodies.append(value)
        return httpx.Response(
            self.status_code,
            json=deepcopy(self.response_body),
        )

    async def aclose(self) -> None:
        """Record closure of the shared HTTP transport."""
        self.close_calls += 1
        await super().aclose()


def _successful_response() -> JsonObject:
    """Return a realistic Anthropic SDK response body."""
    return {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [
            {
                "type": "thinking",
                "thinking": "safe thought",
                "signature": "private-signature",
            },
            {
                "type": "redacted_thinking",
                "data": "private-redacted-thinking",
            },
            {"type": "text", "text": "visible answer"},
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 31,
            "output_tokens": 6,
            "cache_creation_input_tokens": 5,
            "cache_read_input_tokens": 9,
        },
    }


async def _make_backend(
    transport: RecordingTransport,
) -> ModelBackend:
    """Create the public adapter with an isolated mock network."""
    return await create_anthropic_messages_backend(
        FakeSettings(),
        transport=transport,
    )


def _expected_generation() -> ModelGeneration:
    """Return the projected result for the successful wire fixture."""
    return ModelGeneration(
        content="visible answer",
        reasoning=(ModelReasoning(type="thinking", text="safe thought"),),
        usage=ModelUsage(
            input_tokens=31,
            output_tokens=6,
            cache_creation_input_tokens=5,
            cache_read_input_tokens=9,
        ),
        request_snapshot=deepcopy(EXPECTED_PAYLOAD),
    )


def _contract_case() -> BackendContractCase:
    """Build a fresh real-SDK Anthropic contract fixture."""
    transport = RecordingTransport(_successful_response())
    return BackendContractCase(
        backend=asyncio.run(_make_backend(transport)),
        expected_model=MODEL,
        conversation=CONVERSATION,
        expected_generation=_expected_generation(),
        upstream_call_count=lambda: len(transport.request_bodies),
        close_call_count=lambda: transport.close_calls,
        forbidden_snapshot_values=(
            API_KEY,
            BASE_URL,
            "private-signature",
            "private-redacted-thinking",
        ),
    )


class TestAnthropicMessagesBackendContract(BackendContractTests):
    """Apply the neutral async backend contract to Anthropic."""

    contract_case_factory = staticmethod(_contract_case)


def test_backend_structurally_satisfies_frozen_port() -> None:
    """The assembled Direct backend implements the neutral port."""
    transport = RecordingTransport(_successful_response())

    async def create_and_close() -> None:
        backend = await _make_backend(transport)
        try:
            assert isinstance(backend, ModelBackend)
        finally:
            await backend.aclose()

    asyncio.run(create_and_close())


def test_wire_preserves_history_and_has_exactly_two_5m_markers() -> None:
    """Only system and current user receive public five-minute markers."""
    transport = RecordingTransport(_successful_response())

    async def generate_and_close() -> ModelGeneration:
        backend = await _make_backend(transport)
        try:
            return await backend.generate(CONVERSATION)
        finally:
            await backend.aclose()

    generation = asyncio.run(generate_and_close())

    assert transport.request_bodies == [EXPECTED_PAYLOAD]
    assert generation.request_snapshot == transport.request_bodies[0]
    body = transport.request_bodies[0]
    markers = [
        block["cache_control"]
        for block in body["system"]
        + [
            content
            for message in body["messages"]
            for content in message["content"]
        ]
        if "cache_control" in block
    ]
    assert markers == [CACHE_CONTROL, CACHE_CONTROL]
    assert "cache_control" not in body["messages"][-2]["content"][-1]
    assert transport.requests[0].extensions["timeout"] == {
        "connect": 60.0,
        "read": 60.0,
        "write": 60.0,
        "pool": 60.0,
    }


def test_wire_without_history_still_caches_current_user() -> None:
    """An empty confirmed history retains both intended cache boundaries."""
    transport = RecordingTransport(_successful_response())
    conversation = ModelConversation(
        system_prompt=SYSTEM_PROMPT,
        turns=(),
        current_input="only current user",
    )

    async def generate_and_close() -> None:
        backend = await _make_backend(transport)
        try:
            await backend.generate(conversation)
        finally:
            await backend.aclose()

    asyncio.run(generate_and_close())

    assert transport.request_bodies == [
        {
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": "only current user",
                            "type": "text",
                            "cache_control": CACHE_CONTROL,
                        }
                    ],
                }
            ],
            "model": MODEL,
            "stream": False,
            "system": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": CACHE_CONTROL,
                }
            ],
        }
    ]


def test_upstream_error_is_sanitized_and_not_retried() -> None:
    """One failed SDK call exposes no provider body or credential."""
    transport = RecordingTransport(
        {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": f"{API_KEY} at {BASE_URL}",
            },
        },
        status_code=500,
    )

    async def generate_and_close() -> None:
        backend = await _make_backend(transport)
        try:
            await backend.generate(CONVERSATION)
        finally:
            await backend.aclose()

    with pytest.raises(
        RuntimeError,
        match="^Pydantic AI model request failed$",
    ) as exc_info:
        asyncio.run(generate_and_close())

    assert len(transport.request_bodies) == 1
    assert API_KEY not in str(exc_info.value)
    assert BASE_URL not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_factory_error_before_client_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A capture-client construction error has no sensitive context."""

    def fail_creation(**_: Any) -> Any:
        raise ValueError(f"leaked {API_KEY} at {BASE_URL}")

    monkeypatch.setattr(
        adapter_module,
        "create_request_capture_client",
        fail_creation,
    )

    with pytest.raises(
        RuntimeError,
        match="^Anthropic Messages client creation failed$",
    ) as exc_info:
        asyncio.run(create_anthropic_messages_backend(FakeSettings()))

    assert API_KEY not in str(exc_info.value)
    assert BASE_URL not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.parametrize(
    "failing_symbol",
    [
        "AsyncAnthropic",
        "AnthropicProvider",
        "AnthropicModel",
        "PydanticAIBackend",
    ],
)
def test_factory_failure_closes_captured_client_once(
    monkeypatch: pytest.MonkeyPatch,
    failing_symbol: str,
) -> None:
    """Every failure after HTTP client creation closes it before raising."""

    def fail_creation(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError(f"leaked {API_KEY} at {BASE_URL}")

    monkeypatch.setattr(adapter_module, failing_symbol, fail_creation)
    transport = RecordingTransport(_successful_response())

    with pytest.raises(
        RuntimeError,
        match="^Anthropic Messages client creation failed$",
    ) as exc_info:
        asyncio.run(
            create_anthropic_messages_backend(
                FakeSettings(),
                transport=transport,
            )
        )

    assert transport.close_calls == 1
    assert API_KEY not in str(exc_info.value)
    assert BASE_URL not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_factory_cancellation_closes_client_and_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construction cancellation is cleaned up but never sanitized away."""

    def cancel_creation(**_: Any) -> Any:
        raise asyncio.CancelledError

    monkeypatch.setattr(adapter_module, "AsyncAnthropic", cancel_creation)
    transport = RecordingTransport(_successful_response())

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            create_anthropic_messages_backend(
                FakeSettings(),
                transport=transport,
            )
        )

    assert transport.close_calls == 1


def test_factory_passes_bounded_official_sdk_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The thin adapter disables SDK retries and uses the shared client."""
    real_constructor = adapter_module.AsyncAnthropic
    constructor_kwargs: dict[str, Any] = {}

    def capture_constructor(**kwargs: Any) -> Any:
        constructor_kwargs.update(kwargs)
        return real_constructor(**kwargs)

    monkeypatch.setattr(
        adapter_module,
        "AsyncAnthropic",
        capture_constructor,
    )
    transport = RecordingTransport(_successful_response())

    async def create_assert_and_close() -> None:
        backend = await _make_backend(transport)
        try:
            assert constructor_kwargs["api_key"] == API_KEY
            assert constructor_kwargs["base_url"] == BASE_URL
            assert constructor_kwargs["timeout"] == 60.0
            assert constructor_kwargs["max_retries"] == 0
            assert isinstance(
                constructor_kwargs["http_client"],
                httpx.AsyncClient,
            )
        finally:
            await backend.aclose()

    asyncio.run(create_assert_and_close())
