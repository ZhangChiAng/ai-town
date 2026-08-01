"""Captured-wire tests for the OpenAI Responses Pydantic AI adapter."""

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
from app.model_backends import openai_responses as adapter_module
from app.model_backends.openai_responses import (
    create_openai_responses_backend,
)
from tests.model_backends.contract import (
    BackendContractCase,
    BackendContractTests,
)

MODEL = "gpt-5.4"
BASE_URL = "https://openai.example/v1"
API_KEY = "openai-secret-never-preview"
SYSTEM_PROMPT = "SYSTEM exact\nsecond line"
CONVERSATION = ModelConversation(
    system_prompt=SYSTEM_PROMPT,
    turns=(
        ModelTurn(input="first user", output="first assistant"),
        ModelTurn(input="second user", output="second assistant"),
    ),
    current_input="current user\nexact",
)
EXPECTED_PAYLOAD: JsonObject = {
    "include": ["reasoning.encrypted_content"],
    "input": [
        {"role": "user", "content": "first user"},
        {"role": "assistant", "content": "first assistant"},
        {"role": "user", "content": "second user"},
        {"role": "assistant", "content": "second assistant"},
        {"role": "user", "content": "current user\nexact"},
    ],
    "instructions": SYSTEM_PROMPT,
    "max_output_tokens": 2048,
    "model": MODEL,
    "reasoning": {"context": "current_turn"},
    "store": False,
    "stream": False,
    "truncation": "disabled",
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
    """Return a realistic OpenAI Responses SDK response body."""
    return {
        "id": "resp_123",
        "object": "response",
        "created_at": 1_750_000_000.0,
        "status": "completed",
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": 2048,
        "max_tool_calls": None,
        "model": MODEL,
        "output": [
            {
                "id": "rs_123",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "short summary"}],
                "content": [
                    {
                        "type": "reasoning_text",
                        "text": "readable reasoning",
                    }
                ],
                "encrypted_content": "private-encrypted-reasoning",
                "status": None,
            },
            {
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "visible answer",
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            },
        ],
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "prompt_cache_key": None,
        "reasoning": {"effort": "medium", "summary": "auto"},
        "safety_identifier": None,
        "service_tier": "default",
        "store": False,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_logprobs": 0,
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 20,
            "input_tokens_details": {"cached_tokens": 5},
            "output_tokens": 7,
            "output_tokens_details": {"reasoning_tokens": 2},
            "total_tokens": 27,
        },
        "user": None,
        "metadata": {},
    }


async def _make_backend(
    transport: RecordingTransport,
) -> ModelBackend:
    """Create the public adapter with an isolated mock network."""
    return await create_openai_responses_backend(
        FakeSettings(),
        transport=transport,
    )


def _expected_generation() -> ModelGeneration:
    """Return the projected result for the successful wire fixture."""
    return ModelGeneration(
        content="visible answer",
        reasoning=(
            ModelReasoning(type="summary_text", text="short summary"),
            ModelReasoning(
                type="reasoning_text",
                text="readable reasoning",
            ),
        ),
        usage=ModelUsage(
            input_tokens=15,
            output_tokens=7,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=5,
        ),
        request_snapshot=deepcopy(EXPECTED_PAYLOAD),
    )


def _contract_case() -> BackendContractCase:
    """Build a fresh real-SDK OpenAI contract fixture."""
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
            "private-encrypted-reasoning",
        ),
    )


class TestOpenAIResponsesBackendContract(BackendContractTests):
    """Apply the neutral async backend contract to OpenAI Responses."""

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


def test_wire_preserves_full_history_and_stateless_boundaries() -> None:
    """The actual body is complete, stateless, and OpenAI-specific."""
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
    assert body["input"] == [
        {"role": "user", "content": "first user"},
        {"role": "assistant", "content": "first assistant"},
        {"role": "user", "content": "second user"},
        {"role": "assistant", "content": "second assistant"},
        {"role": "user", "content": "current user\nexact"},
    ]
    assert all(
        "id" not in item and item.get("type") != "reasoning"
        for item in body["input"]
    )
    assert set(body).isdisjoint(
        {
            "anthropic_cache",
            "cache_control",
            "context_management",
            "conversation",
            "messages",
            "previous_response_id",
            "system",
            "tools",
        }
    )
    assert body["include"] == ["reasoning.encrypted_content"]
    assert transport.requests[0].extensions["timeout"] == {
        "connect": 60.0,
        "read": 60.0,
        "write": 60.0,
        "pool": 60.0,
    }


def test_response_exposes_only_public_readable_reasoning() -> None:
    """Summary and documented raw text survive; encryption does not."""
    transport = RecordingTransport(_successful_response())

    async def generate_and_close() -> ModelGeneration:
        backend = await _make_backend(transport)
        try:
            return await backend.generate(CONVERSATION)
        finally:
            await backend.aclose()

    generation = asyncio.run(generate_and_close())

    assert generation.reasoning == (
        ModelReasoning(type="summary_text", text="short summary"),
        ModelReasoning(type="reasoning_text", text="readable reasoning"),
    )
    assert "private-encrypted-reasoning" not in repr(generation)


def test_upstream_error_is_sanitized_and_not_retried() -> None:
    """One failed SDK call exposes no provider body or credential."""
    transport = RecordingTransport(
        {
            "error": {
                "type": "server_error",
                "message": f"{API_KEY} at {BASE_URL}",
            }
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
        match="^OpenAI Responses client creation failed$",
    ) as exc_info:
        asyncio.run(create_openai_responses_backend(FakeSettings()))

    assert API_KEY not in str(exc_info.value)
    assert BASE_URL not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.parametrize(
    "failing_symbol",
    [
        "AsyncOpenAI",
        "OpenAIProvider",
        "OpenAIResponsesModel",
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
        match="^OpenAI Responses client creation failed$",
    ) as exc_info:
        asyncio.run(
            create_openai_responses_backend(
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

    monkeypatch.setattr(adapter_module, "AsyncOpenAI", cancel_creation)
    transport = RecordingTransport(_successful_response())

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            create_openai_responses_backend(
                FakeSettings(),
                transport=transport,
            )
        )

    assert transport.close_calls == 1


def test_factory_passes_bounded_official_sdk_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The thin adapter disables SDK retries and uses the shared client."""
    real_constructor = adapter_module.AsyncOpenAI
    constructor_kwargs: dict[str, Any] = {}

    def capture_constructor(**kwargs: Any) -> Any:
        constructor_kwargs.update(kwargs)
        return real_constructor(**kwargs)

    monkeypatch.setattr(adapter_module, "AsyncOpenAI", capture_constructor)
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
