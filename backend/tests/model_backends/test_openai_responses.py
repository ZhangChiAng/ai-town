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

GPT_5_6_PROXY_MODELS = (
    "openai/gpt-5.6-luna",
    "openai/gpt-5.6-terra",
    "openai/gpt-5.6-sol",
)
MODEL = GPT_5_6_PROXY_MODELS[0]
BASE_URL = "https://openai-proxy.example/v1"
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
    "model": MODEL,
    "reasoning": {
        "effort": "max",
        "summary": "auto",
        "context": "current_turn",
    },
    "store": False,
    "stream": False,
}
PRIVATE_ENCRYPTED_REASONING = "private-encrypted-reasoning"
PRIVATE_PROVIDER_DETAIL = "private-raw-provider-detail"


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


def _successful_response(*, model: str = MODEL) -> JsonObject:
    """Return a realistic official Responses SDK response body."""
    return {
        "id": "resp_123",
        "object": "response",
        "created_at": 1_750_000_000.0,
        "status": "completed",
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "max_tool_calls": None,
        "model": model,
        "output": [
            {
                "id": "rs_123",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "public summary"}],
                "content": [],
                "encrypted_content": PRIVATE_ENCRYPTED_REASONING,
                "status": None,
                "private_provider_detail": PRIVATE_PROVIDER_DETAIL,
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
        "reasoning": {"effort": "max", "summary": "auto"},
        "safety_identifier": None,
        "service_tier": None,
        "store": False,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}},
        "tool_choice": None,
        "tools": [],
        "top_logprobs": 0,
        "top_p": 1.0,
        "truncation": None,
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
    *,
    model: str = MODEL,
) -> ModelBackend:
    """Create the public adapter with an isolated mock network."""
    return await create_openai_responses_backend(
        FakeSettings(model=model),
        transport=transport,
    )


def _expected_generation() -> ModelGeneration:
    """Return the projected result for the successful wire fixture."""
    return ModelGeneration(
        content="visible answer",
        reasoning=(ModelReasoning(type="summary_text", text="public summary"),),
        usage=ModelUsage(
            input_tokens=15,
            output_tokens=7,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=5,
        ),
        request_snapshot=deepcopy(EXPECTED_PAYLOAD),
    )


def _contract_case() -> BackendContractCase:
    """Build a fresh real-SDK OpenAI Responses contract fixture."""
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
            PRIVATE_ENCRYPTED_REASONING,
            PRIVATE_PROVIDER_DETAIL,
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


@pytest.mark.parametrize("model", GPT_5_6_PROXY_MODELS)
def test_gpt_5_6_proxy_model_names_are_forwarded_exactly(
    model: str,
) -> None:
    """Every supported proxy route keeps the shared OpenAI wire policy."""
    transport = RecordingTransport(_successful_response(model=model))

    async def generate_and_close() -> None:
        backend = await _make_backend(transport, model=model)
        try:
            assert backend.model == model
            await backend.generate(CONVERSATION)
        finally:
            await backend.aclose()

    asyncio.run(generate_and_close())

    assert transport.request_bodies == [EXPECTED_PAYLOAD | {"model": model}]


def test_wire_preserves_full_history_and_openai_boundaries() -> None:
    """The actual body is complete, stateless, and exactly constrained."""
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
    assert body["model"] == MODEL
    assert body["instructions"] == SYSTEM_PROMPT
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
    assert body["reasoning"] == {
        "effort": "max",
        "summary": "auto",
        "context": "current_turn",
    }
    assert body["store"] is False
    assert set(body).isdisjoint(
        {
            "anthropic_cache",
            "cache_control",
            "context_management",
            "conversation",
            "max_output_tokens",
            "messages",
            "previous_response_id",
            "prompt_cache_key",
            "prompt_cache_options",
            "prompt_cache_retention",
            "service_tier",
            "system",
            "tool_choice",
            "tools",
            "truncation",
        }
    )
    assert body["include"] == ["reasoning.encrypted_content"]
    assert PRIVATE_ENCRYPTED_REASONING not in repr(body)
    assert transport.requests[0].extensions["timeout"] == {
        "connect": 60.0,
        "read": 60.0,
        "write": 60.0,
        "pool": 60.0,
    }


def test_response_exposes_summary_without_private_provider_state() -> None:
    """Only public summary text survives provider response projection."""
    transport = RecordingTransport(_successful_response())

    async def generate_and_close() -> ModelGeneration:
        backend = await _make_backend(transport)
        try:
            return await backend.generate(CONVERSATION)
        finally:
            await backend.aclose()

    generation = asyncio.run(generate_and_close())

    assert generation.reasoning == (
        ModelReasoning(type="summary_text", text="public summary"),
    )
    serialized = repr(generation)
    assert PRIVATE_ENCRYPTED_REASONING not in serialized
    assert PRIVATE_PROVIDER_DETAIL not in serialized
    assert API_KEY not in serialized
    assert BASE_URL not in serialized


def test_concurrent_generations_keep_request_snapshots_isolated() -> None:
    """Concurrent official SDK calls cannot exchange captured bodies."""
    transport = RecordingTransport(_successful_response())
    first = ModelConversation(
        system_prompt="system one",
        turns=(),
        current_input="first current",
    )
    second = ModelConversation(
        system_prompt="system two",
        turns=(),
        current_input="second current",
    )

    async def generate_both() -> tuple[ModelGeneration, ModelGeneration]:
        backend = await _make_backend(transport)
        try:
            first_result, second_result = await asyncio.gather(
                backend.generate(first),
                backend.generate(second),
            )
            return first_result, second_result
        finally:
            await backend.aclose()

    first_result, second_result = asyncio.run(generate_both())

    assert first_result.request_snapshot["instructions"] == "system one"
    assert second_result.request_snapshot["instructions"] == "system two"
    assert first_result.request_snapshot["input"] == [
        {"role": "user", "content": "first current"}
    ]
    assert second_result.request_snapshot["input"] == [
        {"role": "user", "content": "second current"}
    ]
    assert len(transport.request_bodies) == 2


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
