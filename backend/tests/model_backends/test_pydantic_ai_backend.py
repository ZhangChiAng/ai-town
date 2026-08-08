"""Tests for the shared Pydantic AI Direct backend implementation."""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, cast

import httpx
import pytest
from pydantic_ai import (
    CompactionPart,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.usage import RequestUsage

from app.model_backends import pydantic_ai_backend as backend_module
from app.model_backends.contracts import (
    ModelConversation,
    ModelReasoning,
    ModelTurn,
    ModelUsage,
)
from app.model_backends.pydantic_ai_backend import (
    PydanticAIBackend,
    create_request_capture_client,
)

MODEL = "provider/Case-Sensitive"
SYSTEM_PROMPT = "SYSTEM exact\n第二行"
CONVERSATION = ModelConversation(
    system_prompt=SYSTEM_PROMPT,
    turns=(
        ModelTurn(input="first user\nexact", output="first assistant"),
        ModelTurn(input="second user", output="second assistant\nexact"),
    ),
    current_input="current user\nexact",
)
SNAPSHOT = {
    "model": MODEL,
    "messages": ["full", "wire", "body"],
    "nested": {"unicode": "原样", "enabled": False},
}
SECRET = "provider-secret-must-not-escape"


@dataclass(frozen=True, slots=True)
class StubDirectModel:
    """Minimal public model identity used around a patched Direct call."""

    model_name: str = MODEL
    system: str = "anthropic"


def _mock_transport() -> httpx.MockTransport:
    """Return an upstream transport that never touches the network."""
    return httpx.MockTransport(
        lambda _request: httpx.Response(200, json={"ok": True})
    )


def _build_backend(
    *,
    provider: str = "anthropic",
    client: httpx.AsyncClient | None = None,
) -> tuple[PydanticAIBackend, httpx.AsyncClient, StubDirectModel]:
    """Build one common backend and its observable constructor inputs."""
    if client is None:
        client = create_request_capture_client(transport=_mock_transport())
    direct_model = StubDirectModel(system=provider)
    backend = PydanticAIBackend(
        model=MODEL,
        direct_model=cast(Model, direct_model),
        model_settings={"max_tokens": 321},
        http_client=client,
    )
    return backend, client, direct_model


def _response(
    parts: list[Any] | None = None,
    *,
    usage: RequestUsage | None = None,
) -> ModelResponse:
    """Build one Pydantic AI response with inclusive input usage."""
    return ModelResponse(
        parts=[TextPart("visible answer")] if parts is None else parts,
        usage=usage
        or RequestUsage(
            input_tokens=20,
            output_tokens=7,
            cache_write_tokens=3,
            cache_read_tokens=5,
        ),
    )


async def _post_json(
    client: httpx.AsyncClient,
    body: object,
) -> None:
    """Send one body through the installed request capture hook."""
    await client.post("https://model.test/request", json=body)


def test_generate_maps_complete_history_and_calls_direct_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """History stays verbatim and only the final user carries instructions."""
    backend, client, direct_model = _build_backend()
    calls: list[tuple[object, list[Any], dict[str, Any]]] = []

    async def fake_model_request(
        model: object,
        messages: list[Any],
        **kwargs: Any,
    ) -> ModelResponse:
        calls.append((model, messages, kwargs))
        await _post_json(client, SNAPSHOT)
        return _response()

    monkeypatch.setattr(backend_module, "model_request", fake_model_request)

    async def scenario():
        try:
            return await backend.generate(CONVERSATION)
        finally:
            await backend.aclose()

    generation = asyncio.run(scenario())

    assert backend.model == MODEL
    assert generation.content == "visible answer"
    assert generation.reasoning == ()
    assert generation.usage == ModelUsage(
        input_tokens=12,
        output_tokens=7,
        cache_creation_input_tokens=3,
        cache_read_input_tokens=5,
    )
    assert generation.request_snapshot == SNAPSHOT
    assert len(calls) == 1
    called_model, messages, kwargs = calls[0]
    assert called_model is direct_model
    assert kwargs == {
        "model_settings": {"max_tokens": 321},
        "instrument": False,
    }
    assert len(messages) == 5
    assert [type(message) for message in messages] == [
        ModelRequest,
        ModelResponse,
        ModelRequest,
        ModelResponse,
        ModelRequest,
    ]

    user_messages = cast(list[ModelRequest], messages[::2])
    assert [message.instructions for message in user_messages] == [
        None,
        None,
        SYSTEM_PROMPT,
    ]
    assert [
        cast(UserPromptPart, message.parts[0]).content
        for message in user_messages
    ] == ["first user\nexact", "second user", "current user\nexact"]
    assistant_messages = cast(list[ModelResponse], messages[1::2])
    assert [
        cast(TextPart, message.parts[0]).content
        for message in assistant_messages
    ] == ["first assistant", "second assistant\nexact"]
    assert isinstance(messages[-1], ModelRequest)


@pytest.mark.parametrize(
    ("configured_model", "direct_model"),
    [
        (MODEL, StubDirectModel(model_name="different")),
        (MODEL, StubDirectModel(system="unsupported")),
    ],
    ids=("model-mismatch", "unsupported-provider"),
)
def test_constructor_rejects_ambiguous_direct_model(
    configured_model: str,
    direct_model: StubDirectModel,
) -> None:
    """Model identity and provider family are fixed at construction."""
    client = create_request_capture_client(transport=_mock_transport())
    try:
        with pytest.raises(ValueError):
            PydanticAIBackend(
                model=configured_model,
                direct_model=cast(Model, direct_model),
                model_settings={},
                http_client=client,
            )
    finally:
        asyncio.run(client.aclose())


@pytest.mark.parametrize(
    "parts",
    [
        [],
        [TextPart("first"), TextPart("second")],
        [TextPart(" \n")],
        [TextPart("visible"), ToolCallPart("tool", {})],
        [TextPart("visible"), CompactionPart(content="summary")],
    ],
    ids=(
        "missing-text",
        "multiple-text",
        "blank-text",
        "tool-call",
        "compaction",
    ),
)
def test_generate_rejects_non_text_or_ambiguous_response_parts(
    monkeypatch: pytest.MonkeyPatch,
    parts: list[Any],
) -> None:
    """Only thinking and exactly one non-blank text part are accepted."""
    backend, client, _direct_model = _build_backend()

    async def fake_model_request(*_args: Any, **_kwargs: Any) -> ModelResponse:
        await _post_json(client, SNAPSHOT)
        return _response(parts)

    monkeypatch.setattr(backend_module, "model_request", fake_model_request)

    async def scenario() -> None:
        try:
            with pytest.raises(ValueError):
                await backend.generate(CONVERSATION)
        finally:
            await backend.aclose()

    asyncio.run(scenario())


def test_projection_error_log_keeps_full_raw_provider_details(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Error logs retain request, context, signatures, and provider details."""
    backend, client, _direct_model = _build_backend()
    long_provider_detail = "raw-provider-detail" * 500
    response = _response(
        [
            ThinkingPart(
                "visible thought",
                signature="raw-signature-for-server-log",
                provider_name="anthropic",
                provider_details={"raw": long_provider_detail},
            ),
            ToolCallPart("unexpected", {}),
        ]
    )

    async def fake_model_request(*_args: Any, **_kwargs: Any) -> ModelResponse:
        await _post_json(client, SNAPSHOT)
        return response

    monkeypatch.setattr(backend_module, "model_request", fake_model_request)
    caplog.set_level(logging.ERROR)

    async def scenario() -> None:
        try:
            with pytest.raises(ValueError):
                await backend.generate(CONVERSATION)
        finally:
            await backend.aclose()

    asyncio.run(scenario())

    record = next(
        record
        for record in caplog.records
        if getattr(record, "event_name", None) == "model.projection.failed"
    )
    fields = record.event_fields
    serialized = json.dumps(fields, ensure_ascii=False)
    assert fields["conversation"]["current_input"] == CONVERSATION.current_input
    assert json.loads(fields["serialized_requests"][0]["body"]) == SNAPSHOT
    assert "raw-signature-for-server-log" in serialized
    assert long_provider_detail in serialized


def test_anthropic_reasoning_exposes_only_non_blank_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anthropic signatures, redacted data, and details stay private."""
    backend, client, _direct_model = _build_backend(provider="anthropic")
    parts = [
        ThinkingPart(
            "public thought",
            id="thinking-id",
            signature="private-signature",
            provider_name="anthropic",
            provider_details={"raw_content": ["private raw"]},
        ),
        ThinkingPart(
            " \n",
            id="redacted_thinking",
            signature="private-redacted-data",
            provider_name="anthropic",
        ),
        TextPart("visible"),
    ]

    async def fake_model_request(*_args: Any, **_kwargs: Any) -> ModelResponse:
        await _post_json(client, SNAPSHOT)
        return _response(parts)

    monkeypatch.setattr(backend_module, "model_request", fake_model_request)

    async def scenario():
        try:
            return await backend.generate(CONVERSATION)
        finally:
            await backend.aclose()

    generation = asyncio.run(scenario())

    assert generation.reasoning == (
        ModelReasoning(type="thinking", text="public thought"),
    )
    serialized = repr(generation)
    assert "private-signature" not in serialized
    assert "private-redacted-data" not in serialized
    assert "private raw" not in serialized


def test_openai_reasoning_whitelists_summary_and_strict_raw_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI maps documented text while ignoring every private detail."""
    backend, client, _direct_model = _build_backend(provider="openai")
    parts = [
        ThinkingPart(
            "public summary",
            id="reasoning-id",
            signature="private-encrypted-content",
            provider_name="openai",
            provider_details={
                "raw_content": ["first raw", " \n", "second raw"],
                "other": "private-other-detail",
            },
        ),
        ThinkingPart(
            "",
            provider_name="openai",
            provider_details={"raw_content": ["third raw"]},
        ),
        ThinkingPart(
            "second summary",
            provider_name="openai",
            provider_details={"raw_content": ["ignored", 12]},
        ),
        TextPart("visible"),
    ]

    async def fake_model_request(*_args: Any, **_kwargs: Any) -> ModelResponse:
        await _post_json(client, SNAPSHOT)
        return _response(parts)

    monkeypatch.setattr(backend_module, "model_request", fake_model_request)

    async def scenario():
        try:
            return await backend.generate(CONVERSATION)
        finally:
            await backend.aclose()

    generation = asyncio.run(scenario())

    assert generation.reasoning == (
        ModelReasoning(type="summary_text", text="public summary"),
        ModelReasoning(type="reasoning_text", text="first raw"),
        ModelReasoning(type="reasoning_text", text="second raw"),
        ModelReasoning(type="reasoning_text", text="third raw"),
        ModelReasoning(type="summary_text", text="second summary"),
    )
    serialized = repr(generation)
    assert "private-encrypted-content" not in serialized
    assert "private-other-detail" not in serialized
    assert '"ignored"' not in serialized


@pytest.mark.parametrize(
    "usage",
    [
        RequestUsage(input_tokens=True, output_tokens=1),
        RequestUsage(input_tokens=-1, output_tokens=1),
        RequestUsage(input_tokens=1, output_tokens=-1),
        RequestUsage(input_tokens=1, cache_write_tokens=-1),
        RequestUsage(input_tokens=1, cache_read_tokens=-1),
        RequestUsage(
            input_tokens=3,
            output_tokens=1,
            cache_write_tokens=2,
            cache_read_tokens=2,
        ),
    ],
    ids=(
        "bool-input",
        "negative-input",
        "negative-output",
        "negative-cache-write",
        "negative-cache-read",
        "cache-exceeds-total",
    ),
)
def test_generate_rejects_invalid_or_inconsistent_usage(
    monkeypatch: pytest.MonkeyPatch,
    usage: RequestUsage,
) -> None:
    """Token buckets require exact integers and a non-negative remainder."""
    backend, client, _direct_model = _build_backend()

    async def fake_model_request(*_args: Any, **_kwargs: Any) -> ModelResponse:
        await _post_json(client, SNAPSHOT)
        return _response(usage=usage)

    monkeypatch.setattr(backend_module, "model_request", fake_model_request)

    async def scenario() -> None:
        try:
            with pytest.raises(ValueError):
                await backend.generate(CONVERSATION)
        finally:
            await backend.aclose()

    asyncio.run(scenario())


def test_concurrent_generations_capture_their_own_request_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Context-local capture prevents gather calls from crossing snapshots."""
    backend, client, _direct_model = _build_backend()
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

    async def fake_model_request(
        _model: object,
        messages: list[Any],
        **_kwargs: Any,
    ) -> ModelResponse:
        request = cast(ModelRequest, messages[-1])
        current = cast(UserPromptPart, request.parts[0]).content
        assert isinstance(current, str)
        if current.startswith("first"):
            await asyncio.sleep(0.01)
        await _post_json(client, {"current": current})
        if current.startswith("second"):
            await asyncio.sleep(0.01)
        return _response([TextPart(f"answer for {current}")])

    monkeypatch.setattr(backend_module, "model_request", fake_model_request)

    async def scenario():
        try:
            return await asyncio.gather(
                backend.generate(first),
                backend.generate(second),
            )
        finally:
            await backend.aclose()

    first_result, second_result = asyncio.run(scenario())

    assert first_result.request_snapshot == {"current": "first current"}
    assert second_result.request_snapshot == {"current": "second current"}
    assert first_result.content == "answer for first current"
    assert second_result.content == "answer for second current"


def test_snapshot_excludes_request_url_headers_and_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture reads the body without retaining HTTP credential metadata."""
    backend, client, _direct_model = _build_backend()

    async def fake_model_request(*_args: Any, **_kwargs: Any) -> ModelResponse:
        await client.post(
            f"https://model.test/private/{SECRET}",
            headers={"Authorization": f"Bearer {SECRET}"},
            json={"safe": "body only"},
        )
        return _response()

    monkeypatch.setattr(backend_module, "model_request", fake_model_request)

    async def scenario():
        try:
            return await backend.generate(CONVERSATION)
        finally:
            await backend.aclose()

    generation = asyncio.run(scenario())

    assert generation.request_snapshot == {"safe": "body only"}
    assert SECRET not in repr(generation)


@pytest.mark.parametrize(
    "bodies",
    [
        [],
        [[{"one": 1}], [{"two": 2}]],
        [["not-an-object"]],
        [[b"{malformed"]],
        [[b'{"duplicate": 1, "duplicate": 2}']],
        [[b'{"bad": NaN}']],
        [[b'{"huge": 1e9999}']],
    ],
    ids=(
        "missing",
        "multiple",
        "non-object",
        "malformed",
        "duplicate-key",
        "nan",
        "infinite-float",
    ),
)
def test_generate_rejects_missing_multiple_or_malformed_capture(
    monkeypatch: pytest.MonkeyPatch,
    bodies: list[list[object]],
) -> None:
    """Only one strict JSON object can become a successful snapshot."""
    backend, client, _direct_model = _build_backend()

    async def fake_model_request(*_args: Any, **_kwargs: Any) -> ModelResponse:
        for body_group in bodies:
            for body in body_group:
                if isinstance(body, bytes):
                    await client.post(
                        "https://model.test/request",
                        content=body,
                        headers={"Content-Type": "application/json"},
                    )
                else:
                    await _post_json(client, body)
        return _response()

    monkeypatch.setattr(backend_module, "model_request", fake_model_request)

    async def scenario() -> None:
        try:
            with pytest.raises(ValueError, match="snapshot"):
                await backend.generate(CONVERSATION)
        finally:
            await backend.aclose()

    asyncio.run(scenario())


def test_failed_call_is_sanitized_and_leaves_no_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed request cannot expose or donate its body to the next call."""
    backend, client, _direct_model = _build_backend()
    call_count = 0

    async def fake_model_request(*_args: Any, **_kwargs: Any) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            await _post_json(client, {"failed": "stale body"})
            raise RuntimeError(f"response body contains {SECRET}")
        await _post_json(client, {"successful": "fresh body"})
        return _response()

    monkeypatch.setattr(backend_module, "model_request", fake_model_request)

    async def scenario():
        with pytest.raises(RuntimeError) as raised:
            await backend.generate(CONVERSATION)
        result = await backend.generate(CONVERSATION)
        await backend.aclose()
        return raised.value, result

    error, generation = asyncio.run(scenario())

    assert str(error) == "Pydantic AI model request failed"
    assert SECRET not in str(error)
    assert SECRET not in repr(error)
    assert error.__cause__ is None
    assert error.__context__ is None
    assert generation.request_snapshot == {"successful": "fresh body"}
    assert "stale body" not in repr(generation)
    assert call_count == 2


def test_malformed_capture_does_not_poison_later_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each generation starts fresh after a local capture failure."""
    backend, client, _direct_model = _build_backend()
    call_count = 0

    async def fake_model_request(*_args: Any, **_kwargs: Any) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            await client.post(
                "https://model.test/request",
                content=b"[]",
                headers={"Content-Type": "application/json"},
            )
        else:
            await _post_json(client, {"call": "second"})
        return _response()

    monkeypatch.setattr(backend_module, "model_request", fake_model_request)

    async def scenario():
        with pytest.raises(ValueError, match="snapshot"):
            await backend.generate(CONVERSATION)
        result = await backend.generate(CONVERSATION)
        await backend.aclose()
        return result

    generation = asyncio.run(scenario())

    assert generation.request_snapshot == {"call": "second"}


class RecordingAsyncClient(httpx.AsyncClient):
    """HTTP client double that counts actual resource close calls."""

    def __init__(self) -> None:
        """Initialize one isolated no-network client."""
        super().__init__(transport=_mock_transport())
        self.close_calls = 0

    async def aclose(self) -> None:
        """Count and delegate each actual close invocation."""
        self.close_calls += 1
        await super().aclose()


def test_aclose_closes_given_http_client_once() -> None:
    """The common backend owns only the shared HTTP client and is idempotent."""
    client = RecordingAsyncClient()
    backend, _client, _direct_model = _build_backend(client=client)

    async def scenario() -> None:
        await backend.aclose()
        await backend.aclose()

    asyncio.run(scenario())

    assert client.close_calls == 1


def test_capture_client_uses_sixty_second_timeout() -> None:
    """The shared client bounds every HTTPX timeout phase at 60 seconds."""
    client = create_request_capture_client(transport=_mock_transport())
    try:
        assert client.timeout.connect == 60.0
        assert client.timeout.read == 60.0
        assert client.timeout.write == 60.0
        assert client.timeout.pool == 60.0
    finally:
        asyncio.run(client.aclose())
