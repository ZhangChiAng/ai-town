"""Tests for one-step observable message draft generation."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.drafting import (
    CACHE_CONTROL,
    RUNTIME_TURN_PROMPT,
    DraftGenerationError,
    MessageDraftService,
    build_message_request,
    select_model_protocol,
)
from app.main import create_app
from app.models import CreateMessageRequest, add_message, create_scene
from app.storage import SceneStorage
from tests.client import TestClient

ANTHROPIC_MODEL = "anthropic/claude-test"
RESPONSES_MODEL = "gpt-test"


class FakeResource:
    """Capture provider requests and return a configured response."""

    def __init__(self, response: Any) -> None:
        """Initialize with one provider-shaped response."""
        self.response = response
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        """Capture keyword arguments and return the configured response."""
        self.requests.append(deepcopy(kwargs))
        return self.response


class FakeAnthropicClient:
    """Minimal Anthropic client accepted by the draft service."""

    def __init__(self, response: Any) -> None:
        """Initialize a fake Messages resource."""
        self.messages = FakeResource(response)


class FakeResponsesClient:
    """Minimal Responses client accepted by the draft service."""

    def __init__(self, response: Any) -> None:
        """Initialize a fake Responses resource."""
        self.responses = FakeResource(response)


def anthropic_response(
    text: str,
    *,
    leading_blocks: list[Any] | None = None,
    usage: Any | None = None,
) -> SimpleNamespace:
    """Create an Anthropic-shaped response."""
    return SimpleNamespace(
        content=[
            *(leading_blocks or []),
            SimpleNamespace(type="text", text=text),
        ],
        usage=usage
        or SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=2,
            cache_read_input_tokens=3,
        ),
    )


def assistant_message(
    *blocks: Any,
    role: str = "assistant",
) -> SimpleNamespace:
    """Create one Responses output message item."""
    return SimpleNamespace(type="message", role=role, content=list(blocks))


def output_text(text: str) -> SimpleNamespace:
    """Create one Responses output_text block."""
    return SimpleNamespace(type="output_text", text=text)


def responses_response(
    text: str = "To B: 正文",
    *,
    status: str = "completed",
    output: list[Any] | None = None,
    usage: Any | None = None,
) -> SimpleNamespace:
    """Create an OpenAI Responses-shaped response."""
    return SimpleNamespace(
        status=status,
        output=output
        if output is not None
        else [assistant_message(output_text(text))],
        usage=usage
        or SimpleNamespace(
            input_tokens=20,
            output_tokens=5,
            total_tokens=25,
            input_tokens_details=SimpleNamespace(
                cache_write_tokens=2,
                cached_tokens=3,
            ),
        ),
    )


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-sonnet-4-5", "anthropic"),
        ("anthropic/claude-haiku-4.5", "anthropic"),
        ("ANTHROPIC/CLAUDE-OPUS", "anthropic"),
        ("prefix-ClAuDe-suffix", "anthropic"),
        ("gpt-5", "responses"),
        ("openai/o4-mini", "responses"),
    ],
)
def test_model_name_selects_protocol(model: str, expected: str) -> None:
    """Any case-insensitive Claude substring selects Anthropic."""
    assert select_model_protocol(model) == expected


def _scene_with_alternating_timeline() -> Any:
    """Create a scene whose Agent A has one received and one sent record."""
    scene = create_scene("可观测", ANTHROPIC_MODEL)
    scene = add_message(
        scene,
        CreateMessageRequest(sender_id="B", content="To A: 第一条"),
    )
    return add_message(
        scene,
        CreateMessageRequest(sender_id="A", content="To C: 第二条"),
    )


def test_anthropic_request_preserves_text_and_cache_breakpoints() -> None:
    """Anthropic receives exact text plus its two explicit cache points."""
    scene = _scene_with_alternating_timeline()

    request = build_message_request(scene, "A", ANTHROPIC_MODEL)

    assert request == {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 512,
        "system": [
            {
                "type": "text",
                "text": scene.agents[0].system_prompt,
                "cache_control": CACHE_CONTROL,
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "From B: 第一条"}],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "To C: 第二条",
                        "cache_control": CACHE_CONTROL,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": RUNTIME_TURN_PROMPT.format(
                            recipient_ids="B、C"
                        ),
                    }
                ],
            },
        ],
    }


def test_responses_request_preserves_text_without_cache_metadata() -> None:
    """Responses receives stateless input_text turns and no cache options."""
    scene = _scene_with_alternating_timeline()

    request = build_message_request(scene, "A", RESPONSES_MODEL)

    assert request == {
        "model": RESPONSES_MODEL,
        "instructions": scene.agents[0].system_prompt,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "From B: 第一条"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "input_text", "text": "To C: 第二条"}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": RUNTIME_TURN_PROMPT.format(
                            recipient_ids="B、C"
                        ),
                    }
                ],
            },
        ],
        "max_output_tokens": 2048,
        "store": False,
    }
    assert "cache" not in json.dumps(request)
    assert "tools" not in request
    assert "previous_response_id" not in request


@pytest.mark.parametrize(
    ("model", "turns_key", "block_type"),
    [
        (ANTHROPIC_MODEL, "messages", "text"),
        (RESPONSES_MODEL, "input", "input_text"),
    ],
)
def test_both_protocols_merge_same_roles_without_reordering(
    model: str,
    turns_key: str,
    block_type: str,
) -> None:
    """Shared context construction keeps each record as an ordered block."""
    scene = create_scene("同角色", model)
    for sender_id, content in (
        ("B", "To A: 收到一"),
        ("C", "To A: 收到二"),
        ("A", "To B: 发出一"),
        ("A", "To C: 发出二"),
    ):
        scene = add_message(
            scene,
            CreateMessageRequest(sender_id=sender_id, content=content),
        )

    turns = build_message_request(scene, "A", model)[turns_key]

    assert [turn["role"] for turn in turns] == [
        "user",
        "assistant",
        "user",
    ]
    assert [[block["text"] for block in turn["content"]] for turn in turns] == [
        ["From B: 收到一", "From C: 收到二"],
        ["To B: 发出一", "To C: 发出二"],
        [RUNTIME_TURN_PROMPT.format(recipient_ids="B、C")],
    ]
    assert all(
        block["type"] == block_type
        for turn in turns
        for block in turn["content"]
    )


@pytest.mark.parametrize("model", [ANTHROPIC_MODEL, RESPONSES_MODEL])
@pytest.mark.parametrize(
    "messages_before",
    [
        [],
        [("B", "To A: 收到结尾")],
        [("A", "To B: 发出结尾")],
    ],
)
def test_every_request_appends_the_same_runtime_instruction(
    model: str,
    messages_before: list[tuple[str, str]],
) -> None:
    """Empty, user-ending, and assistant-ending contexts share one trigger."""
    scene = create_scene("格式", model)
    for message_sender, content in messages_before:
        scene = add_message(
            scene,
            CreateMessageRequest(sender_id=message_sender, content=content),
        )

    request = build_message_request(scene, "A", model)
    turns = request.get("messages", request.get("input"))
    final_block = turns[-1]["content"][-1]

    assert final_block["text"] == RUNTIME_TURN_PROMPT.format(
        recipient_ids="B、C"
    )
    assert "cache_control" not in final_block


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("To B: 正文", "To B: 正文"),
        ("To   B ：  正文", "To   B ：  正文"),
        ("  To C: 正文  ", "To C: 正文"),
    ],
)
def test_valid_anthropic_text_is_returned_complete(
    text: str,
    expected: str,
) -> None:
    """Readable thinking is returned separately from editable visible text."""
    client = FakeAnthropicClient(
        anthropic_response(
            text,
            leading_blocks=[
                SimpleNamespace(
                    type="thinking",
                    thinking="先判断 B 是否方便。",
                    signature="PRIVATE SIGNATURE",
                ),
                SimpleNamespace(
                    type="redacted_thinking",
                    data="PRIVATE REDACTED DATA",
                ),
            ],
        )
    )
    result = MessageDraftService(client, ANTHROPIC_MODEL).generate(
        create_scene("草稿", ANTHROPIC_MODEL), "A"
    )

    assert result.content == expected
    assert [block.model_dump() for block in result.reasoning] == [
        {"type": "thinking", "text": "先判断 B 是否方便。"}
    ]
    assert "PRIVATE" not in str(result.model_dump())
    assert result.usage.model_dump() == {
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_creation_input_tokens": 2,
        "cache_read_input_tokens": 3,
    }
    assert result.request_snapshot == client.messages.requests[0]


def test_completed_responses_reasoning_and_one_text_are_accepted() -> None:
    """Responses exposes summaries and raw reasoning apart from the draft."""
    client = FakeResponsesClient(
        responses_response(
            output=[
                SimpleNamespace(
                    type="reasoning",
                    summary=[
                        SimpleNamespace(
                            type="summary_text",
                            text="准备发出一个低压力邀请。",
                        )
                    ],
                    content=[
                        SimpleNamespace(
                            type="reasoning_text",
                            text="B 之前提过钟楼，因此选择 C。",
                        )
                    ],
                    encrypted_content="PRIVATE ENCRYPTED DATA",
                ),
                assistant_message(output_text("To C: 去钟楼。")),
            ]
        )
    )

    result = MessageDraftService(client, RESPONSES_MODEL).generate(
        create_scene("Responses", RESPONSES_MODEL), "A"
    )

    assert result.content == "To C: 去钟楼。"
    assert [block.model_dump() for block in result.reasoning] == [
        {"type": "summary_text", "text": "准备发出一个低压力邀请。"},
        {"type": "reasoning_text", "text": "B 之前提过钟楼，因此选择 C。"},
    ]
    assert "PRIVATE" not in str(result.model_dump())
    assert result.usage.model_dump() == {
        "input_tokens": 15,
        "output_tokens": 5,
        "cache_creation_input_tokens": 2,
        "cache_read_input_tokens": 3,
    }
    assert result.request_snapshot == client.responses.requests[0]


def test_invalid_responses_reasoning_is_rejected() -> None:
    """Malformed readable reasoning fails without leaking provider content."""
    client = FakeResponsesClient(
        responses_response(
            output=[
                SimpleNamespace(
                    type="reasoning",
                    summary=[
                        SimpleNamespace(
                            type="unexpected",
                            text="SENSITIVE REASONING",
                        )
                    ],
                ),
                assistant_message(output_text("To C: 去钟楼。")),
            ]
        )
    )

    with pytest.raises(
        DraftGenerationError,
        match="Model returned an invalid message draft.",
    ) as error:
        MessageDraftService(client, RESPONSES_MODEL).generate(
            create_scene("非法 reasoning", RESPONSES_MODEL), "A"
        )

    assert "SENSITIVE" not in str(error.value)


def test_missing_responses_cache_details_map_to_zero() -> None:
    """Optional Responses cache detail fields default to zero."""
    usage = SimpleNamespace(
        input_tokens=12,
        output_tokens=4,
        total_tokens=16,
    )
    client = FakeResponsesClient(responses_response(usage=usage))

    result = MessageDraftService(client, RESPONSES_MODEL).generate(
        create_scene("无缓存明细", RESPONSES_MODEL), "A"
    )

    assert result.usage.model_dump() == {
        "input_tokens": 12,
        "output_tokens": 4,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


def test_null_responses_cache_write_maps_to_zero() -> None:
    """A nullable cache write count coexists with a real cache hit count."""
    usage = SimpleNamespace(
        input_tokens=12,
        output_tokens=4,
        total_tokens=16,
        input_tokens_details=SimpleNamespace(
            cache_write_tokens=None,
            cached_tokens=3,
        ),
    )
    client = FakeResponsesClient(responses_response(usage=usage))

    result = MessageDraftService(client, RESPONSES_MODEL).generate(
        create_scene("空缓存明细", RESPONSES_MODEL), "A"
    )

    assert result.usage.model_dump() == {
        "input_tokens": 9,
        "output_tokens": 4,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 3,
    }


@pytest.mark.parametrize(
    "text",
    [
        "To A: 自言自语",
        "To D: 无效",
        "To B:",
        "To B: 第一行\n第二行",
        "To B: 正文\n",
        "说明：\nTo B: 正文",
        "先说明 To B: 正文",
    ],
)
@pytest.mark.parametrize("protocol", ["anthropic", "responses"])
def test_invalid_visible_text_fails_once_without_exposing_details(
    text: str,
    protocol: str,
) -> None:
    """Unsafe visible output is rejected rather than repaired or retried."""
    if protocol == "anthropic":
        client = FakeAnthropicClient(anthropic_response(text))
        model = ANTHROPIC_MODEL
        resource = client.messages
    else:
        client = FakeResponsesClient(responses_response(text))
        model = RESPONSES_MODEL
        resource = client.responses
    service = MessageDraftService(client, model)

    with pytest.raises(
        DraftGenerationError,
        match="Model returned an invalid message draft.",
    ):
        service.generate(create_scene("无效", model), "A")

    assert len(resource.requests) == 1


def test_unexpected_anthropic_response_block_is_rejected() -> None:
    """Anthropic allows only thinking variants and one text block."""
    client = FakeAnthropicClient(
        anthropic_response(
            "To B: 正文",
            leading_blocks=[SimpleNamespace(type="tool_use")],
        )
    )

    with pytest.raises(DraftGenerationError):
        MessageDraftService(client, ANTHROPIC_MODEL).generate(
            create_scene("块", ANTHROPIC_MODEL), "A"
        )


@pytest.mark.parametrize(
    ("status", "output"),
    [
        ("incomplete", [assistant_message(output_text("To B: 正文"))]),
        (
            "completed",
            [
                assistant_message(
                    SimpleNamespace(type="refusal", refusal="不能回答")
                )
            ],
        ),
        (
            "completed",
            [
                SimpleNamespace(type="function_call", name="tool"),
                assistant_message(output_text("To B: 正文")),
            ],
        ),
        (
            "completed",
            [
                assistant_message(output_text("To B: 第一段")),
                assistant_message(output_text("To C: 第二段")),
            ],
        ),
        (
            "completed",
            [
                assistant_message(
                    output_text("To B: 第一段"),
                    output_text("To C: 第二段"),
                )
            ],
        ),
        (
            "completed",
            [assistant_message(output_text("To B: 正文"), role="user")],
        ),
    ],
)
def test_invalid_responses_output_shapes_are_rejected(
    status: str,
    output: list[Any],
) -> None:
    """Incomplete, refusal, tool, and ambiguous Responses outputs fail."""
    client = FakeResponsesClient(
        responses_response(status=status, output=output)
    )

    with pytest.raises(
        DraftGenerationError,
        match="Model returned an invalid message draft.",
    ):
        MessageDraftService(client, RESPONSES_MODEL).generate(
            create_scene("非法 Responses", RESPONSES_MODEL), "A"
        )

    assert len(client.responses.requests) == 1


@pytest.mark.parametrize(
    "usage",
    [
        SimpleNamespace(input_tokens=-1, output_tokens=1),
        SimpleNamespace(input_tokens=True, output_tokens=1),
        SimpleNamespace(input_tokens=5, output_tokens="1"),
        SimpleNamespace(
            input_tokens=4,
            output_tokens=1,
            input_tokens_details=SimpleNamespace(
                cache_write_tokens=3,
                cached_tokens=2,
            ),
        ),
        SimpleNamespace(
            input_tokens=10,
            output_tokens=2,
            total_tokens=99,
        ),
        SimpleNamespace(
            input_tokens=10,
            output_tokens=2,
            cache_write_tokens=1,
            input_tokens_details=SimpleNamespace(cache_write_tokens=2),
        ),
        SimpleNamespace(
            input_tokens=10,
            output_tokens=2,
            input_tokens_details=None,
        ),
    ],
)
def test_invalid_or_contradictory_responses_usage_is_rejected(
    usage: Any,
) -> None:
    """Malformed usage returns the same sanitized generation failure."""
    client = FakeResponsesClient(responses_response(usage=usage))

    with pytest.raises(
        DraftGenerationError,
        match="Model returned an invalid message draft.",
    ):
        MessageDraftService(client, RESPONSES_MODEL).generate(
            create_scene("非法 usage", RESPONSES_MODEL), "A"
        )

    assert len(client.responses.requests) == 1


@pytest.mark.parametrize(
    ("scene_model", "expected_protocol"),
    [
        (ANTHROPIC_MODEL, "anthropic"),
        (RESPONSES_MODEL, "responses"),
    ],
)
def test_http_generation_uses_the_scene_bound_service(
    tmp_path: Path,
    scene_model: str,
    expected_protocol: str,
) -> None:
    """The route selects by binding rather than a request parameter."""
    scene = create_scene("按场景路由", scene_model)
    storage = SceneStorage(tmp_path / "scenes")
    storage.create(scene)
    anthropic_client = FakeAnthropicClient(
        anthropic_response("To B: Claude 草稿")
    )
    responses_client = FakeResponsesClient(
        responses_response("To C: Responses 草稿")
    )
    application = create_app(
        storage,
        {
            ANTHROPIC_MODEL: MessageDraftService(
                anthropic_client, ANTHROPIC_MODEL
            ),
            RESPONSES_MODEL: MessageDraftService(
                responses_client, RESPONSES_MODEL
            ),
        },
    )

    response = TestClient(application).post(
        f"/api/scenes/{scene.id}/agents/A/message-drafts"
    )

    assert response.status_code == 200
    assert response.json()["request_snapshot"]["model"] == scene_model
    assert len(anthropic_client.messages.requests) == (
        1 if expected_protocol == "anthropic" else 0
    )
    assert len(responses_client.responses.requests) == (
        1 if expected_protocol == "responses" else 0
    )


def test_invalid_responses_usage_returns_sanitized_502(tmp_path: Path) -> None:
    """The HTTP boundary does not expose upstream text for invalid usage."""
    scene = create_scene("usage 502", RESPONSES_MODEL)
    storage = SceneStorage(tmp_path / "scenes")
    storage.create(scene)
    client = FakeResponsesClient(
        responses_response(
            text="To B: SENSITIVE UPSTREAM TEXT",
            usage=SimpleNamespace(
                input_tokens=1,
                output_tokens=1,
                input_tokens_details=SimpleNamespace(cached_tokens=2),
            ),
        )
    )
    application = create_app(
        storage,
        MessageDraftService(client, RESPONSES_MODEL),
    )

    response = TestClient(application).post(
        f"/api/scenes/{scene.id}/agents/A/message-drafts"
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Model returned an invalid message draft."
    }
    assert "SENSITIVE" not in response.text
