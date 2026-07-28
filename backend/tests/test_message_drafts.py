"""Tests for one-step observable message draft generation."""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from app.drafting import (
    CACHE_CONTROL,
    RUNTIME_TURN_PROMPT,
    DraftGenerationError,
    MessageDraftService,
    build_message_request,
)
from app.models import CreateMessageRequest, add_message, create_scene

MODEL = "test-model"


class FakeMessages:
    """Capture one provider request and return a configured response."""

    def __init__(self, response: Any) -> None:
        """Initialize with a provider-shaped response."""
        self.response = response
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        """Capture keyword arguments and return the configured response."""
        self.requests.append(deepcopy(kwargs))
        return self.response


class FakeClient:
    """Minimal client accepted by the draft service."""

    def __init__(self, response: Any) -> None:
        """Initialize a fake messages resource."""
        self.messages = FakeMessages(response)


def model_response(
    text: str,
    *,
    leading_blocks: list[Any] | None = None,
) -> SimpleNamespace:
    """Create a provider-shaped response."""
    return SimpleNamespace(
        content=[
            *(leading_blocks or []),
            SimpleNamespace(type="text", text=text),
        ],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=2,
            cache_read_input_tokens=3,
        ),
    )


def test_timeline_text_and_runtime_instruction_are_exact_request_blocks() -> (
    None
):
    """Request construction never decorates authoritative timeline content."""
    scene = create_scene("可观测")
    scene = add_message(
        scene,
        CreateMessageRequest(sender_id="B", content="To A: 第一条"),
    )
    scene = add_message(
        scene,
        CreateMessageRequest(sender_id="A", content="To C: 第二条"),
    )

    request = build_message_request(scene, "A", MODEL)

    assert request["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "From B: 第一条",
                }
            ],
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
                    "text": RUNTIME_TURN_PROMPT.format(recipient_ids="B、C"),
                }
            ],
        },
    ]


@pytest.mark.parametrize(
    ("sender_id", "messages_before"),
    [
        ("A", []),
        ("A", [("B", "To A: 收到结尾")]),
        ("A", [("A", "To B: 发出结尾")]),
    ],
)
def test_every_request_appends_the_same_uncached_format_instruction(
    sender_id: str,
    messages_before: list[tuple[str, str]],
) -> None:
    """Empty, user-ending, and assistant-ending timelines all get a prompt."""
    scene = create_scene("格式")
    for message_sender, content in messages_before:
        scene = add_message(
            scene,
            CreateMessageRequest(sender_id=message_sender, content=content),
        )

    request = build_message_request(scene, sender_id, MODEL)
    final_block = request["messages"][-1]["content"][-1]

    assert final_block == {
        "type": "text",
        "text": RUNTIME_TURN_PROMPT.format(recipient_ids="B、C"),
    }
    assert "cache_control" not in final_block
    if messages_before:
        cache_blocks = [
            block
            for message in request["messages"]
            for block in message["content"]
            if block.get("cache_control") == CACHE_CONTROL
        ]
        assert len(cache_blocks) == 1
        assert cache_blocks[0]["text"] != final_block["text"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("To B: 正文", "To B: 正文"),
        ("To   B ：  正文", "To   B ：  正文"),
        ("  To C: 正文  ", "To C: 正文"),
    ],
)
def test_valid_visible_to_text_is_returned_complete(
    text: str,
    expected: str,
) -> None:
    """Allowed whitespace variants remain one complete editable draft."""
    client = FakeClient(
        model_response(
            text,
            leading_blocks=[
                SimpleNamespace(type="thinking", thinking="PRIVATE")
            ],
        )
    )
    result = MessageDraftService(client, MODEL).generate(
        create_scene("草稿"), "A"
    )

    assert result.content == expected
    assert "recipient_id" not in result.model_dump()
    assert result.request_snapshot == client.messages.requests[0]


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
def test_invalid_visible_text_fails_once_without_exposing_details(
    text: str,
) -> None:
    """Unsafe visible output is rejected rather than repaired or retried."""
    client = FakeClient(model_response(text))
    service = MessageDraftService(client, MODEL)

    with pytest.raises(
        DraftGenerationError,
        match="Model returned an invalid message draft.",
    ):
        service.generate(create_scene("无效"), "A")

    assert len(client.messages.requests) == 1


def test_unexpected_response_block_is_rejected() -> None:
    """Only thinking variants and one text block are accepted."""
    client = FakeClient(
        model_response(
            "To B: 正文",
            leading_blocks=[SimpleNamespace(type="tool_use")],
        )
    )

    with pytest.raises(DraftGenerationError):
        MessageDraftService(client, MODEL).generate(create_scene("块"), "A")
