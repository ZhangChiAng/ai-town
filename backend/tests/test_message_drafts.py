"""Tests for isolated inner/outer model requests and draft generation."""

import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from app.drafting import (
    CACHE_CONTROL,
    DraftGenerationError,
    LayerDraftService,
    build_model_request,
    select_model_protocol,
)
from app.models import (
    ConfirmLayerRequest,
    SceneConflictError,
    add_manual_event,
    create_scene,
)

MODEL = "anthropic/claude-test"
RESPONSES_MODEL = "gpt-test"


class FakeMessages:
    """Capture provider requests and return queued provider-shaped responses."""

    def __init__(self, responses: list[Any]) -> None:
        """Store responses returned once in FIFO order."""
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        """Record one call and return its configured response."""
        self.requests.append(deepcopy(kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    """Minimal injectable Anthropic client."""

    def __init__(self, responses: list[Any]) -> None:
        """Expose one fake messages resource."""
        self.messages = FakeMessages(responses)


class FakeResponses:
    """Capture Responses requests and return queued provider responses."""

    def __init__(self, responses: list[Any]) -> None:
        """Store responses returned once in FIFO order."""
        self.responses_to_return = responses
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        """Record one call and return its configured response."""
        self.requests.append(deepcopy(kwargs))
        response = self.responses_to_return.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeResponsesClient:
    """Minimal injectable OpenAI client."""

    def __init__(self, responses: list[Any]) -> None:
        """Expose one fake Responses resource."""
        self.responses = FakeResponses(responses)


def model_response(
    text: str,
    *,
    leading_blocks: list[Any] | None = None,
) -> SimpleNamespace:
    """Build one provider-shaped response."""
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


def responses_model_response(text: str) -> dict[str, Any]:
    """Build one completed Responses result with temporary reasoning."""
    return {
        "status": "completed",
        "output": [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "简短摘要"}],
                "content": [{"type": "reasoning_text", "text": "临时推理"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
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


def confirmation(draft: Any, content: str | None = None) -> ConfirmLayerRequest:
    """Convert one browser draft to its confirmation payload."""
    return ConfirmLayerRequest(
        call_id=draft.call_id,
        event_id=draft.event_id,
        content=draft.content if content is None else content,
        state_token=draft.state_token,
    )


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (MODEL, "anthropic"),
        ("vendor/CLAUDE-custom", "anthropic"),
        (RESPONSES_MODEL, "responses"),
    ],
)
def test_model_name_selects_protocol(model: str, expected: str) -> None:
    """Concrete configured names deterministically select one protocol."""
    assert select_model_protocol(model) == expected


def test_responses_request_preserves_layer_input_without_cache_metadata() -> (
    None
):
    """Responses receives the same isolated text through stateless fields."""
    scene = add_manual_event(
        create_scene("Responses", RESPONSES_MODEL),
        "A",
        "门外传来两声敲门。",
    )

    request = build_model_request(scene, "A", "inner", RESPONSES_MODEL)

    assert request == {
        "model": RESPONSES_MODEL,
        "instructions": scene.agents[0].inner_context.system_prompt,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "外部事件：\n门外传来两声敲门。",
                    }
                ],
            }
        ],
        "max_output_tokens": 2048,
        "store": False,
    }
    assert "cache_control" not in json.dumps(request)
    assert "messages" not in request
    assert "system" not in request


def test_responses_generation_returns_reasoning_usage_and_actual_request() -> (
    None
):
    """Responses reasoning is observer-only and usage is normalized."""
    scene = add_manual_event(
        create_scene("Responses", RESPONSES_MODEL),
        "C",
        "一个事件",
    )
    client = FakeResponsesClient([responses_model_response("内层回答")])

    result = LayerDraftService(client, RESPONSES_MODEL).generate(
        scene,
        "C",
        "inner",
    )

    assert result.content == "内层回答"
    assert [block.model_dump() for block in result.reasoning] == [
        {"type": "summary_text", "text": "简短摘要"},
        {"type": "reasoning_text", "text": "临时推理"},
    ]
    assert result.usage.model_dump() == {
        "input_tokens": 12,
        "output_tokens": 7,
        "cache_creation_input_tokens": 3,
        "cache_read_input_tokens": 5,
    }
    assert result.request_snapshot == client.responses.requests[0]
    assert "临时推理" not in json.dumps(
        result.request_snapshot,
        ensure_ascii=False,
    )


def test_invalid_responses_output_fails_once_with_sanitized_error() -> None:
    """Unsafe Responses shapes are rejected without retry or provider data."""
    scene = add_manual_event(
        create_scene("坏响应", RESPONSES_MODEL),
        "A",
        "事件",
    )
    response = responses_model_response("文本")
    response["status"] = "failed"
    client = FakeResponsesClient([response])

    with pytest.raises(DraftGenerationError, match="invalid inner draft"):
        LayerDraftService(client, RESPONSES_MODEL).generate(
            scene,
            "A",
            "inner",
        )

    assert len(client.responses.requests) == 1


def test_first_inner_and_outer_inputs_are_exact() -> None:
    """The first round uses the two fixed user-text formats verbatim."""
    scene = add_manual_event(
        create_scene("格式", MODEL), "A", "门外传来两声敲门。"
    )
    client = FakeClient(
        [
            model_response("先别开门。\n问清楚是谁。"),
            model_response("To B: 你在门外吗？"),
        ]
    )
    service = LayerDraftService(client, MODEL)

    inner_draft = service.generate(scene, "A", "inner")
    assert client.messages.requests[0]["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "外部事件：\n门外传来两声敲门。",
                }
            ],
        }
    ]
    scene = service.confirm(
        scene,
        "A",
        "inner",
        confirmation(inner_draft),
    )

    outer_draft = service.generate(scene, "A", "outer")
    assert client.messages.requests[1]["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "外部事件：\n门外传来两声敲门。\n\n"
                        "你内心有一个声音：\n"
                        "先别开门。\n问清楚是谁。"
                    ),
                }
            ],
        }
    ]
    assert outer_draft.event_id == inner_draft.event_id


def test_later_inner_input_includes_only_previous_outer_output_and_event() -> (
    None
):
    """A later inner call gets the prior outer output in the fixed prefix."""
    scene = add_manual_event(create_scene("后续", MODEL), "A", "第一件事")
    client = FakeClient(
        [
            model_response("内层一"),
            model_response("To B: 外层一"),
            model_response("内层二"),
        ]
    )
    service = LayerDraftService(client, MODEL)

    inner = service.generate(scene, "A", "inner")
    scene = service.confirm(scene, "A", "inner", confirmation(inner))
    outer = service.generate(scene, "A", "outer")
    scene = service.confirm(scene, "A", "outer", confirmation(outer))
    scene = add_manual_event(scene, "A", "第二件事")

    service.generate(scene, "A", "inner")

    assert client.messages.requests[2]["messages"][-1] == {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": ("外层人格：\nTo B: 外层一\n\n外部事件：\n第二件事"),
            }
        ],
    }


def test_two_layers_send_only_their_own_complete_history() -> None:
    """System prompts, turns, and cache breakpoints stay layer-private."""
    scene = create_scene("隔离", MODEL)
    scene = scene.model_copy(
        update={
            "agents": [
                agent.model_copy(
                    update={
                        "inner_context": agent.inner_context.model_copy(
                            update={"system_prompt": f"INNER {agent.id}"}
                        ),
                        "outer_context": agent.outer_context.model_copy(
                            update={"system_prompt": f"OUTER {agent.id}"}
                        ),
                    }
                )
                for agent in scene.agents
            ]
        }
    )
    scene = add_manual_event(scene, "A", "A 的第一件事")
    client = FakeClient(
        [
            model_response("A 的内层秘密"),
            model_response("To B: A 的公开话"),
        ]
    )
    service = LayerDraftService(client, MODEL)
    inner = service.generate(scene, "A", "inner")
    scene = service.confirm(scene, "A", "inner", confirmation(inner))
    outer = service.generate(scene, "A", "outer")
    scene = service.confirm(scene, "A", "outer", confirmation(outer))
    scene = add_manual_event(scene, "A", "A 的第二件事")

    inner_request = build_model_request(scene, "A", "inner", MODEL)
    inner_json = json.dumps(inner_request, ensure_ascii=False)
    assert inner_request["system"][0]["text"] == "INNER A"
    assert "OUTER A" not in inner_json
    assert "INNER B" not in inner_json
    assert "From A: A 的公开话" not in inner_json
    assert str(scene.rollback_stack[0].call_id) not in inner_json
    assert inner_request["messages"][1]["content"][0]["cache_control"] == (
        CACHE_CONTROL
    )
    assert "cache_control" not in inner_request["messages"][-1]["content"][0]

    responses_request = build_model_request(
        scene,
        "A",
        "inner",
        RESPONSES_MODEL,
    )
    responses_json = json.dumps(responses_request, ensure_ascii=False)
    assert responses_request["instructions"] == "INNER A"
    assert "OUTER A" not in responses_json
    assert "INNER B" not in responses_json
    assert "From A: A 的公开话" not in responses_json
    assert "cache_control" not in responses_json

    pending_for_b = scene.agents[1].pending_events[0]
    scene = add_manual_event(scene, "B", "排在后面的 B 事件")
    outer_json = json.dumps(
        build_model_request(
            # B has no inner half-round, so outer preview must remain blocked.
            scene,
            "A",
            "inner",
            MODEL,
        ),
        ensure_ascii=False,
    )
    assert str(pending_for_b.id) not in outer_json
    assert "排在后面的 B 事件" not in outer_json


def test_each_layer_keeps_every_confirmed_turn_without_truncation() -> None:
    """All earlier turns remain verbatim in the selected layer request."""
    scene = create_scene("完整历史", MODEL)
    responses: list[Any] = []
    for index in range(4):
        responses.extend(
            (
                model_response(f"INNER-{index}"),
                model_response(f"To B: OUTER-{index}"),
            )
        )
    service = LayerDraftService(FakeClient(responses), MODEL)

    for index in range(4):
        scene = add_manual_event(scene, "A", f"EVENT-{index}")
        inner = service.generate(scene, "A", "inner")
        scene = service.confirm(scene, "A", "inner", confirmation(inner))
        outer = service.generate(scene, "A", "outer")
        scene = service.confirm(scene, "A", "outer", confirmation(outer))

    scene = add_manual_event(scene, "A", "EVENT-next")
    request = build_model_request(scene, "A", "inner", MODEL)
    text = json.dumps(request, ensure_ascii=False)
    for index in range(4):
        assert f"EVENT-{index}" in text
        assert f"INNER-{index}" in text
    assert len(request["messages"]) == 9


def test_generation_is_one_call_and_never_mutates_scene() -> None:
    """Allow multiline inner text and expose thinking without state writes."""
    scene = add_manual_event(create_scene("零写入", MODEL), "C", "一个事件")
    before = scene.model_dump_json()
    client = FakeClient(
        [
            model_response(
                "第一行\n第二行",
                leading_blocks=[
                    SimpleNamespace(type="thinking", thinking="PRIVATE")
                ],
            )
        ]
    )

    result = LayerDraftService(client, MODEL).generate(scene, "C", "inner")

    assert result.content == "第一行\n第二行"
    assert [block.model_dump() for block in result.reasoning] == [
        {"type": "thinking", "text": "PRIVATE"}
    ]
    assert "PRIVATE" not in json.dumps(result.request_snapshot)
    assert result.request_snapshot == client.messages.requests[0]
    assert len(client.messages.requests) == 1
    assert scene.model_dump_json() == before


@pytest.mark.parametrize(
    "text",
    [
        "To A: 发给自己",
        "To D: 无效",
        "To B:",
        "To B: 第一行\n第二行",
        "说明\nTo B: 正文",
    ],
)
def test_invalid_outer_generation_fails_without_retry(text: str) -> None:
    """Outer model output is rejected once instead of repaired or retried."""
    scene = add_manual_event(create_scene("无效外层", MODEL), "A", "事件")
    inner_client = FakeClient([model_response("内层")])
    inner_service = LayerDraftService(inner_client, MODEL)
    inner = inner_service.generate(scene, "A", "inner")
    scene = inner_service.confirm(
        scene,
        "A",
        "inner",
        confirmation(inner),
    )
    client = FakeClient([model_response(text)])

    with pytest.raises(DraftGenerationError, match="invalid outer draft"):
        LayerDraftService(client, MODEL).generate(scene, "A", "outer")

    assert len(client.messages.requests) == 1


def test_outer_preview_is_rejected_before_inner_confirmation() -> None:
    """Outer context is unavailable until a matching inner turn is saved."""
    scene = add_manual_event(create_scene("阶段", MODEL), "A", "事件")

    with pytest.raises(SceneConflictError, match="no confirmed inner"):
        build_model_request(scene, "A", "outer", MODEL)


def test_stale_token_rejects_confirmation_without_model_call() -> None:
    """A prompt edit after generation invalidates confirmation."""
    scene = add_manual_event(create_scene("过期", MODEL), "A", "事件")
    client = FakeClient([model_response("内层输出")])
    service = LayerDraftService(client, MODEL)
    draft = service.generate(scene, "A", "inner")
    agent = scene.agents[0]
    changed_agent = agent.model_copy(
        update={
            "inner_context": agent.inner_context.model_copy(
                update={"system_prompt": "CHANGED"}
            )
        }
    )
    changed_scene = scene.model_copy(
        update={"agents": [changed_agent, *scene.agents[1:]]}
    )

    with pytest.raises(SceneConflictError, match="scene changed"):
        service.confirm(
            changed_scene,
            "A",
            "inner",
            confirmation(draft),
        )

    assert len(client.messages.requests) == 1
