"""Tests for model-generated, explicitly cached message drafts."""

import json
import logging
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import pytest

from app.drafting import (
    CACHE_CONTROL,
    RUNTIME_TURN_PROMPT,
    MessageDraftService,
    build_message_request,
)
from app.main import create_app
from app.models import (
    CreateMessageRequest,
    Scene,
    add_message,
    create_scene,
)
from app.storage import SceneStorage
from tests.client import TestClient

MODEL = "anthropic/claude-haiku-4.5"


class FakeMessages:
    """Capture requests and return or raise a configured result."""

    def __init__(self, result: Any) -> None:
        """Initialize with a response object or exception."""
        self.result = result
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        """Capture one Messages API request."""
        self.requests.append(deepcopy(kwargs))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeAnthropic:
    """Minimal client accepted by MessageDraftService."""

    def __init__(self, result: Any) -> None:
        """Initialize the fake Messages API resource."""
        self.messages = FakeMessages(result)


class DraftFixture(NamedTuple):
    """Collapsed draft_client fixture values."""

    client: TestClient
    fake: FakeAnthropic
    scene_directory: Path


def model_response(
    *,
    content: str = "今晚去灯塔。",
    include_thinking: bool = False,
    input_tokens: int = 120,
    output_tokens: int = 48,
    cache_creation_input_tokens: int = 17,
    cache_read_input_tokens: int = 91,
) -> SimpleNamespace:
    """Build a fake successful Anthropic response."""
    blocks = []
    if include_thinking:
        blocks.append(SimpleNamespace(type="thinking", thinking="PRIVATE"))
    blocks.append(
        SimpleNamespace(
            type="text",
            text=content,
        )
    )
    return SimpleNamespace(
        content=blocks,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        ),
    )


def test_response_allows_thinking_before_the_visible_text() -> None:
    """Provider reasoning blocks do not invalidate a plain-text draft."""
    service = MessageDraftService(
        FakeAnthropic(model_response(include_thinking=True)),
        MODEL,
    )
    scene = private_scene()

    result = service.generate(scene, "A")

    assert result.recipient_id is None
    assert result.content == "今晚去灯塔。"
    assert result.request_snapshot["model"] == MODEL


def private_scene() -> Scene:
    """Create a scene with distinguishable public and private fields."""
    scene = create_scene("SECRET_SCENE_NAME")
    updated_agents = []
    for agent in scene.agents:
        updated_agents.append(
            agent.model_copy(
                update={
                    "name": f"NAME_{agent.id}",
                    "persona": f"PRIVATE_PERSONA_{agent.id}",
                    "desire": f"PRIVATE_DESIRE_{agent.id}",
                    "fear": f"PRIVATE_FEAR_{agent.id}",
                    "memory": f"PRIVATE_MEMORY_{agent.id}",
                    "system_prompt": f"AUTHORITATIVE_SYSTEM_{agent.id}",
                }
            )
        )
    scene = scene.model_copy(update={"agents": updated_agents})
    scene = add_message(
        scene,
        CreateMessageRequest(
            sender_id="B",
            recipient_id="C",
            content="PRIVATE_BC_TIMELINE",
        ),
    )
    return add_message(
        scene,
        CreateMessageRequest(
            sender_id="C",
            recipient_id="A",
            content="VISIBLE_A_TIMELINE",
        ),
    )


@pytest.fixture
def draft_client(
    tmp_path: Path,
) -> DraftFixture:
    """Provide an API client with a successful fake Anthropic service."""
    scene_directory = tmp_path / "scenes"
    fake = FakeAnthropic(model_response())
    service = MessageDraftService(fake, MODEL)
    application = create_app(SceneStorage(scene_directory), service)
    return DraftFixture(TestClient(application), fake, scene_directory)


def persist_scene(
    client: TestClient,
    scene_directory: Path,
    scene: Scene,
) -> bytes:
    """Persist *scene* and return its exact serialized bytes."""
    SceneStorage(scene_directory).create(scene)
    response = client.get(f"/api/scenes/{scene.id}")
    assert response.status_code == 200
    return (scene_directory / f"{scene.id}.json").read_bytes()


def test_draft_endpoint_preserves_information_boundaries_and_disk(
    draft_client: DraftFixture,
) -> None:
    """Only the selected Agent's permitted context reaches the model."""
    client, fake, scene_directory = draft_client
    scene = private_scene()
    original_bytes = persist_scene(client, scene_directory, scene)

    response = client.post(f"/api/scenes/{scene.id}/agents/A/message-drafts")

    assert response.status_code == 200
    body = response.json()
    assert {key: body[key] for key in ("recipient_id", "content", "usage")} == {
        "recipient_id": None,
        "content": "今晚去灯塔。",
        "usage": {
            "input_tokens": 120,
            "output_tokens": 48,
            "cache_creation_input_tokens": 17,
            "cache_read_input_tokens": 91,
        },
    }
    assert body["request_snapshot"] == fake.messages.requests[0]
    request_text = json.dumps(
        fake.messages.requests[0],
        ensure_ascii=False,
    )
    for excluded in (
        "SECRET_SCENE_NAME",
        "PRIVATE_PERSONA_B",
        "PRIVATE_DESIRE_B",
        "PRIVATE_FEAR_C",
        "PRIVATE_MEMORY_C",
        "PRIVATE_BC_TIMELINE",
        "PRIVATE_PERSONA_A",
        "PRIVATE_DESIRE_A",
        "PRIVATE_FEAR_A",
        "PRIVATE_MEMORY_A",
        "NAME_A",
        "NAME_B",
        "NAME_C",
        "当前 Agent",
        "候选接收人",
        "已确认时间线记录",
        "正文:",
    ):
        assert excluded not in request_text
    for included in ("AUTHORITATIVE_SYSTEM_A", "VISIBLE_A_TIMELINE"):
        assert included in request_text
    assert (scene_directory / f"{scene.id}.json").read_bytes() == original_bytes


def test_preview_matches_actual_request_without_calling_model_or_writing(
    draft_client: DraftFixture,
) -> None:
    """Preview and generation share one exact, side-effect-free payload."""
    client, fake, scene_directory = draft_client
    scene = private_scene()
    original_bytes = persist_scene(client, scene_directory, scene)

    preview = client.get(
        f"/api/scenes/{scene.id}/agents/A/model-request-preview"
    )

    assert preview.status_code == 200
    assert fake.messages.requests == []
    assert (scene_directory / f"{scene.id}.json").read_bytes() == original_bytes

    draft = client.post(f"/api/scenes/{scene.id}/agents/A/message-drafts")

    assert draft.status_code == 200
    assert preview.json()["request"] == fake.messages.requests[0]
    assert draft.json()["request_snapshot"] == preview.json()["request"]


def test_compose_prompt_endpoint_uses_canonical_template() -> None:
    """The no-side-effect endpoint is the sole template implementation."""
    client = TestClient(create_app(SceneStorage(Path("/tmp/unused-scenes"))))

    response = client.post(
        "/api/system-prompts/compose",
        json={
            "persona": "人设正文",
            "desire": "欲望正文",
            "fear": "恐惧正文",
            "memory": "记忆正文",
        },
    )

    assert response.status_code == 200
    prompt = response.json()["system_prompt"]
    assert "像真人一样说话" in prompt
    assert "一个人最本质的东西是他的欲望和恐惧" in prompt
    assert prompt.endswith("【记忆】\n记忆正文")


def test_request_has_no_tools_and_two_explicit_5m_breakpoints() -> None:
    """Keep plain-text generation and both cache layers deterministic."""
    scene = private_scene()
    expected_prompt = next(
        agent.system_prompt for agent in scene.agents if agent.id == "A"
    )
    request = build_message_request(scene, "A", MODEL)

    assert request["max_tokens"] == 512
    assert request["model"] == MODEL
    assert request["system"][0]["text"] == expected_prompt
    assert "tools" not in request
    assert "tool_choice" not in request
    assert request["system"][0]["cache_control"] == CACHE_CONTROL
    last_block = request["messages"][-1]["content"][-1]
    assert last_block["cache_control"] == CACHE_CONTROL
    serialized = json.dumps(request)
    assert serialized.count('"cache_control"') == 2


def test_timeline_maps_to_alternating_native_messages_in_exact_order() -> None:
    """Adjacent same-role records remain separate blocks in one API turn."""
    scene = create_scene("原生消息映射")
    for sender_id, recipient_id, content in (
        ("B", "A", "第一条收到"),
        ("A", "C", "第一条发出"),
        ("A", "B", "连续发出"),
        ("B", "A", "连续收到之一"),
        ("C", "A", "最后收到"),
    ):
        scene = add_message(
            scene,
            CreateMessageRequest(
                sender_id=sender_id,
                recipient_id=recipient_id,
                content=content,
            ),
        )

    request = build_message_request(scene, "A", MODEL)

    assert request["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "B：第一条收到"}],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "第一条发出"},
                {"type": "text", "text": "连续发出"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "B：连续收到之一"},
                {
                    "type": "text",
                    "text": "C：最后收到",
                    "cache_control": CACHE_CONTROL,
                },
            ],
        },
    ]
    assert all(
        current["role"] != following["role"]
        for current, following in zip(
            request["messages"], request["messages"][1:], strict=False
        )
    )


def test_empty_timeline_adds_uncached_runtime_user_turn() -> None:
    """An empty log gets one runtime turn and no rolling breakpoint."""
    request = build_message_request(create_scene("空时间线"), "A", MODEL)

    assert request["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": RUNTIME_TURN_PROMPT}],
        }
    ]
    assert request["system"][0]["cache_control"] == CACHE_CONTROL
    assert json.dumps(request).count('"cache_control"') == 1


def test_timeline_ending_in_sent_adds_uncached_runtime_user_turn() -> None:
    """An outgoing final record is cached before a runtime user turn."""
    scene = add_message(
        create_scene("发出结尾"),
        CreateMessageRequest(
            sender_id="A",
            recipient_id="B",
            content="由 assistant 结尾",
        ),
    )

    request = build_message_request(scene, "A", MODEL)

    assert request["messages"][-2]["role"] == "assistant"
    assert request["messages"][-2]["content"][0]["text"] == "由 assistant 结尾"
    assert request["messages"][-2]["content"][0]["cache_control"] == (
        CACHE_CONTROL
    )
    assert request["messages"][-1] == {
        "role": "user",
        "content": [{"type": "text", "text": RUNTIME_TURN_PROMPT}],
    }


def test_timeline_ending_in_user_does_not_add_runtime_turn() -> None:
    """A received message already provides the user turn to continue."""
    scene = add_message(
        create_scene("收到结尾"),
        CreateMessageRequest(
            sender_id="B",
            recipient_id="A",
            content="由 user 结尾",
        ),
    )

    request = build_message_request(scene, "A", MODEL)

    assert len(request["messages"]) == 1
    assert request["messages"][0]["role"] == "user"
    assert request["messages"][0]["content"][0]["text"] == "B：由 user 结尾"


def test_context_order_and_cache_prefix_are_stable_as_timeline_grows() -> None:
    """Repeat requests match and appended records preserve prior messages."""
    scene = private_scene()
    first = build_message_request(scene, "A", MODEL)
    repeated = build_message_request(scene, "A", MODEL)
    assert repeated == first

    grown_scene = add_message(
        scene,
        CreateMessageRequest(
            sender_id="A",
            recipient_id="B",
            content="APPENDED_TIMELINE",
        ),
    )
    grown = build_message_request(grown_scene, "A", MODEL)
    old_messages = first["messages"]
    grown_messages = grown["messages"]

    # Cache marker placement moves, but immutable role and text stay byte-equal.
    assert [
        {
            "role": message["role"],
            "text": [block["text"] for block in message["content"]],
        }
        for message in grown_messages[: len(old_messages)]
    ] == [
        {
            "role": message["role"],
            "text": [block["text"] for block in message["content"]],
        }
        for message in old_messages
    ]
    assert grown_messages[-2]["content"][0]["text"] == "APPENDED_TIMELINE"
    assert grown_messages[-2]["content"][0]["cache_control"] == CACHE_CONTROL
    assert grown_messages[-1]["content"][0]["text"] == RUNTIME_TURN_PROMPT


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(content=[], usage=SimpleNamespace()),
        SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use",
                    name="compose_message",
                    input={"recipient_id": "B", "content": "正文"},
                )
            ],
            usage=SimpleNamespace(),
        ),
        SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="正文"),
                SimpleNamespace(type="text", text="第二条"),
            ],
            usage=SimpleNamespace(),
        ),
        model_response(content=" \t "),
        SimpleNamespace(
            content=[SimpleNamespace(type="image", source={})],
            usage=SimpleNamespace(),
        ),
    ],
)
def test_invalid_model_outputs_return_one_sanitized_502(
    tmp_path: Path,
    response: SimpleNamespace,
) -> None:
    """Invalid blocks or content fail without a repair request."""
    fake = FakeAnthropic(response)
    service = MessageDraftService(fake, MODEL)
    storage = SceneStorage(tmp_path / "scenes")
    scene = private_scene()
    storage.create(scene)

    client = TestClient(create_app(storage, service))
    result = client.post(f"/api/scenes/{scene.id}/agents/A/message-drafts")

    assert result.status_code == 502
    assert result.json() == {
        "detail": "Model returned an invalid message draft."
    }
    assert len(fake.messages.requests) == 1


def test_upstream_failure_is_sanitized_and_not_retried(tmp_path: Path) -> None:
    """Provider response details and credentials do not reach the API."""
    secret = "sk-sensitive"
    provider_body = "raw provider response"
    fake = FakeAnthropic(RuntimeError(f"{secret}: {provider_body}"))
    service = MessageDraftService(fake, MODEL)
    storage = SceneStorage(tmp_path / "scenes")
    scene = private_scene()
    storage.create(scene)

    client = TestClient(create_app(storage, service))
    result = client.post(f"/api/scenes/{scene.id}/agents/A/message-drafts")

    serialized = result.text
    assert result.status_code == 502
    assert result.json() == {"detail": "Model request failed."}
    assert secret not in serialized
    assert provider_body not in serialized
    assert len(fake.messages.requests) == 1


def test_usage_is_logged_without_prompt_or_draft(
    draft_client: DraftFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Successful generation logs IDs, model, and all four usage fields."""
    client, _fake, scene_directory = draft_client
    scene = private_scene()
    persist_scene(client, scene_directory, scene)

    with caplog.at_level(logging.INFO, logger="app.drafting"):
        response = client.post(
            f"/api/scenes/{scene.id}/agents/A/message-drafts"
        )

    assert response.status_code == 200
    record = json.loads(caplog.records[-1].message)
    assert record == {
        "event": "message_draft_generated",
        "scene_id": str(scene.id),
        "agent_id": "A",
        "model": MODEL,
        "input_tokens": 120,
        "output_tokens": 48,
        "cache_creation_input_tokens": 17,
        "cache_read_input_tokens": 91,
    }
    log_text = caplog.text
    assert "sk-" not in log_text
    assert "PRIVATE_PERSONA_A" not in log_text
    assert "今晚去灯塔。" not in log_text


def test_invalid_agent_and_missing_scene_have_expected_statuses(
    draft_client: DraftFixture,
) -> None:
    """Path validation remains distinct from storage lookup failures."""
    client, fake, scene_directory = draft_client
    scene = private_scene()
    persist_scene(client, scene_directory, scene)

    invalid_agent = client.post(
        f"/api/scenes/{scene.id}/agents/D/message-drafts"
    )
    missing_scene = client.post(
        "/api/scenes/00000000-0000-4000-8000-000000000000/"
        "agents/A/message-drafts"
    )

    assert invalid_agent.status_code == 422
    assert missing_scene.status_code == 404
    assert fake.messages.requests == []
