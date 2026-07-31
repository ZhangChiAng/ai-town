"""API and persistence tests for schema-v6 two-layer scenes."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

from app.drafting import LayerDraftService
from app.main import create_app
from app.models import SCHEMA_VERSION, create_scene
from app.storage import SceneReadError, SceneStorage
from tests.client import TestClient

MODEL = "anthropic/claude-test"


class FakeMessages:
    """Capture model calls and return configured responses in order."""

    def __init__(self, texts: list[str]) -> None:
        """Create provider responses from visible text strings."""
        self.texts = texts
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        """Record a request and return its next visible text."""
        self.requests.append(deepcopy(kwargs))
        text = self.texts.pop(0)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(
                input_tokens=8,
                output_tokens=4,
                cache_creation_input_tokens=1,
                cache_read_input_tokens=2,
            ),
        )


class FakeClient:
    """Minimal model client for route tests."""

    def __init__(self, texts: list[str]) -> None:
        """Expose a fake messages resource."""
        self.messages = FakeMessages(texts)


@pytest.fixture
def scene_directory(tmp_path: Path) -> Path:
    """Provide isolated JSON scene storage."""
    return tmp_path / "scenes"


def make_client(
    scene_directory: Path,
    texts: list[str] | None = None,
) -> tuple[TestClient, FakeClient]:
    """Create an API client with one injectable global model."""
    model_client = FakeClient(texts or [])
    application = create_app(
        SceneStorage(scene_directory),
        LayerDraftService(model_client, MODEL),
    )
    return TestClient(application), model_client


def post_scene(client: TestClient, name: str = "港口") -> dict[str, Any]:
    """Create and return one scene."""
    response = client.post(
        "/api/scenes",
        json={"name": name, "model": MODEL},
    )
    assert response.status_code == 201
    return response.json()


def post_event(
    client: TestClient,
    scene_id: str,
    agent_id: str,
    content: str,
) -> dict[str, Any]:
    """Append a manual event and return the updated scene."""
    response = client.post(
        f"/api/scenes/{scene_id}/agents/{agent_id}/events",
        json={"content": content},
    )
    assert response.status_code == 201
    return response.json()


def generate(
    client: TestClient,
    scene_id: str,
    agent_id: str,
    layer: str,
) -> dict[str, Any]:
    """Generate one browser-only layer draft."""
    response = client.post(
        f"/api/scenes/{scene_id}/agents/{agent_id}/{layer}-drafts"
    )
    assert response.status_code == 200
    return response.json()


def confirm(
    client: TestClient,
    scene_id: str,
    agent_id: str,
    layer: str,
    draft: dict[str, Any],
    content: str | None = None,
) -> Any:
    """Confirm one draft, optionally replacing its editable content."""
    return client.post(
        (f"/api/scenes/{scene_id}/agents/{agent_id}/{layer}-confirmations"),
        json={
            "call_id": draft["call_id"],
            "event_id": draft["event_id"],
            "content": draft["content"] if content is None else content,
            "state_token": draft["state_token"],
        },
    )


def test_model_options_are_ordered_and_hide_credentials(
    scene_directory: Path,
) -> None:
    """The public registry exposes only protocol and concrete model name."""
    services = {
        "gpt-test": LayerDraftService(FakeClient([]), "gpt-test"),
        MODEL: LayerDraftService(FakeClient([]), MODEL),
    }
    client = TestClient(create_app(SceneStorage(scene_directory), services))

    response = client.get("/api/model-options")

    assert response.status_code == 200
    assert response.json() == {
        "options": [
            {"protocol": "anthropic", "model": MODEL},
            {"protocol": "responses", "model": "gpt-test"},
        ]
    }
    assert "key" not in response.text.casefold()
    assert "url" not in response.text.casefold()


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "缺模型"},
        {"name": "空模型", "model": "  "},
        {"name": "未知模型", "model": "gpt-missing"},
    ],
)
def test_new_scene_requires_an_available_explicit_model(
    scene_directory: Path,
    payload: dict[str, str],
) -> None:
    """Creation never silently chooses or accepts an unavailable model."""
    client, _model = make_client(scene_directory)

    response = client.post("/api/scenes", json=payload)

    assert response.status_code == 422
    assert not scene_directory.exists()


def test_unbound_scene_can_bind_once(
    scene_directory: Path,
) -> None:
    """An explicit null binding can be filled once but never replaced."""
    storage = SceneStorage(scene_directory)
    scene = create_scene("待绑定", MODEL).model_copy(update={"model": None})
    storage.create(scene)
    client, _model = make_client(scene_directory)

    response = client.put(
        f"/api/scenes/{scene.id}/model",
        json={"model": MODEL},
    )

    assert response.status_code == 200
    assert response.json()["model"] == MODEL
    replacement = client.put(
        f"/api/scenes/{scene.id}/model",
        json={"model": "gpt-missing"},
    )
    assert replacement.status_code == 409
    assert SceneStorage(scene_directory).get(scene.id).model == MODEL


@pytest.mark.parametrize(
    ("method", "suffix", "body"),
    [
        ("POST", "/agents/A/inner-drafts", None),
        (
            "POST",
            "/agents/A/inner-confirmations",
            {
                "call_id": "33333333-3333-4333-8333-333333333333",
                "event_id": "22222222-2222-4222-8222-222222222222",
                "content": "草稿",
                "state_token": "a" * 64,
            },
        ),
        (
            "GET",
            "/agents/A/model-request-preview?layer=inner",
            None,
        ),
    ],
)
@pytest.mark.parametrize("bound_model", [None, "removed-model"])
def test_unbound_or_unavailable_model_blocks_only_model_operations(
    scene_directory: Path,
    method: str,
    suffix: str,
    body: dict[str, str] | None,
    bound_model: str | None,
) -> None:
    """Model operations return 409 while ordinary scene state stays editable."""
    storage = SceneStorage(scene_directory)
    scene = create_scene("失效模型", MODEL).model_copy(
        update={"model": bound_model}
    )
    storage.create(scene)
    client, _model = make_client(scene_directory)

    path = f"/api/scenes/{scene.id}{suffix}"
    response = (
        client.get(path) if method == "GET" else client.post(path, json=body)
    )

    assert response.status_code == 409
    event = client.post(
        f"/api/scenes/{scene.id}/agents/A/events",
        json={"content": "仍可编辑事件"},
    )
    assert event.status_code == 201


def test_new_schema_round_trips_without_legacy_agent_fields(
    scene_directory: Path,
) -> None:
    """A new file contains only the two contexts, queue, and scene metadata."""
    client, _model = make_client(scene_directory)
    scene = post_scene(client)

    assert scene["schema_version"] == SCHEMA_VERSION == 6
    assert [agent["id"] for agent in scene["agents"]] == ["A", "B", "C"]
    assert set(scene) == {
        "schema_version",
        "id",
        "name",
        "model",
        "agents",
        "rollback_stack",
        "next_sequence",
    }
    for agent in scene["agents"]:
        assert set(agent) == {
            "id",
            "name",
            "inner_context",
            "outer_context",
            "pending_events",
        }
        assert set(agent["inner_context"]) == {"system_prompt", "turns"}
        assert set(agent["outer_context"]) == {"system_prompt", "turns"}
        for removed in (
            "persona",
            "desire",
            "fear",
            "memory",
            "system_prompt",
            "timeline",
        ):
            assert removed not in agent

    restarted = SceneStorage(scene_directory).get(UUID(scene["id"]))
    assert restarted.model_dump(mode="json") == scene


@pytest.mark.parametrize("schema_version", [1, 2, 3, 4, 5])
def test_old_schema_is_rejected_without_migration(
    scene_directory: Path,
    schema_version: int,
) -> None:
    """Legacy single-layer files are not upgraded or rewritten."""
    scene_directory.mkdir()
    scene_id = uuid4()
    path = scene_directory / f"{scene_id}.json"
    raw = {
        "schema_version": schema_version,
        "id": str(scene_id),
        "name": "旧场景",
        "agents": [],
    }
    original = json.dumps(raw).encode()
    path.write_bytes(original)

    with pytest.raises(SceneReadError, match="not schema v6"):
        SceneStorage(scene_directory).get(scene_id)

    assert path.read_bytes() == original


def test_scene_edit_saves_two_complete_prompts_and_preserves_state(
    scene_directory: Path,
) -> None:
    """PUT only changes names and prompt text, never queue or call history."""
    client, _model = make_client(scene_directory)
    scene = post_scene(client)
    scene = post_event(client, scene["id"], "A", "保留的事件")
    payload = {
        "name": "新名称",
        "agents": [
            {
                "id": agent["id"],
                "name": f"居民 {agent['id']}",
                "inner_context": {
                    "system_prompt": f"INNER {agent['id']}\n完整文本"
                },
                "outer_context": {
                    "system_prompt": f"OUTER {agent['id']}\n完整文本"
                },
            }
            for agent in scene["agents"]
        ],
    }

    response = client.put(f"/api/scenes/{scene['id']}", json=payload)

    assert response.status_code == 200
    updated = response.json()
    assert updated["name"] == "新名称"
    assert (
        updated["agents"][0]["pending_events"]
        == (scene["agents"][0]["pending_events"])
    )
    assert updated["agents"][0]["inner_context"]["system_prompt"] == (
        "INNER A\n完整文本"
    )
    obsolete = deepcopy(payload)
    obsolete["agents"][0]["persona"] = "不再支持"
    assert (
        client.put(f"/api/scenes/{scene['id']}", json=obsolete).status_code
        == 422
    )


def test_manual_events_are_fifo_editable_and_deletable_only_while_queued(
    scene_directory: Path,
) -> None:
    """Manual event APIs preserve order and enforce queue ownership."""
    client, _model = make_client(scene_directory)
    scene = post_scene(client)
    scene = post_event(client, scene["id"], "A", "  第一件事  ")
    scene = post_event(client, scene["id"], "A", "第二件事")
    events = scene["agents"][0]["pending_events"]
    first_id = events[0]["id"]
    second_id = events[1]["id"]
    assert [event["content"] for event in events] == ["第一件事", "第二件事"]
    assert [event["sequence"] for event in events] == [1, 2]

    edited = client.put(
        f"/api/scenes/{scene['id']}/agents/A/events/{second_id}",
        json={"content": "改过的第二件事"},
    )
    assert edited.status_code == 200
    assert [
        event["content"]
        for event in edited.json()["agents"][0]["pending_events"]
    ] == ["第一件事", "改过的第二件事"]

    wrong_owner = client.put(
        f"/api/scenes/{scene['id']}/agents/B/events/{second_id}",
        json={"content": "越权"},
    )
    assert wrong_owner.status_code == 404
    deleted = client.delete(
        f"/api/scenes/{scene['id']}/agents/A/events/{first_id}"
    )
    assert deleted.status_code == 200
    assert [
        event["id"] for event in deleted.json()["agents"][0]["pending_events"]
    ] == [second_id]


def test_generation_writes_nothing_and_two_confirmations_route_atomically(
    scene_directory: Path,
) -> None:
    """Only confirmations persist turns; outer confirmation routes one event."""
    client, model_client = make_client(
        scene_directory,
        ["先观察。\n别急着表态。", "To   C ：  去码头等我。  "],
    )
    scene = post_scene(client)
    scene = post_event(client, scene["id"], "A", "潮水突然退了。")
    path = scene_directory / f"{scene['id']}.json"

    before_inner_generation = path.read_bytes()
    inner = generate(client, scene["id"], "A", "inner")
    assert path.read_bytes() == before_inner_generation
    assert inner["event_id"] == scene["agents"][0]["pending_events"][0]["id"]
    assert inner["content"] == "先观察。\n别急着表态。"

    inner_response = confirm(client, scene["id"], "A", "inner", inner)
    assert inner_response.status_code == 200
    half_round = inner_response.json()
    assert half_round["agents"][0]["pending_events"] == []
    inner_turn = half_round["agents"][0]["inner_context"]["turns"][0]
    assert inner_turn["input"] == "外部事件：\n潮水突然退了。"
    assert inner_turn["output"] == inner["content"]
    assert inner_turn["consumed_event"]["id"] == inner["event_id"]
    assert half_round["agents"][0]["outer_context"]["turns"] == []

    before_outer_generation = path.read_bytes()
    outer = generate(client, scene["id"], "A", "outer")
    assert path.read_bytes() == before_outer_generation
    outer_response = confirm(
        client,
        scene["id"],
        "A",
        "outer",
        outer,
    )
    assert outer_response.status_code == 200
    completed = outer_response.json()
    outer_turn = completed["agents"][0]["outer_context"]["turns"][0]
    assert outer_turn["output"] == "To C: 去码头等我。"
    assert outer_turn["recipient_id"] == "C"
    assert completed["agents"][1]["pending_events"] == []
    received = completed["agents"][2]["pending_events"][0]
    assert received["content"] == "From A: 去码头等我。"
    assert received["kind"] == "agent_message"
    assert received["source_call_id"] == outer_turn["call_id"]
    assert received["id"] == outer_turn["generated_event_id"]
    assert len(model_client.messages.requests) == 2

    restarted = SceneStorage(scene_directory).get(UUID(scene["id"]))
    assert restarted.model_dump(mode="json") == completed


def test_half_round_restores_outer_stage_after_restart(
    scene_directory: Path,
) -> None:
    """A saved inner turn remains ready for outer generation after reload."""
    client, _model = make_client(scene_directory, ["内层已确认", "To B: 继续"])
    scene = post_event(
        client,
        post_scene(client)["id"],
        "A",
        "事件",
    )
    inner = generate(client, scene["id"], "A", "inner")
    assert confirm(client, scene["id"], "A", "inner", inner).status_code == 200

    restarted_client, _restarted_model = make_client(
        scene_directory,
        ["To B: 继续"],
    )
    preview = restarted_client.get(
        f"/api/scenes/{scene['id']}/agents/A/model-request-preview?layer=outer"
    )

    assert preview.status_code == 200
    assert preview.json()["layer"] == "outer"
    assert (
        preview.json()["request"]["messages"][-1]["content"][0]["text"]
        == "外部事件：\n事件\n\n你内心有一个声音：\n内层已确认"
    )


def test_stale_and_invalid_confirmations_never_partially_write(
    scene_directory: Path,
) -> None:
    """Changed events, replayed calls, and invalid outer text fail safely."""
    client, _model = make_client(
        scene_directory,
        ["原内层", "新内层", "To B: 合法外层"],
    )
    scene = post_event(
        client,
        post_scene(client)["id"],
        "A",
        "原事件",
    )
    event_id = scene["agents"][0]["pending_events"][0]["id"]
    stale = generate(client, scene["id"], "A", "inner")
    edited = client.put(
        f"/api/scenes/{scene['id']}/agents/A/events/{event_id}",
        json={"content": "新事件"},
    )
    assert edited.status_code == 200
    after_edit = (scene_directory / f"{scene['id']}.json").read_bytes()

    stale_response = confirm(client, scene["id"], "A", "inner", stale)
    assert stale_response.status_code == 409
    assert (scene_directory / f"{scene['id']}.json").read_bytes() == after_edit

    fresh = generate(client, scene["id"], "A", "inner")
    confirmed = confirm(client, scene["id"], "A", "inner", fresh)
    assert confirmed.status_code == 200
    confirmed_bytes = (scene_directory / f"{scene['id']}.json").read_bytes()
    assert confirm(client, scene["id"], "A", "inner", fresh).status_code == 409
    assert (
        scene_directory / f"{scene['id']}.json"
    ).read_bytes() == confirmed_bytes

    outer = generate(client, scene["id"], "A", "outer")
    invalid = confirm(
        client,
        scene["id"],
        "A",
        "outer",
        outer,
        "To A: 不能发给自己",
    )
    assert invalid.status_code == 422
    assert (
        scene_directory / f"{scene['id']}.json"
    ).read_bytes() == confirmed_bytes


def test_agent_events_cannot_be_edited_deleted_or_skipped(
    scene_directory: Path,
) -> None:
    """Routed messages are immutable and remain behind earlier FIFO events."""
    client, _model = make_client(
        scene_directory,
        ["A 内层", "To B: A 发出的消息"],
    )
    scene = post_scene(client)
    scene = post_event(client, scene["id"], "B", "B 的既有事件")
    scene = post_event(client, scene["id"], "A", "A 的事件")
    inner = generate(client, scene["id"], "A", "inner")
    assert confirm(client, scene["id"], "A", "inner", inner).status_code == 200
    outer = generate(client, scene["id"], "A", "outer")
    completed = confirm(client, scene["id"], "A", "outer", outer).json()
    b_events = completed["agents"][1]["pending_events"]
    generated_id = b_events[1]["id"]
    assert [event["content"] for event in b_events] == [
        "B 的既有事件",
        "From A: A 发出的消息",
    ]

    edit = client.put(
        f"/api/scenes/{scene['id']}/agents/B/events/{generated_id}",
        json={"content": "篡改"},
    )
    delete = client.delete(
        f"/api/scenes/{scene['id']}/agents/B/events/{generated_id}"
    )
    assert edit.status_code == 409
    assert delete.status_code == 409


def test_global_rollback_undoes_exactly_one_confirmed_call(
    scene_directory: Path,
) -> None:
    """Inner and outer rollback restore their distinct half-round states."""
    client, _model = make_client(
        scene_directory,
        ["A 内层", "To B: A 外层", "B 内层"],
    )
    scene = post_scene(client)
    scene = post_event(client, scene["id"], "A", "A 手工事件")
    scene = post_event(client, scene["id"], "B", "B 手工事件")
    a_event = scene["agents"][0]["pending_events"][0]
    b_event = scene["agents"][1]["pending_events"][0]

    a_inner = generate(client, scene["id"], "A", "inner")
    scene = confirm(client, scene["id"], "A", "inner", a_inner).json()
    a_outer = generate(client, scene["id"], "A", "outer")
    scene = confirm(client, scene["id"], "A", "outer", a_outer).json()
    routed_id = scene["agents"][1]["pending_events"][1]["id"]
    b_inner = generate(client, scene["id"], "B", "inner")
    scene = confirm(client, scene["id"], "B", "inner", b_inner).json()
    assert len(scene["rollback_stack"]) == 3
    next_sequence = scene["next_sequence"]

    undo_b_inner = client.post(f"/api/scenes/{scene['id']}/rollback")
    assert undo_b_inner.status_code == 200
    scene = undo_b_inner.json()
    assert scene["agents"][1]["inner_context"]["turns"] == []
    assert [event["id"] for event in scene["agents"][1]["pending_events"]] == [
        b_event["id"],
        routed_id,
    ]

    undo_a_outer = client.post(f"/api/scenes/{scene['id']}/rollback")
    assert undo_a_outer.status_code == 200
    scene = undo_a_outer.json()
    assert scene["agents"][0]["inner_context"]["turns"]
    assert scene["agents"][0]["outer_context"]["turns"] == []
    assert [event["id"] for event in scene["agents"][1]["pending_events"]] == [
        b_event["id"]
    ]

    undo_a_inner = client.post(f"/api/scenes/{scene['id']}/rollback")
    assert undo_a_inner.status_code == 200
    scene = undo_a_inner.json()
    assert scene["agents"][0]["inner_context"]["turns"] == []
    assert scene["agents"][0]["pending_events"][0]["id"] == a_event["id"]
    assert scene["rollback_stack"] == []
    assert scene["next_sequence"] == next_sequence

    before_conflict = (scene_directory / f"{scene['id']}.json").read_bytes()
    conflict = client.post(f"/api/scenes/{scene['id']}/rollback")
    assert conflict.status_code == 409
    assert (
        scene_directory / f"{scene['id']}.json"
    ).read_bytes() == before_conflict


def test_removed_single_layer_endpoints_are_absent(
    scene_directory: Path,
) -> None:
    """Old message, compose, and message-draft routes are not retained."""
    client, _model = make_client(scene_directory)
    scene = post_scene(client)

    assert (
        client.post(
            f"/api/scenes/{scene['id']}/messages",
            json={"sender_id": "A", "content": "To B: 旧消息"},
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/api/system-prompts/compose",
            json={"persona": "", "desire": "", "fear": "", "memory": ""},
        ).status_code
        == 404
    )
    assert (
        client.post(
            f"/api/scenes/{scene['id']}/agents/A/message-drafts"
        ).status_code
        == 404
    )
