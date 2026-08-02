"""API and persistence tests for schema-v7 two-layer scenes."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from app.main import create_app
from app.models import SCHEMA_VERSION, create_scene
from app.storage import SceneReadError, SceneStorage
from tests import helpers
from tests.client import TestClient

FakeBackend = helpers.FakeBackend
confirm = helpers.confirm_draft
generate = helpers.generate_draft
post_event = helpers.post_event

MODEL = "anthropic/claude-test"


@pytest.fixture
def scene_directory(tmp_path: Path) -> Path:
    """Provide isolated JSON scene storage."""
    return tmp_path / "scenes"


def make_client(
    scene_directory: Path,
    texts: list[str] | None = None,
) -> tuple[TestClient, FakeBackend]:
    """Create an API client with one injectable global model."""
    backend = FakeBackend(texts or [], model=MODEL)
    return helpers.make_client(scene_directory, backend), backend


def post_scene(client: TestClient, name: str = "港口") -> dict[str, Any]:
    """Create and return one scene bound to the test model."""
    return helpers.post_scene(client, model=MODEL, name=name)


def test_model_options_are_ordered_and_hide_credentials(
    scene_directory: Path,
) -> None:
    """The public registry exposes only ordered concrete model names."""
    services = {
        "gpt-test": FakeBackend([], model="gpt-test"),
        MODEL: FakeBackend([], model=MODEL),
    }
    client = TestClient(create_app(SceneStorage(scene_directory), services))

    response = client.get("/api/model-options")

    assert response.status_code == 200
    assert response.json() == {
        "options": [
            {"model": "gpt-test"},
            {"model": MODEL},
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

    assert scene["schema_version"] == SCHEMA_VERSION == 7
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

    with pytest.raises(SceneReadError, match="not schema v7 or v6"):
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
    assert inner["event_ids"] == [scene["agents"][0]["pending_events"][0]["id"]]
    assert inner["content"] == "先观察。\n别急着表态。"

    inner_response = confirm(client, scene["id"], "A", "inner", inner)
    assert inner_response.status_code == 200
    half_round = inner_response.json()
    assert half_round["agents"][0]["pending_events"] == []
    inner_turn = half_round["agents"][0]["inner_context"]["turns"][0]
    assert inner_turn["input"] == "外部事件：\n潮水突然退了。"
    assert inner_turn["output"] == inner["content"]
    assert [event["id"] for event in inner_turn["consumed_events"]] == (
        inner["event_ids"]
    )
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
    assert len(model_client.generate_calls) == 2

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
    assert preview.json()["context"][-1] == {
        "role": "user",
        "text": "外部事件：\n事件\n\n你内心有一个声音：\n内层已确认",
    }


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
        content="To A: 不能发给自己",
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


def test_inner_round_consumes_the_whole_pending_batch_at_once(
    scene_directory: Path,
) -> None:
    """One inner confirmation consumes and persists the entire FIFO batch."""
    client, _model = make_client(
        scene_directory,
        ["A 内层", "To B: A 外层"],
    )
    scene = post_scene(client)
    scene = post_event(client, scene["id"], "A", "第一件事")
    scene = post_event(client, scene["id"], "A", "第二件事")
    event_ids = [event["id"] for event in scene["agents"][0]["pending_events"]]
    assert len(event_ids) == 2

    inner = generate(client, scene["id"], "A", "inner")
    assert inner["event_ids"] == event_ids
    # Both events appear in one block, separated by a blank line.
    assert inner["content"] == "A 内层"
    preview = client.get(
        f"/api/scenes/{scene['id']}/agents/A/model-request-preview?layer=inner"
    ).json()
    assert preview["event_ids"] == event_ids
    assert preview["context"][-1]["text"] == (
        "外部事件：\n第一件事\n\n第二件事"
    )

    scene = confirm(client, scene["id"], "A", "inner", inner).json()
    assert scene["agents"][0]["pending_events"] == []
    inner_turn = scene["agents"][0]["inner_context"]["turns"][0]
    assert inner_turn["event_ids"] == event_ids
    assert [event["content"] for event in inner_turn["consumed_events"]] == [
        "第一件事",
        "第二件事",
    ]

    # Outer input reproduces the same batch verbatim.
    outer = generate(client, scene["id"], "A", "outer")
    assert outer["event_ids"] == event_ids
    assert client.get(
        f"/api/scenes/{scene['id']}/agents/A/model-request-preview?layer=outer"
    ).json()["context"][-1]["text"] == (
        "外部事件：\n第一件事\n\n第二件事\n\n你内心有一个声音：\nA 内层"
    )
    scene = confirm(client, scene["id"], "A", "outer", outer).json()
    outer_turn = scene["agents"][0]["outer_context"]["turns"][0]
    assert outer_turn["event_ids"] == event_ids


def test_rollback_inner_restores_the_consumed_batch_in_fifo_order(
    scene_directory: Path,
) -> None:
    """Undoing an inner batch restores every consumed event in order."""
    client, _model = make_client(
        scene_directory,
        ["A 内层", "To C: A 外层"],
    )
    scene = post_scene(client)
    scene = post_event(client, scene["id"], "A", "第一件事")
    scene = post_event(client, scene["id"], "A", "第二件事")
    first_id = scene["agents"][0]["pending_events"][0]["id"]
    second_id = scene["agents"][0]["pending_events"][1]["id"]

    inner = generate(client, scene["id"], "A", "inner")
    scene = confirm(client, scene["id"], "A", "inner", inner).json()
    assert scene["agents"][0]["pending_events"] == []

    undone = client.post(f"/api/scenes/{scene['id']}/rollback").json()
    restored = undone["agents"][0]["pending_events"]
    assert [event["id"] for event in restored] == [first_id, second_id]
    assert undone["agents"][0]["inner_context"]["turns"] == []


def test_v6_scene_is_migrated_to_v7_on_read(
    scene_directory: Path,
) -> None:
    """A legacy single-event v6 file loads as a v7 batch scene and persists."""
    scene_directory.mkdir()
    scene_id = uuid4()
    path = scene_directory / f"{scene_id}.json"
    legacy = {
        "schema_version": 6,
        "id": str(scene_id),
        "name": "旧 v6 场景",
        "model": MODEL,
        "agents": [
            {
                "id": agent_id,
                "name": agent_id,
                "inner_context": {
                    "system_prompt": f"INNER {agent_id}",
                    "turns": [],
                },
                "outer_context": {
                    "system_prompt": f"OUTER {agent_id}",
                    "turns": [],
                },
                "pending_events": [],
            }
            for agent_id in ("A", "B", "C")
        ],
        "rollback_stack": [],
        "next_sequence": 1,
    }
    path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")

    scene = SceneStorage(scene_directory).get(scene_id)
    assert scene.schema_version == 7
    # A migrated v6 scene round-trips through validation and back to disk.
    SceneStorage(scene_directory).save(scene)
    reloaded = SceneStorage(scene_directory).get(scene_id)
    assert reloaded == scene
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == 7


def test_v6_scene_with_confirmed_turn_is_migrated_to_v7_batches(
    scene_directory: Path,
) -> None:
    """Legacy inner/outer turns keep references via single-item v7 batches."""  # noqa: E501
    scene_directory.mkdir()
    scene_id = uuid4()
    path = scene_directory / f"{scene_id}.json"
    event_id = str(uuid4())
    inner_call_id = str(uuid4())
    outer_call_id = str(uuid4())
    outer_sequence = 3
    inner_sequence = 2
    event_sequence = 1
    legacy = {
        "schema_version": 6,
        "id": str(scene_id),
        "name": "v6 带回合",
        "model": MODEL,
        "agents": [
            {
                "id": "A",
                "name": "A",
                "inner_context": {
                    "system_prompt": "INNER A",
                    "turns": [
                        {
                            "call_id": inner_call_id,
                            "event_id": event_id,
                            "sequence": inner_sequence,
                            "input": "外部事件：\n单一事件",
                            "output": "内层判断",
                            "consumed_event": {
                                "id": event_id,
                                "sequence": event_sequence,
                                "kind": "manual",
                                "content": "单一事件",
                                "source_agent_id": None,
                                "source_call_id": None,
                            },
                        }
                    ],
                },
                "outer_context": {
                    "system_prompt": "OUTER A",
                    "turns": [
                        {
                            "call_id": outer_call_id,
                            "event_id": event_id,
                            "sequence": outer_sequence,
                            "input": "外部事件：\n单一事件\n\n",
                            "output": "To B: 外层台词",
                            "recipient_id": "B",
                            "generated_event_id": "ignored",
                        }
                    ],
                },
                "pending_events": [],
            },
            {
                "id": "B",
                "name": "B",
                "inner_context": {
                    "system_prompt": "INNER B",
                    "turns": [],
                },
                "outer_context": {
                    "system_prompt": "OUTER B",
                    "turns": [],
                },
                "pending_events": [],
            },
            {
                "id": "C",
                "name": "C",
                "inner_context": {
                    "system_prompt": "INNER C",
                    "turns": [],
                },
                "outer_context": {
                    "system_prompt": "OUTER C",
                    "turns": [],
                },
                "pending_events": [],
            },
        ],
        "rollback_stack": [
            {
                "call_id": inner_call_id,
                "agent_id": "A",
                "layer": "inner",
            },
            {
                "call_id": outer_call_id,
                "agent_id": "A",
                "layer": "outer",
            },
        ],
        "next_sequence": 4,
    }
    # Make the routed event observable for B so outer routing is consistent.
    generated_event_id = str(uuid4())
    legacy["agents"][1]["pending_events"].append(
        {
            "id": generated_event_id,
            "sequence": outer_sequence,
            "kind": "agent_message",
            "content": "From A: 外层台词",
            "source_agent_id": "A",
            "source_call_id": outer_call_id,
        }
    )
    legacy["agents"][0]["outer_context"]["turns"][0]["generated_event_id"] = (
        generated_event_id
    )
    path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")

    scene = SceneStorage(scene_directory).get(scene_id)
    inner_turn = scene.agents[0].inner_context.turns[0]
    outer_turn = scene.agents[0].outer_context.turns[0]
    assert scene.schema_version == 7
    assert inner_turn.event_ids == [UUID(event_id)]
    assert [event.id for event in inner_turn.consumed_events] == [
        UUID(event_id)
    ]
    assert outer_turn.event_ids == [UUID(event_id)]
    assert outer_turn.generated_event_id == UUID(generated_event_id)


def test_inner_draft_goes_stale_when_a_new_event_arrives_before_confirm(
    scene_directory: Path,
) -> None:
    """A generated inner draft is rejected once the queue content changes."""
    client, _model = make_client(
        scene_directory,
        ["A 内层", "To C: A 外层"],
    )
    scene = post_scene(client)
    scene = post_event(client, scene["id"], "A", "第一件事")
    draft = generate(client, scene["id"], "A", "inner")
    assert draft["event_ids"] == [scene["agents"][0]["pending_events"][0]["id"]]

    newer = post_event(client, scene["id"], "A", "第二件事")
    assert len(newer["agents"][0]["pending_events"]) == 2

    stale_response = confirm(client, scene["id"], "A", "inner", draft)
    assert stale_response.status_code == 409
    # No partial write happened: the queue still holds both events.
    persisted = SceneStorage(scene_directory).get(UUID(scene["id"]))
    assert [event.content for event in persisted.agents[0].pending_events] == [
        "第一件事",
        "第二件事",
    ]


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
