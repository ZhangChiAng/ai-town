"""API and persistence tests for versioned two-layer scenes."""

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from app.main import create_app
from app.models import CURRENT_SCENE_SCHEMA, create_scene
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
    """Create one scene with complete prompts bound to the test model."""
    return helpers.post_prompted_scene(client, model=MODEL, name=name)


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
def test_unavailable_model_blocks_only_model_operations(
    scene_directory: Path,
    method: str,
    suffix: str,
    body: dict[str, str] | None,
) -> None:
    """Model operations return 409 while ordinary scene state stays editable."""
    storage = SceneStorage(scene_directory)
    scene = create_scene("失效模型", "removed-model")
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


def test_null_model_is_corrupt_and_model_binding_route_is_absent(
    scene_directory: Path,
) -> None:
    """Every persisted scene has one immutable non-empty model string."""
    storage = SceneStorage(scene_directory)
    scene = create_scene("模型必需", MODEL)
    raw = scene.model_dump(mode="json", by_alias=True)
    raw["model"] = None
    scene_directory.mkdir()
    path = scene_directory / f"{scene.id}.json"
    original = json.dumps(raw, ensure_ascii=False).encode()
    path.write_bytes(original)

    with pytest.raises(SceneReadError, match="corrupted"):
        storage.get(scene.id)
    assert path.read_bytes() == original

    path.unlink()
    client, _model = make_client(scene_directory)
    created = post_scene(client)
    response = client.put(
        f"/api/scenes/{created['id']}/model",
        json={"model": MODEL},
    )
    assert response.status_code == 404


def test_new_schema_round_trips_without_legacy_agent_fields(
    scene_directory: Path,
) -> None:
    """A new file contains only the two contexts, queue, and scene metadata."""
    client, _model = make_client(scene_directory)
    scene = post_scene(client)

    assert scene["schema"] == CURRENT_SCENE_SCHEMA == "ai-town.scene/1.0"
    assert [agent["id"] for agent in scene["agents"]] == ["A", "B", "C"]
    assert set(scene) == {
        "schema",
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
            "prompt_profile",
            "interactions",
            "inner_context",
            "outer_context",
            "pending_events",
        }
        assert set(agent["inner_context"]) == {"turns"}
        assert set(agent["outer_context"]) == {"turns"}
        assert set(agent["prompt_profile"]) == {
            "pronoun",
            "hidden_beliefs",
            "inner_memories",
            "outer_memories",
        }
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
    assert restarted.model_dump(mode="json", by_alias=True) == scene


def test_new_scene_starts_with_blank_variables_and_blocks_model_calls(
    scene_directory: Path,
) -> None:
    """Blank profiles and interactions save but cannot reach a model."""
    client, model_client = make_client(scene_directory, ["内层", "对B说：外层"])
    scene = helpers.post_scene(client, model=MODEL)
    for agent in scene["agents"]:
        assert agent["prompt_profile"] == {
            "pronoun": "",
            "hidden_beliefs": "",
            "inner_memories": "",
            "outer_memories": "",
        }
        assert agent["interactions"] == {}

    post_event(client, scene["id"], "A", "事件")
    assert (
        client.post(
            f"/api/scenes/{scene['id']}/agents/A/inner-drafts"
        ).status_code
        == 409
    )
    assert (
        client.get(
            f"/api/scenes/{scene['id']}/agents/A/model-request-preview"
            "?layer=inner"
        ).status_code
        == 409
    )
    assert (
        client.post(
            f"/api/scenes/{scene['id']}/agents/A/outer-drafts"
        ).status_code
        == 409
    )
    assert model_client.generate_calls == []

    blank_update = {
        "name": scene["name"],
        "agents": [
            {
                "id": agent["id"],
                "name": agent["name"],
                "prompt_profile": {
                    "pronoun": "",
                    "hidden_beliefs": "",
                    "inner_memories": "",
                    "outer_memories": "  ",
                },
                "interactions": {},
            }
            for agent in scene["agents"]
        ],
    }
    saved = client.put(f"/api/scenes/{scene['id']}", json=blank_update)
    assert saved.status_code == 200
    assert saved.json()["agents"][0]["prompt_profile"]["pronoun"] == ""
    assert (
        client.post(
            f"/api/scenes/{scene['id']}/agents/A/inner-drafts"
        ).status_code
        == 409
    )

    filled = client.put(
        f"/api/scenes/{scene['id']}",
        json={
            "name": scene["name"],
            "agents": [
                {
                    "id": agent["id"],
                    "name": agent["name"],
                    "prompt_profile": {
                        "pronoun": "她",
                        "hidden_beliefs": f"HIDDEN {agent['id']}",
                        "inner_memories": f"INNER {agent['id']}",
                        "outer_memories": f"OUTER {agent['id']}",
                    },
                    "interactions": (
                        {
                            "B": {
                                "description": "你的儿子。",
                                "addresses": {"B": "一般场合"},
                            }
                        }
                        if agent["id"] == "A"
                        else {}
                    ),
                }
                for agent in scene["agents"]
            ],
        },
    )
    assert filled.status_code == 200
    inner = generate(client, scene["id"], "A", "inner")
    assert inner["content"] == "内层"


@pytest.mark.parametrize(
    "schema",
    ["ai-town.scene/1.0", "ai-town.scene/1.1", "ai-town.scene/1.999"],
)
def test_same_major_schema_round_trips_without_changing_minor(
    scene_directory: Path,
    schema: str,
) -> None:
    """Every legal 1.x file remains on its persisted schema identifier."""
    storage = SceneStorage(scene_directory)
    scene = helpers.create_prompted_scene("同主版本", MODEL)
    raw = scene.model_dump(mode="json", by_alias=True)
    raw["schema"] = schema
    scene_directory.mkdir()
    path = scene_directory / f"{scene.id}.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    loaded = storage.get(scene.id)
    assert loaded.schema_id == schema

    storage.mutate(
        scene.id,
        lambda current: current.model_copy(update={"name": "保存后"}),
    )
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["schema"] == schema


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        ("other.scene/1.0", "invalid scene schema"),
        ("ai-town.scene/1", "invalid scene schema"),
        ("ai-town.scene/1.0.0", "invalid scene schema"),
        ("ai-town.scene/01.0", "invalid scene schema"),
        ("ai-town.scene/1.01", "invalid scene schema"),
        ("ai-town.scene/-1.0", "invalid scene schema"),
        ("ai-town.scene/0.9", "incompatible schema major 0"),
        ("ai-town.scene/2.0", "incompatible schema major 2"),
    ],
)
def test_invalid_or_incompatible_schema_is_rejected_before_structure(
    scene_directory: Path,
    schema: str,
    message: str,
) -> None:
    """Schema identity is checked first and failed files stay untouched."""
    scene_directory.mkdir()
    scene_id = uuid4()
    path = scene_directory / f"{scene_id}.json"
    raw = {
        "schema": schema,
        "id": str(scene_id),
    }
    original = json.dumps(raw).encode()
    path.write_bytes(original)

    with pytest.raises(SceneReadError, match=message):
        SceneStorage(scene_directory).get(scene_id)

    assert path.read_bytes() == original


def test_numeric_schema_version_and_legacy_interactions_are_not_converted(
    scene_directory: Path,
) -> None:
    """Obsolete version fields and relationship maps remain invalid data."""
    storage = SceneStorage(scene_directory)
    scene = helpers.create_prompted_scene("旧结构", MODEL)
    raw = scene.model_dump(mode="json", by_alias=True)
    raw["schema_version"] = 9
    del raw["schema"]
    scene_directory.mkdir()
    path = scene_directory / f"{scene.id}.json"
    original = json.dumps(raw, ensure_ascii=False).encode()
    path.write_bytes(original)

    with pytest.raises(SceneReadError, match="invalid scene schema"):
        storage.get(scene.id)
    assert path.read_bytes() == original

    raw["schema"] = CURRENT_SCENE_SCHEMA
    del raw["schema_version"]
    for agent in raw["agents"]:
        agent["interactions"] = {
            target_id: relationship["addresses"]
            for target_id, relationship in agent["interactions"].items()
        }
    original = json.dumps(raw, ensure_ascii=False, indent=2).encode()
    path.write_bytes(original)

    with pytest.raises(SceneReadError, match="corrupted"):
        storage.get(scene.id)
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "missing_field",
    [
        "rollback_stack",
        "next_sequence",
        "pending_events",
        "inner_turns",
        "outer_turns",
        "event_source",
        "reasoning",
    ],
)
def test_same_major_scene_rejects_missing_persisted_fields(
    scene_directory: Path,
    missing_field: str,
) -> None:
    """A 1.x identifier never enables defaults for corrupted structures."""
    client, _model = make_client(scene_directory, ["内层输出"])
    scene = post_event(client, post_scene(client)["id"], "A", "事件")
    draft = generate(client, scene["id"], "A", "inner")
    scene = confirm(client, scene["id"], "A", "inner", draft).json()
    path = scene_directory / f"{scene['id']}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))

    if missing_field in {"rollback_stack", "next_sequence"}:
        del raw[missing_field]
    elif missing_field == "pending_events":
        del raw["agents"][1]["pending_events"]
    elif missing_field == "inner_turns":
        del raw["agents"][0]["inner_context"]["turns"]
    elif missing_field == "outer_turns":
        del raw["agents"][0]["outer_context"]["turns"]
    elif missing_field == "event_source":
        del raw["agents"][0]["inner_context"]["turns"][0]["consumed_events"][0][
            "source_agent_id"
        ]
    else:
        del raw["agents"][0]["inner_context"]["turns"][0]["reasoning"]

    original = json.dumps(raw, ensure_ascii=False).encode()
    path.write_bytes(original)

    with pytest.raises(SceneReadError, match="corrupted"):
        SceneStorage(scene_directory).get(UUID(scene["id"]))
    assert path.read_bytes() == original


def test_explicit_empty_persisted_arrays_round_trip(
    scene_directory: Path,
) -> None:
    """Empty queues, histories, rollback state, and reasoning stay legal."""
    client, _model = make_client(scene_directory, ["内层输出"])
    scene = post_event(client, post_scene(client)["id"], "A", "事件")
    draft = generate(client, scene["id"], "A", "inner")
    scene = confirm(client, scene["id"], "A", "inner", draft).json()
    path = scene_directory / f"{scene['id']}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["agents"][0]["inner_context"]["turns"][0]["reasoning"] = []
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    loaded = SceneStorage(scene_directory).get(UUID(scene["id"]))

    assert loaded.agents[0].inner_context.turns[0].reasoning == []
    assert loaded.agents[0].pending_events == []
    assert loaded.agents[0].outer_context.turns == []
    assert loaded.agents[1].inner_context.turns == []


def test_scene_api_rejects_duplicate_trimmed_agent_names_without_writing(
    scene_directory: Path,
) -> None:
    """Names are unique after trimming with case-sensitive comparison."""
    client, _model = make_client(scene_directory)
    scene = post_scene(client)
    payload = {
        "name": scene["name"],
        "agents": [
            {
                "id": agent["id"],
                "name": (
                    " 同名 "
                    if agent["id"] == "A"
                    else "同名"
                    if agent["id"] == "B"
                    else "第三人"
                ),
                "prompt_profile": agent["prompt_profile"],
                "interactions": agent["interactions"],
            }
            for agent in scene["agents"]
        ],
    }
    path = scene_directory / f"{scene['id']}.json"
    before = path.read_bytes()

    response = client.put(f"/api/scenes/{scene['id']}", json=payload)

    assert response.status_code == 422
    assert path.read_bytes() == before

    payload["agents"][0]["name"] = "Person"
    payload["agents"][1]["name"] = "person"
    assert (
        client.put(f"/api/scenes/{scene['id']}", json=payload).status_code
        == 200
    )


def test_scene_list_skips_one_unreadable_file(
    scene_directory: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One corrupt file is logged and does not hide readable scenes."""
    client, _model = make_client(scene_directory)
    later = post_scene(client, name="码头")
    earlier = post_scene(client, name="仓库")
    corrupt_path = scene_directory / f"{uuid4()}.json"
    corrupt_contents = b"not valid JSON"
    corrupt_path.write_bytes(corrupt_contents)
    caplog.set_level(logging.ERROR, logger="app.storage")

    response = client.get("/api/scenes")

    assert response.status_code == 200
    assert response.json() == [
        {"id": earlier["id"], "name": earlier["name"]},
        {"id": later["id"], "name": later["name"]},
    ]
    assert corrupt_path.read_bytes() == corrupt_contents
    [record] = [
        record
        for record in caplog.records
        if getattr(record, "event_name", None) == "scene.load.failed"
    ]
    assert record.event_fields["scene_file"] == corrupt_path.name
    assert record.event_fields["exception_type"] == "SceneReadError"


def test_scene_edit_saves_profiles_interactions_and_preserves_state(
    scene_directory: Path,
) -> None:
    """PUT changes scene settings but never queue or call history."""
    client, _model = make_client(scene_directory)
    scene = post_scene(client)
    scene = post_event(client, scene["id"], "A", "保留的事件")
    payload = {
        "name": "新名称",
        "agents": [
            {
                "id": agent["id"],
                "name": f"居民 {agent['id']}",
                "prompt_profile": {
                    "pronoun": "她",
                    "hidden_beliefs": f"HIDDEN {agent['id']}",
                    "inner_memories": f"INNER {agent['id']}\n完整文本",
                    "outer_memories": f"OUTER {agent['id']}\n完整文本",
                },
                "interactions": {
                    ("B" if agent["id"] == "A" else "A"): {
                        "description": "关系简介。",
                        "addresses": {"家人": "一般场合"},
                    }
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
    assert updated["agents"][0]["prompt_profile"]["inner_memories"] == (
        "INNER A\n完整文本"
    )
    obsolete = deepcopy(payload)
    obsolete["agents"][0]["persona"] = "不再支持"
    assert (
        client.put(f"/api/scenes/{scene['id']}", json=obsolete).status_code
        == 422
    )


@pytest.mark.parametrize(
    "bad_interactions",
    [
        {
            "A": {
                "description": "自己。",
                "addresses": {"自己": "任何场合"},
            }
        },
        {
            "B": {
                "description": "家人。",
                "addresses": {"家人": "一般场合"},
            },
            "C": {
                "description": "另一个家人。",
                "addresses": {"家人": "正式场合"},
            },
        },
    ],
)
def test_scene_api_rejects_self_targets_and_duplicate_addresses(
    scene_directory: Path,
    bad_interactions: dict[str, object],
) -> None:
    """Invalid reverse-routing configuration never reaches scene JSON."""
    client, _model = make_client(scene_directory)
    scene = post_scene(client)
    payload = {
        "name": scene["name"],
        "agents": [
            {
                "id": agent["id"],
                "name": agent["name"],
                "prompt_profile": agent["prompt_profile"],
                "interactions": (
                    bad_interactions
                    if agent["id"] == "A"
                    else agent["interactions"]
                ),
            }
            for agent in scene["agents"]
        ],
    }
    path = scene_directory / f"{scene['id']}.json"
    before = path.read_bytes()

    response = client.put(f"/api/scenes/{scene['id']}", json=payload)

    assert response.status_code == 422
    assert path.read_bytes() == before


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
    assert [event["content"] for event in events] == [
        "  第一件事  ",
        "第二件事",
    ]
    assert [event["sequence"] for event in events] == [1, 2]

    edited = client.put(
        f"/api/scenes/{scene['id']}/agents/A/events/{second_id}",
        json={"content": "  改过的第二件事\n"},
    )
    assert edited.status_code == 200
    assert [
        event["content"]
        for event in edited.json()["agents"][0]["pending_events"]
    ] == ["  第一件事  ", "  改过的第二件事\n"]

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


def test_scene_can_be_deleted_and_disappears_from_storage(
    scene_directory: Path,
) -> None:
    """DELETE removes the JSON file and 404s on a second attempt."""
    client, _model = make_client(scene_directory)
    scene = post_scene(client, name="被删除")
    other = post_scene(client, name="保留")
    path = scene_directory / f"{scene['id']}.json"

    missing = client.delete(f"/api/scenes/{uuid4()}")
    assert missing.status_code == 404
    assert path.exists()

    response = client.delete(f"/api/scenes/{scene['id']}")

    assert response.status_code == 204
    assert response.content == b""
    assert not path.exists()
    summaries = client.get("/api/scenes").json()
    assert [summary["id"] for summary in summaries] == [other["id"]]
    assert client.get(f"/api/scenes/{scene['id']}").status_code == 404
    # A second delete of the now-removed scene is also 404.
    assert client.delete(f"/api/scenes/{scene['id']}").status_code == 404
    assert client.get(f"/api/scenes/{other['id']}").status_code == 200


def test_generation_writes_nothing_and_two_confirmations_route_atomically(
    scene_directory: Path,
) -> None:
    """Only confirmations persist turns; outer confirmation routes one event."""
    client, model_client = make_client(
        scene_directory,
        ["先观察。\n别急着表态。", "  对 C 说 ：  去码头等我。  "],
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
    assert outer_turn["output"] == "对C说：去码头等我。"
    assert outer_turn["recipient_id"] == "C"
    assert completed["agents"][1]["pending_events"] == []
    received = completed["agents"][2]["pending_events"][0]
    assert received["content"] == "去码头等我。"
    assert received["kind"] == "agent_message"
    assert received["source_call_id"] == outer_turn["call_id"]
    assert received["id"] == outer_turn["generated_event_id"]
    assert len(model_client.generate_calls) == 2

    restarted = SceneStorage(scene_directory).get(UUID(scene["id"]))
    assert restarted.model_dump(mode="json", by_alias=True) == completed


def test_half_round_restores_outer_stage_after_restart(
    scene_directory: Path,
) -> None:
    """A saved inner turn remains ready for outer generation after reload."""
    client, _model = make_client(scene_directory, ["内层已确认", "对B说：继续"])
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
        ["对B说：继续"],
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
        ["原内层", "新内层", "对B说：合法外层"],
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
        content="对A说：不能发给自己",
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
        ["A 内层", "对B说：A 发出的消息"],
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
        "A 发出的消息",
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
        ["A 内层", "对B说：A 外层", "B 内层"],
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
        ["A 内层", "对B说：A 外层"],
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
        ["A 内层", "对C说：A 外层"],
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


def test_scene_keeps_validated_historic_reasoning(
    scene_directory: Path,
) -> None:
    """Persisted reasoning validates and round-trips while empty stays legal."""  # noqa: E501
    client, _model = make_client(
        scene_directory,
        ["A 内层", "对B说：A 外层"],
    )
    scene = post_scene(client)
    scene = post_event(client, scene["id"], "A", "事件")
    inner = generate(client, scene["id"], "A", "inner")
    scene = confirm(client, scene["id"], "A", "inner", inner).json()
    outer = generate(client, scene["id"], "A", "outer")
    scene = confirm(client, scene["id"], "A", "outer", outer).json()
    path = scene_directory / f"{scene['id']}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["agents"][0]["inner_context"]["turns"][0]["reasoning"] = [
        {"type": "thinking", "text": "historic inner reasoning"}
    ]
    raw["agents"][0]["outer_context"]["turns"][0]["reasoning"] = [
        {"type": "summary_text", "text": "historic outer reasoning"}
    ]
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    reloaded = SceneStorage(scene_directory).get(UUID(scene["id"]))

    assert reloaded.schema_id == CURRENT_SCENE_SCHEMA
    assert [
        block.model_dump()
        for block in reloaded.agents[0].inner_context.turns[0].reasoning
    ] == [{"type": "thinking", "text": "historic inner reasoning"}]
    assert [
        block.model_dump()
        for block in reloaded.agents[0].outer_context.turns[0].reasoning
    ] == [{"type": "summary_text", "text": "historic outer reasoning"}]


def test_inner_draft_goes_stale_when_a_new_event_arrives_before_confirm(
    scene_directory: Path,
) -> None:
    """A generated inner draft is rejected once the queue content changes."""
    client, _model = make_client(
        scene_directory,
        ["A 内层", "对C说：A 外层"],
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


def test_stop_confirmation_has_null_route_and_rolls_back_without_event(
    scene_directory: Path,
) -> None:
    """The HTTP workflow persists STOP independently from event routing."""
    client, _model = make_client(scene_directory, ["A 内层", "STOP"])
    scene = post_event(client, post_scene(client)["id"], "A", "事件")
    inner = generate(client, scene["id"], "A", "inner")
    scene = confirm(client, scene["id"], "A", "inner", inner).json()
    outer = generate(client, scene["id"], "A", "outer")

    completed = confirm(client, scene["id"], "A", "outer", outer)

    assert completed.status_code == 200
    saved = completed.json()
    turn = saved["agents"][0]["outer_context"]["turns"][-1]
    assert turn["output"] == "STOP"
    assert turn["recipient_id"] is None
    assert turn["generated_event_id"] is None
    assert all(not agent["pending_events"] for agent in saved["agents"])

    rolled_back = client.post(f"/api/scenes/{scene['id']}/rollback")
    assert rolled_back.status_code == 200
    assert rolled_back.json()["agents"][0]["outer_context"]["turns"] == []
    assert all(
        not agent["pending_events"] for agent in rolled_back.json()["agents"]
    )
