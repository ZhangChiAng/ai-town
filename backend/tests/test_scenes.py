"""Integration tests for v5 scene persistence and model binding."""

import json
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from app.drafting import MessageDraftService
from app.main import create_app
from app.models import CreateMessageRequest, add_message, create_scene
from app.storage import SceneReadError, SceneStorage
from tests.client import TestClient

ANTHROPIC_MODEL = "anthropic/claude-test"
RESPONSES_MODEL = "gpt-test"


def model_services() -> dict[str, MessageDraftService]:
    """Provide both configured model names without making upstream calls."""
    return {
        ANTHROPIC_MODEL: MessageDraftService(object(), ANTHROPIC_MODEL),
        RESPONSES_MODEL: MessageDraftService(object(), RESPONSES_MODEL),
    }


@pytest.fixture
def scene_directory(tmp_path: Path) -> Path:
    """Provide isolated JSON scene storage."""
    return tmp_path / "scenes"


@pytest.fixture
def client(scene_directory: Path) -> TestClient:
    """Provide an API client backed by isolated storage."""
    return TestClient(
        create_app(SceneStorage(scene_directory), model_services())
    )


def post_scene(
    client: TestClient,
    model: str = ANTHROPIC_MODEL,
) -> dict[str, Any]:
    """Create and return one scene."""
    response = client.post("/api/scenes", json={"name": "港口", "model": model})
    assert response.status_code == 201
    return response.json()


def test_model_options_are_ordered_and_do_not_expose_credentials(
    client: TestClient,
) -> None:
    """The public registry contains only protocol labels and model names."""
    response = client.get("/api/model-options")

    assert response.status_code == 200
    assert response.json() == {
        "options": [
            {"protocol": "anthropic", "model": ANTHROPIC_MODEL},
            {"protocol": "responses", "model": RESPONSES_MODEL},
        ]
    }
    assert "key" not in response.text.casefold()
    assert "url" not in response.text.casefold()


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "缺模型"},
        {"name": "空模型", "model": " "},
        {"name": "未知模型", "model": "gpt-not-configured"},
    ],
)
def test_new_scene_requires_one_currently_configured_model(
    client: TestClient,
    scene_directory: Path,
    payload: dict[str, str],
) -> None:
    """Missing, blank, and unknown model selections cannot create files."""
    response = client.post("/api/scenes", json=payload)

    assert response.status_code == 422
    assert not scene_directory.exists()


@pytest.mark.parametrize("model", [ANTHROPIC_MODEL, RESPONSES_MODEL])
def test_new_scene_is_schema_v5_bound_and_survives_restart(
    client: TestClient,
    scene_directory: Path,
    model: str,
) -> None:
    """New files persist the current schema and all three Agents."""
    scene = post_scene(client, model)

    assert scene["schema_version"] == 5
    assert scene["model"] == model
    assert [agent["id"] for agent in scene["agents"]] == ["A", "B", "C"]
    assert (
        SceneStorage(scene_directory)
        .get(UUID(scene["id"]))
        .model_dump(mode="json")
        == scene
    )


def test_confirmation_parses_recipient_and_writes_perspective_text(
    client: TestClient,
) -> None:
    """The complete draft controls routing and both authoritative records."""
    scene = post_scene(client)
    response = client.post(
        f"/api/scenes/{scene['id']}/messages",
        json={"sender_id": "A", "content": "To   C ：  灯塔下见。  "},
    )

    assert response.status_code == 201
    updated = response.json()
    sent = updated["agents"][0]["timeline"][0]
    received = updated["agents"][2]["timeline"][0]
    assert sent == {
        "type": "message",
        "message_id": received["message_id"],
        "direction": "sent",
        "counterpart_id": "C",
        "content": "To C: 灯塔下见。",
    }
    assert received == {
        "type": "message",
        "message_id": sent["message_id"],
        "direction": "received",
        "counterpart_id": "A",
        "content": "From A: 灯塔下见。",
    }
    assert updated["agents"][1]["timeline"] == []


@pytest.mark.parametrize(
    "payload",
    [
        {"sender_id": "A", "content": "正文"},
        {"sender_id": "A", "content": "To A: 自己"},
        {"sender_id": "A", "content": "To D: 无效"},
        {"sender_id": "A", "content": "To B:"},
        {"sender_id": "A", "content": "To B: 第一行\n第二行"},
        {"sender_id": "A", "content": "To B: 正文\n"},
        {"sender_id": "A", "content": "说明\nTo B: 正文"},
        {"sender_id": "A", "recipient_id": "B", "content": "To B: 正文"},
    ],
)
def test_invalid_confirmation_is_422_and_does_not_write(
    client: TestClient,
    scene_directory: Path,
    payload: dict,
) -> None:
    """Malformed or obsolete payloads cannot partially mutate timelines."""
    scene = post_scene(client)
    path = scene_directory / f"{scene['id']}.json"
    before = path.read_bytes()

    response = client.post(
        f"/api/scenes/{scene['id']}/messages",
        json=payload,
    )

    assert response.status_code == 422
    assert path.read_bytes() == before


def test_manual_to_edit_changes_recipient(client: TestClient) -> None:
    """Routing follows edited visible content, not a separate UI field."""
    scene = post_scene(client)
    response = client.post(
        f"/api/scenes/{scene['id']}/messages",
        json={"sender_id": "A", "content": "To C: 改发给 C"},
    )

    assert response.status_code == 201
    timelines = [agent["timeline"] for agent in response.json()["agents"]]
    assert len(timelines[0]) == 1
    assert timelines[1] == []
    assert len(timelines[2]) == 1


def test_delete_validates_perspective_pair_and_both_timeline_tops(
    client: TestClient,
) -> None:
    """Only a consistent To/From pair at both absolute tops is deletable."""
    scene = post_scene(client)
    first = client.post(
        f"/api/scenes/{scene['id']}/messages",
        json={"sender_id": "A", "content": "To B: 第一条"},
    ).json()
    first_id = first["agents"][0]["timeline"][0]["message_id"]
    second = client.post(
        f"/api/scenes/{scene['id']}/messages",
        json={"sender_id": "A", "content": "To C: 第二条"},
    ).json()
    second_id = second["agents"][0]["timeline"][-1]["message_id"]

    conflict = client.delete(f"/api/scenes/{scene['id']}/messages/{first_id}")
    assert conflict.status_code == 409
    assert (
        client.delete(
            f"/api/scenes/{scene['id']}/messages/{second_id}"
        ).status_code
        == 200
    )
    deleted = client.delete(f"/api/scenes/{scene['id']}/messages/{first_id}")
    assert deleted.status_code == 200
    assert [agent["timeline"] for agent in deleted.json()["agents"]] == [
        [],
        [],
        [],
    ]


@pytest.mark.parametrize("schema_version", [1, 2, 3, 4])
def test_legacy_message_upgrade_is_in_memory_until_save(
    tmp_path: Path,
    schema_version: int,
) -> None:
    """v1-v4 scenes become unbound v5 snapshots without eager file writes."""
    directory = tmp_path / f"v{schema_version}"
    directory.mkdir()
    scene_id = uuid4()
    message_id = uuid4()
    raw = create_scene("旧场景", ANTHROPIC_MODEL).model_dump(mode="json")
    raw["schema_version"] = schema_version
    raw["id"] = str(scene_id)
    raw.pop("model")
    if schema_version == 1:
        for agent in raw["agents"]:
            agent.pop("system_prompt")
    raw["agents"][0]["timeline"] = [
        {
            **({"type": "message"} if schema_version == 3 else {}),
            "message_id": str(message_id),
            "direction": "sent",
            "counterpart_id": "B",
            "content": "旧消息",
        }
    ]
    raw["agents"][1]["timeline"] = [
        {
            **({"type": "message"} if schema_version == 3 else {}),
            "message_id": str(message_id),
            "direction": "received",
            "counterpart_id": "A",
            "content": "旧消息",
        }
    ]
    path = directory / f"{scene_id}.json"
    original = json.dumps(raw, ensure_ascii=False).encode()
    path.write_bytes(original)
    storage = SceneStorage(directory)

    upgraded = storage.get(scene_id)

    assert path.read_bytes() == original
    assert upgraded.schema_version == 5
    assert upgraded.model is None
    assert upgraded.agents[0].timeline[0].content == "To B: 旧消息"
    assert upgraded.agents[1].timeline[0].content == "From A: 旧消息"
    storage.save(upgraded)
    saved = json.loads(path.read_text())
    assert saved["schema_version"] == 5
    assert saved["model"] is None


def test_existing_correct_legacy_prefix_is_not_duplicated(
    tmp_path: Path,
) -> None:
    """An already perspective-tagged v3 record stays unchanged."""
    directory = tmp_path / "scenes"
    directory.mkdir()
    scene = add_message(
        create_scene("已有标签", ANTHROPIC_MODEL),
        CreateMessageRequest(sender_id="A", content="To B: 正文"),
    )
    raw = scene.model_dump(mode="json")
    raw["schema_version"] = 3
    raw.pop("model")
    path = directory / f"{scene.id}.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = SceneStorage(directory).get(scene.id)

    assert loaded.agents[0].timeline[0].content == "To B: 正文"
    assert loaded.agents[1].timeline[0].content == "From A: 正文"


def test_legacy_inner_voice_is_rejected_with_clear_error(
    tmp_path: Path,
) -> None:
    """Removed private prompt records cannot be silently imported."""
    directory = tmp_path / "scenes"
    directory.mkdir()
    scene = create_scene("旧内心声音", ANTHROPIC_MODEL)
    raw = scene.model_dump(mode="json")
    raw["schema_version"] = 3
    raw.pop("model")
    raw["agents"][0]["timeline"] = [
        {
            "type": "inner_voice",
            "inner_voice_id": str(uuid4()),
            "content": "旧提示",
        }
    ]
    (directory / f"{scene.id}.json").write_text(
        json.dumps(raw), encoding="utf-8"
    )

    with pytest.raises(SceneReadError, match="removed 'inner_voice'"):
        SceneStorage(directory).get(scene.id)


def test_legacy_scene_can_bind_once_and_persists_v5(
    client: TestClient,
    scene_directory: Path,
) -> None:
    """An unbound upgrade accepts one configured concrete model name."""
    scene_id = uuid4()
    raw = create_scene("待绑定", ANTHROPIC_MODEL).model_dump(mode="json")
    raw["schema_version"] = 4
    raw["id"] = str(scene_id)
    raw.pop("model")
    scene_directory.mkdir()
    path = scene_directory / f"{scene_id}.json"
    original = json.dumps(raw, ensure_ascii=False).encode()
    path.write_bytes(original)

    loaded = client.get(f"/api/scenes/{scene_id}")
    assert loaded.status_code == 200
    assert loaded.json()["model"] is None
    assert path.read_bytes() == original
    assert (
        client.get(
            f"/api/scenes/{scene_id}/agents/A/model-request-preview"
        ).status_code
        == 409
    )
    assert (
        client.post(
            f"/api/scenes/{scene_id}/agents/A/message-drafts"
        ).status_code
        == 409
    )
    assert path.read_bytes() == original

    bound = client.put(
        f"/api/scenes/{scene_id}/model",
        json={"model": RESPONSES_MODEL},
    )
    assert bound.status_code == 200
    assert bound.json()["model"] == RESPONSES_MODEL
    assert json.loads(path.read_text()) | {
        "agents": [],
    } == bound.json() | {"agents": []}

    repeated = client.put(
        f"/api/scenes/{scene_id}/model",
        json={"model": ANTHROPIC_MODEL},
    )
    assert repeated.status_code == 409
    assert json.loads(path.read_text())["model"] == RESPONSES_MODEL
    repeated_unknown = client.put(
        f"/api/scenes/{scene_id}/model",
        json={"model": "gpt-unknown"},
    )
    assert repeated_unknown.status_code == 409


def test_unknown_model_cannot_bind_legacy_scene(
    client: TestClient,
    scene_directory: Path,
) -> None:
    """The one-time migration only accepts a current model option."""
    scene_id = uuid4()
    raw = create_scene("待绑定", ANTHROPIC_MODEL).model_dump(mode="json")
    raw["schema_version"] = 4
    raw["id"] = str(scene_id)
    raw.pop("model")
    scene_directory.mkdir()
    path = scene_directory / f"{scene_id}.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    before = path.read_bytes()

    response = client.put(
        f"/api/scenes/{scene_id}/model",
        json={"model": "gpt-unknown"},
    )

    assert response.status_code == 422
    assert path.read_bytes() == before


def test_unavailable_bound_model_remains_editable_but_cannot_run(
    tmp_path: Path,
) -> None:
    """Configuration changes never rewrite or unlock a scene binding."""
    storage = SceneStorage(tmp_path / "scenes")
    scene = create_scene("旧配置模型", "gpt-retired")
    storage.create(scene)
    client = TestClient(create_app(storage, model_services()))

    loaded = client.get(f"/api/scenes/{scene.id}")
    assert loaded.status_code == 200
    payload = loaded.json()
    payload["name"] = "仍可编辑"
    update = {
        "name": payload["name"],
        "agents": [
            {
                key: value
                for key, value in agent.items()
                if key
                in {
                    "id",
                    "name",
                    "persona",
                    "desire",
                    "fear",
                    "memory",
                    "system_prompt",
                }
            }
            for agent in payload["agents"]
        ],
    }
    assert client.put(f"/api/scenes/{scene.id}", json=update).status_code == 200
    assert (
        client.post(
            f"/api/scenes/{scene.id}/messages",
            json={"sender_id": "A", "content": "To B: 手工消息"},
        ).status_code
        == 201
    )
    assert (
        client.get(
            f"/api/scenes/{scene.id}/agents/A/model-request-preview"
        ).status_code
        == 409
    )
    assert (
        client.post(
            f"/api/scenes/{scene.id}/agents/A/message-drafts"
        ).status_code
        == 409
    )
    assert (
        client.put(
            f"/api/scenes/{scene.id}/model",
            json={"model": ANTHROPIC_MODEL},
        ).status_code
        == 409
    )
