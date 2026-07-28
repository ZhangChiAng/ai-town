"""Integration tests for scene persistence and v4 messages."""

import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from app.main import create_app
from app.models import CreateMessageRequest, add_message, create_scene
from app.storage import SceneReadError, SceneStorage
from tests.client import TestClient


@pytest.fixture
def scene_directory(tmp_path: Path) -> Path:
    """Provide isolated JSON scene storage."""
    return tmp_path / "scenes"


@pytest.fixture
def client(scene_directory: Path) -> TestClient:
    """Provide an API client backed by isolated storage."""
    return TestClient(create_app(SceneStorage(scene_directory)))


def post_scene(client: TestClient) -> dict:
    """Create and return one scene."""
    response = client.post("/api/scenes", json={"name": "港口"})
    assert response.status_code == 201
    return response.json()


def test_new_scene_is_schema_v4_and_survives_restart(
    client: TestClient,
    scene_directory: Path,
) -> None:
    """New files persist the current schema and all three Agents."""
    scene = post_scene(client)

    assert scene["schema_version"] == 4
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


@pytest.mark.parametrize("schema_version", [1, 2, 3])
def test_legacy_message_upgrade_is_in_memory_until_save(
    tmp_path: Path,
    schema_version: int,
) -> None:
    """v1-v3 messages gain perspective prefixes without eager file writes."""
    directory = tmp_path / f"v{schema_version}"
    directory.mkdir()
    scene_id = uuid4()
    message_id = uuid4()
    raw = create_scene("旧场景").model_dump(mode="json")
    raw["schema_version"] = schema_version
    raw["id"] = str(scene_id)
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
    assert upgraded.schema_version == 4
    assert upgraded.agents[0].timeline[0].content == "To B: 旧消息"
    assert upgraded.agents[1].timeline[0].content == "From A: 旧消息"
    storage.save(upgraded)
    assert json.loads(path.read_text())["schema_version"] == 4


def test_existing_correct_legacy_prefix_is_not_duplicated(
    tmp_path: Path,
) -> None:
    """An already perspective-tagged v3 record stays unchanged."""
    directory = tmp_path / "scenes"
    directory.mkdir()
    scene = add_message(
        create_scene("已有标签"),
        CreateMessageRequest(sender_id="A", content="To B: 正文"),
    )
    raw = scene.model_dump(mode="json")
    raw["schema_version"] = 3
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
    scene = create_scene("旧内心声音")
    raw = scene.model_dump(mode="json")
    raw["schema_version"] = 3
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
