"""Tests for private, manually appended inner voices."""

import json
from pathlib import Path
from uuid import UUID, uuid4

from app.drafting import CACHE_CONTROL, build_message_request
from app.main import create_app
from app.models import CreateMessageRequest, add_message, create_scene
from app.storage import SceneStorage
from tests.client import TestClient

MODEL = "test-model"


def _create_scene(client: TestClient) -> dict:
    response = client.post("/api/scenes", json={"name": "内心声音场景"})
    assert response.status_code == 201
    return response.json()


def _write_voice(
    client: TestClient,
    scene_id: str,
    agent_id: str = "A",
    content: str = "  去问问发生了什么。  ",
) -> dict:
    response = client.post(
        f"/api/scenes/{scene_id}/agents/{agent_id}/inner-voices",
        json={"content": content},
    )
    assert response.status_code == 201
    return response.json()


def test_inner_voice_is_trimmed_private_persisted_and_restored(
    tmp_path: Path,
) -> None:
    """Only the target timeline receives the record across restarts."""
    directory = tmp_path / "scenes"
    client = TestClient(create_app(SceneStorage(directory)))
    scene = _create_scene(client)

    updated = _write_voice(client, scene["id"])

    voice = updated["agents"][0]["timeline"][0]
    assert voice["type"] == "inner_voice"
    assert voice["content"] == "去问问发生了什么。"
    assert set(voice) == {"type", "inner_voice_id", "content"}
    assert updated["agents"][1]["timeline"] == []
    assert updated["agents"][2]["timeline"] == []

    restarted = TestClient(create_app(SceneStorage(directory)))
    reopened = restarted.get(f"/api/scenes/{scene['id']}")
    assert reopened.status_code == 200
    assert reopened.json() == updated


def test_blank_inner_voice_is_rejected_without_disk_change(
    tmp_path: Path,
) -> None:
    """Validation failure cannot mutate the persisted scene."""
    directory = tmp_path / "scenes"
    client = TestClient(create_app(SceneStorage(directory)))
    scene = _create_scene(client)
    path = directory / f"{scene['id']}.json"
    original = path.read_bytes()

    response = client.post(
        f"/api/scenes/{scene['id']}/agents/A/inner-voices",
        json={"content": " \t\n "},
    )

    assert response.status_code == 422
    assert path.read_bytes() == original


def test_inner_voice_delete_requires_target_absolute_timeline_top(
    tmp_path: Path,
) -> None:
    """Later messages block deletion; wrong Agent is a 404."""
    directory = tmp_path / "scenes"
    client = TestClient(create_app(SceneStorage(directory)))
    scene = _create_scene(client)
    voiced = _write_voice(client, scene["id"])
    voice_id = voiced["agents"][0]["timeline"][0]["inner_voice_id"]

    wrong_agent = client.delete(
        f"/api/scenes/{scene['id']}/agents/B/inner-voices/{voice_id}"
    )
    assert wrong_agent.status_code == 404

    message = client.post(
        f"/api/scenes/{scene['id']}/messages",
        json={"sender_id": "A", "recipient_id": "B", "content": "后来消息"},
    )
    assert message.status_code == 201
    path = directory / f"{scene['id']}.json"
    before_conflict = path.read_bytes()

    conflict = client.delete(
        f"/api/scenes/{scene['id']}/agents/A/inner-voices/{voice_id}"
    )
    assert conflict.status_code == 409
    assert path.read_bytes() == before_conflict

    message_id = message.json()["agents"][0]["timeline"][-1]["message_id"]
    assert (
        client.delete(
            f"/api/scenes/{scene['id']}/messages/{message_id}"
        ).status_code
        == 200
    )
    deleted = client.delete(
        f"/api/scenes/{scene['id']}/agents/A/inner-voices/{voice_id}"
    )
    assert deleted.status_code == 200
    assert deleted.json()["agents"][0]["timeline"] == []


def test_mixed_timeline_maps_in_order_and_keeps_private_boundary(
    tmp_path: Path,
) -> None:
    """Inner voices use a marked user turn and own the rolling breakpoint."""
    directory = tmp_path / "scenes"
    client = TestClient(create_app(SceneStorage(directory)))
    scene_data = _create_scene(client)
    scene = SceneStorage(directory).get(UUID(scene_data["id"]))
    scene = add_message(
        scene,
        CreateMessageRequest(
            sender_id="B", recipient_id="A", content="第一条收到"
        ),
    )
    SceneStorage(directory).save(scene)
    _write_voice(client, scene_data["id"], "A", "只给 A 的提示")
    _write_voice(client, scene_data["id"], "B", "不能泄露给 A")
    scene = SceneStorage(directory).get(UUID(scene_data["id"]))

    request = build_message_request(scene, "A", MODEL)

    assert request["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "From B: 第一条收到"}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "内心的声音：只给 A 的提示",
                    "cache_control": CACHE_CONTROL,
                }
            ],
        },
    ]
    assert "不能泄露给 A" not in json.dumps(request, ensure_ascii=False)


def test_v2_read_upgrade_is_non_mutating_until_explicit_write(
    tmp_path: Path,
) -> None:
    """A v2 message gains its discriminator in memory and writes as v3."""
    directory = tmp_path / "scenes"
    directory.mkdir()
    scene_id = uuid4()
    message_id = uuid4()
    v2 = create_scene("旧 v2").model_dump(mode="json")
    v2["schema_version"] = 2
    v2["id"] = str(scene_id)
    for agent in v2["agents"]:
        agent["timeline"] = []
    v2["agents"][0]["timeline"] = [
        {
            "message_id": str(message_id),
            "direction": "sent",
            "counterpart_id": "B",
            "content": "旧消息",
        }
    ]
    path = directory / f"{scene_id}.json"
    original = json.dumps(v2, ensure_ascii=False).encode()
    path.write_bytes(original)
    storage = SceneStorage(directory)

    upgraded = storage.get(scene_id)

    assert path.read_bytes() == original
    assert upgraded.schema_version == 3
    assert upgraded.agents[0].timeline[0].type == "message"

    storage.save(upgraded)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == 3
    assert persisted["agents"][0]["timeline"][0]["type"] == "message"
