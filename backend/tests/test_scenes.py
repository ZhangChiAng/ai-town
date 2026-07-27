"""Integration tests for the scene CRUD API."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

import app.storage as storage_module
from app.main import create_app
from app.storage import SceneStorage
from tests.client import TestClient


@pytest.fixture
def scene_directory(tmp_path: Path) -> Path:
    """A temporary scene storage directory inside *tmp_path*."""
    return tmp_path / "scenes"


@pytest.fixture
def client(scene_directory: Path) -> TestClient:
    """A TestClient wired to a fresh SceneStorage in *scene_directory*."""
    return TestClient(create_app(SceneStorage(scene_directory)))


def create_scene(client: TestClient, name: str = "港口") -> dict[str, Any]:
    """POST a new scene and return the parsed JSON response."""
    response = client.post("/api/scenes", json={"name": name})
    assert response.status_code == 201
    return response.json()


def update_payload(name: str = "雨夜港口") -> dict[str, Any]:
    """Build a valid PUT payload that updates name and agent fields."""
    return {
        "name": name,
        "agents": [
            {
                "id": agent_id,
                "name": f"居民 {agent_id}",
                "persona": f"{agent_id} 的人设",
                "desire": f"{agent_id} 的欲望",
                "fear": f"{agent_id} 的恐惧",
                "memory": f"{agent_id} 的当前压缩记忆",
            }
            for agent_id in ("A", "B", "C")
        ],
    }


def send_message(
    client: TestClient,
    scene_id: str,
    *,
    sender_id: str = "A",
    recipient_id: str = "B",
    content: str = "你今晚会来码头吗？",
) -> dict[str, Any]:
    """Confirm a message and return the updated scene response."""
    response = client.post(
        f"/api/scenes/{scene_id}/messages",
        json={
            "sender_id": sender_id,
            "recipient_id": recipient_id,
            "content": content,
        },
    )
    assert response.status_code == 201
    return response.json()


def test_create_scene_writes_one_utf8_json_file(
    client: TestClient, scene_directory: Path
) -> None:
    """Creating a scene persists a single UTF-8 JSON file on disk."""
    scene = create_scene(client, "  海边小镇  ")

    assert scene["schema_version"] == 1
    assert scene["name"] == "海边小镇"
    UUID(scene["id"])
    assert [
        {
            "id": agent["id"],
            "name": agent["name"],
            "persona": agent["persona"],
            "desire": agent["desire"],
            "fear": agent["fear"],
            "memory": agent["memory"],
            "timeline": agent["timeline"],
        }
        for agent in scene["agents"]
    ] == [
        {
            "id": agent_id,
            "name": agent_id,
            "persona": "",
            "desire": "",
            "fear": "",
            "memory": "",
            "timeline": [],
        }
        for agent_id in ("A", "B", "C")
    ]

    files = list(scene_directory.glob("*.json"))
    assert files == [scene_directory / f"{scene['id']}.json"]
    contents = files[0].read_text(encoding="utf-8")
    assert "海边小镇" in contents
    assert json.loads(contents) == scene


def test_each_scene_uses_a_separate_file(
    client: TestClient, scene_directory: Path
) -> None:
    """Each scene is stored in its own JSON file."""
    first = create_scene(client, "场景一")
    second = create_scene(client, "场景二")

    assert {path.name for path in scene_directory.glob("*.json")} == {
        f"{first['id']}.json",
        f"{second['id']}.json",
    }


def test_scene_can_be_reopened_with_a_new_app_instance(
    scene_directory: Path,
) -> None:
    """A scene written by one app instance can be read by another."""
    first_client = TestClient(create_app(SceneStorage(scene_directory)))
    scene = create_scene(first_client, "可恢复场景")

    restarted_client = TestClient(create_app(SceneStorage(scene_directory)))

    list_response = restarted_client.get("/api/scenes")
    open_response = restarted_client.get(f"/api/scenes/{scene['id']}")

    assert list_response.status_code == 200
    assert list_response.json() == [{"id": scene["id"], "name": "可恢复场景"}]
    assert open_response.status_code == 200
    assert open_response.json() == scene


def test_scene_list_has_stable_name_and_id_order(
    client: TestClient,
) -> None:
    """Scene list is sorted by (name, id) for deterministic ordering."""
    scenes = [
        create_scene(client, name) for name in ("同名", "北岸", "同名", "南岸")
    ]

    response = client.get("/api/scenes")

    expected = sorted(
        [{"id": scene["id"], "name": scene["name"]} for scene in scenes],
        key=lambda scene: (scene["name"], scene["id"]),
    )
    assert response.status_code == 200
    assert response.json() == expected


def test_missing_scene_directory_is_an_empty_store(
    client: TestClient,
) -> None:
    """Listing scenes returns [] when the storage directory doesn't exist."""
    response = client.get("/api/scenes")

    assert response.status_code == 200
    assert response.json() == []


def test_scene_list_reports_when_storage_path_is_not_a_directory(
    client: TestClient, scene_directory: Path
) -> None:
    """A file (not a directory) at the storage path triggers a 500 error."""
    scene_directory.write_text("not a directory", encoding="utf-8")

    response = client.get("/api/scenes")

    assert response.status_code == 500
    assert (
        "Could not list the scene storage directory"
        in response.json()["detail"]
    )


def test_update_replaces_only_editable_fields(
    client: TestClient, scene_directory: Path
) -> None:
    """PUT updates name and agent fields.

    Preserves server-owned fields: ID, schema_version, and timeline.
    """
    original = create_scene(client)
    payload = update_payload()

    response = client.put(
        f"/api/scenes/{original['id']}",
        json=payload,
    )

    assert response.status_code == 200
    updated = response.json()
    assert updated["id"] == original["id"]
    assert updated["schema_version"] == original["schema_version"]
    assert updated["name"] == payload["name"]
    assert [
        {
            key: agent[key]
            for key in ("id", "name", "persona", "desire", "fear", "memory")
        }
        for agent in updated["agents"]
    ] == payload["agents"]
    assert [agent["timeline"] for agent in updated["agents"]] == [[], [], []]

    saved = json.loads(
        (scene_directory / f"{original['id']}.json").read_text(encoding="utf-8")
    )
    assert saved == updated


def test_message_appends_matching_records_only_to_participants(
    client: TestClient,
) -> None:
    """A confirmed message gives matching records only to its participants."""
    scene = create_scene(client)

    updated = send_message(
        client,
        scene["id"],
        sender_id="A",
        recipient_id="C",
        content="  灯塔下见。  ",
    )

    sender_record = updated["agents"][0]["timeline"][0]
    recipient_record = updated["agents"][2]["timeline"][0]
    UUID(sender_record["message_id"])
    assert sender_record == {
        "message_id": recipient_record["message_id"],
        "direction": "sent",
        "counterpart_id": "C",
        "content": "灯塔下见。",
    }
    assert recipient_record == {
        "message_id": sender_record["message_id"],
        "direction": "received",
        "counterpart_id": "A",
        "content": "灯塔下见。",
    }
    assert updated["agents"][1]["timeline"] == []


def test_messages_preserve_each_agents_timeline_order(
    client: TestClient,
) -> None:
    """Successive messages remain in confirmation order per agent."""
    scene = create_scene(client)

    send_message(client, scene["id"], content="第一条")
    send_message(
        client,
        scene["id"],
        sender_id="C",
        recipient_id="A",
        content="第二条",
    )
    updated = send_message(
        client,
        scene["id"],
        sender_id="A",
        recipient_id="B",
        content="第三条",
    )

    assert [
        (record["direction"], record["counterpart_id"], record["content"])
        for record in updated["agents"][0]["timeline"]
    ] == [
        ("sent", "B", "第一条"),
        ("received", "C", "第二条"),
        ("sent", "B", "第三条"),
    ]
    assert [
        record["content"] for record in updated["agents"][1]["timeline"]
    ] == ["第一条", "第三条"]
    assert [
        record["content"] for record in updated["agents"][2]["timeline"]
    ] == ["第二条"]


@pytest.mark.parametrize(
    "payload",
    [
        {"sender_id": "A", "recipient_id": "B", "content": " \t "},
        {"sender_id": "A", "recipient_id": "A", "content": "自言自语"},
        {"sender_id": "D", "recipient_id": "A", "content": "非法发送者"},
        {"sender_id": "A", "recipient_id": "D", "content": "非法接收者"},
        {
            "sender_id": "A",
            "recipient_id": "B",
            "content": "正文",
            "unexpected": True,
        },
    ],
)
def test_message_rejects_invalid_payloads(
    client: TestClient,
    payload: dict[str, Any],
) -> None:
    """Invalid content, participants, and extra fields return 422."""
    scene = create_scene(client)

    response = client.post(
        f"/api/scenes/{scene['id']}/messages",
        json=payload,
    )

    assert response.status_code == 422
    reopened = client.get(f"/api/scenes/{scene['id']}")
    assert [agent["timeline"] for agent in reopened.json()["agents"]] == [
        [],
        [],
        [],
    ]


def test_message_for_missing_scene_returns_404(client: TestClient) -> None:
    """Confirming a message in a missing scene returns 404."""
    response = client.post(
        f"/api/scenes/{uuid4()}/messages",
        json={
            "sender_id": "A",
            "recipient_id": "B",
            "content": "无人收到",
        },
    )

    assert response.status_code == 404


def test_message_timeline_survives_restart_and_later_scene_update(
    scene_directory: Path,
) -> None:
    """Messages reload after restart and survive editable-field updates."""
    first_client = TestClient(create_app(SceneStorage(scene_directory)))
    scene = create_scene(first_client)
    messaged = send_message(first_client, scene["id"], content="保留这条消息")

    restarted_client = TestClient(create_app(SceneStorage(scene_directory)))
    reopened = restarted_client.get(f"/api/scenes/{scene['id']}")
    assert reopened.status_code == 200
    assert reopened.json() == messaged

    update_response = restarted_client.put(
        f"/api/scenes/{scene['id']}",
        json=update_payload("消息后的场景"),
    )
    assert update_response.status_code == 200
    assert [
        agent["timeline"] for agent in update_response.json()["agents"]
    ] == [agent["timeline"] for agent in messaged["agents"]]


def test_failed_message_write_leaves_no_single_sided_record(
    client: TestClient,
    scene_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed atomic save keeps both timelines and disk unchanged."""
    scene = create_scene(client)
    scene_path = scene_directory / f"{scene['id']}.json"
    original_contents = scene_path.read_bytes()

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(storage_module.os, "replace", fail_replace)

    response = client.post(
        f"/api/scenes/{scene['id']}/messages",
        json={
            "sender_id": "A",
            "recipient_id": "B",
            "content": "不能只写给一方",
        },
    )

    assert response.status_code == 500
    assert scene_path.read_bytes() == original_contents
    assert list(scene_directory.glob("*.tmp")) == []
    assert [
        agent["timeline"] for agent in json.loads(original_contents)["agents"]
    ] == [[], [], []]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update({"id": str(uuid4())}),
        lambda payload: payload.update({"schema_version": 1}),
        lambda payload: payload["agents"][0].update({"timeline": []}),
    ],
)
def test_update_rejects_server_owned_fields(
    client: TestClient,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    """PUT rejects payloads that include id, schema_version, or timeline."""
    scene = create_scene(client)
    payload = update_payload()
    mutate(payload)

    response = client.put(f"/api/scenes/{scene['id']}", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "agent_ids",
    [
        ("A", "B"),
        ("A", "A", "C"),
        ("B", "A", "C"),
    ],
)
def test_update_requires_exactly_a_b_c_in_order(
    client: TestClient, agent_ids: tuple[str, ...]
) -> None:
    """PUT rejects agent lists that are not exactly A, B, C in order."""
    scene = create_scene(client)
    payload = update_payload()
    payload["agents"] = [
        {
            **payload["agents"][index],
            "id": agent_id,
        }
        for index, agent_id in enumerate(agent_ids)
    ]

    response = client.put(f"/api/scenes/{scene['id']}", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("post", "/api/scenes", {"name": "   "}),
        ("put", "", update_payload(name=" ")),
        (
            "put",
            "",
            {
                **update_payload(),
                "agents": [
                    {**agent, "name": "\t"} if agent["id"] == "B" else agent
                    for agent in update_payload()["agents"]
                ],
            },
        ),
    ],
)
def test_scene_and_agent_names_must_not_be_blank(
    client: TestClient,
    method: str,
    path: str,
    payload: dict[str, Any],
) -> None:
    """Scene and agent names that are blank or whitespace-only return 422."""
    if method == "post":
        response = client.post(path, json=payload)
    else:
        scene = create_scene(client)
        response = client.put(f"/api/scenes/{scene['id']}", json=payload)

    assert response.status_code == 422


def test_missing_scene_returns_404(client: TestClient) -> None:
    """GET for a non-existent scene ID returns a 404 error."""
    response = client.get(f"/api/scenes/{uuid4()}")

    assert response.status_code == 404
    assert "does not exist" in response.json()["detail"]


def test_corrupted_json_is_reported_instead_of_ignored(
    client: TestClient, scene_directory: Path
) -> None:
    """Corrupt JSON on disk produces 500 errors, not silent failures."""
    corrupt_id = uuid4()
    scene_directory.mkdir(parents=True)
    (scene_directory / f"{corrupt_id}.json").write_text(
        "{ definitely not json",
        encoding="utf-8",
    )

    open_response = client.get(f"/api/scenes/{corrupt_id}")
    list_response = client.get("/api/scenes")

    assert open_response.status_code == 500
    assert "invalid or corrupted" in open_response.json()["detail"]
    assert list_response.status_code == 500
    assert "invalid or corrupted" in list_response.json()["detail"]


def test_failed_atomic_replace_preserves_previous_file(
    client: TestClient,
    scene_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When os.replace fails, the original file is untouched.

    No .tmp files remain after a failed write.
    """
    scene = create_scene(client, "原始名称")
    scene_path = scene_directory / f"{scene['id']}.json"
    original_contents = scene_path.read_bytes()

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(storage_module.os, "replace", fail_replace)

    response = client.put(
        f"/api/scenes/{scene['id']}",
        json=update_payload("不应落盘"),
    )

    assert response.status_code == 500
    assert "Could not save scene" in response.json()["detail"]
    assert scene_path.read_bytes() == original_contents
    # No .tmp files lingering after a failed write.
    assert list(scene_directory.glob("*.tmp")) == []
