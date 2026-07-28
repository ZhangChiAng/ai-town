"""JSON-file persistence layer for scene data."""

import contextlib
import json
import os
import tempfile
from pathlib import Path
from uuid import UUID

from pydantic import ValidationError

from app.models import (
    SCHEMA_VERSION,
    Scene,
    SceneSummary,
    compose_system_prompt,
)


class SceneStorageError(RuntimeError):
    """Base class for scene persistence failures."""


class SceneNotFoundError(SceneStorageError):
    """Raised when a requested scene does not exist on disk."""


class SceneAlreadyExistsError(SceneStorageError):
    """Raised when attempting to create a scene with a duplicate ID."""


class SceneReadError(SceneStorageError):
    """Raised when a scene file cannot be read or parsed."""


class SceneWriteError(SceneStorageError):
    """Raised when a scene file cannot be written."""


class SceneStorage:
    """Reads, writes, and lists scene JSON files in a single directory.

    Each scene is stored as ``<uuid>.json``. Writes are atomic (temp file
    followed by ``os.replace``) so that crashes cannot produce partial files.
    """

    def __init__(self, directory: Path) -> None:
        """Initialize storage rooted at *directory*.

        Args:
            directory: Path to the directory that holds ``.json`` scene
                files. The directory is created automatically on first write.
        """
        self.directory = directory

    def list_scenes(self) -> list[SceneSummary]:
        """Return every scene in the storage directory.

        Scenes are sorted by ``(name, id)`` for deterministic ordering.

        Returns:
            Scene summaries ordered by ``(name, id)``. An empty list is
            returned when the directory does not exist yet.

        Raises:
            SceneReadError: If the directory exists but cannot be listed or
                any of its scene files cannot be read.
        """
        try:
            paths = [
                path
                for path in self.directory.iterdir()
                if path.name.endswith(".json")
            ]
        except FileNotFoundError:
            return []
        except OSError as error:
            raise SceneReadError(
                f"Could not list the scene storage directory: {error}"
            ) from error

        scenes = [self._read_path(path) for path in paths]
        scenes.sort(key=lambda scene: (scene.name, str(scene.id)))
        return [SceneSummary(id=scene.id, name=scene.name) for scene in scenes]

    def get(self, scene_id: UUID) -> Scene:
        """Read a single scene by its ID.

        Args:
            scene_id: The UUID of the scene to retrieve.

        Returns:
            The deserialized Scene.

        Raises:
            SceneNotFoundError: If no file exists for *scene_id*.
            SceneReadError: If the file exists but cannot be read or
                validated.
        """
        return self._read_path(self._path_for(scene_id), expected_id=scene_id)

    def create(self, scene: Scene) -> None:
        """Persist a new scene, failing if it already exists.

        Args:
            scene: The scene to write.

        Raises:
            SceneAlreadyExistsError: If a file for *scene.id* already exists.
            SceneWriteError: If the write operation fails.
        """
        self._write(scene, must_not_exist=True)

    def save(self, scene: Scene) -> None:
        """Persist an existing scene, overwriting the current file.

        Args:
            scene: The scene to write.

        Raises:
            SceneWriteError: If the write operation fails.
        """
        self._write(scene, must_not_exist=False)

    def _path_for(self, scene_id: UUID) -> Path:
        return self.directory / f"{scene_id}.json"

    def _read_path(self, path: Path, expected_id: UUID | None = None) -> Scene:
        try:
            contents = path.read_text(encoding="utf-8")
        except FileNotFoundError as error:
            raise SceneNotFoundError(
                f"Scene '{path.stem}' does not exist."
            ) from error
        except (OSError, UnicodeError) as error:
            raise SceneReadError(
                f"Could not read scene file '{path.name}': {error}"
            ) from error

        try:
            raw_scene = json.loads(contents)
            scene = Scene.model_validate(_upgrade_scene(raw_scene))
        except (json.JSONDecodeError, TypeError, ValidationError) as error:
            raise SceneReadError(
                f"Scene file '{path.name}' is invalid or corrupted."
            ) from error

        file_id = expected_id
        if file_id is None:
            try:
                file_id = UUID(path.stem)
            except ValueError as error:
                raise SceneReadError(
                    f"Scene file '{path.name}' does not have"
                    " a valid UUID filename."
                ) from error

        if scene.id != file_id:
            raise SceneReadError(
                f"Scene file '{path.name}' contains an ID that"
                " does not match its filename."
            )

        return scene

    def _write(self, scene: Scene, *, must_not_exist: bool) -> None:
        target = self._path_for(scene.id)
        temporary_path: Path | None = None

        try:
            self.directory.mkdir(parents=True, exist_ok=True)

            if must_not_exist and target.exists():
                raise SceneAlreadyExistsError(
                    f"Scene '{scene.id}' already exists."
                )

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{scene.id}.",
                suffix=".tmp",
                dir=self.directory,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(scene.model_dump_json(indent=2))
                temporary_file.write("\n")
                # Force-flush to disk so os.replace sees complete data.
                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            os.replace(temporary_path, target)
        except SceneAlreadyExistsError:
            raise
        except (OSError, UnicodeError) as error:
            raise SceneWriteError(
                f"Could not save scene '{scene.id}': {error}"
            ) from error
        finally:
            if temporary_path is not None:
                # Clean up the temp file if the replace failed.
                with contextlib.suppress(OSError):
                    temporary_path.unlink(missing_ok=True)


def _upgrade_scene(raw_scene: object) -> object:
    """Return an in-memory v3 representation of a v1 or v2 scene."""
    if not isinstance(raw_scene, dict) or raw_scene.get(
        "schema_version"
    ) not in (1, 2):
        return raw_scene

    upgraded = dict(raw_scene)
    agents = upgraded.get("agents")
    if not isinstance(agents, list):
        return upgraded

    upgraded_agents = []
    for raw_agent in agents:
        if not isinstance(raw_agent, dict):
            upgraded_agents.append(raw_agent)
            continue
        agent = dict(raw_agent)
        if upgraded["schema_version"] == 1:
            agent["system_prompt"] = compose_system_prompt(
                agent.get("persona", ""),
                agent.get("desire", ""),
                agent.get("fear", ""),
                agent.get("memory", ""),
            )
        timeline = agent.get("timeline")
        if isinstance(timeline, list):
            agent["timeline"] = [
                {"type": "message", **record}
                if isinstance(record, dict) and "type" not in record
                else record
                for record in timeline
            ]
        upgraded_agents.append(agent)

    upgraded["agents"] = upgraded_agents
    upgraded["schema_version"] = SCHEMA_VERSION
    return upgraded
