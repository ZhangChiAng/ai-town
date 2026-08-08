"""Atomic JSON-file persistence for namespaced scene contracts."""

import contextlib
import json
import logging
import os
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

from pydantic import ValidationError

from app.models import (
    SCENE_SCHEMA_MAJOR,
    Scene,
    SceneSummary,
    parse_scene_schema,
)
from app.structured_logging import log_event

LOGGER = logging.getLogger(__name__)


class SceneStorageError(RuntimeError):
    """Base class for scene persistence failures."""


class SceneNotFoundError(SceneStorageError):
    """Raised when a requested scene does not exist."""


class SceneAlreadyExistsError(SceneStorageError):
    """Raised when creating a duplicate scene ID."""


class SceneReadError(SceneStorageError):
    """Raised when a scene cannot be read as the current schema."""


class SceneWriteError(SceneStorageError):
    """Raised when a scene cannot be written."""


class SceneStorage:
    """Read and atomically write one JSON file per scene."""

    def __init__(self, directory: Path) -> None:
        """Initialize storage rooted at *directory*."""
        self.directory = directory
        self._lock = threading.RLock()

    def list_scenes(self) -> list[SceneSummary]:
        """Return readable scene summaries ordered by name and UUID."""
        with self._lock:
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

            scenes: list[Scene] = []
            for path in paths:
                try:
                    scenes.append(self._read_path(path))
                except SceneStorageError as error:
                    # Leave the failed file untouched for diagnosis or repair
                    # while keeping every other valid scene discoverable.
                    log_event(
                        LOGGER,
                        logging.ERROR,
                        "scene.load.failed",
                        "Scene file was skipped while listing scenes.",
                        exception=error,
                        scene_file=path.name,
                    )
            scenes.sort(key=lambda scene: (scene.name, str(scene.id)))
            return [
                SceneSummary(id=scene.id, name=scene.name) for scene in scenes
            ]

    def get(self, scene_id: UUID) -> Scene:
        """Read and validate one current-schema scene."""
        with self._lock:
            return self._read_path(
                self._path_for(scene_id),
                expected_id=scene_id,
            )

    def create(self, scene: Scene) -> None:
        """Persist a new scene and fail if its UUID already exists."""
        with self._lock:
            self._write(scene, must_not_exist=True)

    def save(self, scene: Scene) -> None:
        """Atomically overwrite one scene."""
        with self._lock:
            self._write(scene, must_not_exist=False)

    def delete(self, scene_id: UUID) -> None:
        """Remove one scene file; a missing scene is treated as not found."""
        with self._lock:
            path = self._path_for(scene_id)
            try:
                path.unlink(missing_ok=False)
            except FileNotFoundError as error:
                raise SceneNotFoundError(
                    f"Scene '{scene_id}' does not exist."
                ) from error
            except OSError as error:
                raise SceneWriteError(
                    f"Could not delete scene '{scene_id}': {error}"
                ) from error

    def mutate(
        self,
        scene_id: UUID,
        operation: Callable[[Scene], Scene],
    ) -> Scene:
        """Read, transform, validate, and save one scene under a lock."""
        with self._lock:
            current = self._read_path(
                self._path_for(scene_id),
                expected_id=scene_id,
            )
            updated = operation(current)
            if updated.id != scene_id:
                raise SceneWriteError(
                    "A scene mutation cannot change the scene ID."
                )
            # Revalidation makes partial or structurally inconsistent writes
            # impossible even if an operation used Pydantic model_copy.
            validated = Scene.model_validate(updated.model_dump(by_alias=True))
            self._write(validated, must_not_exist=False)
            return validated

    def _path_for(self, scene_id: UUID) -> Path:
        """Return the authoritative file path for one UUID."""
        return self.directory / f"{scene_id}.json"

    def _read_path(
        self,
        path: Path,
        expected_id: UUID | None = None,
    ) -> Scene:
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
            raw = json.loads(contents)
        except (json.JSONDecodeError, TypeError) as error:
            raise SceneReadError(
                f"Scene file '{path.name}' is not valid JSON or is corrupted."
            ) from error

        schema = raw.get("schema") if isinstance(raw, dict) else None
        try:
            major, _minor = parse_scene_schema(schema)
        except ValueError as error:
            raise SceneReadError(
                f"Scene file '{path.name}' has an invalid scene schema."
            ) from error
        if major != SCENE_SCHEMA_MAJOR:
            raise SceneReadError(
                f"Scene file '{path.name}' uses incompatible schema major "
                f"{major}; this application supports major "
                f"{SCENE_SCHEMA_MAJOR}."
            )

        try:
            scene = Scene.model_validate(raw)
        except (ValidationError, TypeError, ValueError) as error:
            raise SceneReadError(
                f"Scene file '{path.name}' is corrupted."
            ) from error

        file_id = expected_id
        if file_id is None:
            try:
                file_id = UUID(path.stem)
            except ValueError as error:
                raise SceneReadError(
                    f"Scene file '{path.name}' has no valid UUID filename."
                ) from error
        if scene.id != file_id:
            raise SceneReadError(
                f"Scene file '{path.name}' contains a mismatched ID."
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
                temporary_file.write(
                    scene.model_dump_json(indent=2, by_alias=True)
                )
                temporary_file.write("\n")
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
                with contextlib.suppress(OSError):
                    temporary_path.unlink(missing_ok=True)
