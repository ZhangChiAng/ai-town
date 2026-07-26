"""FastAPI application factory and route definitions for the AI Town API."""

from pathlib import Path
from typing import Literal, TypedDict
from uuid import UUID

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.models import (
    CreateSceneRequest,
    Scene,
    SceneSummary,
    UpdateSceneRequest,
    create_scene,
    update_scene,
)
from app.storage import (
    SceneNotFoundError,
    SceneStorage,
    SceneStorageError,
)


class HealthResponse(TypedDict):
    """Response shape for the health check endpoint."""

    status: Literal["ok"]


DEFAULT_SCENE_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "data" / "scenes"
)


def create_app(scene_storage: SceneStorage | None = None) -> FastAPI:
    """Build and configure a FastAPI application instance.

    Args:
        scene_storage: Pre-configured SceneStorage instance. When omitted,
            a default storage pointed at ``data/scenes/`` is used.

    Returns:
        A fully configured FastAPI application with routes and exception
        handlers registered.
    """
    application = FastAPI(title="AI Town API")
    application.state.scene_storage = scene_storage or SceneStorage(
        DEFAULT_SCENE_DIRECTORY
    )

    def storage(request: Request) -> SceneStorage:
        return request.app.state.scene_storage

    @application.exception_handler(SceneNotFoundError)
    async def handle_scene_not_found(
        _request: Request, error: SceneNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": str(error)},
        )

    @application.exception_handler(SceneStorageError)
    async def handle_scene_storage_error(
        _request: Request, error: SceneStorageError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(error)},
        )

    @application.get("/api/health")
    def health() -> HealthResponse:
        return {"status": "ok"}

    @application.get("/api/scenes")
    def list_scenes(request: Request) -> list[SceneSummary]:
        return storage(request).list_scenes()

    @application.post(
        "/api/scenes",
        status_code=status.HTTP_201_CREATED,
    )
    def add_scene(payload: CreateSceneRequest, request: Request) -> Scene:
        scene = create_scene(payload.name)
        storage(request).create(scene)
        return scene

    @application.get("/api/scenes/{scene_id}")
    def get_scene(scene_id: UUID, request: Request) -> Scene:
        return storage(request).get(scene_id)

    @application.put("/api/scenes/{scene_id}")
    def put_scene(
        scene_id: UUID,
        payload: UpdateSceneRequest,
        request: Request,
    ) -> Scene:
        current_scene = storage(request).get(scene_id)
        updated_scene = update_scene(current_scene, payload)
        storage(request).save(updated_scene)
        return updated_scene

    return application


app = create_app()
