"""FastAPI application factory and route definitions for the AI Town API."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, TypedDict
from uuid import UUID

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.config import load_model_settings
from app.drafting import (
    DraftGenerationError,
    MessageDraftService,
    create_anthropic_client,
)
from app.models import (
    AgentId,
    CreateMessageRequest,
    CreateSceneRequest,
    MessageDraftResponse,
    Scene,
    SceneSummary,
    UpdateSceneRequest,
    add_message,
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


def create_app(
    scene_storage: SceneStorage | None = None,
    message_draft_service: MessageDraftService | None = None,
) -> FastAPI:
    """Build and configure a FastAPI application instance.

    Args:
        scene_storage: Pre-configured SceneStorage instance. When omitted, a
            default storage pointed at ``data/scenes/`` is used.
        message_draft_service: Injectable draft service for tests. When
            omitted, startup loads model settings and creates an Anthropic
            client.

    Returns:
        A fully configured FastAPI application with routes and exception
        handlers registered.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        client = None
        if message_draft_service is None:
            settings = load_model_settings()
            client = create_anthropic_client(settings)
            app.state.message_draft_service = MessageDraftService(
                client,
                settings.model,
            )

        try:
            yield
        finally:
            if client is not None:
                client.close()

    application = FastAPI(title="AI Town API", lifespan=lifespan)
    application.state.scene_storage = scene_storage or SceneStorage(
        DEFAULT_SCENE_DIRECTORY
    )
    if message_draft_service is not None:
        application.state.message_draft_service = message_draft_service

    def storage(request: Request) -> SceneStorage:
        return request.app.state.scene_storage

    def drafts(request: Request) -> MessageDraftService:
        return request.app.state.message_draft_service

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

    @application.exception_handler(DraftGenerationError)
    async def handle_draft_generation_error(
        _request: Request, error: DraftGenerationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"detail": str(error)},
        )

    @application.get("/api/health")
    async def health() -> HealthResponse:
        return {"status": "ok"}

    @application.get("/api/scenes")
    async def list_scenes(request: Request) -> list[SceneSummary]:
        return storage(request).list_scenes()

    @application.post(
        "/api/scenes",
        status_code=status.HTTP_201_CREATED,
    )
    async def add_scene(payload: CreateSceneRequest, request: Request) -> Scene:
        scene = create_scene(payload.name)
        storage(request).create(scene)
        return scene

    @application.get("/api/scenes/{scene_id}")
    async def get_scene(scene_id: UUID, request: Request) -> Scene:
        return storage(request).get(scene_id)

    @application.put("/api/scenes/{scene_id}")
    async def put_scene(
        scene_id: UUID,
        payload: UpdateSceneRequest,
        request: Request,
    ) -> Scene:
        current_scene = storage(request).get(scene_id)
        updated_scene = update_scene(current_scene, payload)
        storage(request).save(updated_scene)
        return updated_scene

    @application.post(
        "/api/scenes/{scene_id}/messages",
        status_code=status.HTTP_201_CREATED,
    )
    async def post_message(
        scene_id: UUID,
        payload: CreateMessageRequest,
        request: Request,
    ) -> Scene:
        current_scene = storage(request).get(scene_id)
        updated_scene = add_message(current_scene, payload)
        storage(request).save(updated_scene)
        return updated_scene

    @application.post("/api/scenes/{scene_id}/agents/{agent_id}/message-drafts")
    async def post_message_draft(
        scene_id: UUID,
        agent_id: AgentId,
        request: Request,
    ) -> MessageDraftResponse:
        current_scene = storage(request).get(scene_id)
        return drafts(request).generate(current_scene, agent_id)

    return application


app = create_app()
