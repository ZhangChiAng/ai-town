"""FastAPI application factory and route definitions for the AI Town API."""

from collections.abc import AsyncIterator, Mapping
from contextlib import ExitStack, asynccontextmanager
from pathlib import Path
from typing import Literal, TypedDict
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from app.config import load_model_settings
from app.drafting import (
    DraftGenerationError,
    MessageDraftService,
    create_anthropic_client,
    create_responses_client,
)
from app.models import (
    AgentId,
    BindSceneModelRequest,
    ComposeSystemPromptRequest,
    ComposeSystemPromptResponse,
    CreateMessageRequest,
    CreateSceneRequest,
    MessageDeletionConflictError,
    MessageDraftResponse,
    MessageNotFoundError,
    ModelOption,
    ModelOptionsResponse,
    ModelRequestPreviewResponse,
    Scene,
    SceneModelBindingConflictError,
    SceneSummary,
    UpdateSceneRequest,
    add_message,
    bind_scene_model,
    compose_system_prompt,
    create_scene,
    delete_message,
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
    message_draft_services: (
        Mapping[str, MessageDraftService] | MessageDraftService | None
    ) = None,
) -> FastAPI:
    """Build and configure a FastAPI application instance.

    Args:
        scene_storage: Pre-configured SceneStorage instance. When omitted, a
            default storage pointed at ``data/scenes/`` is used.
        message_draft_services: Injectable model-name registry for tests.
            Passing one service remains supported for focused route tests.
            When omitted, startup creates both configured SDK clients.

    Returns:
        A fully configured FastAPI application with routes and exception
        handlers registered.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if injected_services is not None:
            yield
            return

        # ExitStack also closes the first client if the second fails to start.
        with ExitStack() as clients:
            settings = load_model_settings()
            anthropic_client = create_anthropic_client(settings.anthropic)
            clients.callback(anthropic_client.close)
            responses_client = create_responses_client(settings.responses)
            clients.callback(responses_client.close)
            install_model_services(
                app,
                {
                    settings.anthropic.model: MessageDraftService(
                        anthropic_client,
                        settings.anthropic.model,
                    ),
                    settings.responses.model: MessageDraftService(
                        responses_client,
                        settings.responses.model,
                    ),
                },
            )
            yield

    application = FastAPI(title="AI Town API", lifespan=lifespan)
    application.state.scene_storage = scene_storage or SceneStorage(
        DEFAULT_SCENE_DIRECTORY
    )
    injected_services = _normalize_services(message_draft_services)
    if injected_services is not None:
        install_model_services(application, injected_services)

    def storage(request: Request) -> SceneStorage:
        return request.app.state.scene_storage

    def model_services(request: Request) -> dict[str, MessageDraftService]:
        return request.app.state.message_draft_services

    def require_available_model(model: str, request: Request) -> None:
        if model not in model_services(request):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Selected model is not available.",
            )

    def drafts_for_scene(
        scene: Scene,
        request: Request,
    ) -> MessageDraftService:
        if scene.model is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Scene must be bound to a model before model requests.",
            )
        service = model_services(request).get(scene.model)
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "The scene's bound model is not available in the "
                    "current configuration."
                ),
            )
        return service

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

    @application.exception_handler(MessageNotFoundError)
    async def handle_message_not_found(
        _request: Request, error: MessageNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": str(error)},
        )

    @application.exception_handler(MessageDeletionConflictError)
    async def handle_message_deletion_conflict(
        _request: Request, error: MessageDeletionConflictError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": str(error)},
        )

    @application.exception_handler(SceneModelBindingConflictError)
    async def handle_scene_model_binding_conflict(
        _request: Request, error: SceneModelBindingConflictError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
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

    @application.get("/api/model-options")
    async def get_model_options(request: Request) -> ModelOptionsResponse:
        return ModelOptionsResponse(options=request.app.state.model_options)

    @application.get("/api/scenes")
    async def list_scenes(request: Request) -> list[SceneSummary]:
        return storage(request).list_scenes()

    @application.post("/api/system-prompts/compose")
    async def post_composed_system_prompt(
        payload: ComposeSystemPromptRequest,
    ) -> ComposeSystemPromptResponse:
        return ComposeSystemPromptResponse(
            system_prompt=compose_system_prompt(
                payload.persona,
                payload.desire,
                payload.fear,
                payload.memory,
            )
        )

    @application.post(
        "/api/scenes",
        status_code=status.HTTP_201_CREATED,
    )
    async def add_scene(payload: CreateSceneRequest, request: Request) -> Scene:
        require_available_model(payload.model, request)
        scene = create_scene(payload.name, payload.model)
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

    @application.put("/api/scenes/{scene_id}/model")
    async def put_scene_model(
        scene_id: UUID,
        payload: BindSceneModelRequest,
        request: Request,
    ) -> Scene:
        current_scene = storage(request).get(scene_id)
        if current_scene.model is not None:
            # A bound scene conflicts regardless of the replacement value.
            bind_scene_model(current_scene, payload.model)
        require_available_model(payload.model, request)
        updated_scene = bind_scene_model(current_scene, payload.model)
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

    @application.delete("/api/scenes/{scene_id}/messages/{message_id}")
    async def remove_message(
        scene_id: UUID,
        message_id: UUID,
        request: Request,
    ) -> Scene:
        current_scene = storage(request).get(scene_id)
        updated_scene = delete_message(current_scene, message_id)
        storage(request).save(updated_scene)
        return updated_scene

    @application.post("/api/scenes/{scene_id}/agents/{agent_id}/message-drafts")
    async def post_message_draft(
        scene_id: UUID,
        agent_id: AgentId,
        request: Request,
    ) -> MessageDraftResponse:
        current_scene = storage(request).get(scene_id)
        return drafts_for_scene(current_scene, request).generate(
            current_scene, agent_id
        )

    @application.get(
        "/api/scenes/{scene_id}/agents/{agent_id}/model-request-preview"
    )
    async def get_model_request_preview(
        scene_id: UUID,
        agent_id: AgentId,
        request: Request,
    ) -> ModelRequestPreviewResponse:
        current_scene = storage(request).get(scene_id)
        return ModelRequestPreviewResponse(
            request=drafts_for_scene(current_scene, request).preview(
                current_scene, agent_id
            )
        )

    return application


def _normalize_services(
    services: Mapping[str, MessageDraftService] | MessageDraftService | None,
) -> dict[str, MessageDraftService] | None:
    """Normalize the injectable single-service convenience form."""
    if services is None:
        return None
    if isinstance(services, MessageDraftService):
        return {services.model: services}
    return dict(services)


def install_model_services(
    application: FastAPI,
    services: Mapping[str, MessageDraftService],
) -> None:
    """Install a concrete-name service registry and stable public options."""
    registry = dict(services)
    options: list[ModelOption] = []
    for protocol in ("anthropic", "responses"):
        options.extend(
            ModelOption(protocol=service.protocol, model=model)
            for model, service in registry.items()
            if service.protocol == protocol
        )
    application.state.message_draft_services = registry
    application.state.model_options = options


app = create_app()
