"""FastAPI routes for the two-layer AI Town experiment."""

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, TypedDict
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from app.draft_workflow import (
    DraftGenerationError,
    DraftWorkflow,
    confirm_draft,
)
from app.model_backends import (
    BackendFactory,
    ModelBackend,
    create_anthropic_messages_backend,
    create_model_backend_registry,
    create_openai_responses_backend,
)
from app.model_config import load_model_settings
from app.models import (
    AgentId,
    BindSceneModelRequest,
    ConfirmLayerRequest,
    CreateSceneRequest,
    EventContentRequest,
    EventNotFoundError,
    InvalidLayerOutputError,
    Layer,
    LayerDraftResponse,
    ModelOption,
    ModelOptionsResponse,
    ModelRequestContextItem,
    ModelRequestPreviewResponse,
    Scene,
    SceneConflictError,
    SceneModelBindingConflictError,
    SceneSummary,
    UpdateSceneRequest,
    add_manual_event,
    bind_scene_model,
    create_scene,
    delete_manual_event,
    edit_manual_event,
    rollback_latest_call,
    update_scene,
)
from app.storage import SceneNotFoundError, SceneStorage, SceneStorageError


class HealthResponse(TypedDict):
    """Response shape for the health endpoint."""

    status: Literal["ok"]


DEFAULT_SCENE_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "data" / "scenes"
)
BACKEND_FACTORIES: dict[str, BackendFactory] = {
    "anthropic_messages": create_anthropic_messages_backend,
    "openai_responses": create_openai_responses_backend,
}


def create_app(
    scene_storage: SceneStorage | None = None,
    model_backends: Mapping[str, ModelBackend] | ModelBackend | None = None,
) -> FastAPI:
    """Create the application with injectable storage and model backends."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if injected_backends is not None:
            yield
            return

        settings = load_model_settings()
        registry = create_model_backend_registry(
            settings,
            BACKEND_FACTORIES,
        )
        try:
            install_model_backends(application, registry)
            yield
        finally:
            registry.close()

    application = FastAPI(title="AI Town API", lifespan=lifespan)
    application.state.scene_storage = scene_storage or SceneStorage(
        DEFAULT_SCENE_DIRECTORY
    )
    injected_backends = _normalize_backends(model_backends)
    if injected_backends is not None:
        install_model_backends(application, injected_backends)

    def storage(request: Request) -> SceneStorage:
        """Return request-scoped access to the configured storage."""
        return request.app.state.scene_storage

    def draft_workflows(request: Request) -> dict[str, DraftWorkflow]:
        """Return workflows keyed by exact configured model name."""
        return request.app.state.draft_workflows

    def require_available_model(model: str, request: Request) -> None:
        """Reject a model name that is not configured in this process."""
        if model not in draft_workflows(request):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Selected model is not available.",
            )

    def drafts_for_scene(
        scene: Scene,
        request: Request,
    ) -> DraftWorkflow:
        """Resolve the immutable scene binding to an available service."""
        if scene.model is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Scene must be bound to a model before model requests.",
            )
        workflow = draft_workflows(request).get(scene.model)
        if workflow is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "The scene's bound model is not available in the "
                    "current configuration."
                ),
            )
        return workflow

    @application.exception_handler(SceneNotFoundError)
    async def handle_scene_not_found(
        _request: Request,
        error: SceneNotFoundError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": str(error)},
        )

    @application.exception_handler(EventNotFoundError)
    async def handle_event_not_found(
        _request: Request,
        error: EventNotFoundError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": str(error)},
        )

    @application.exception_handler(SceneConflictError)
    async def handle_scene_conflict(
        _request: Request,
        error: SceneConflictError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": str(error)},
        )

    @application.exception_handler(InvalidLayerOutputError)
    async def handle_invalid_layer_output(
        _request: Request,
        error: InvalidLayerOutputError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": str(error)},
        )

    @application.exception_handler(SceneModelBindingConflictError)
    async def handle_scene_model_binding_conflict(
        _request: Request,
        error: SceneModelBindingConflictError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": str(error)},
        )

    @application.exception_handler(DraftGenerationError)
    async def handle_generation_error(
        _request: Request,
        error: DraftGenerationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"detail": str(error)},
        )

    @application.exception_handler(SceneStorageError)
    async def handle_storage_error(
        _request: Request,
        error: SceneStorageError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
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

    @application.post(
        "/api/scenes",
        status_code=status.HTTP_201_CREATED,
    )
    async def post_scene(
        payload: CreateSceneRequest,
        request: Request,
    ) -> Scene:
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
        return storage(request).mutate(
            scene_id,
            lambda scene: update_scene(scene, payload),
        )

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
        return storage(request).mutate(
            scene_id,
            lambda scene: bind_scene_model(scene, payload.model),
        )

    @application.post(
        "/api/scenes/{scene_id}/agents/{agent_id}/events",
        status_code=status.HTTP_201_CREATED,
    )
    async def post_manual_event(
        scene_id: UUID,
        agent_id: AgentId,
        payload: EventContentRequest,
        request: Request,
    ) -> Scene:
        return storage(request).mutate(
            scene_id,
            lambda scene: add_manual_event(
                scene,
                agent_id,
                payload.content,
            ),
        )

    @application.put(
        "/api/scenes/{scene_id}/agents/{agent_id}/events/{event_id}"
    )
    async def put_manual_event(
        scene_id: UUID,
        agent_id: AgentId,
        event_id: UUID,
        payload: EventContentRequest,
        request: Request,
    ) -> Scene:
        return storage(request).mutate(
            scene_id,
            lambda scene: edit_manual_event(
                scene,
                agent_id,
                event_id,
                payload.content,
            ),
        )

    @application.delete(
        "/api/scenes/{scene_id}/agents/{agent_id}/events/{event_id}"
    )
    async def remove_manual_event(
        scene_id: UUID,
        agent_id: AgentId,
        event_id: UUID,
        request: Request,
    ) -> Scene:
        return storage(request).mutate(
            scene_id,
            lambda scene: delete_manual_event(
                scene,
                agent_id,
                event_id,
            ),
        )

    @application.post("/api/scenes/{scene_id}/agents/{agent_id}/inner-drafts")
    async def post_inner_draft(
        scene_id: UUID,
        agent_id: AgentId,
        request: Request,
    ) -> LayerDraftResponse:
        scene = storage(request).get(scene_id)
        return drafts_for_scene(scene, request).generate(
            scene,
            agent_id,
            "inner",
        )

    @application.post("/api/scenes/{scene_id}/agents/{agent_id}/outer-drafts")
    async def post_outer_draft(
        scene_id: UUID,
        agent_id: AgentId,
        request: Request,
    ) -> LayerDraftResponse:
        scene = storage(request).get(scene_id)
        return drafts_for_scene(scene, request).generate(
            scene,
            agent_id,
            "outer",
        )

    @application.post(
        "/api/scenes/{scene_id}/agents/{agent_id}/inner-confirmations"
    )
    async def post_inner_confirmation(
        scene_id: UUID,
        agent_id: AgentId,
        payload: ConfirmLayerRequest,
        request: Request,
    ) -> Scene:
        return storage(request).mutate(
            scene_id,
            lambda scene: confirm_draft(
                scene,
                agent_id,
                "inner",
                payload,
            ),
        )

    @application.post(
        "/api/scenes/{scene_id}/agents/{agent_id}/outer-confirmations"
    )
    async def post_outer_confirmation(
        scene_id: UUID,
        agent_id: AgentId,
        payload: ConfirmLayerRequest,
        request: Request,
    ) -> Scene:
        return storage(request).mutate(
            scene_id,
            lambda scene: confirm_draft(
                scene,
                agent_id,
                "outer",
                payload,
            ),
        )

    @application.get(
        "/api/scenes/{scene_id}/agents/{agent_id}/model-request-preview"
    )
    async def get_model_request_preview(
        scene_id: UUID,
        agent_id: AgentId,
        layer: Layer,
        request: Request,
    ) -> ModelRequestPreviewResponse:
        scene = storage(request).get(scene_id)
        preview = drafts_for_scene(scene, request).preview(
            scene,
            agent_id,
            layer,
        )
        return ModelRequestPreviewResponse(
            layer=preview.layer,
            event_id=preview.event_id,
            context=[
                ModelRequestContextItem(role=item.role, text=item.text)
                for item in preview.context
            ],
            request=preview.request,
        )

    @application.post("/api/scenes/{scene_id}/rollback")
    async def post_rollback(scene_id: UUID, request: Request) -> Scene:
        return storage(request).mutate(scene_id, rollback_latest_call)

    return application


def _normalize_backends(
    backends: Mapping[str, ModelBackend] | ModelBackend | None,
) -> dict[str, ModelBackend] | None:
    """Normalize the injectable single-backend convenience form."""
    if backends is None:
        return None
    if isinstance(backends, ModelBackend):
        return {backends.model: backends}
    return dict(backends)


def install_model_backends(
    application: FastAPI,
    backends: Mapping[str, ModelBackend],
) -> None:
    """Install ordered backends, workflows, and protocol-free options."""
    registry = dict(backends)
    application.state.model_backends = registry
    application.state.draft_workflows = {
        model: DraftWorkflow(backend) for model, backend in registry.items()
    }
    application.state.model_options = [
        ModelOption(model=model) for model in registry
    ]


app = create_app()
