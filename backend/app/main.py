"""FastAPI routes for the two-layer AI Town experiment."""

import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, TypedDict
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from app.draft_workflow import (
    DraftGenerationError,
    DraftWorkflow,
    confirm_draft,
)
from app.http_logging import RequestLoggingMiddleware
from app.model_backends import (
    BackendFactory,
    ModelBackend,
    create_anthropic_messages_backend,
    create_deepseek_responses_backend,
    create_minimax_responses_backend,
    create_model_backend_registry,
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
    get_agent,
    rollback_latest_call,
    update_scene,
)
from app.storage import SceneNotFoundError, SceneStorage, SceneStorageError
from app.structured_logging import (
    bind_log_context,
    log_event,
    register_secrets,
    text_metadata,
)

LOGGER = logging.getLogger(__name__)


class HealthResponse(TypedDict):
    """Response shape for the health endpoint."""

    status: Literal["ok"]


DEFAULT_SCENE_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "data" / "scenes"
)
BACKEND_FACTORIES: dict[str, BackendFactory] = {
    "anthropic_messages": create_anthropic_messages_backend,
    "deepseek_responses": create_deepseek_responses_backend,
    "minimax_responses": create_minimax_responses_backend,
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

        try:
            settings = load_model_settings()
            # Register every resolved key before any provider object exists.
            register_secrets(setting.api_key for setting in settings)
            registry = await create_model_backend_registry(
                settings,
                BACKEND_FACTORIES,
            )
        except Exception as error:
            log_event(
                LOGGER,
                logging.ERROR,
                "application.startup.failed",
                "Application startup failed.",
                exception=error,
            )
            raise
        try:
            install_model_backends(application, registry)
            log_event(
                LOGGER,
                logging.INFO,
                "application.started",
                "Application startup completed.",
                model_count=len(registry),
                models=list(registry),
            )
            yield
        finally:
            try:
                await registry.aclose()
            except BaseException as error:
                log_event(
                    LOGGER,
                    logging.ERROR,
                    "application.shutdown.failed",
                    "Application shutdown failed.",
                    exception=error,
                )
                raise
            log_event(
                LOGGER,
                logging.INFO,
                "application.stopped",
                "Application shutdown completed.",
            )

    application = FastAPI(title="AI Town API", lifespan=lifespan)
    application.add_middleware(RequestLoggingMiddleware)
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
        log_event(
            LOGGER,
            logging.INFO,
            "scene.created",
            "Scene created.",
            scene_id=scene.id,
            model=scene.model,
            agent_count=len(scene.agents),
            **text_metadata("name", scene.name),
        )
        return scene

    @application.get("/api/scenes/{scene_id}")
    async def get_scene(scene_id: UUID, request: Request) -> Scene:
        return storage(request).get(scene_id)

    @application.delete(
        "/api/scenes/{scene_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def delete_scene(
        scene_id: UUID,
        request: Request,
    ) -> Response:
        # Fetch before delete so the 404 path and log metadata reuse the
        # same not-found handler and persisted scene identity.
        scene = storage(request).get(scene_id)
        storage(request).delete(scene_id)
        log_event(
            LOGGER,
            logging.INFO,
            "scene.deleted",
            "Scene deleted.",
            scene_id=scene.id,
            model=scene.model,
            **text_metadata("name", scene.name),
        )
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @application.put("/api/scenes/{scene_id}")
    async def put_scene(
        scene_id: UUID,
        payload: UpdateSceneRequest,
        request: Request,
    ) -> Scene:
        scene = storage(request).mutate(
            scene_id,
            lambda scene: update_scene(scene, payload),
        )
        log_event(
            LOGGER,
            logging.INFO,
            "scene.updated",
            "Scene settings updated.",
            scene_id=scene.id,
            model=scene.model,
            agents=[
                {
                    "agent_id": agent.id,
                    **text_metadata("name", agent.name),
                    **text_metadata(
                        "inner_prompt", agent.inner_context.system_prompt
                    ),
                    **text_metadata(
                        "outer_prompt", agent.outer_context.system_prompt
                    ),
                }
                for agent in scene.agents
            ],
        )
        return scene

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
        scene = storage(request).mutate(
            scene_id,
            lambda scene: bind_scene_model(scene, payload.model),
        )
        log_event(
            LOGGER,
            logging.INFO,
            "scene.model_bound",
            "Scene model bound.",
            scene_id=scene.id,
            model=scene.model,
        )
        return scene

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
        scene = storage(request).mutate(
            scene_id,
            lambda scene: add_manual_event(
                scene,
                agent_id,
                payload.content,
            ),
        )
        event = get_agent(scene, agent_id).pending_events[-1]
        log_event(
            LOGGER,
            logging.INFO,
            "event.created",
            "Manual event created.",
            scene_id=scene.id,
            agent_id=agent_id,
            event_id=event.id,
            event_kind=event.kind,
            pending_event_count=len(get_agent(scene, agent_id).pending_events),
            **text_metadata("content", event.content),
        )
        return scene

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
        scene = storage(request).mutate(
            scene_id,
            lambda scene: edit_manual_event(
                scene,
                agent_id,
                event_id,
                payload.content,
            ),
        )
        log_event(
            LOGGER,
            logging.INFO,
            "event.updated",
            "Manual event updated.",
            scene_id=scene.id,
            agent_id=agent_id,
            event_id=event_id,
            **text_metadata("content", payload.content),
        )
        return scene

    @application.delete(
        "/api/scenes/{scene_id}/agents/{agent_id}/events/{event_id}"
    )
    async def remove_manual_event(
        scene_id: UUID,
        agent_id: AgentId,
        event_id: UUID,
        request: Request,
    ) -> Scene:
        deleted_events = []

        def delete_with_metadata(scene: Scene) -> Scene:
            """Capture metadata from the same queued event being deleted."""
            updated = delete_manual_event(scene, agent_id, event_id)
            event = next(
                event
                for event in get_agent(scene, agent_id).pending_events
                if event.id == event_id
            )
            deleted_events.append(event)
            return updated

        scene = storage(request).mutate(
            scene_id,
            delete_with_metadata,
        )
        [deleted_event] = deleted_events
        log_event(
            LOGGER,
            logging.INFO,
            "event.deleted",
            "Manual event deleted.",
            scene_id=scene.id,
            agent_id=agent_id,
            event_id=event_id,
            pending_event_count=len(get_agent(scene, agent_id).pending_events),
            **text_metadata("content", deleted_event.content),
        )
        return scene

    @application.post("/api/scenes/{scene_id}/agents/{agent_id}/inner-drafts")
    async def post_inner_draft(
        scene_id: UUID,
        agent_id: AgentId,
        request: Request,
    ) -> LayerDraftResponse:
        scene = storage(request).get(scene_id)
        with bind_log_context(
            scene_id=scene_id, agent_id=agent_id, layer="inner"
        ):
            return await drafts_for_scene(scene, request).generate(
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
        with bind_log_context(
            scene_id=scene_id, agent_id=agent_id, layer="outer"
        ):
            return await drafts_for_scene(scene, request).generate(
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
        scene = storage(request).mutate(
            scene_id,
            lambda scene: confirm_draft(
                scene,
                agent_id,
                "inner",
                payload,
            ),
        )
        turn = get_agent(scene, agent_id).inner_context.turns[-1]
        log_event(
            LOGGER,
            logging.INFO,
            "draft.inner_confirmed",
            "Inner draft confirmed.",
            scene_id=scene.id,
            agent_id=agent_id,
            layer="inner",
            call_id=turn.call_id,
            model=scene.model,
            event_ids=turn.event_ids,
            event_count=len(turn.event_ids),
            reasoning_count=len(turn.reasoning),
            **text_metadata("input", turn.input),
            **text_metadata("output", turn.output),
        )
        return scene

    @application.post(
        "/api/scenes/{scene_id}/agents/{agent_id}/outer-confirmations"
    )
    async def post_outer_confirmation(
        scene_id: UUID,
        agent_id: AgentId,
        payload: ConfirmLayerRequest,
        request: Request,
    ) -> Scene:
        scene = storage(request).mutate(
            scene_id,
            lambda scene: confirm_draft(
                scene,
                agent_id,
                "outer",
                payload,
            ),
        )
        turn = get_agent(scene, agent_id).outer_context.turns[-1]
        log_event(
            LOGGER,
            logging.INFO,
            "draft.outer_confirmed",
            "Outer draft confirmed and routed.",
            scene_id=scene.id,
            agent_id=agent_id,
            layer="outer",
            call_id=turn.call_id,
            model=scene.model,
            event_ids=turn.event_ids,
            event_count=len(turn.event_ids),
            recipient_id=turn.recipient_id,
            generated_event_id=turn.generated_event_id,
            reasoning_count=len(turn.reasoning),
            **text_metadata("input", turn.input),
            **text_metadata("output", turn.output),
        )
        return scene

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
            event_ids=list(preview.event_ids),
            context=[
                ModelRequestContextItem(role=item.role, text=item.text)
                for item in preview.context
            ],
        )

    @application.post("/api/scenes/{scene_id}/rollback")
    async def post_rollback(scene_id: UUID, request: Request) -> Scene:
        references = []
        rolled_back_metadata = []

        def rollback_with_reference(scene: Scene) -> Scene:
            """Capture the same stack top changed under the storage lock."""
            updated = rollback_latest_call(scene)
            reference = scene.rollback_stack[-1]
            agent = get_agent(scene, reference.agent_id)
            turn = (
                agent.inner_context.turns[-1]
                if reference.layer == "inner"
                else agent.outer_context.turns[-1]
            )
            references.append(reference)
            rolled_back_metadata.append(
                {
                    "event_ids": turn.event_ids,
                    "event_count": len(turn.event_ids),
                    "reasoning_count": len(turn.reasoning),
                    **text_metadata("input", turn.input),
                    **text_metadata("output", turn.output),
                }
            )
            return updated

        scene = storage(request).mutate(scene_id, rollback_with_reference)
        [reference] = references
        [turn_metadata] = rolled_back_metadata
        log_event(
            LOGGER,
            logging.INFO,
            "draft.rolled_back",
            "Latest confirmed draft rolled back.",
            scene_id=scene.id,
            agent_id=reference.agent_id,
            layer=reference.layer,
            call_id=reference.call_id,
            model=scene.model,
            rollback_depth=len(scene.rollback_stack),
            **turn_metadata,
        )
        return scene

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
