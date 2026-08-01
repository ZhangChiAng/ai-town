import type {
  Agent,
  AgentId,
  ConfirmLayerRequest,
  ExternalEvent,
  InnerTurn,
  Layer,
  LayerDraftResponse,
  ModelOption,
  ModelOptionsResponse,
  ModelReasoningBlock,
  ModelRequestContextItem,
  ModelRequestPreviewResponse,
  OuterTurn,
  Scene,
  SceneSummary,
  SceneUpdate,
  TokenUsage,
} from "./types";

type UnknownRecord = Record<string, unknown>;

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === "object" && value !== null;
}

function isAgentId(value: unknown): value is AgentId {
  return value === "A" || value === "B" || value === "C";
}

function isLayer(value: unknown): value is Layer {
  return value === "inner" || value === "outer";
}

function isPositiveInteger(value: unknown): value is number {
  return Number.isInteger(value) && Number(value) >= 1;
}

function isTokenCount(value: unknown): value is number {
  return Number.isInteger(value) && Number(value) >= 0;
}

function isUsage(value: unknown): value is TokenUsage {
  return (
    isRecord(value) &&
    isTokenCount(value.input_tokens) &&
    isTokenCount(value.output_tokens) &&
    isTokenCount(value.cache_creation_input_tokens) &&
    isTokenCount(value.cache_read_input_tokens)
  );
}

function isReasoningBlock(value: unknown): value is ModelReasoningBlock {
  return (
    isRecord(value) &&
    (value.type === "thinking" ||
      value.type === "summary_text" ||
      value.type === "reasoning_text") &&
    typeof value.text === "string" &&
    value.text.trim() !== ""
  );
}

function isModelOption(value: unknown): value is ModelOption {
  return (
    isRecord(value) &&
    Object.keys(value).length === 1 &&
    typeof value.model === "string" &&
    value.model.trim() !== ""
  );
}

function isModelRequestContextItem(
  value: unknown,
): value is ModelRequestContextItem {
  return (
    isRecord(value) &&
    Object.keys(value).length === 2 &&
    (value.role === "system" ||
      value.role === "user" ||
      value.role === "assistant") &&
    typeof value.text === "string"
  );
}

function isEvent(value: unknown): value is ExternalEvent {
  if (
    !isRecord(value) ||
    typeof value.id !== "string" ||
    !isPositiveInteger(value.sequence) ||
    typeof value.content !== "string" ||
    (value.kind !== "manual" && value.kind !== "agent_message")
  ) {
    return false;
  }
  return (
    (value.source_agent_id === null ||
      isAgentId(value.source_agent_id)) &&
    (value.source_call_id === null ||
      typeof value.source_call_id === "string")
  );
}

function isInnerTurn(value: unknown): value is InnerTurn {
  return (
    isRecord(value) &&
    typeof value.call_id === "string" &&
    typeof value.event_id === "string" &&
    isPositiveInteger(value.sequence) &&
    typeof value.input === "string" &&
    typeof value.output === "string" &&
    isEvent(value.consumed_event)
  );
}

function isOuterTurn(value: unknown): value is OuterTurn {
  return (
    isRecord(value) &&
    typeof value.call_id === "string" &&
    typeof value.event_id === "string" &&
    isPositiveInteger(value.sequence) &&
    typeof value.input === "string" &&
    typeof value.output === "string" &&
    isAgentId(value.recipient_id) &&
    typeof value.generated_event_id === "string"
  );
}

function isAgent(value: unknown): value is Agent {
  return (
    isRecord(value) &&
    isAgentId(value.id) &&
    typeof value.name === "string" &&
    isRecord(value.inner_context) &&
    typeof value.inner_context.system_prompt === "string" &&
    value.inner_context.system_prompt.trim() !== "" &&
    Array.isArray(value.inner_context.turns) &&
    value.inner_context.turns.every(isInnerTurn) &&
    isRecord(value.outer_context) &&
    typeof value.outer_context.system_prompt === "string" &&
    value.outer_context.system_prompt.trim() !== "" &&
    Array.isArray(value.outer_context.turns) &&
    value.outer_context.turns.every(isOuterTurn) &&
    Array.isArray(value.pending_events) &&
    value.pending_events.every(isEvent)
  );
}

function isScene(value: unknown): value is Scene {
  return (
    isRecord(value) &&
    value.schema_version === 6 &&
    typeof value.id === "string" &&
    typeof value.name === "string" &&
    (value.model === null ||
      (typeof value.model === "string" && value.model.trim() !== "")) &&
    Array.isArray(value.agents) &&
    value.agents.length === 3 &&
    value.agents.every(isAgent) &&
    value.agents.map((agent) => agent.id).join("") === "ABC" &&
    Array.isArray(value.rollback_stack) &&
    value.rollback_stack.every(
      (reference) =>
        isRecord(reference) &&
        typeof reference.call_id === "string" &&
        isAgentId(reference.agent_id) &&
        isLayer(reference.layer),
    ) &&
    isPositiveInteger(value.next_sequence)
  );
}

function isSceneSummary(value: unknown): value is SceneSummary {
  return (
    isRecord(value) &&
    typeof value.id === "string" &&
    typeof value.name === "string"
  );
}

function isLayerDraft(value: unknown): value is LayerDraftResponse {
  return (
    isRecord(value) &&
    isLayer(value.layer) &&
    typeof value.call_id === "string" &&
    typeof value.event_id === "string" &&
    typeof value.content === "string" &&
    value.content.trim() !== "" &&
    Array.isArray(value.reasoning) &&
    value.reasoning.every(isReasoningBlock) &&
    isUsage(value.usage) &&
    isRecord(value.request_snapshot) &&
    typeof value.state_token === "string"
  );
}

function isPreview(
  value: unknown,
): value is ModelRequestPreviewResponse {
  return (
    isRecord(value) &&
    Object.keys(value).length === 3 &&
    isLayer(value.layer) &&
    typeof value.event_id === "string" &&
    Array.isArray(value.context) &&
    value.context.every(isModelRequestContextItem)
  );
}

function detailMessage(body: unknown): string | undefined {
  if (!isRecord(body)) {
    return undefined;
  }
  if (typeof body.detail === "string") {
    return body.detail;
  }
  if (!Array.isArray(body.detail)) {
    return undefined;
  }
  const messages = body.detail
    .map((item) =>
      isRecord(item) && typeof item.msg === "string"
        ? item.msg
        : undefined,
    )
    .filter((message): message is string => message !== undefined);
  return messages.length > 0 ? messages.join("；") : undefined;
}

async function requestJson(
  path: string,
  options?: RequestInit,
): Promise<unknown> {
  let response: Response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new ApiError("无法连接后端，请确认服务正在运行。");
  }

  let body: unknown;
  try {
    body = await response.json();
  } catch {
    throw new ApiError(
      response.ok
        ? "后端返回了无法识别的数据。"
        : `请求失败（${response.status}）。`,
      response.status,
    );
  }
  if (!response.ok) {
    throw new ApiError(
      detailMessage(body) ?? `请求失败（${response.status}）。`,
      response.status,
    );
  }
  return body;
}

function requireScene(body: unknown): Scene {
  if (!isScene(body)) {
    throw new ApiError("后端返回了无法识别的场景数据。");
  }
  return body;
}

export async function listScenes(): Promise<SceneSummary[]> {
  const body = await requestJson("/api/scenes");
  if (!Array.isArray(body) || !body.every(isSceneSummary)) {
    throw new ApiError("后端返回了无法识别的场景列表。");
  }
  return body;
}

export async function getModelOptions(): Promise<ModelOptionsResponse> {
  const body = await requestJson("/api/model-options");
  if (
    !isRecord(body) ||
    !Array.isArray(body.options) ||
    !body.options.every(isModelOption)
  ) {
    throw new ApiError("后端返回了无法识别的模型列表。");
  }
  return { options: body.options };
}

export async function createScene(
  name: string,
  model: string,
): Promise<Scene> {
  return requireScene(
    await requestJson("/api/scenes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, model }),
    }),
  );
}

export async function bindSceneModel(
  sceneId: string,
  model: string,
): Promise<Scene> {
  return requireScene(
    await requestJson(
      `/api/scenes/${encodeURIComponent(sceneId)}/model`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model }),
      },
    ),
  );
}

export async function getScene(sceneId: string): Promise<Scene> {
  return requireScene(
    await requestJson(`/api/scenes/${encodeURIComponent(sceneId)}`),
  );
}

export async function saveScene(
  sceneId: string,
  update: SceneUpdate,
): Promise<Scene> {
  return requireScene(
    await requestJson(`/api/scenes/${encodeURIComponent(sceneId)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(update),
    }),
  );
}

export async function createManualEvent(
  sceneId: string,
  agentId: AgentId,
  content: string,
): Promise<Scene> {
  return requireScene(
    await requestJson(
      `/api/scenes/${encodeURIComponent(sceneId)}/agents/${agentId}/events`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content }),
      },
    ),
  );
}

export async function editManualEvent(
  sceneId: string,
  agentId: AgentId,
  eventId: string,
  content: string,
): Promise<Scene> {
  return requireScene(
    await requestJson(
      `/api/scenes/${encodeURIComponent(sceneId)}/agents/${agentId}/events/${encodeURIComponent(eventId)}`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content }),
      },
    ),
  );
}

export async function deleteManualEvent(
  sceneId: string,
  agentId: AgentId,
  eventId: string,
): Promise<Scene> {
  return requireScene(
    await requestJson(
      `/api/scenes/${encodeURIComponent(sceneId)}/agents/${agentId}/events/${encodeURIComponent(eventId)}`,
      { method: "DELETE" },
    ),
  );
}

export async function generateLayerDraft(
  sceneId: string,
  agentId: AgentId,
  layer: Layer,
): Promise<LayerDraftResponse> {
  const body = await requestJson(
    `/api/scenes/${encodeURIComponent(sceneId)}/agents/${agentId}/${layer}-drafts`,
    { method: "POST" },
  );
  if (!isLayerDraft(body)) {
    throw new ApiError("后端返回了无法识别的人格草稿。");
  }
  return body;
}

export async function confirmLayerDraft(
  sceneId: string,
  agentId: AgentId,
  layer: Layer,
  confirmation: ConfirmLayerRequest,
): Promise<Scene> {
  return requireScene(
    await requestJson(
      `/api/scenes/${encodeURIComponent(sceneId)}/agents/${agentId}/${layer}-confirmations`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(confirmation),
      },
    ),
  );
}

export async function getModelRequestPreview(
  sceneId: string,
  agentId: AgentId,
  layer: Layer,
): Promise<ModelRequestPreviewResponse> {
  const body = await requestJson(
    `/api/scenes/${encodeURIComponent(sceneId)}/agents/${agentId}/model-request-preview?layer=${layer}`,
  );
  if (!isPreview(body)) {
    throw new ApiError("后端返回了无法识别的请求预览。");
  }
  return body;
}

export async function rollbackLatestCall(
  sceneId: string,
): Promise<Scene> {
  return requireScene(
    await requestJson(
      `/api/scenes/${encodeURIComponent(sceneId)}/rollback`,
      { method: "POST" },
    ),
  );
}
