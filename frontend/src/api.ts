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
  PromptProfile,
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

function hasExactKeys(
  value: UnknownRecord,
  expected: readonly string[],
): boolean {
  const actual = Object.keys(value);
  return (
    actual.length === expected.length &&
    expected.every((key) => Object.hasOwn(value, key))
  );
}

function isSceneSchema(value: unknown): value is Scene["schema"] {
  return (
    typeof value === "string" &&
    /^ai-town\.scene\/1\.(0|[1-9][0-9]*)$/.test(value)
  );
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
    hasExactKeys(value, [
      "input_tokens",
      "output_tokens",
      "cache_creation_input_tokens",
      "cache_read_input_tokens",
    ]) &&
    isTokenCount(value.input_tokens) &&
    isTokenCount(value.output_tokens) &&
    isTokenCount(value.cache_creation_input_tokens) &&
    isTokenCount(value.cache_read_input_tokens)
  );
}

function isReasoningBlock(value: unknown): value is ModelReasoningBlock {
  return (
    isRecord(value) &&
    hasExactKeys(value, ["type", "text"]) &&
    (value.type === "thinking" ||
      value.type === "summary_text" ||
      value.type === "reasoning_text") &&
    typeof value.text === "string" &&
    value.text.trim() !== ""
  );
}

function isReasoningList(value: unknown): value is ModelReasoningBlock[] {
  return Array.isArray(value) && value.every(isReasoningBlock);
}

function isModelOption(value: unknown): value is ModelOption {
  return (
    isRecord(value) &&
    hasExactKeys(value, ["model"]) &&
    typeof value.model === "string" &&
    value.model.trim() !== ""
  );
}

function isModelRequestContextItem(
  value: unknown,
): value is ModelRequestContextItem {
  return (
    isRecord(value) &&
    hasExactKeys(value, ["role", "text"]) &&
    (value.role === "system" ||
      value.role === "user" ||
      value.role === "assistant") &&
    typeof value.text === "string"
  );
}

function isEvent(value: unknown): value is ExternalEvent {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "id",
      "sequence",
      "kind",
      "content",
      "source_agent_id",
      "source_call_id",
    ]) ||
    typeof value.id !== "string" ||
    !isPositiveInteger(value.sequence) ||
    typeof value.content !== "string" ||
    value.content.trim() === "" ||
    (value.kind !== "manual" && value.kind !== "agent_message")
  ) {
    return false;
  }
  return (
    (value.kind === "manual" &&
      value.source_agent_id === null &&
      value.source_call_id === null) ||
    (value.kind === "agent_message" &&
      isAgentId(value.source_agent_id) &&
      typeof value.source_call_id === "string")
  );
}

function isInnerTurn(value: unknown): value is InnerTurn {
  return (
    isRecord(value) &&
    hasExactKeys(value, [
      "call_id",
      "event_ids",
      "sequence",
      "input",
      "output",
      "consumed_events",
      "reasoning",
    ]) &&
    typeof value.call_id === "string" &&
    Array.isArray(value.event_ids) &&
    value.event_ids.every((id): id is string => typeof id === "string") &&
    value.event_ids.length > 0 &&
    isPositiveInteger(value.sequence) &&
    typeof value.input === "string" &&
    value.input.trim() !== "" &&
    typeof value.output === "string" &&
    value.output.trim() !== "" &&
    Array.isArray(value.consumed_events) &&
    value.consumed_events.length > 0 &&
    value.consumed_events.every(isEvent) &&
    value.event_ids.every(
      (id, index) =>
        (value.consumed_events as ExternalEvent[])[index]?.id === id,
    ) &&
    isReasoningList(value.reasoning)
  );
}

function isOuterTurn(value: unknown): value is OuterTurn {
  return (
    isRecord(value) &&
    hasExactKeys(value, [
      "call_id",
      "event_ids",
      "sequence",
      "input",
      "output",
      "recipient_id",
      "generated_event_id",
      "reasoning",
    ]) &&
    typeof value.call_id === "string" &&
    Array.isArray(value.event_ids) &&
    value.event_ids.every((id): id is string => typeof id === "string") &&
    value.event_ids.length > 0 &&
    isPositiveInteger(value.sequence) &&
    typeof value.input === "string" &&
    value.input.trim() !== "" &&
    typeof value.output === "string" &&
    value.output.trim() !== "" &&
    (value.recipient_id === null || isAgentId(value.recipient_id)) &&
    (value.generated_event_id === null ||
      typeof value.generated_event_id === "string") &&
    ((value.output === "STOP" &&
      value.recipient_id === null &&
      value.generated_event_id === null) ||
      (value.output !== "STOP" &&
        isAgentId(value.recipient_id) &&
        typeof value.generated_event_id === "string")) &&
    isReasoningList(value.reasoning)
  );
}

function isPromptProfile(value: unknown): value is PromptProfile {
  return (
    isRecord(value) &&
    hasExactKeys(value, [
      "pronoun",
      "hidden_beliefs",
      "inner_memories",
      "outer_memories",
    ]) &&
    typeof value.pronoun === "string" &&
    typeof value.hidden_beliefs === "string" &&
    typeof value.inner_memories === "string" &&
    typeof value.outer_memories === "string"
  );
}

function hasValidInteractions(
  value: unknown,
  senderId: AgentId,
): boolean {
  if (!isRecord(value)) {
    return false;
  }
  const addresses = new Set<string>();
  return Object.entries(value).every(([targetId, relationship]) => {
    if (
      !isAgentId(targetId) ||
      targetId === senderId ||
      !isRecord(relationship) ||
      !hasExactKeys(relationship, ["description", "addresses"]) ||
      typeof relationship.description !== "string" ||
      !isRecord(relationship.addresses)
    ) {
      return false;
    }
    return Object.entries(relationship.addresses).every(
      ([address, occasion]) => {
      if (
        address.trim() === "" ||
        typeof occasion !== "string" ||
        occasion.trim() === "" ||
        addresses.has(address)
      ) {
        return false;
      }
      addresses.add(address);
      return true;
      },
    );
  });
}

function isAgent(value: unknown): value is Agent {
  return (
    isRecord(value) &&
    hasExactKeys(value, [
      "id",
      "name",
      "prompt_profile",
      "interactions",
      "inner_context",
      "outer_context",
      "pending_events",
    ]) &&
    isAgentId(value.id) &&
    typeof value.name === "string" &&
    value.name.trim() !== "" &&
    isPromptProfile(value.prompt_profile) &&
    hasValidInteractions(value.interactions, value.id) &&
    isRecord(value.inner_context) &&
    hasExactKeys(value.inner_context, ["turns"]) &&
    Array.isArray(value.inner_context.turns) &&
    value.inner_context.turns.every(isInnerTurn) &&
    isRecord(value.outer_context) &&
    hasExactKeys(value.outer_context, ["turns"]) &&
    Array.isArray(value.outer_context.turns) &&
    value.outer_context.turns.every(isOuterTurn) &&
    Array.isArray(value.pending_events) &&
    value.pending_events.every(isEvent)
  );
}

function isScene(value: unknown): value is Scene {
  return (
    isRecord(value) &&
    hasExactKeys(value, [
      "schema",
      "id",
      "name",
      "model",
      "agents",
      "rollback_stack",
      "next_sequence",
    ]) &&
    isSceneSchema(value.schema) &&
    typeof value.id === "string" &&
    typeof value.name === "string" &&
    value.name.trim() !== "" &&
    typeof value.model === "string" &&
    value.model.trim() !== "" &&
    Array.isArray(value.agents) &&
    value.agents.length === 3 &&
    value.agents.every(isAgent) &&
    value.agents.map((agent) => agent.id).join("") === "ABC" &&
    value.agents.every((agent) => agent.name.trim() !== "") &&
    new Set(value.agents.map((agent) => agent.name.trim())).size === 3 &&
    Array.isArray(value.rollback_stack) &&
    value.rollback_stack.every(
      (reference) =>
        isRecord(reference) &&
        hasExactKeys(reference, ["call_id", "agent_id", "layer"]) &&
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
    hasExactKeys(value, ["id", "name"]) &&
    typeof value.id === "string" &&
    typeof value.name === "string"
  );
}

function isLayerDraft(value: unknown): value is LayerDraftResponse {
  return (
    isRecord(value) &&
    hasExactKeys(value, [
      "layer",
      "call_id",
      "event_ids",
      "content",
      "reasoning",
      "usage",
      "request_snapshot",
      "state_token",
    ]) &&
    isLayer(value.layer) &&
    typeof value.call_id === "string" &&
    Array.isArray(value.event_ids) &&
    value.event_ids.every((id): id is string => typeof id === "string") &&
    value.event_ids.length > 0 &&
    typeof value.content === "string" &&
    value.content.trim() !== "" &&
    isReasoningList(value.reasoning) &&
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
    hasExactKeys(value, ["layer", "event_ids", "context"]) &&
    isLayer(value.layer) &&
    Array.isArray(value.event_ids) &&
    value.event_ids.every((id): id is string => typeof id === "string") &&
    value.event_ids.length > 0 &&
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

async function requestNoContent(
  path: string,
  options?: RequestInit,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new ApiError("无法连接后端，请确认服务正在运行。");
  }
  // A 204 has no body; only parse JSON to surface error details on 4xx/5xx.
  if (response.status === 204) {
    return;
  }
  if (response.ok) {
    throw new ApiError("后端返回了无法识别的数据。", response.status);
  }
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    throw new ApiError(`请求失败（${response.status}）。`, response.status);
  }
  throw new ApiError(
    detailMessage(body) ?? `请求失败（${response.status}）。`,
    response.status,
  );
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

export async function getScene(sceneId: string): Promise<Scene> {
  return requireScene(
    await requestJson(`/api/scenes/${encodeURIComponent(sceneId)}`),
  );
}

export async function deleteScene(sceneId: string): Promise<void> {
  await requestNoContent(
    `/api/scenes/${encodeURIComponent(sceneId)}`,
    { method: "DELETE" },
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
