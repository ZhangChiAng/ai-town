export const AGENT_IDS = ["A", "B", "C"] as const;

export type AgentId = (typeof AGENT_IDS)[number];
export type Layer = "inner" | "outer";
export type EventKind = "manual" | "agent_message";

export interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens: number;
  cache_read_input_tokens: number;
}

export interface ModelReasoningBlock {
  type: "thinking" | "summary_text" | "reasoning_text";
  text: string;
}

export interface ModelOption {
  model: string;
}

export interface ModelOptionsResponse {
  options: ModelOption[];
}

export interface ExternalEvent {
  id: string;
  sequence: number;
  kind: EventKind;
  content: string;
  source_agent_id: AgentId | null;
  source_call_id: string | null;
}

export interface InnerTurn {
  call_id: string;
  event_ids: string[];
  sequence: number;
  input: string;
  output: string;
  consumed_events: ExternalEvent[];
}

export interface OuterTurn {
  call_id: string;
  event_ids: string[];
  sequence: number;
  input: string;
  output: string;
  recipient_id: AgentId;
  generated_event_id: string;
}

export interface InnerContext {
  system_prompt: string;
  turns: InnerTurn[];
}

export interface OuterContext {
  system_prompt: string;
  turns: OuterTurn[];
}

export interface Agent {
  id: AgentId;
  name: string;
  inner_context: InnerContext;
  outer_context: OuterContext;
  pending_events: ExternalEvent[];
}

export interface ConfirmedCallReference {
  call_id: string;
  agent_id: AgentId;
  layer: Layer;
}

export interface Scene {
  schema_version: 7;
  id: string;
  name: string;
  model: string | null;
  agents: Agent[];
  rollback_stack: ConfirmedCallReference[];
  next_sequence: number;
}

export interface SceneSummary {
  id: string;
  name: string;
}

export interface ContextUpdate {
  system_prompt: string;
}

export interface AgentUpdate {
  id: AgentId;
  name: string;
  inner_context: ContextUpdate;
  outer_context: ContextUpdate;
}

export interface SceneUpdate {
  name: string;
  agents: AgentUpdate[];
}

export type JsonObject = Record<string, unknown>;

export interface ModelRequestContextItem {
  role: "system" | "user" | "assistant";
  text: string;
}

export interface LayerDraftResponse {
  layer: Layer;
  call_id: string;
  event_ids: string[];
  content: string;
  reasoning: ModelReasoningBlock[];
  usage: TokenUsage;
  request_snapshot: JsonObject;
  state_token: string;
}

export interface ConfirmLayerRequest {
  call_id: string;
  event_ids: string[];
  content: string;
  state_token: string;
}

export interface ModelRequestPreviewResponse {
  layer: Layer;
  event_ids: string[];
  context: ModelRequestContextItem[];
}
