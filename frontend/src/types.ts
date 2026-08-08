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
  reasoning: ModelReasoningBlock[];
}

export interface OuterTurn {
  call_id: string;
  event_ids: string[];
  sequence: number;
  input: string;
  output: string;
  recipient_id: AgentId | null;
  generated_event_id: string | null;
  reasoning: ModelReasoningBlock[];
}

export interface InnerContext {
  turns: InnerTurn[];
}

export interface OuterContext {
  turns: OuterTurn[];
}

export interface PromptProfile {
  pronoun: string;
  hidden_beliefs: string;
  inner_memories: string;
  outer_memories: string;
}

export interface Interaction {
  description: string;
  addresses: Record<string, string>;
}

export type Interactions = Partial<Record<AgentId, Interaction>>;

export interface Agent {
  id: AgentId;
  name: string;
  prompt_profile: PromptProfile;
  interactions: Interactions;
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
  schema: `ai-town.scene/1.${number}`;
  id: string;
  name: string;
  model: string;
  agents: Agent[];
  rollback_stack: ConfirmedCallReference[];
  next_sequence: number;
}

export interface SceneSummary {
  id: string;
  name: string;
}

export interface AgentUpdate {
  id: AgentId;
  name: string;
  prompt_profile: PromptProfile;
  interactions: Interactions;
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
  reasoning: ModelReasoningBlock[];
}

export interface ModelRequestPreviewResponse {
  layer: Layer;
  event_ids: string[];
  context: ModelRequestContextItem[];
}
