export const AGENT_IDS = ["A", "B", "C"] as const;

export type AgentId = (typeof AGENT_IDS)[number];

export interface MessageTimelineRecord {
  type: "message";
  message_id: string;
  direction: "sent" | "received";
  counterpart_id: AgentId;
  content: string;
}

export interface Agent {
  id: AgentId;
  name: string;
  persona: string;
  desire: string;
  fear: string;
  memory: string;
  system_prompt: string;
  timeline: MessageTimelineRecord[];
}

export interface Scene {
  schema_version: 5;
  id: string;
  name: string;
  model: string | null;
  agents: Agent[];
}

export interface ModelOption {
  protocol: "anthropic" | "responses";
  model: string;
}

export interface SceneSummary {
  id: string;
  name: string;
}

export interface AgentUpdate {
  id: AgentId;
  name: string;
  persona: string;
  desire: string;
  fear: string;
  memory: string;
  system_prompt: string;
}

export interface SceneUpdate {
  name: string;
  agents: AgentUpdate[];
}

export interface MessageCreate {
  sender_id: AgentId;
  content: string;
}

export interface MessageDraftUsage {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens: number;
  cache_read_input_tokens: number;
}

export interface ModelReasoningBlock {
  type: "thinking" | "summary_text" | "reasoning_text";
  text: string;
}

export interface MessageDraftResponse {
  content: string;
  reasoning: ModelReasoningBlock[];
  usage: MessageDraftUsage;
  request_snapshot: ModelRequest;
}

export type ModelRequest = Record<string, unknown>;

export interface ModelRequestPreviewResponse {
  request: ModelRequest;
}
