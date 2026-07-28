export const AGENT_IDS = ["A", "B", "C"] as const;

export type AgentId = (typeof AGENT_IDS)[number];

export interface MessageTimelineRecord {
  type: "message";
  message_id: string;
  direction: "sent" | "received";
  counterpart_id: AgentId;
  content: string;
}

export interface InnerVoiceTimelineRecord {
  type: "inner_voice";
  inner_voice_id: string;
  content: string;
}

export type TimelineRecord =
  | MessageTimelineRecord
  | InnerVoiceTimelineRecord;

export interface Agent {
  id: AgentId;
  name: string;
  persona: string;
  desire: string;
  fear: string;
  memory: string;
  system_prompt: string;
  timeline: TimelineRecord[];
}

export interface Scene {
  schema_version: 3;
  id: string;
  name: string;
  agents: Agent[];
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
  recipient_id: AgentId;
  content: string;
}

export interface MessageDraftUsage {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens: number;
  cache_read_input_tokens: number;
}

export interface MessageDraftResponse {
  recipient_id: AgentId;
  content: string;
  usage: MessageDraftUsage;
  request_snapshot: ModelRequest;
}

export type ModelRequest = Record<string, unknown>;

export interface ModelRequestPreviewResponse {
  request: ModelRequest;
}
