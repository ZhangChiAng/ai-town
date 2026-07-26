export const AGENT_IDS = ["A", "B", "C"] as const;

export type AgentId = (typeof AGENT_IDS)[number];

export interface TimelineRecord {
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
  timeline: TimelineRecord[];
}

export interface Scene {
  schema_version: 1;
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
