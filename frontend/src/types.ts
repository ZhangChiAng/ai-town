export const AGENT_IDS = ["A", "B", "C"] as const;

export type AgentId = (typeof AGENT_IDS)[number];

export interface Agent {
  id: AgentId;
  name: string;
  persona: string;
  desire: string;
  fear: string;
  memory: string;
  timeline: unknown[];
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
