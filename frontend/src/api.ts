import type {
  Agent,
  AgentId,
  Scene,
  SceneSummary,
  SceneUpdate,
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

function isAgent(value: unknown): value is Agent {
  return (
    isRecord(value) &&
    isAgentId(value.id) &&
    typeof value.name === "string" &&
    typeof value.persona === "string" &&
    typeof value.desire === "string" &&
    typeof value.fear === "string" &&
    typeof value.memory === "string" &&
    Array.isArray(value.timeline)
  );
}

function isScene(value: unknown): value is Scene {
  return (
    isRecord(value) &&
    value.schema_version === 1 &&
    typeof value.id === "string" &&
    typeof value.name === "string" &&
    Array.isArray(value.agents) &&
    value.agents.length === 3 &&
    value.agents.every(isAgent) &&
    value.agents.map((agent) => agent.id).join("") === "ABC"
  );
}

function isSceneSummary(value: unknown): value is SceneSummary {
  return (
    isRecord(value) &&
    typeof value.id === "string" &&
    typeof value.name === "string"
  );
}

function detailMessage(body: unknown): string | undefined {
  if (!isRecord(body)) {
    return undefined;
  }

  if (typeof body.detail === "string") {
    return body.detail;
  }

  if (Array.isArray(body.detail)) {
    const messages = body.detail
      .map((item) => {
        if (!isRecord(item) || typeof item.msg !== "string") {
          return undefined;
        }
        return item.msg;
      })
      .filter((message): message is string => message !== undefined);

    if (messages.length > 0) {
      return messages.join("；");
    }
  }

  return undefined;
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

export async function listScenes(): Promise<SceneSummary[]> {
  const body = await requestJson("/api/scenes");
  if (!Array.isArray(body) || !body.every(isSceneSummary)) {
    throw new ApiError("后端返回了无法识别的场景列表。");
  }
  return body;
}

export async function createScene(name: string): Promise<Scene> {
  const body = await requestJson("/api/scenes", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!isScene(body)) {
    throw new ApiError("后端返回了无法识别的场景数据。");
  }
  return body;
}

export async function getScene(sceneId: string): Promise<Scene> {
  const body = await requestJson(
    `/api/scenes/${encodeURIComponent(sceneId)}`,
  );
  if (!isScene(body)) {
    throw new ApiError("后端返回了无法识别的场景数据。");
  }
  return body;
}

export async function saveScene(
  sceneId: string,
  update: SceneUpdate,
): Promise<Scene> {
  const body = await requestJson(
    `/api/scenes/${encodeURIComponent(sceneId)}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(update),
    },
  );
  if (!isScene(body)) {
    throw new ApiError("后端返回了无法识别的场景数据。");
  }
  return body;
}
