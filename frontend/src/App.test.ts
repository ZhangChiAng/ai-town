import { createApp, nextTick } from "vue";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App.vue";
import type { Agent, AgentId, Scene } from "./types";

const FIRST_ID = "11111111-1111-4111-8111-111111111111";
const SECOND_ID = "22222222-2222-4222-8222-222222222222";

let container: HTMLDivElement;
let unmountApp: (() => void) | undefined;

function makeAgent(id: AgentId, name = `居民 ${id}`): Agent {
  return {
    id,
    name,
    persona: `${id} 的人设`,
    desire: `${id} 的欲望`,
    fear: `${id} 的恐惧`,
    memory: `${id} 的记忆`,
    timeline: [],
  };
}

function makeScene(
  id = FIRST_ID,
  name = "雨夜港口",
): Scene {
  return {
    schema_version: 1,
    id,
    name,
    agents: [
      makeAgent("A", "阿岚"),
      makeAgent("B", "北辰"),
      makeAgent("C", "迟夏"),
    ],
  };
}

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: vi.fn().mockResolvedValue(body),
  } as unknown as Response;
}

function mountApp(): void {
  const app = createApp(App);
  app.mount(container);
  unmountApp = () => app.unmount();
}

async function flushAsyncUpdates(): Promise<void> {
  for (let index = 0; index < 8; index += 1) {
    await Promise.resolve();
  }
  await nextTick();
}

function findButton(text: string): HTMLButtonElement {
  const button = Array.from(
    container.querySelectorAll<HTMLButtonElement>("button"),
  ).find((candidate) => candidate.textContent?.includes(text));

  if (button === undefined) {
    throw new Error(`Could not find button containing "${text}"`);
  }
  return button;
}

async function setFieldValue(
  selector: string,
  value: string,
): Promise<void> {
  const field = container.querySelector<
    HTMLInputElement | HTMLTextAreaElement
  >(selector);
  if (field === null) {
    throw new Error(`Could not find field "${selector}"`);
  }
  field.value = value;
  field.dispatchEvent(new Event("input", { bubbles: true }));
  await nextTick();
}

async function openFirstScene(): Promise<void> {
  const button = container.querySelector<HTMLButtonElement>(
    ".scene-list-button",
  );
  if (button === null) {
    throw new Error("Could not find first scene button");
  }
  button.click();
  await flushAsyncUpdates();
}

beforeEach(() => {
  container = document.createElement("div");
  document.body.append(container);
});

afterEach(() => {
  unmountApp?.();
  unmountApp = undefined;
  container.remove();
  vi.unstubAllGlobals();
});

describe("App", () => {
  it("loads the scene list and opens a selected scene", async () => {
    const scene = makeScene();
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      if (input === "/api/scenes") {
        return jsonResponse([{ id: scene.id, name: scene.name }]);
      }
      if (input === `/api/scenes/${scene.id}`) {
        return jsonResponse(scene);
      }
      throw new Error(`Unexpected request: ${String(input)}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    mountApp();
    await flushAsyncUpdates();

    expect(fetchMock).toHaveBeenCalledWith("/api/scenes", undefined);
    expect(container.textContent).toContain("雨夜港口");

    await openFirstScene();

    expect(fetchMock).toHaveBeenCalledWith(
      `/api/scenes/${scene.id}`,
      undefined,
    );
    expect(
      container.querySelector<HTMLInputElement>("#scene-name")?.value,
    ).toBe("雨夜港口");
    expect(
      container.querySelector<HTMLInputElement>("#agent-name")?.value,
    ).toBe("阿岚");

    findButton("Agent B").click();
    await nextTick();

    expect(
      container.querySelector<HTMLInputElement>("#agent-name")?.value,
    ).toBe("北辰");
    expect(container.textContent).toContain("时间线为空");
  });

  it("creates a named scene and immediately opens it", async () => {
    const created = makeScene(FIRST_ID, "海边小镇");
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse([]))
      .mockResolvedValueOnce(jsonResponse(created, 201));
    vi.stubGlobal("fetch", fetchMock);

    mountApp();
    await flushAsyncUpdates();
    await setFieldValue("#new-scene-name", "  海边小镇  ");

    findButton("创建").click();
    await flushAsyncUpdates();

    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "/api/scenes",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: "海边小镇" }),
      }),
    );
    expect(
      container.querySelector<HTMLInputElement>("#scene-name")?.value,
    ).toBe("海边小镇");
    expect(container.querySelectorAll(".scene-list-button")).toHaveLength(
      1,
    );
    expect(container.textContent).toContain("当前内容已保存");
  });

  it("merges a created scene with an older list response", async () => {
    const existing = makeScene(SECOND_ID, "已有场景");
    const created = makeScene(FIRST_ID, "海边小镇");
    let resolveList!: (response: Response) => void;
    const pendingList = new Promise<Response>((resolve) => {
      resolveList = resolve;
    });
    const fetchMock = vi.fn(
      (
        input: RequestInfo | URL,
        options?: RequestInit,
      ): Promise<Response> => {
        if (input === "/api/scenes" && options === undefined) {
          return pendingList;
        }
        if (input === "/api/scenes" && options?.method === "POST") {
          return Promise.resolve(jsonResponse(created, 201));
        }
        return Promise.reject(
          new Error(`Unexpected request: ${String(input)}`),
        );
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    mountApp();
    await nextTick();
    await setFieldValue("#new-scene-name", "海边小镇");

    findButton("创建").click();
    await flushAsyncUpdates();

    expect(container.querySelectorAll(".scene-list-button")).toHaveLength(
      1,
    );
    expect(container.textContent).not.toContain("正在加载场景");

    resolveList(
      jsonResponse([{ id: existing.id, name: existing.name }]),
    );
    await flushAsyncUpdates();

    expect(container.querySelectorAll(".scene-list-button")).toHaveLength(
      2,
    );
    expect(container.textContent).toContain("已有场景");
    expect(container.textContent).toContain("海边小镇");
    expect(container.textContent).not.toContain("正在加载场景");
  });

  it("keeps a created scene visible when an older list request fails", async () => {
    const created = makeScene(FIRST_ID, "海边小镇");
    let rejectList!: (reason: Error) => void;
    const pendingList = new Promise<Response>((_resolve, reject) => {
      rejectList = reject;
    });
    const fetchMock = vi.fn(
      (
        input: RequestInfo | URL,
        options?: RequestInit,
      ): Promise<Response> => {
        if (input === "/api/scenes" && options === undefined) {
          return pendingList;
        }
        if (input === "/api/scenes" && options?.method === "POST") {
          return Promise.resolve(jsonResponse(created, 201));
        }
        return Promise.reject(
          new Error(`Unexpected request: ${String(input)}`),
        );
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    mountApp();
    await nextTick();
    await setFieldValue("#new-scene-name", "海边小镇");

    findButton("创建").click();
    await flushAsyncUpdates();

    rejectList(new Error("Scene storage is temporarily unavailable"));
    await flushAsyncUpdates();

    expect(container.querySelectorAll(".scene-list-button")).toHaveLength(
      1,
    );
    expect(container.textContent).toContain("海边小镇");
    expect(container.textContent).not.toContain(
      "Scene storage is temporarily unavailable",
    );
    expect(container.textContent).not.toContain("重新加载");
  });

  it("marks edits dirty and writes only editable fields on explicit save", async () => {
    const scene = makeScene();
    let putBody: unknown;
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, options?: RequestInit) => {
        if (input === "/api/scenes" && options === undefined) {
          return jsonResponse([{ id: scene.id, name: scene.name }]);
        }
        if (
          input === `/api/scenes/${scene.id}` &&
          options === undefined
        ) {
          return jsonResponse(scene);
        }
        if (
          input === `/api/scenes/${scene.id}` &&
          options?.method === "PUT"
        ) {
          putBody = JSON.parse(String(options.body));
          const update = putBody as {
            name: string;
            agents: Omit<Agent, "timeline">[];
          };
          return jsonResponse({
            ...scene,
            name: update.name,
            agents: update.agents.map((agent) => ({
              ...agent,
              timeline: [],
            })),
          });
        }
        throw new Error(`Unexpected request: ${String(input)}`);
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    mountApp();
    await flushAsyncUpdates();
    await openFirstScene();
    await setFieldValue("#scene-name", "清晨港口");
    await setFieldValue("#agent-desire", "找到失踪的船");

    expect(container.textContent).toContain("有未保存的更改");
    expect(fetchMock).toHaveBeenCalledTimes(2);

    findButton("保存场景").click();
    await flushAsyncUpdates();

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(putBody).toEqual({
      name: "清晨港口",
      agents: scene.agents.map(
        ({ id, name, persona, desire, fear, memory }) => ({
          id,
          name,
          persona,
          desire: id === "A" ? "找到失踪的船" : desire,
          fear,
          memory,
        }),
      ),
    });
    expect(putBody).not.toHaveProperty("id");
    expect(putBody).not.toHaveProperty("schema_version");
    expect(
      (putBody as { agents: Record<string, unknown>[] }).agents[0],
    ).not.toHaveProperty("timeline");
    expect(container.textContent).toContain("保存成功");
    expect(
      container.querySelector<HTMLButtonElement>(".save-button")
        ?.disabled,
    ).toBe(true);
  });

  it("keeps the selected agent and editor active after saving", async () => {
    const scene = makeScene();
    const savedMemory = "迟夏保存后的记忆";
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, options?: RequestInit) => {
        if (input === "/api/scenes" && options === undefined) {
          return jsonResponse([{ id: scene.id, name: scene.name }]);
        }
        if (
          input === `/api/scenes/${scene.id}` &&
          options === undefined
        ) {
          return jsonResponse(scene);
        }
        if (
          input === `/api/scenes/${scene.id}` &&
          options?.method === "PUT"
        ) {
          return jsonResponse({
            ...scene,
            agents: scene.agents.map((agent) =>
              agent.id === "C"
                ? { ...agent, memory: savedMemory }
                : agent,
            ),
          });
        }
        throw new Error(`Unexpected request: ${String(input)}`);
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    mountApp();
    await flushAsyncUpdates();
    await openFirstScene();

    findButton("Agent C").click();
    await nextTick();
    await setFieldValue("#agent-memory", savedMemory);

    findButton("保存场景").click();
    await flushAsyncUpdates();

    expect(findButton("Agent C").getAttribute("aria-selected")).toBe(
      "true",
    );
    expect(
      container.querySelector<HTMLInputElement>("#agent-name")?.value,
    ).toBe("迟夏");
    expect(
      container.querySelector<HTMLTextAreaElement>("#agent-memory")
        ?.value,
    ).toBe(savedMemory);
    expect(container.textContent).toContain("保存成功");
  });

  it("keeps unsaved form content and reports an API save failure", async () => {
    const scene = makeScene();
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, options?: RequestInit) => {
        if (input === "/api/scenes") {
          return jsonResponse([{ id: scene.id, name: scene.name }]);
        }
        if (options?.method === "PUT") {
          return jsonResponse(
            { detail: "Could not save scene: disk is full" },
            500,
          );
        }
        return jsonResponse(scene);
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    mountApp();
    await flushAsyncUpdates();
    await openFirstScene();
    await setFieldValue("#agent-memory", "这段内容尚未落盘");

    findButton("保存场景").click();
    await flushAsyncUpdates();

    expect(
      container.querySelector<HTMLTextAreaElement>("#agent-memory")
        ?.value,
    ).toBe("这段内容尚未落盘");
    expect(container.textContent).toContain(
      "Could not save scene: disk is full",
    );
    expect(
      container.querySelector<HTMLButtonElement>(".save-button")
        ?.disabled,
    ).toBe(false);
  });

  it("asks before switching away from dirty scene content", async () => {
    const first = makeScene(FIRST_ID, "第一个场景");
    const second = makeScene(SECOND_ID, "第二个场景");
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      if (input === "/api/scenes") {
        return jsonResponse([
          { id: first.id, name: first.name },
          { id: second.id, name: second.name },
        ]);
      }
      if (input === `/api/scenes/${first.id}`) {
        return jsonResponse(first);
      }
      if (input === `/api/scenes/${second.id}`) {
        return jsonResponse(second);
      }
      throw new Error(`Unexpected request: ${String(input)}`);
    });
    const confirmMock = vi
      .fn()
      .mockReturnValueOnce(false)
      .mockReturnValueOnce(true);
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("confirm", confirmMock);

    mountApp();
    await flushAsyncUpdates();
    await openFirstScene();
    await setFieldValue("#scene-name", "尚未保存的名称");

    const secondButton =
      container.querySelectorAll<HTMLButtonElement>(
        ".scene-list-button",
      )[1];
    secondButton.click();
    await flushAsyncUpdates();

    expect(confirmMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(
      container.querySelector<HTMLInputElement>("#scene-name")?.value,
    ).toBe("尚未保存的名称");

    secondButton.click();
    await flushAsyncUpdates();

    expect(confirmMock).toHaveBeenCalledTimes(2);
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(
      container.querySelector<HTMLInputElement>("#scene-name")?.value,
    ).toBe("第二个场景");
  });

  it("asks before creating a new scene when the editor is dirty", async () => {
    const existing = makeScene(FIRST_ID, "已有场景");
    const created = makeScene(SECOND_ID, "新场景");
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, options?: RequestInit) => {
        if (input === "/api/scenes" && options === undefined) {
          return jsonResponse([
            { id: existing.id, name: existing.name },
          ]);
        }
        if (input === `/api/scenes/${existing.id}`) {
          return jsonResponse(existing);
        }
        if (input === "/api/scenes" && options?.method === "POST") {
          return jsonResponse(created, 201);
        }
        throw new Error(`Unexpected request: ${String(input)}`);
      },
    );
    const confirmMock = vi
      .fn()
      .mockReturnValueOnce(false)
      .mockReturnValueOnce(true);
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("confirm", confirmMock);

    mountApp();
    await flushAsyncUpdates();
    await openFirstScene();
    await setFieldValue("#agent-fear", "未保存的恐惧");
    await setFieldValue("#new-scene-name", "新场景");

    findButton("创建").click();
    await flushAsyncUpdates();

    expect(confirmMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(
      container.querySelector<HTMLTextAreaElement>("#agent-fear")?.value,
    ).toBe("未保存的恐惧");

    findButton("创建").click();
    await flushAsyncUpdates();

    expect(confirmMock).toHaveBeenCalledTimes(2);
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(
      container.querySelector<HTMLInputElement>("#scene-name")?.value,
    ).toBe("新场景");
  });

  it("protects dirty content when the page is about to close", async () => {
    const scene = makeScene();
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValueOnce(
          jsonResponse([{ id: scene.id, name: scene.name }]),
        )
        .mockResolvedValueOnce(jsonResponse(scene)),
    );

    mountApp();
    await flushAsyncUpdates();
    await openFirstScene();

    const cleanEvent = new Event("beforeunload", {
      cancelable: true,
    });
    window.dispatchEvent(cleanEvent);
    expect(cleanEvent.defaultPrevented).toBe(false);

    await setFieldValue("#agent-persona", "还没有保存的人设");
    const dirtyEvent = new Event("beforeunload", {
      cancelable: true,
    });
    window.dispatchEvent(dirtyEvent);

    expect(dirtyEvent.defaultPrevented).toBe(true);
  });

  it("shows scene-list API errors and offers a retry", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          jsonResponse({ detail: "Scene storage is unreadable" }, 500),
        ),
    );

    mountApp();
    await flushAsyncUpdates();

    expect(container.textContent).toContain(
      "Scene storage is unreadable",
    );
    expect(findButton("重新加载")).toBeTruthy();
  });
});
