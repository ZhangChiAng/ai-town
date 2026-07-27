import { createApp, nextTick } from "vue";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App.vue";
import type {
  Agent,
  AgentId,
  MessageDraftResponse,
  Scene,
} from "./types";

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

function makeDraftResponse(
  recipientId: AgentId,
  content: string,
  seed = 0,
): MessageDraftResponse {
  return {
    recipient_id: recipientId,
    content,
    usage: {
      input_tokens: 120 + seed,
      output_tokens: 48 + seed,
      cache_creation_input_tokens: 17 + seed,
      cache_read_input_tokens: 91 + seed,
    },
  };
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

async function setSelectValue(
  selector: string,
  value: string,
): Promise<void> {
  const field = container.querySelector<HTMLSelectElement>(selector);
  if (field === null) {
    throw new Error(`Could not find select "${selector}"`);
  }
  field.value = value;
  field.dispatchEvent(new Event("change", { bubbles: true }));
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

  it("requires an explicit recipient and confirms a message for both timelines", async () => {
    const scene = makeScene();
    const confirmed = makeScene();
    confirmed.agents[0].timeline = [
      {
        message_id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        direction: "sent",
        counterpart_id: "B",
        content: "今晚在灯塔下见。",
      },
    ];
    confirmed.agents[1].timeline = [
      {
        message_id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        direction: "received",
        counterpart_id: "A",
        content: "今晚在灯塔下见。",
      },
    ];
    let messageBody: unknown;
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
          input === `/api/scenes/${scene.id}/messages` &&
          options?.method === "POST"
        ) {
          messageBody = JSON.parse(String(options.body));
          return jsonResponse(confirmed, 201);
        }
        throw new Error(`Unexpected request: ${String(input)}`);
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    mountApp();
    await flushAsyncUpdates();
    await openFirstScene();

    const recipient =
      container.querySelector<HTMLSelectElement>("#message-recipient");
    expect(recipient?.value).toBe("");
    expect(
      Array.from(recipient?.options ?? []).map((option) => option.value),
    ).toEqual(["", "B", "C"]);
    expect(findButton("确认发送").disabled).toBe(true);

    await setFieldValue(
      "#message-content",
      "  今晚在灯塔下见。  ",
    );
    expect(findButton("确认发送").disabled).toBe(true);

    await setSelectValue("#message-recipient", "B");
    expect(findButton("确认发送").disabled).toBe(false);
    findButton("确认发送").click();
    await flushAsyncUpdates();

    expect(messageBody).toEqual({
      sender_id: "A",
      recipient_id: "B",
      content: "今晚在灯塔下见。",
    });
    expect(
      container.querySelector<HTMLSelectElement>(
        "#message-recipient",
      )?.value,
    ).toBe("");
    expect(
      container.querySelector<HTMLTextAreaElement>(
        "#message-content",
      )?.value,
    ).toBe("");
    expect(container.textContent).toContain(
      "发送给 B：今晚在灯塔下见。",
    );

    findButton("Agent B").click();
    await nextTick();
    expect(container.textContent).toContain("A：今晚在灯塔下见。");

    findButton("Agent C").click();
    await nextTick();
    expect(container.textContent).toContain("时间线为空");
    expect(container.textContent).not.toContain("今晚在灯塔下见。");
  });

  it("keeps the draft and reports an error when message confirmation fails", async () => {
    const scene = makeScene();
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
        if (options?.method === "POST") {
          return jsonResponse(
            { detail: "Could not save scene: disk is full" },
            500,
          );
        }
        throw new Error(`Unexpected request: ${String(input)}`);
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    mountApp();
    await flushAsyncUpdates();
    await openFirstScene();
    await setSelectValue("#message-recipient", "C");
    await setFieldValue("#message-content", "不要丢掉这份草稿");

    findButton("确认发送").click();
    await flushAsyncUpdates();

    expect(
      container.querySelector<HTMLSelectElement>(
        "#message-recipient",
      )?.value,
    ).toBe("C");
    expect(
      container.querySelector<HTMLTextAreaElement>(
        "#message-content",
      )?.value,
    ).toBe("不要丢掉这份草稿");
    expect(container.textContent).toContain(
      "Could not save scene: disk is full",
    );
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

  it("blocks message confirmation until scene edits are saved", async () => {
    const scene = makeScene();
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
        if (options?.method === "PUT") {
          return jsonResponse({
            ...scene,
            agents: scene.agents.map((agent) =>
              agent.id === "A"
                ? { ...agent, persona: "尚未保存的人设" }
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
    await setFieldValue("#agent-persona", "尚未保存的人设");
    await setSelectValue("#message-recipient", "B");
    await setFieldValue("#message-content", "保存后才能发送");

    expect(container.textContent).toContain("请先保存场景");
    expect(findButton("确认发送").disabled).toBe(true);
    findButton("确认发送").click();
    await flushAsyncUpdates();
    expect(fetchMock).toHaveBeenCalledTimes(2);

    findButton("保存场景").click();
    await flushAsyncUpdates();

    expect(
      container.querySelector<HTMLSelectElement>(
        "#message-recipient",
      )?.value,
    ).toBe("B");
    expect(
      container.querySelector<HTMLTextAreaElement>(
        "#message-content",
      )?.value,
    ).toBe("保存后才能发送");
    expect(findButton("确认发送").disabled).toBe(false);
  });

  it("keeps a separate in-memory message draft for each agent", async () => {
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
    await setSelectValue("#message-recipient", "B");
    await setFieldValue("#message-content", "A 的草稿");

    findButton("Agent B").click();
    await nextTick();
    expect(
      container.querySelector<HTMLSelectElement>(
        "#message-recipient",
      )?.value,
    ).toBe("");
    await setSelectValue("#message-recipient", "C");
    await setFieldValue("#message-content", "B 的草稿");

    findButton("Agent A").click();
    await nextTick();
    expect(
      container.querySelector<HTMLSelectElement>(
        "#message-recipient",
      )?.value,
    ).toBe("B");
    expect(
      container.querySelector<HTMLTextAreaElement>(
        "#message-content",
      )?.value,
    ).toBe("A 的草稿");

    findButton("Agent B").click();
    await nextTick();
    expect(
      container.querySelector<HTMLSelectElement>(
        "#message-recipient",
      )?.value,
    ).toBe("C");
    expect(
      container.querySelector<HTMLTextAreaElement>(
        "#message-content",
      )?.value,
    ).toBe("B 的草稿");
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

  it("protects drafts on scene exit and discards them only after open succeeds", async () => {
    const first = makeScene(FIRST_ID, "第一个场景");
    const second = makeScene(SECOND_ID, "第二个场景");
    let secondOpenAttempts = 0;
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
        secondOpenAttempts += 1;
        return secondOpenAttempts === 1
          ? jsonResponse({ detail: "Scene temporarily unavailable" }, 500)
          : jsonResponse(second);
      }
      throw new Error(`Unexpected request: ${String(input)}`);
    });
    const confirmMock = vi.fn().mockReturnValue(true);
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("confirm", confirmMock);

    mountApp();
    await flushAsyncUpdates();
    await openFirstScene();
    await setSelectValue("#message-recipient", "B");
    await setFieldValue("#message-content", "还未确认的草稿");

    const secondButton =
      container.querySelectorAll<HTMLButtonElement>(
        ".scene-list-button",
      )[1];
    secondButton.click();
    await flushAsyncUpdates();

    expect(confirmMock).toHaveBeenCalledTimes(1);
    expect(container.textContent).toContain(
      "Scene temporarily unavailable",
    );
    expect(
      container.querySelector<HTMLTextAreaElement>(
        "#message-content",
      )?.value,
    ).toBe("还未确认的草稿");

    secondButton.click();
    await flushAsyncUpdates();

    expect(confirmMock).toHaveBeenCalledTimes(2);
    expect(
      container.querySelector<HTMLInputElement>("#scene-name")?.value,
    ).toBe("第二个场景");
    expect(
      container.querySelector<HTMLSelectElement>(
        "#message-recipient",
      )?.value,
    ).toBe("");
    expect(
      container.querySelector<HTMLTextAreaElement>(
        "#message-content",
      )?.value,
    ).toBe("");
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

  it("protects message drafts when creating a new scene", async () => {
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
    await setSelectValue("#message-recipient", "C");
    await setFieldValue("#message-content", "创建前的草稿");
    await setFieldValue("#new-scene-name", "新场景");

    findButton("创建").click();
    await flushAsyncUpdates();

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(
      container.querySelector<HTMLTextAreaElement>(
        "#message-content",
      )?.value,
    ).toBe("创建前的草稿");

    findButton("创建").click();
    await flushAsyncUpdates();

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(
      container.querySelector<HTMLInputElement>("#scene-name")?.value,
    ).toBe("新场景");
    expect(
      container.querySelector<HTMLTextAreaElement>(
        "#message-content",
      )?.value,
    ).toBe("");
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

    await setFieldValue("#message-content", "关闭前的草稿");
    const draftEvent = new Event("beforeunload", {
      cancelable: true,
    });
    window.dispatchEvent(draftEvent);
    expect(draftEvent.defaultPrevented).toBe(true);

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

  it("generates and replaces an editable draft while showing cache usage", async () => {
    const scene = makeScene();
    const generated = makeDraftResponse(
      "C",
      "模型选择发给迟夏的消息",
    );
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
          input ===
            `/api/scenes/${scene.id}/agents/A/message-drafts` &&
          options?.method === "POST"
        ) {
          return jsonResponse(generated);
        }
        throw new Error(`Unexpected request: ${String(input)}`);
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    mountApp();
    await flushAsyncUpdates();
    await openFirstScene();

    expect(findButton("生成草稿")).toBeTruthy();
    findButton("生成草稿").click();
    await flushAsyncUpdates();

    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      `/api/scenes/${scene.id}/agents/A/message-drafts`,
      { method: "POST" },
    );
    expect(
      container.querySelector<HTMLSelectElement>(
        "#message-recipient",
      )?.value,
    ).toBe("C");
    expect(
      container.querySelector<HTMLTextAreaElement>(
        "#message-content",
      )?.value,
    ).toBe("模型选择发给迟夏的消息");
    expect(findButton("重新生成")).toBeTruthy();
    expect(
      Array.from(
        container.querySelectorAll<HTMLElement>(".usage-metrics dd"),
      ).map((element) => element.textContent?.trim()),
    ).toEqual(["17", "91", "120", "48"]);

    await setSelectValue("#message-recipient", "B");
    await setFieldValue("#message-content", "人工编辑后的正文");
    expect(
      container.querySelector<HTMLTextAreaElement>(
        "#message-content",
      )?.value,
    ).toBe("人工编辑后的正文");
    expect(findButton("确认发送").disabled).toBe(false);
  });

  it("requires saved scene state before generating a draft", async () => {
    const scene = makeScene();
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse([{ id: scene.id, name: scene.name }]),
      )
      .mockResolvedValueOnce(jsonResponse(scene));
    vi.stubGlobal("fetch", fetchMock);

    mountApp();
    await flushAsyncUpdates();
    await openFirstScene();
    await setFieldValue("#agent-memory", "尚未保存的记忆");

    expect(container.textContent).toContain(
      "请先保存场景，再生成或确认发送。",
    );
    expect(findButton("生成草稿").disabled).toBe(true);
    findButton("生成草稿").click();
    await flushAsyncUpdates();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("locks conflicting editor actions while generation is pending", async () => {
    const scene = makeScene();
    let resolveDraft!: (response: Response) => void;
    const pendingDraft = new Promise<Response>((resolve) => {
      resolveDraft = resolve;
    });
    const fetchMock = vi.fn(
      (
        input: RequestInfo | URL,
        options?: RequestInit,
      ): Promise<Response> => {
        if (input === "/api/scenes" && options === undefined) {
          return Promise.resolve(
            jsonResponse([{ id: scene.id, name: scene.name }]),
          );
        }
        if (
          input === `/api/scenes/${scene.id}` &&
          options === undefined
        ) {
          return Promise.resolve(jsonResponse(scene));
        }
        if (options?.method === "POST") {
          return pendingDraft;
        }
        return Promise.reject(
          new Error(`Unexpected request: ${String(input)}`),
        );
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    mountApp();
    await flushAsyncUpdates();
    await openFirstScene();
    findButton("生成草稿").click();
    await nextTick();

    expect(findButton("生成中…").disabled).toBe(true);
    expect(findButton("确认发送").disabled).toBe(true);
    expect(findButton("Agent B").disabled).toBe(true);
    expect(
      container.querySelector<HTMLInputElement>("#scene-name")
        ?.disabled,
    ).toBe(true);
    expect(
      container.querySelector<HTMLTextAreaElement>("#agent-persona")
        ?.disabled,
    ).toBe(true);
    expect(
      container.querySelector<HTMLSelectElement>(
        "#message-recipient",
      )?.disabled,
    ).toBe(true);
    expect(
      container.querySelector<HTMLButtonElement>(
        ".scene-list-button",
      )?.disabled,
    ).toBe(true);

    resolveDraft(jsonResponse(makeDraftResponse("B", "等待后生成")));
    await flushAsyncUpdates();

    expect(findButton("重新生成").disabled).toBe(false);
    expect(
      container.querySelector<HTMLInputElement>("#scene-name")
        ?.disabled,
    ).toBe(false);
  });

  it("keeps the prior edited draft and usage when regeneration fails", async () => {
    const scene = makeScene();
    let generationCount = 0;
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
        if (options?.method === "POST") {
          generationCount += 1;
          return generationCount === 1
            ? jsonResponse(makeDraftResponse("B", "第一次草稿"))
            : jsonResponse({ detail: "模型暂时不可用" }, 502);
        }
        throw new Error(`Unexpected request: ${String(input)}`);
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    mountApp();
    await flushAsyncUpdates();
    await openFirstScene();
    findButton("生成草稿").click();
    await flushAsyncUpdates();
    await setFieldValue("#message-content", "保留人工编辑");

    findButton("重新生成").click();
    await flushAsyncUpdates();

    expect(
      container.querySelector<HTMLSelectElement>(
        "#message-recipient",
      )?.value,
    ).toBe("B");
    expect(
      container.querySelector<HTMLTextAreaElement>(
        "#message-content",
      )?.value,
    ).toBe("保留人工编辑");
    expect(container.textContent).toContain("模型暂时不可用");
    expect(
      Array.from(
        container.querySelectorAll<HTMLElement>(".usage-metrics dd"),
      ).map((element) => element.textContent?.trim()),
    ).toEqual(["17", "91", "120", "48"]);
  });

  it("isolates generated drafts per Agent and clears usage after confirmation", async () => {
    const scene = makeScene();
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
        if (String(input).endsWith("/agents/A/message-drafts")) {
          return jsonResponse(makeDraftResponse("B", "A 的模型草稿"));
        }
        if (String(input).endsWith("/agents/B/message-drafts")) {
          return jsonResponse(
            makeDraftResponse("C", "B 的模型草稿", 100),
          );
        }
        if (
          input === `/api/scenes/${scene.id}/messages` &&
          options?.method === "POST"
        ) {
          return jsonResponse(scene, 201);
        }
        throw new Error(`Unexpected request: ${String(input)}`);
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    mountApp();
    await flushAsyncUpdates();
    await openFirstScene();
    findButton("生成草稿").click();
    await flushAsyncUpdates();

    findButton("Agent B").click();
    await nextTick();
    findButton("生成草稿").click();
    await flushAsyncUpdates();
    expect(
      container.querySelector<HTMLTextAreaElement>(
        "#message-content",
      )?.value,
    ).toBe("B 的模型草稿");
    expect(container.textContent).toContain("117");

    findButton("Agent A").click();
    await nextTick();
    expect(
      container.querySelector<HTMLTextAreaElement>(
        "#message-content",
      )?.value,
    ).toBe("A 的模型草稿");
    expect(container.textContent).not.toContain("117");

    findButton("确认发送").click();
    await flushAsyncUpdates();
    expect(
      container.querySelector<HTMLTextAreaElement>(
        "#message-content",
      )?.value,
    ).toBe("");
    expect(container.querySelector(".usage-metrics")).toBeNull();

    findButton("Agent B").click();
    await nextTick();
    expect(
      container.querySelector<HTMLTextAreaElement>(
        "#message-content",
      )?.value,
    ).toBe("B 的模型草稿");
    expect(container.querySelector(".usage-metrics")).not.toBeNull();
  });
});
