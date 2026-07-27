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
    system_prompt: `Agent ${id} 的最终系统提示词`,
    timeline: [],
  };
}

function makeScene(
  id = FIRST_ID,
  name = "雨夜港口",
): Scene {
  return {
    schema_version: 2,
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
    request_snapshot: {
      model: "test-model",
      system: [{ type: "text", text: `Agent ${recipientId}` }],
      messages: [],
      tools: [],
      tool_choice: {},
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

// ---- fetch stubbing helpers ----

type RouteHandler =
  | ((options: RequestInit | undefined) => Response | Promise<Response>)
  | Response
  | Promise<Response>;

function stubFetch(routes: Record<string, RouteHandler>) {
  const mock = vi.fn(
    async (input: RequestInfo | URL, options?: RequestInit): Promise<Response> => {
      const key = `${options?.method ?? "GET"} ${String(input)}`;
      const handler = routes[key];
      if (handler === undefined) {
        throw new Error(`Unexpected request: ${String(input)}`);
      }
      return typeof handler === "function" ? handler(options) : handler;
    },
  );
  vi.stubGlobal("fetch", mock);
  return mock;
}

// ---- DOM assertion helpers ----

function expectText(text: string): void {
  expect(container.textContent).toContain(text);
}

function expectNoText(text: string): void {
  expect(container.textContent).not.toContain(text);
}

function queryValue(selector: string): string | undefined {
  return container.querySelector<
    HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement
  >(selector)?.value;
}

function expectValue(selector: string, expected: string): void {
  expect(queryValue(selector)).toBe(expected);
}

function expectButtonDisabled(text: string): void {
  expect(findButton(text).disabled).toBe(true);
}

function expectButtonEnabled(text: string): void {
  expect(findButton(text).disabled).toBe(false);
}

// ---- lifecycle ----

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
    const fetchMock = stubFetch({
      "GET /api/scenes": jsonResponse([{ id: scene.id, name: scene.name }]),
      [`GET /api/scenes/${scene.id}`]: jsonResponse(scene),
    });

    mountApp();
    await flushAsyncUpdates();

    expect(fetchMock).toHaveBeenCalledWith("/api/scenes", undefined);
    expectText("雨夜港口");

    await openFirstScene();

    expect(fetchMock).toHaveBeenCalledWith(
      `/api/scenes/${scene.id}`,
      undefined,
    );
    expectValue("#scene-name", "雨夜港口");
    expectValue("#agent-name", "阿岚");

    findButton("Agent B").click();
    await nextTick();

    expectValue("#agent-name", "北辰");
    expectText("时间线为空");
  });

  it("creates a named scene and immediately opens it", async () => {
    const created = makeScene(FIRST_ID, "海边小镇");
    let firstListCall = true;
    const fetchMock = stubFetch({
      "GET /api/scenes": () => {
        if (firstListCall) {
          firstListCall = false;
          return jsonResponse([]);
        }
        return jsonResponse([{ id: created.id, name: created.name }]);
      },
      [`GET /api/scenes/${created.id}`]: jsonResponse(created),
      "POST /api/scenes": jsonResponse(created, 201),
    });

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
    expectValue("#scene-name", "海边小镇");
    expect(container.querySelectorAll(".scene-list-button")).toHaveLength(
      1,
    );
    expectText("当前内容已保存");
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
    stubFetch({
      "GET /api/scenes": jsonResponse([{ id: scene.id, name: scene.name }]),
      [`GET /api/scenes/${scene.id}`]: jsonResponse(scene),
      [`POST /api/scenes/${scene.id}/messages`]: (options) => {
        messageBody = JSON.parse(String(options!.body));
        return jsonResponse(confirmed, 201);
      },
    });

    mountApp();
    await flushAsyncUpdates();
    await openFirstScene();

    const recipient =
      container.querySelector<HTMLSelectElement>("#message-recipient");
    expect(recipient?.value).toBe("");
    expect(
      Array.from(recipient?.options ?? []).map((option) => option.value),
    ).toEqual(["", "B", "C"]);
    expectButtonDisabled("确认发送");

    await setFieldValue("#message-content", "  今晚在灯塔下见。  ");
    expectButtonDisabled("确认发送");

    await setSelectValue("#message-recipient", "B");
    expectButtonEnabled("确认发送");
    findButton("确认发送").click();
    await flushAsyncUpdates();

    expect(messageBody).toEqual({
      sender_id: "A",
      recipient_id: "B",
      content: "今晚在灯塔下见。",
    });
    expectValue("#message-recipient", "");
    expectValue("#message-content", "");
    expectText("发送给 B：今晚在灯塔下见。");

    findButton("Agent B").click();
    await nextTick();
    expectText("A：今晚在灯塔下见。");

    findButton("Agent C").click();
    await nextTick();
    expectText("时间线为空");
    expectNoText("今晚在灯塔下见。");
  });

  it("keeps the draft and reports an error when message confirmation fails", async () => {
    const scene = makeScene();
    stubFetch({
      "GET /api/scenes": jsonResponse([{ id: scene.id, name: scene.name }]),
      [`GET /api/scenes/${scene.id}`]: jsonResponse(scene),
      [`POST /api/scenes/${scene.id}/messages`]: jsonResponse(
        { detail: "Could not save scene: disk is full" },
        500,
      ),
    });

    mountApp();
    await flushAsyncUpdates();
    await openFirstScene();
    await setSelectValue("#message-recipient", "C");
    await setFieldValue("#message-content", "不要丢掉这份草稿");

    findButton("确认发送").click();
    await flushAsyncUpdates();

    expectValue("#message-recipient", "C");
    expectValue("#message-content", "不要丢掉这份草稿");
    expectText("Could not save scene: disk is full");
  });

  it("merges a created scene with an older list response", async () => {
    const existing = makeScene(SECOND_ID, "已有场景");
    const created = makeScene(FIRST_ID, "海边小镇");
    let resolveList!: (response: Response) => void;
    const pendingList = new Promise<Response>((resolve) => {
      resolveList = resolve;
    });
    stubFetch({
      "GET /api/scenes": pendingList,
      [`GET /api/scenes/${created.id}`]: jsonResponse(created),
      "POST /api/scenes": jsonResponse(created, 201),
    });

    mountApp();
    await nextTick();
    await setFieldValue("#new-scene-name", "海边小镇");

    findButton("创建").click();
    await flushAsyncUpdates();

    expect(container.querySelectorAll(".scene-list-button")).toHaveLength(
      1,
    );
    expectNoText("正在加载场景");

    resolveList(
      jsonResponse([{ id: existing.id, name: existing.name }]),
    );
    await flushAsyncUpdates();

    expect(container.querySelectorAll(".scene-list-button")).toHaveLength(
      2,
    );
    expectText("已有场景");
    expectText("海边小镇");
    expectNoText("正在加载场景");
  });

  it("keeps a created scene visible when an older list request fails", async () => {
    const created = makeScene(FIRST_ID, "海边小镇");
    let rejectList!: (reason: Error) => void;
    const pendingList = new Promise<Response>((_resolve, reject) => {
      rejectList = reject;
    });
    stubFetch({
      "GET /api/scenes": pendingList,
      [`GET /api/scenes/${created.id}`]: jsonResponse(created),
      "POST /api/scenes": jsonResponse(created, 201),
    });

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
    expectText("海边小镇");
    expectNoText("Scene storage is temporarily unavailable");
    expectNoText("重新加载");
  });

  it("marks edits dirty and writes only editable fields on explicit save", async () => {
    const scene = makeScene();
    let putBody: unknown;
    const fetchMock = stubFetch({
      "GET /api/scenes": jsonResponse([{ id: scene.id, name: scene.name }]),
      [`GET /api/scenes/${scene.id}`]: jsonResponse(scene),
      [`PUT /api/scenes/${scene.id}`]: (options) => {
        putBody = JSON.parse(String(options!.body));
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
      },
    });

    mountApp();
    await flushAsyncUpdates();
    await openFirstScene();
    await setFieldValue("#scene-name", "清晨港口");
    await setFieldValue("#agent-desire", "找到失踪的船");

    expectText("有未保存的更改");
    expect(fetchMock).toHaveBeenCalledTimes(2);

    findButton("保存场景").click();
    await flushAsyncUpdates();

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(putBody).toEqual({
      name: "清晨港口",
      agents: scene.agents.map(
        ({ id, name, persona, desire, fear, memory, system_prompt }) => ({
          id,
          name,
          persona,
          desire: id === "A" ? "找到失踪的船" : desire,
          fear,
          memory,
          system_prompt,
        }),
      ),
    });
    expect(putBody).not.toHaveProperty("id");
    expect(putBody).not.toHaveProperty("schema_version");
    expect(
      (putBody as { agents: Record<string, unknown>[] }).agents[0],
    ).not.toHaveProperty("timeline");
    expectText("保存成功");
    expect(container.querySelector<HTMLButtonElement>(".save-button")?.disabled).toBe(true);
  });

  it("keeps the selected agent and editor active after saving", async () => {
    const scene = makeScene();
    const savedMemory = "迟夏保存后的记忆";
    stubFetch({
      "GET /api/scenes": jsonResponse([{ id: scene.id, name: scene.name }]),
      [`GET /api/scenes/${scene.id}`]: jsonResponse(scene),
      [`PUT /api/scenes/${scene.id}`]: () =>
        jsonResponse({
          ...scene,
          agents: scene.agents.map((agent) =>
            agent.id === "C" ? { ...agent, memory: savedMemory } : agent,
          ),
        }),
    });

    mountApp();
    await flushAsyncUpdates();
    await openFirstScene();

    findButton("Agent C").click();
    await nextTick();
    await setFieldValue("#agent-memory", savedMemory);

    findButton("保存场景").click();
    await flushAsyncUpdates();

    expect(findButton("Agent C").getAttribute("aria-selected")).toBe("true");
    expectValue("#agent-name", "迟夏");
    expectValue("#agent-memory", savedMemory);
    expectText("保存成功");
  });

  it("keeps unsaved form content and reports an API save failure", async () => {
    const scene = makeScene();
    stubFetch({
      "GET /api/scenes": jsonResponse([{ id: scene.id, name: scene.name }]),
      [`GET /api/scenes/${scene.id}`]: jsonResponse(scene),
      [`PUT /api/scenes/${scene.id}`]: jsonResponse(
        { detail: "Could not save scene: disk is full" },
        500,
      ),
    });

    mountApp();
    await flushAsyncUpdates();
    await openFirstScene();
    await setFieldValue("#agent-memory", "这段内容尚未落盘");

    findButton("保存场景").click();
    await flushAsyncUpdates();

    expectValue("#agent-memory", "这段内容尚未落盘");
    expectText("Could not save scene: disk is full");
    expect(container.querySelector<HTMLButtonElement>(".save-button")?.disabled).toBe(false);
  });

  it("blocks message confirmation until scene edits are saved", async () => {
    const scene = makeScene();
    const fetchMock = stubFetch({
      "GET /api/scenes": jsonResponse([{ id: scene.id, name: scene.name }]),
      [`GET /api/scenes/${scene.id}`]: jsonResponse(scene),
      [`PUT /api/scenes/${scene.id}`]: () =>
        jsonResponse({
          ...scene,
          agents: scene.agents.map((agent) =>
            agent.id === "A" ? { ...agent, persona: "尚未保存的人设" } : agent,
          ),
        }),
    });

    mountApp();
    await flushAsyncUpdates();
    await openFirstScene();
    await setFieldValue("#agent-persona", "尚未保存的人设");
    await setSelectValue("#message-recipient", "B");
    await setFieldValue("#message-content", "保存后才能发送");

    expectText("请先保存场景");
    expectButtonDisabled("确认发送");
    findButton("确认发送").click();
    await flushAsyncUpdates();
    expect(fetchMock).toHaveBeenCalledTimes(2);

    findButton("保存场景").click();
    await flushAsyncUpdates();

    expectValue("#message-recipient", "B");
    expectValue("#message-content", "保存后才能发送");
    expectButtonEnabled("确认发送");
  });

  it("keeps a separate in-memory message draft for each agent", async () => {
    const scene = makeScene();
    stubFetch({
      "GET /api/scenes": jsonResponse([{ id: scene.id, name: scene.name }]),
      [`GET /api/scenes/${scene.id}`]: jsonResponse(scene),
    });

    mountApp();
    await flushAsyncUpdates();
    await openFirstScene();
    await setSelectValue("#message-recipient", "B");
    await setFieldValue("#message-content", "A 的草稿");

    findButton("Agent B").click();
    await nextTick();
    expectValue("#message-recipient", "");
    await setSelectValue("#message-recipient", "C");
    await setFieldValue("#message-content", "B 的草稿");

    findButton("Agent A").click();
    await nextTick();
    expectValue("#message-recipient", "B");
    expectValue("#message-content", "A 的草稿");

    findButton("Agent B").click();
    await nextTick();
    expectValue("#message-recipient", "C");
    expectValue("#message-content", "B 的草稿");
  });

  it("asks before switching away from dirty scene content", async () => {
    const first = makeScene(FIRST_ID, "第一个场景");
    const second = makeScene(SECOND_ID, "第二个场景");
    const fetchMock = stubFetch({
      "GET /api/scenes": jsonResponse([
        { id: first.id, name: first.name },
        { id: second.id, name: second.name },
      ]),
      [`GET /api/scenes/${first.id}`]: jsonResponse(first),
      [`GET /api/scenes/${second.id}`]: jsonResponse(second),
    });
    const confirmMock = vi
      .fn()
      .mockReturnValueOnce(false)
      .mockReturnValueOnce(true);
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
    expectValue("#scene-name", "尚未保存的名称");

    secondButton.click();
    await flushAsyncUpdates();

    expect(confirmMock).toHaveBeenCalledTimes(2);
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expectValue("#scene-name", "第二个场景");
  });

  it("protects drafts on scene exit and discards them only after open succeeds", async () => {
    const first = makeScene(FIRST_ID, "第一个场景");
    const second = makeScene(SECOND_ID, "第二个场景");
    let secondOpenAttempts = 0;
    stubFetch({
      "GET /api/scenes": jsonResponse([
        { id: first.id, name: first.name },
        { id: second.id, name: second.name },
      ]),
      [`GET /api/scenes/${first.id}`]: jsonResponse(first),
      [`GET /api/scenes/${second.id}`]: () => {
        secondOpenAttempts += 1;
        return secondOpenAttempts === 1
          ? jsonResponse({ detail: "Scene temporarily unavailable" }, 500)
          : jsonResponse(second);
      },
    });
    const confirmMock = vi.fn().mockReturnValue(true);
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
    expectText("Scene temporarily unavailable");
    expectValue("#message-content", "还未确认的草稿");

    secondButton.click();
    await flushAsyncUpdates();

    expect(confirmMock).toHaveBeenCalledTimes(2);
    expectValue("#scene-name", "第二个场景");
    expectValue("#message-recipient", "");
    expectValue("#message-content", "");
  });

  it("asks before creating a new scene when the editor is dirty", async () => {
    const existing = makeScene(FIRST_ID, "已有场景");
    const created = makeScene(SECOND_ID, "新场景");
    const fetchMock = stubFetch({
      "GET /api/scenes": jsonResponse([{ id: existing.id, name: existing.name }]),
      [`GET /api/scenes/${existing.id}`]: jsonResponse(existing),
      [`GET /api/scenes/${created.id}`]: jsonResponse(created),
      "POST /api/scenes": jsonResponse(created, 201),
    });
    const confirmMock = vi
      .fn()
      .mockReturnValueOnce(false)
      .mockReturnValueOnce(true);
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
    expectValue("#agent-fear", "未保存的恐惧");

    findButton("创建").click();
    await flushAsyncUpdates();

    expect(confirmMock).toHaveBeenCalledTimes(2);
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expectValue("#scene-name", "新场景");
  });

  it("protects message drafts when creating a new scene", async () => {
    const existing = makeScene(FIRST_ID, "已有场景");
    const created = makeScene(SECOND_ID, "新场景");
    const fetchMock = stubFetch({
      "GET /api/scenes": jsonResponse([{ id: existing.id, name: existing.name }]),
      [`GET /api/scenes/${existing.id}`]: jsonResponse(existing),
      [`GET /api/scenes/${created.id}`]: jsonResponse(created),
      "POST /api/scenes": jsonResponse(created, 201),
    });
    const confirmMock = vi
      .fn()
      .mockReturnValueOnce(false)
      .mockReturnValueOnce(true);
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
    expectValue("#message-content", "创建前的草稿");

    findButton("创建").click();
    await flushAsyncUpdates();

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expectValue("#scene-name", "新场景");
    expectValue("#message-content", "");
  });

  it("protects dirty content when the page is about to close", async () => {
    const scene = makeScene();
    stubFetch({
      "GET /api/scenes": jsonResponse([{ id: scene.id, name: scene.name }]),
      [`GET /api/scenes/${scene.id}`]: jsonResponse(scene),
    });

    mountApp();
    await flushAsyncUpdates();
    await openFirstScene();

    const cleanEvent = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(cleanEvent);
    expect(cleanEvent.defaultPrevented).toBe(false);

    await setFieldValue("#message-content", "关闭前的草稿");
    const draftEvent = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(draftEvent);
    expect(draftEvent.defaultPrevented).toBe(true);

    await setFieldValue("#agent-persona", "还没有保存的人设");
    const dirtyEvent = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(dirtyEvent);
    expect(dirtyEvent.defaultPrevented).toBe(true);
  });

  it("shows scene-list API errors and offers a retry", async () => {
    stubFetch({
      "GET /api/scenes": jsonResponse({ detail: "Scene storage is unreadable" }, 500),
    });

    mountApp();
    await flushAsyncUpdates();

    expectText("Scene storage is unreadable");
    expect(findButton("重新加载")).toBeTruthy();
  });

  it("generates and replaces an editable draft while showing cache usage", async () => {
    const scene = makeScene();
    const generated = makeDraftResponse("C", "模型选择发给迟夏的消息");
    const fetchMock = stubFetch({
      "GET /api/scenes": jsonResponse([{ id: scene.id, name: scene.name }]),
      [`GET /api/scenes/${scene.id}`]: jsonResponse(scene),
      [`POST /api/scenes/${scene.id}/agents/A/message-drafts`]: jsonResponse(generated),
    });

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
    expectValue("#message-recipient", "C");
    expectValue("#message-content", "模型选择发给迟夏的消息");
    expect(findButton("重新生成")).toBeTruthy();
    expect(
      Array.from(
        container.querySelectorAll<HTMLElement>(".usage-metrics dd"),
      ).map((element) => element.textContent?.trim()),
    ).toEqual(["17", "91", "120", "48"]);

    await setSelectValue("#message-recipient", "B");
    await setFieldValue("#message-content", "人工编辑后的正文");
    expectValue("#message-content", "人工编辑后的正文");
    expectButtonEnabled("确认发送");
  });

  it("requires saved scene state before generating a draft", async () => {
    const scene = makeScene();
    const fetchMock = stubFetch({
      "GET /api/scenes": jsonResponse([{ id: scene.id, name: scene.name }]),
      [`GET /api/scenes/${scene.id}`]: jsonResponse(scene),
    });

    mountApp();
    await flushAsyncUpdates();
    await openFirstScene();
    await setFieldValue("#agent-memory", "尚未保存的记忆");

    expectText("请先保存场景，再生成或确认发送。");
    expectButtonDisabled("生成草稿");
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
    stubFetch({
      "GET /api/scenes": jsonResponse([{ id: scene.id, name: scene.name }]),
      [`GET /api/scenes/${scene.id}`]: jsonResponse(scene),
      [`POST /api/scenes/${scene.id}/agents/A/message-drafts`]: pendingDraft,
    });

    mountApp();
    await flushAsyncUpdates();
    await openFirstScene();
    findButton("生成草稿").click();
    await nextTick();

    expectButtonDisabled("生成中…");
    expectButtonDisabled("确认发送");
    expectButtonDisabled("Agent B");
    expect(container.querySelector<HTMLInputElement>("#scene-name")?.disabled).toBe(true);
    expect(container.querySelector<HTMLTextAreaElement>("#agent-persona")?.disabled).toBe(true);
    expect(container.querySelector<HTMLSelectElement>("#message-recipient")?.disabled).toBe(true);
    expect(container.querySelector<HTMLButtonElement>(".scene-list-button")?.disabled).toBe(true);

    resolveDraft(jsonResponse(makeDraftResponse("B", "等待后生成")));
    await flushAsyncUpdates();

    expect(findButton("重新生成").disabled).toBe(false);
    expect(container.querySelector<HTMLInputElement>("#scene-name")?.disabled).toBe(false);
  });

  it("keeps the prior edited draft and usage when regeneration fails", async () => {
    const scene = makeScene();
    let generationCount = 0;
    stubFetch({
      "GET /api/scenes": jsonResponse([{ id: scene.id, name: scene.name }]),
      [`GET /api/scenes/${scene.id}`]: jsonResponse(scene),
      [`POST /api/scenes/${scene.id}/agents/A/message-drafts`]: () => {
        generationCount += 1;
        return generationCount === 1
          ? jsonResponse(makeDraftResponse("B", "第一次草稿"))
          : jsonResponse({ detail: "模型暂时不可用" }, 502);
      },
    });

    mountApp();
    await flushAsyncUpdates();
    await openFirstScene();
    findButton("生成草稿").click();
    await flushAsyncUpdates();
    await setFieldValue("#message-content", "保留人工编辑");

    findButton("重新生成").click();
    await flushAsyncUpdates();

    expectValue("#message-recipient", "B");
    expectValue("#message-content", "保留人工编辑");
    expectText("模型暂时不可用");
    expect(
      Array.from(
        container.querySelectorAll<HTMLElement>(".usage-metrics dd"),
      ).map((element) => element.textContent?.trim()),
    ).toEqual(["17", "91", "120", "48"]);
  });

  it("isolates generated drafts per Agent and clears usage after confirmation", async () => {
    const scene = makeScene();
    stubFetch({
      "GET /api/scenes": jsonResponse([{ id: scene.id, name: scene.name }]),
      [`GET /api/scenes/${scene.id}`]: jsonResponse(scene),
      [`POST /api/scenes/${scene.id}/agents/A/message-drafts`]: jsonResponse(
        makeDraftResponse("B", "A 的模型草稿"),
      ),
      [`POST /api/scenes/${scene.id}/agents/B/message-drafts`]: jsonResponse(
        makeDraftResponse("C", "B 的模型草稿", 100),
      ),
      [`POST /api/scenes/${scene.id}/messages`]: jsonResponse(scene, 201),
    });

    mountApp();
    await flushAsyncUpdates();
    await openFirstScene();
    findButton("生成草稿").click();
    await flushAsyncUpdates();

    findButton("Agent B").click();
    await nextTick();
    findButton("生成草稿").click();
    await flushAsyncUpdates();
    expectValue("#message-content", "B 的模型草稿");
    expectText("117");

    findButton("Agent A").click();
    await nextTick();
    expectValue("#message-content", "A 的模型草稿");
    expectNoText("117");

    findButton("确认发送").click();
    await flushAsyncUpdates();
    expectValue("#message-content", "");
    expect(container.querySelector(".usage-metrics")).toBeNull();

    findButton("Agent B").click();
    await nextTick();
    expectValue("#message-content", "B 的模型草稿");
    expect(container.querySelector(".usage-metrics")).not.toBeNull();
  });

  it("recomposes only after confirmation and preserves manual prompt edits", async () => {
    const scene = makeScene();
    const confirmMock = vi
      .spyOn(window, "confirm")
      .mockReturnValueOnce(false)
      .mockReturnValueOnce(true);
    stubFetch({
      "GET /api/scenes": jsonResponse([{ id: scene.id, name: scene.name }]),
      [`GET /api/scenes/${scene.id}`]: jsonResponse(scene),
      "POST /api/system-prompts/compose": jsonResponse({
        system_prompt: "后端模板拼接结果",
      }),
    });

    mountApp();
    await flushAsyncUpdates();
    await openFirstScene();
    await setFieldValue("#agent-persona", "修改后的拼接素材");

    findButton("从槽位重新拼接").click();
    await nextTick();
    expectValue("#agent-system-prompt", "Agent A 的最终系统提示词");

    findButton("从槽位重新拼接").click();
    await flushAsyncUpdates();
    expect(confirmMock).toHaveBeenCalledTimes(2);
    expectValue("#agent-system-prompt", "后端模板拼接结果");

    await setFieldValue("#agent-system-prompt", "用户逐字手工版本");
    expectValue("#agent-system-prompt", "用户逐字手工版本");
  });

  it("rejects an empty final system prompt before saving", async () => {
    const scene = makeScene();
    const fetchMock = stubFetch({
      "GET /api/scenes": jsonResponse([{ id: scene.id, name: scene.name }]),
      [`GET /api/scenes/${scene.id}`]: jsonResponse(scene),
    });

    mountApp();
    await flushAsyncUpdates();
    await openFirstScene();
    await setFieldValue("#agent-system-prompt", " \n ");
    findButton("保存场景").click();
    await nextTick();

    expectText("三个 Agent 的最终系统提示词均不能为空");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("marks saved-scene previews stale and renders readable and raw input", async () => {
    const scene = makeScene();
    const request = {
      model: "observable-model",
      max_tokens: 512,
      system: [
        {
          type: "text",
          text: "PREVIEW_SYSTEM_TEXT",
          cache_control: { type: "ephemeral", ttl: "5m" },
        },
      ],
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "From B: 你今晚有空吗？" },
          ],
        },
        {
          role: "assistant",
          content: [
            {
              type: "text",
              text: "To B: 有空，灯塔见。",
              cache_control: { type: "ephemeral", ttl: "5m" },
            },
          ],
        },
      ],
      tools: [{ name: "compose_message", strict: true }],
      tool_choice: { type: "tool", name: "compose_message" },
    };
    stubFetch({
      "GET /api/scenes": jsonResponse([{ id: scene.id, name: scene.name }]),
      [`GET /api/scenes/${scene.id}`]: jsonResponse(scene),
      [`GET /api/scenes/${scene.id}/agents/A/model-request-preview`]: jsonResponse({ request }),
    });

    mountApp();
    await flushAsyncUpdates();
    await openFirstScene();
    await setFieldValue("#agent-desire", "尚未保存的修改");
    findButton("加载预览").click();
    await flushAsyncUpdates();

    expectText("当前显示为旧版本");
    expectText("PREVIEW_SYSTEM_TEXT");
    expectText("observable-model");
    expectText("user");
    expectText("From B: 你今晚有空吗？");
    expectText("assistant");
    expectText("To B: 有空，灯塔见。");
    expectText("工具输出约束与强制选择");
    expectText("缓存断点");
    expectNoText("当前 Agent\nID:");
    expectNoText("候选接收人");
    expectNoText("个人时间线 1");
  });
});
