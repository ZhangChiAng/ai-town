import { createApp, nextTick, type App as VueApp } from "vue";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App.vue";
import type { MessageDraftResponse, Scene } from "./types";

const SCENE_ID = "11111111-1111-4111-8111-111111111111";
const ANTHROPIC_MODEL = "anthropic/claude-test";
const RESPONSES_MODEL = "gpt-test";
const MODEL_OPTIONS = {
  options: [
    { protocol: "anthropic", model: ANTHROPIC_MODEL },
    { protocol: "responses", model: RESPONSES_MODEL },
  ],
};

function makeScene(model: string | null = ANTHROPIC_MODEL): Scene {
  return {
    schema_version: 5,
    id: SCENE_ID,
    name: "海边小镇",
    model,
    agents: (["A", "B", "C"] as const).map((id) => ({
      id,
      name: `居民 ${id}`,
      persona: "",
      desire: "",
      fear: "",
      memory: "",
      system_prompt: `SYSTEM ${id}`,
      timeline: [],
    })),
  };
}

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: vi.fn().mockResolvedValue(body),
  } as unknown as Response;
}

async function flush(): Promise<void> {
  await new Promise((resolve) => window.setTimeout(resolve, 0));
  await nextTick();
  await new Promise((resolve) => window.setTimeout(resolve, 0));
  await nextTick();
}

describe("App", () => {
  let container: HTMLDivElement;
  let app: VueApp<Element> | undefined;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.append(container);
  });

  afterEach(() => {
    app?.unmount();
    container.remove();
    vi.unstubAllGlobals();
  });

  async function mountOpenedScene(
    routes: Record<string, Response | (() => Response)>,
  ): Promise<ReturnType<typeof vi.fn>> {
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        const method = init?.method ?? "GET";
        const route = `${method} ${String(input)}`;
        if (route === "GET /api/model-options") {
          return jsonResponse(MODEL_OPTIONS);
        }
        const result = routes[route];
        if (result === undefined) {
          throw new Error(`Unexpected request: ${route}`);
        }
        return typeof result === "function" ? result() : result;
      },
    );
    vi.stubGlobal("fetch", fetchMock);
    app = createApp(App);
    app.mount(container);
    await flush();
    (
      [...container.querySelectorAll("button")].find((button) =>
        button.textContent?.includes("海边小镇"),
      ) as HTMLButtonElement
    ).click();
    await flush();
    return fetchMock;
  }

  it("shows authoritative timeline text without generated direction labels", async () => {
    const scene = makeScene();
    scene.agents[0].timeline.push({
      type: "message",
      message_id: "22222222-2222-4222-8222-222222222222",
      direction: "sent",
      counterpart_id: "B",
      content: "To B: 灯塔下见。",
    });

    await mountOpenedScene({
      "GET /api/scenes": jsonResponse([{ id: scene.id, name: scene.name }]),
      [`GET /api/scenes/${scene.id}`]: jsonResponse(scene),
    });

    expect(container.textContent).toContain("To B: 灯塔下见。");
    expect(container.textContent).not.toContain("发送给 B：");
    expect(container.textContent).not.toContain("发给 居民 B");
    expect(container.textContent).not.toContain("内心的声音");
  });

  it("edits and submits the complete To-prefixed draft unchanged", async () => {
    const scene = makeScene();
    const confirmed = makeScene();
    confirmed.agents[0].timeline.push({
      type: "message",
      message_id: "33333333-3333-4333-8333-333333333333",
      direction: "sent",
      counterpart_id: "C",
      content: "To C: 改去码头。",
    });
    confirmed.agents[2].timeline.push({
      type: "message",
      message_id: "33333333-3333-4333-8333-333333333333",
      direction: "received",
      counterpart_id: "A",
      content: "From A: 改去码头。",
    });
    const generated: MessageDraftResponse = {
      content: "To B: 灯塔下见。",
      reasoning: [
        {
          type: "thinking",
          text: "UNIQUE READONLY REASONING",
        },
        {
          type: "summary_text",
          text: "先做一个低压力的邀请。",
        },
      ],
      usage: {
        input_tokens: 10,
        output_tokens: 5,
        cache_creation_input_tokens: 2,
        cache_read_input_tokens: 3,
      },
      request_snapshot: {},
    };
    let sentBody: unknown;
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        const method = init?.method ?? "GET";
        const path = String(input);
        if (method === "GET" && path === "/api/scenes") {
          return jsonResponse([{ id: scene.id, name: scene.name }]);
        }
        if (method === "GET" && path === "/api/model-options") {
          return jsonResponse(MODEL_OPTIONS);
        }
        if (method === "GET" && path === `/api/scenes/${scene.id}`) {
          return jsonResponse(scene);
        }
        if (
          method === "POST" &&
          path === `/api/scenes/${scene.id}/agents/A/message-drafts`
        ) {
          return jsonResponse(generated);
        }
        if (method === "POST" && path === `/api/scenes/${scene.id}/messages`) {
          sentBody = JSON.parse(String(init?.body));
          return jsonResponse(confirmed, 201);
        }
        throw new Error(`Unexpected request: ${method} ${path}`);
      },
    );
    vi.stubGlobal("fetch", fetchMock);
    app = createApp(App);
    app.mount(container);
    await flush();
    (
      [...container.querySelectorAll("button")].find((button) =>
        button.textContent?.includes("海边小镇"),
      ) as HTMLButtonElement
    ).click();
    await flush();
    (
      [...container.querySelectorAll("button")].find((button) =>
        button.textContent?.includes("生成草稿"),
      ) as HTMLButtonElement
    ).click();
    await flush();

    const textarea = container.querySelector(
      "#message-content",
    ) as HTMLTextAreaElement;
    expect(textarea.value).toBe("To B: 灯塔下见。");
    expect(container.textContent).toContain("模型思维");
    expect(container.textContent).toContain("Claude thinking");
    expect(container.textContent).toContain("UNIQUE READONLY REASONING");
    expect(container.textContent).toContain("推理摘要");
    expect(container.textContent).toContain("缓存写入");
    expect(container.textContent).not.toContain("5 分钟缓存写入");
    textarea.value = "To C: 改去码头。";
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
    await nextTick();
    (
      [...container.querySelectorAll("button")].find((button) =>
        button.textContent?.includes("确认发送"),
      ) as HTMLButtonElement
    ).click();
    await flush();

    expect(sentBody).toEqual({
      sender_id: "A",
      content: "To C: 改去码头。",
    });
    expect(container.textContent).not.toContain("UNIQUE READONLY REASONING");
    expect(container.querySelector("#message-recipient")).toBeNull();
  });

  it("states when a generated response has no readable reasoning", async () => {
    const scene = makeScene();
    const generated: MessageDraftResponse = {
      content: "To B: 下午喝茶吗？",
      reasoning: [],
      usage: {
        input_tokens: 10,
        output_tokens: 5,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
      },
      request_snapshot: {},
    };
    await mountOpenedScene({
      "GET /api/scenes": jsonResponse([
        { id: scene.id, name: scene.name },
      ]),
      [`GET /api/scenes/${scene.id}`]: jsonResponse(scene),
      [`POST /api/scenes/${scene.id}/agents/A/message-drafts`]:
        jsonResponse(generated),
    });

    (
      [...container.querySelectorAll("button")].find((button) =>
        button.textContent?.includes("生成草稿"),
      ) as HTMLButtonElement
    ).click();
    await flush();

    expect(container.textContent).toContain(
      "本次响应没有返回可读思维内容。",
    );
  });

  it("renders Anthropic context and raw JSON from one snapshot", async () => {
    const scene = makeScene();
    const request = {
      model: "test-model",
      system: [{ type: "text", text: "SYSTEM A", cache_control: {} }],
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "From B: 你好。" },
            { type: "text", text: "FORMAT INSTRUCTION" },
          ],
        },
      ],
    };
    await mountOpenedScene({
      "GET /api/scenes": jsonResponse([{ id: scene.id, name: scene.name }]),
      [`GET /api/scenes/${scene.id}`]: jsonResponse(scene),
      [`GET /api/scenes/${scene.id}/agents/A/model-request-preview`]:
        jsonResponse({ request }),
    });

    const details = [...container.querySelectorAll("details")].find(
      (element) => element.textContent?.includes("模型请求预览"),
    ) as HTMLDetailsElement;
    details.open = true;
    (
      [...details.querySelectorAll("button")].find((button) =>
        button.textContent?.includes("加载预览"),
      ) as HTMLButtonElement
    ).click();
    await flush();

    expect(details.textContent).toContain("SYSTEM A");
    expect(details.textContent).toContain("From B: 你好。");
    expect(details.textContent).toContain("FORMAT INSTRUCTION");
    (
      [...details.querySelectorAll("button")].find((button) =>
        button.textContent?.includes("原始 JSON"),
      ) as HTMLButtonElement
    ).click();
    await nextTick();
    expect(details.querySelector(".request-preview > pre")?.textContent).toBe(
      JSON.stringify(request, null, 2),
    );
  });

  it("renders Responses instructions and input from one snapshot", async () => {
    const scene = makeScene();
    const request = {
      model: "gpt-test",
      instructions: "SYSTEM A",
      input: [
        {
          role: "assistant",
          content: [
            { type: "input_text", text: "To B: 之前的消息。" },
          ],
        },
        {
          role: "user",
          content: [
            { type: "input_text", text: "FORMAT INSTRUCTION" },
          ],
        },
      ],
      max_output_tokens: 2048,
      store: false,
    };
    await mountOpenedScene({
      "GET /api/scenes": jsonResponse([
        { id: scene.id, name: scene.name },
      ]),
      [`GET /api/scenes/${scene.id}`]: jsonResponse(scene),
      [`GET /api/scenes/${scene.id}/agents/A/model-request-preview`]:
        jsonResponse({ request }),
    });

    const details = [...container.querySelectorAll("details")].find(
      (element) => element.textContent?.includes("模型请求预览"),
    ) as HTMLDetailsElement;
    details.open = true;
    (
      [...details.querySelectorAll("button")].find((button) =>
        button.textContent?.includes("加载预览"),
      ) as HTMLButtonElement
    ).click();
    await flush();

    expect(details.textContent).toContain("instructions");
    expect(details.textContent).toContain("SYSTEM A");
    expect(details.textContent).toContain("assistant");
    expect(details.textContent).toContain("To B: 之前的消息。");
    expect(details.textContent).toContain("FORMAT INSTRUCTION");
    (
      [...details.querySelectorAll("button")].find((button) =>
        button.textContent?.includes("原始 JSON"),
      ) as HTMLButtonElement
    ).click();
    await nextTick();
    expect(details.querySelector(".request-preview > pre")?.textContent).toBe(
      JSON.stringify(request, null, 2),
    );
  });

  it("shows both actual model names with no creation default", async () => {
    let createdBody: unknown;
    const created = makeScene(RESPONSES_MODEL);
    created.name = "雨夜港口";
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        const method = init?.method ?? "GET";
        const path = String(input);
        if (method === "GET" && path === "/api/scenes") {
          return jsonResponse([]);
        }
        if (method === "GET" && path === "/api/model-options") {
          return jsonResponse(MODEL_OPTIONS);
        }
        if (method === "POST" && path === "/api/scenes") {
          createdBody = JSON.parse(String(init?.body));
          return jsonResponse(created, 201);
        }
        throw new Error(`Unexpected request: ${method} ${path}`);
      },
    );
    vi.stubGlobal("fetch", fetchMock);
    app = createApp(App);
    app.mount(container);
    await flush();

    const select = container.querySelector(
      "#new-scene-model",
    ) as HTMLSelectElement;
    expect(select.value).toBe("");
    expect(select.textContent).toContain(`Claude — ${ANTHROPIC_MODEL}`);
    expect(select.textContent).toContain(`Responses — ${RESPONSES_MODEL}`);
    const name = container.querySelector(
      "#new-scene-name",
    ) as HTMLInputElement;
    name.value = "雨夜港口";
    name.dispatchEvent(new Event("input", { bubbles: true }));
    select.value = RESPONSES_MODEL;
    select.dispatchEvent(new Event("change", { bubbles: true }));
    (
      container.querySelector(".create-button") as HTMLButtonElement
    ).click();
    await flush();

    expect(createdBody).toEqual({
      name: "雨夜港口",
      model: RESPONSES_MODEL,
    });
    expect(container.textContent).toContain(
      `Responses — ${RESPONSES_MODEL}`,
    );
    expect(container.querySelector("#legacy-scene-model")).toBeNull();
  });

  it("binds a legacy scene once and then renders the model read-only", async () => {
    const scene = makeScene(null);
    const bound = makeScene(RESPONSES_MODEL);
    const fetchMock = await mountOpenedScene({
      "GET /api/scenes": jsonResponse([{ id: scene.id, name: scene.name }]),
      [`GET /api/scenes/${scene.id}`]: jsonResponse(scene),
      [`PUT /api/scenes/${scene.id}/model`]: jsonResponse(bound),
    });

    expect(container.textContent).toContain("旧场景尚未绑定模型");
    const select = container.querySelector(
      "#legacy-scene-model",
    ) as HTMLSelectElement;
    expect(select.value).toBe("");
    select.value = RESPONSES_MODEL;
    select.dispatchEvent(new Event("change", { bubbles: true }));
    await nextTick();
    (
      [...container.querySelectorAll("button")].find((button) =>
        button.textContent?.includes("确认绑定"),
      ) as HTMLButtonElement
    ).click();
    await flush();

    const bindingCall = fetchMock.mock.calls.find(
      ([input, init]) =>
        String(input) === `/api/scenes/${scene.id}/model` &&
        init?.method === "PUT",
    );
    expect(JSON.parse(String(bindingCall?.[1]?.body))).toEqual({
      model: RESPONSES_MODEL,
    });
    expect(container.querySelector("#legacy-scene-model")).toBeNull();
    expect(container.textContent).toContain(
      `Responses — ${RESPONSES_MODEL}`,
    );
    expect(container.textContent).toContain("创建后固定，不可切换");
  });

  it("keeps manual messaging enabled when the bound model is unavailable", async () => {
    const scene = makeScene("gpt-retired");
    await mountOpenedScene({
      "GET /api/scenes": jsonResponse([{ id: scene.id, name: scene.name }]),
      [`GET /api/scenes/${scene.id}`]: jsonResponse(scene),
    });

    expect(container.textContent).toContain("当前配置不再提供此模型");
    const generate = [...container.querySelectorAll("button")].find(
      (button) => button.textContent?.includes("生成草稿"),
    ) as HTMLButtonElement;
    const preview = container.querySelector(
      ".preview-button",
    ) as HTMLButtonElement;
    const textarea = container.querySelector(
      "#message-content",
    ) as HTMLTextAreaElement;
    expect(generate.disabled).toBe(true);
    expect(preview.disabled).toBe(true);
    expect(textarea.disabled).toBe(false);
    textarea.value = "To B: 手工消息";
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
    await nextTick();
    expect(
      (container.querySelector(".message-send-button") as HTMLButtonElement)
        .disabled,
    ).toBe(false);
  });
});
