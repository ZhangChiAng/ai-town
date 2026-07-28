import { createApp, nextTick, type App as VueApp } from "vue";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App.vue";
import type { MessageDraftResponse, Scene } from "./types";

const SCENE_ID = "11111111-1111-4111-8111-111111111111";

function makeScene(): Scene {
  return {
    schema_version: 4,
    id: SCENE_ID,
    name: "海边小镇",
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
    expect(container.querySelector("#message-recipient")).toBeNull();
  });

  it("renders readable context and raw JSON from one preview snapshot", async () => {
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
});
