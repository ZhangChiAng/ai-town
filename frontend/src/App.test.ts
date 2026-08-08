import {
  createApp,
  nextTick,
  type App as VueApp,
} from "vue";
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";

import App from "./App.vue";
import type {
  ExternalEvent,
  Interactions,
  LayerDraftResponse,
  ModelOption,
  ModelRequestPreviewResponse,
  Scene,
} from "./types";

const SCENE_ID = "11111111-1111-4111-8111-111111111111";
const EVENT_ID = "22222222-2222-4222-8222-222222222222";
const INNER_CALL_ID = "33333333-3333-4333-8333-333333333333";
const OUTER_CALL_ID = "44444444-4444-4444-8444-444444444444";
const GENERATED_EVENT_ID = "55555555-5555-4555-8555-555555555555";
const STATE_TOKEN = "a".repeat(64);
const MODEL = "anthropic/claude-test";

type RouteHandler =
  | Response
  | ((init?: RequestInit) => Response | Promise<Response>);

function interactionsFor(id: "A" | "B" | "C"): Interactions {
  if (id === "A") {
    return {
      B: {
        description: "你的儿子，最近工作不顺。",
        addresses: { 儿子: "一般场合", 小名: "亲昵场合" },
      },
      C: {
        description: "住在隔壁的邻居。",
        addresses: { 邻居: "邻里场合" },
      },
    };
  }
  return id === "B"
    ? {
        A: {
          description: "你的母亲。",
          addresses: { 母亲: "家庭场合" },
        },
      }
    : {
        A: {
          description: "住在隔壁的邻居。",
          addresses: { 邻居: "邻里场合" },
        },
      };
}

function makeScene(): Scene {
  return {
    schema: "ai-town.scene/1.0",
    id: SCENE_ID,
    name: "海边小镇",
    model: MODEL,
    agents: (["A", "B", "C"] as const).map((id) => ({
      id,
      name: `居民 ${id}`,
      prompt_profile: {
        pronoun: "她",
        hidden_beliefs: `HIDDEN ${id}`,
        inner_memories: `INNER ${id}`,
        outer_memories: `OUTER ${id}`,
      },
      interactions: interactionsFor(id),
      inner_context: {
        turns: [],
      },
      outer_context: {
        turns: [],
      },
      pending_events: [],
    })),
    rollback_stack: [],
    next_sequence: 1,
  };
}

function manualEvent(
  content = "海面突然起雾。",
  id = EVENT_ID,
  sequence = 1,
): ExternalEvent {
  return {
    id,
    sequence,
    kind: "manual",
    content,
    source_agent_id: null,
    source_call_id: null,
  };
}

function pendingScene(): Scene {
  const scene = makeScene();
  scene.agents[0].pending_events.push(manualEvent());
  scene.next_sequence = 2;
  return scene;
}

function halfRoundScene(): Scene {
  const scene = makeScene();
  scene.agents[0].inner_context.turns.push({
    call_id: INNER_CALL_ID,
    event_ids: [EVENT_ID],
    sequence: 2,
    input: "外部事件：\n海面突然起雾。",
    output: "先观察风向。\n别急着靠岸。",
    consumed_events: [manualEvent()],
    reasoning: [],
  });
  scene.rollback_stack.push({
    call_id: INNER_CALL_ID,
    agent_id: "A",
    layer: "inner",
  });
  scene.next_sequence = 3;
  return scene;
}

function completeScene(): Scene {
  const scene = halfRoundScene();
  scene.agents[0].outer_context.turns.push({
    call_id: OUTER_CALL_ID,
    event_ids: [EVENT_ID],
    sequence: 3,
    input:
      "外部事件：\n海面突然起雾。\n\n" +
      "你内心有一个声音：\n先观察风向。\n别急着靠岸。",
    output: "对儿子说：今晚先别出海。",
    recipient_id: "B",
    generated_event_id: GENERATED_EVENT_ID,
    reasoning: [],
  });
  scene.agents[1].pending_events.push({
    id: GENERATED_EVENT_ID,
    sequence: 3,
    kind: "agent_message",
    content: "今晚先别出海。",
    source_agent_id: "A",
    source_call_id: OUTER_CALL_ID,
  });
  scene.rollback_stack.push({
    call_id: OUTER_CALL_ID,
    agent_id: "A",
    layer: "outer",
  });
  scene.next_sequence = 4;
  return scene;
}

function historyReasoningScene(): Scene {
  const scene = completeScene();
  scene.agents[0].inner_context.turns[0].reasoning = [
    { type: "thinking", text: "历史内层思考" },
  ];
  scene.agents[0].outer_context.turns[0].reasoning = [
    { type: "summary_text", text: "历史外层总结" },
  ];
  return scene;
}

function innerDraft(): LayerDraftResponse {
  return {
    layer: "inner",
    call_id: INNER_CALL_ID,
    event_ids: [EVENT_ID],
    content: "先观察风向。",
    reasoning: [
      { type: "thinking", text: "临时判断，不应持久化" },
    ],
    usage: {
      input_tokens: 10,
      output_tokens: 5,
      cache_creation_input_tokens: 2,
      cache_read_input_tokens: 3,
    },
    request_snapshot: {
      model: "test-model",
      system: [{ type: "text", text: "INNER A" }],
      messages: [
        {
          role: "user",
          content: [{ type: "text", text: "外部事件：\n海面突然起雾。" }],
        },
      ],
    },
    state_token: STATE_TOKEN,
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

function findButton(
  container: HTMLElement,
  text: string,
): HTMLButtonElement {
  const button = [...container.querySelectorAll("button")].find(
    (candidate) => candidate.textContent?.includes(text),
  );
  if (button === undefined) {
    throw new Error(`Button not found: ${text}`);
  }
  return button as HTMLButtonElement;
}

function findButtonByAriaLabel(
  container: HTMLElement,
  label: string,
): HTMLButtonElement {
  const button = container.querySelector(`button[aria-label="${label}"]`);
  if (button === null) {
    throw new Error(`Button not found by aria-label: ${label}`);
  }
  return button as HTMLButtonElement;
}

function emptyResponse(status = 204): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    content: "",
    text: vi.fn().mockResolvedValue(""),
    json: vi.fn().mockResolvedValue(null),
  } as unknown as Response;
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
    vi.restoreAllMocks();
  });

  async function mountOpenedScene(
    scene: Scene,
    routes: Record<string, RouteHandler> = {},
    availableModels: () => ModelOption[] = () => [
      { model: MODEL },
      { model: "gpt-test" },
    ],
  ): Promise<ReturnType<typeof vi.fn>> {
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        const method = init?.method ?? "GET";
        const path = String(input);
        if (method === "GET" && path === "/api/scenes") {
          return jsonResponse([{ id: scene.id, name: scene.name }]);
        }
        if (method === "GET" && path === "/api/model-options") {
          return jsonResponse({ options: availableModels() });
        }
        if (method === "GET" && path === `/api/scenes/${scene.id}`) {
          return jsonResponse(scene);
        }
        const handler = routes[`${method} ${path}`];
        if (handler === undefined) {
          throw new Error(`Unexpected request: ${method} ${path}`);
        }
        return typeof handler === "function"
          ? await handler(init)
          : handler;
      },
    );
    vi.stubGlobal("fetch", fetchMock);
    app = createApp(App);
    app.mount(container);
    await flush();
    findButton(container, scene.name).click();
    await flush();
    return fetchMock;
  }

  it("creates a scene with the explicitly selected model", async () => {
    const created = makeScene();
    created.name = "新场景";
    created.model = "gpt-test";
    let createBody: unknown;
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        const method = init?.method ?? "GET";
        const path = String(input);
        if (method === "GET" && path === "/api/scenes") {
          return jsonResponse([]);
        }
        if (method === "GET" && path === "/api/model-options") {
          return jsonResponse({
            options: [{ model: MODEL }, { model: "gpt-test" }],
          });
        }
        if (method === "POST" && path === "/api/scenes") {
          createBody = JSON.parse(String(init?.body));
          return jsonResponse(created, 201);
        }
        throw new Error(`Unexpected request: ${method} ${path}`);
      },
    );
    vi.stubGlobal("fetch", fetchMock);
    app = createApp(App);
    app.mount(container);
    await flush();

    const name = container.querySelector(
      "#new-scene-name",
    ) as HTMLInputElement;
    const model = container.querySelector(
      '[aria-label="新场景模型"]',
    ) as HTMLSelectElement;
    name.value = "新场景";
    name.dispatchEvent(new Event("input", { bubbles: true }));
    model.value = "gpt-test";
    model.dispatchEvent(new Event("change", { bubbles: true }));
    await nextTick();
    findButton(container, "创建").click();
    await flush();

    expect(createBody).toEqual({ name: "新场景", model: "gpt-test" });
    expect(container.textContent).toContain("gpt-test");
    expect([...model.options].map((option) => option.text)).toEqual([
      MODEL,
      "gpt-test",
    ]);
  });

  it("rejects model options containing internal metadata", async () => {
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL) => {
        const path = String(input);
        if (path === "/api/scenes") {
          return jsonResponse([]);
        }
        if (path === "/api/model-options") {
          return jsonResponse({
            options: [{ model: MODEL, internal_transport: "hidden" }],
          });
        }
        throw new Error(`Unexpected request: GET ${path}`);
      },
    );
    vi.stubGlobal("fetch", fetchMock);
    app = createApp(App);
    app.mount(container);
    await flush();

    const model = container.querySelector(
      '[aria-label="新场景模型"]',
    ) as HTMLSelectElement;
    expect(model.options).toHaveLength(0);
    expect(model.disabled).toBe(true);
    expect(container.textContent).toContain(
      "后端返回了无法识别的模型列表。",
    );
  });

  it("opens scenes from any valid current-major schema", async () => {
    const scene = pendingScene();
    scene.schema = "ai-town.scene/1.999";

    await mountOpenedScene(scene);

    expect(container.textContent).toContain(scene.name);
    expect(findButton(container, "生成内层草稿").disabled).toBe(false);
  });

  it("rejects scene responses with fields outside the contract", async () => {
    const scene = pendingScene() as Scene & { unexpected?: boolean };
    scene.unexpected = true;

    await mountOpenedScene(scene);

    expect(container.textContent).toContain(
      "后端返回了无法识别的场景数据。",
    );
    expect(container.querySelector(".scene-toolbar")).toBeNull();
  });

  it("keeps editing available while a bound model is unavailable", async () => {
    const scene = pendingScene();
    scene.model = "removed-model";
    await mountOpenedScene(scene);

    expect(container.textContent).toContain("removed-model 当前不可用");
    expect(findButton(container, "生成内层草稿").disabled).toBe(true);
    expect(findButton(container, "添加到队尾").disabled).toBe(false);
    expect(findButton(container, "加载预览").disabled).toBe(true);
  });

  it("allows only A to be configured and blocks blank B independently", async () => {
    const scene = pendingScene();
    scene.agents[1].prompt_profile = {
      pronoun: "",
      hidden_beliefs: "",
      inner_memories: "",
      outer_memories: "",
    };
    scene.agents[1].interactions = {};
    scene.agents[1].pending_events.push(
      manualEvent("B 的事件", "99999999-9999-4999-8999-999999999999", 2),
    );
    scene.next_sequence = 3;
    await mountOpenedScene(scene);

    expect(findButton(container, "生成内层草稿").disabled).toBe(false);
    findButton(container, "居民 B").click();
    await nextTick();

    expect(findButton(container, "生成内层草稿").disabled).toBe(true);
    expect(container.textContent).toContain(
      "请填写四个提示词变量。",
    );
    expect(findButton(container, "加载预览").disabled).toBe(true);
  });

  it("blocks model controls when an addressed person lacks a description", async () => {
    const scene = pendingScene();
    const relationship = scene.agents[0].interactions.B;
    if (relationship === undefined) {
      throw new Error("fixture relationship missing");
    }
    relationship.description = "";

    await mountOpenedScene(scene);

    expect(findButton(container, "生成内层草稿").disabled).toBe(true);
    expect(findButton(container, "加载预览").disabled).toBe(true);
    expect(container.textContent).toContain(
      "请补全所有已配置互动人物的简介。",
    );
  });

  it("rejects duplicate trimmed person names before saving", async () => {
    const scene = makeScene();
    const fetchMock = await mountOpenedScene(scene);
    const settings = container.querySelector(
      ".settings-card",
    ) as HTMLDetailsElement;
    settings.open = true;
    const name = settings.querySelector("#agent-name") as HTMLInputElement;
    name.value = " 居民 B ";
    name.dispatchEvent(new Event("input", { bubbles: true }));
    await nextTick();

    findButton(container, "保存设定").click();
    await flush();

    expect(container.textContent).toContain("三位人物的姓名不能重复。");
    expect(fetchMock).not.toHaveBeenCalledWith(
      `/api/scenes/${SCENE_ID}`,
      expect.objectContaining({ method: "PUT" }),
    );
  });

  it("edits prompt variables and ordered interactions in the scene payload", async () => {
    const scene = makeScene();
    let savedBody: unknown;
    await mountOpenedScene(scene, {
      [`PUT /api/scenes/${SCENE_ID}`]: (init) => {
        savedBody = JSON.parse(String(init?.body));
        const update = savedBody as {
          name: string;
          agents: Scene["agents"];
        };
        const updated = structuredClone(scene);
        updated.name = update.name;
        updated.agents.forEach((agent, index) => {
          agent.name = update.agents[index].name;
          agent.prompt_profile = update.agents[index].prompt_profile;
          agent.interactions = update.agents[index].interactions;
        });
        return jsonResponse(updated);
      },
    });

    const settings = container.querySelector(
      ".settings-card",
    ) as HTMLDetailsElement;
    settings.open = true;
    const pronoun = settings.querySelector(
      "#agent-pronoun",
    ) as HTMLInputElement;
    pronoun.value = "她自己";
    pronoun.dispatchEvent(new Event("input", { bubbles: true }));

    expect(settings.textContent).toContain("居民 B");
    expect(settings.textContent).not.toContain("Agent B");
    const description = settings.querySelector(
      '[aria-label="居民 B的人物简介"]',
    ) as HTMLTextAreaElement;
    description.value = "你的成年儿子，最近工作不顺。";
    description.dispatchEvent(new Event("input", { bubbles: true }));

    const firstAddress = settings.querySelector(
      ".interaction-row input",
    ) as HTMLInputElement;
    firstAddress.value = "孩子";
    firstAddress.dispatchEvent(new Event("change", { bubbles: true }));

    const addInputs = settings.querySelectorAll(
      ".interaction-add input",
    );
    const newAddress = addInputs[0] as HTMLInputElement;
    const newOccasion = addInputs[1] as HTMLInputElement;
    newAddress.value = "宝贝";
    newAddress.dispatchEvent(new Event("input", { bubbles: true }));
    newOccasion.value = "安慰场合";
    newOccasion.dispatchEvent(new Event("input", { bubbles: true }));
    findButton(settings, "添加称呼").click();
    await nextTick();

    const nicknameRow = [...settings.querySelectorAll(".interaction-row")].find(
      (row) =>
        (row.querySelector("input") as HTMLInputElement).value === "小名",
    ) as HTMLElement;
    findButton(nicknameRow, "删除").click();
    await nextTick();
    findButton(container, "保存设定").click();
    await flush();

    const savedUpdate = savedBody as {
      name: string;
      agents: Scene["agents"];
    };
    expect(savedUpdate.name).toBe(scene.name);
    expect(savedUpdate.agents[0]).toMatchObject({
      id: "A",
      prompt_profile: {
        pronoun: "她自己",
        hidden_beliefs: "HIDDEN A",
        inner_memories: "INNER A",
        outer_memories: "OUTER A",
      },
      interactions: {
        B: {
          description: "你的成年儿子，最近工作不顺。",
          addresses: {
            孩子: "一般场合",
            宝贝: "安慰场合",
          },
        },
        C: {
          description: "住在隔壁的邻居。",
          addresses: { 邻居: "邻里场合" },
        },
      },
    });
    expect(JSON.stringify(savedBody)).not.toContain("system_prompt");
  });

  it("renders one mixed history with explicit neutral, inner, and outer labels", async () => {
    await mountOpenedScene(completeScene());

    const timeline = container.querySelector(".timeline") as HTMLElement;
    expect(timeline.textContent).toContain("外部事件");
    expect(timeline.textContent).toContain("内层输出");
    expect(timeline.textContent).toContain("外层输出");
    expect(timeline.querySelectorAll(".timeline-item--event")).toHaveLength(1);
    expect(timeline.querySelectorAll(".timeline-item--inner")).toHaveLength(1);
    expect(timeline.querySelectorAll(".timeline-item--outer")).toHaveLength(1);
    expect(timeline.querySelectorAll(".call-input")).toHaveLength(2);
    expect(
      timeline.querySelector(".timeline-item--event article > p")
        ?.textContent,
    ).toBe("海面突然起雾。");

    expect(container.querySelector("#agent-persona")).toBeNull();
    expect(container.querySelector("#agent-desire")).toBeNull();
    expect(container.querySelector("#agent-fear")).toBeNull();
    expect(container.querySelector("#agent-memory")).toBeNull();
    expect(container.querySelector("#message-content")).toBeNull();
  });

  it("renders persisted reasoning blocks inside the scene history", async () => {
    await mountOpenedScene(historyReasoningScene());

    const timeline = container.querySelector(".timeline") as HTMLElement;
    const blocks = timeline.querySelectorAll(".draft-reasoning");
    expect(blocks).toHaveLength(2);
    expect(
      timeline.querySelector(".timeline-item--inner .draft-reasoning")
        ?.textContent,
    ).toContain("历史内层思考");
    expect(
      timeline.querySelector(".timeline-item--outer .draft-reasoning")
        ?.textContent,
    ).toContain("历史外层总结");
    expect(
      timeline.querySelectorAll(".timeline-item--event .draft-reasoning"),
    ).toHaveLength(0);
  });

  it("does not render reasoning when the scene history holds none", async () => {
    const scene = completeScene();
    scene.agents[0].inner_context.turns[0].reasoning = [];
    scene.agents[0].outer_context.turns[0].reasoning = [];
    await mountOpenedScene(scene);

    const timeline = container.querySelector(".timeline") as HTMLElement;
    expect(timeline.querySelectorAll(".draft-reasoning")).toHaveLength(0);
  });

  it("edits and confirms an inner draft, then exposes only the outer stage", async () => {
    const initial = pendingScene();
    const confirmed = halfRoundScene();
    let confirmationBody: unknown;
    await mountOpenedScene(initial, {
      [`POST /api/scenes/${SCENE_ID}/agents/A/inner-drafts`]:
        jsonResponse(innerDraft()),
      [`POST /api/scenes/${SCENE_ID}/agents/A/inner-confirmations`]: (
        init,
      ) => {
        confirmationBody = JSON.parse(String(init?.body));
        return jsonResponse(confirmed);
      },
    });

    expect(findButton(container, "生成内层草稿").disabled).toBe(false);
    findButton(container, "生成内层草稿").click();
    await flush();

    const textarea = container.querySelector(
      "#layer-draft-content",
    ) as HTMLTextAreaElement;
    expect(textarea.value).toBe("先观察风向。");
    expect(container.querySelector(".draft-reasoning")?.textContent).toContain(
      "临时判断，不应持久化",
    );
    expect(container.querySelector(".draft-request pre")?.textContent).toBe(
      JSON.stringify(innerDraft().request_snapshot, null, 2),
    );
    textarea.value = "先观察。\n也许只是天气。";
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
    await nextTick();
    findButton(container, "确认内层").click();
    await flush();

    expect(confirmationBody).toEqual({
      call_id: INNER_CALL_ID,
      event_ids: [EVENT_ID],
      content: "先观察。\n也许只是天气。",
      state_token: STATE_TOKEN,
      reasoning: innerDraft().reasoning,
    });
    expect(container.textContent).toContain("等待外层人格");
    expect(container.querySelector(".draft-reasoning")).toBeNull();
    expect(findButton(container, "生成外层草稿").disabled).toBe(false);
    expect(
      [...container.querySelectorAll("button")].some((button) =>
        button.textContent?.includes("确认发送"),
      ),
    ).toBe(false);
  });

  it("keeps the edited draft when regeneration fails", async () => {
    let generationCount = 0;
    await mountOpenedScene(pendingScene(), {
      [`POST /api/scenes/${SCENE_ID}/agents/A/inner-drafts`]: () => {
        generationCount += 1;
        return generationCount === 1
          ? jsonResponse(innerDraft())
          : jsonResponse({ detail: "Model request failed." }, 502);
      },
    });

    findButton(container, "生成内层草稿").click();
    await flush();
    const textarea = container.querySelector(
      "#layer-draft-content",
    ) as HTMLTextAreaElement;
    textarea.value = "用户保留的修改";
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
    await nextTick();
    findButton(container, "重新生成").click();
    await flush();

    expect(textarea.value).toBe("用户保留的修改");
    expect(container.textContent).toContain("Model request failed.");
    expect(generationCount).toBe(2);
  });

  it("confirms an existing draft after its model is removed", async () => {
    let availableModels: ModelOption[] = [
      { model: MODEL },
      { model: "gpt-test" },
    ];
    let generationCount = 0;
    let optionsRequestCount = 0;
    let confirmationBody: unknown;
    await mountOpenedScene(
      pendingScene(),
      {
        [`POST /api/scenes/${SCENE_ID}/agents/A/inner-drafts`]: () => {
          generationCount += 1;
          return jsonResponse(innerDraft());
        },
        [`POST /api/scenes/${SCENE_ID}/agents/A/inner-confirmations`]: (
          init,
        ) => {
          confirmationBody = JSON.parse(String(init?.body));
          return jsonResponse(halfRoundScene());
        },
      },
      () => {
        optionsRequestCount += 1;
        // Return a fresh array, matching a real JSON response boundary.
        return availableModels.map((option) => ({ ...option }));
      },
    );
    findButton(container, "生成内层草稿").click();
    await flush();

    availableModels = [{ model: "gpt-test" }];
    findButton(container, "重新生成").click();
    await flush();

    expect(container.textContent).toContain(`${MODEL} 当前不可用`);
    expect(generationCount).toBe(1);
    expect(optionsRequestCount).toBeGreaterThanOrEqual(3);
    expect(findButton(container, "重新生成").disabled).toBe(true);
    expect(findButton(container, "加载预览").disabled).toBe(true);
    expect(findButton(container, "确认内层").disabled).toBe(false);
    findButton(container, "确认内层").click();
    await flush();

    expect(confirmationBody).toEqual({
      call_id: INNER_CALL_ID,
      event_ids: [EVENT_ID],
      content: "先观察风向。",
      state_token: STATE_TOKEN,
      reasoning: innerDraft().reasoning,
    });
    expect(container.textContent).toContain("等待外层人格");
  });

  it("creates, edits, and deletes only manual queued events", async () => {
    const scene = pendingScene();
    const agentEvent: ExternalEvent = {
      id: GENERATED_EVENT_ID,
      sequence: 2,
      kind: "agent_message",
      content: "码头见。",
      source_agent_id: "B",
      source_call_id: OUTER_CALL_ID,
    };
    scene.agents[0].pending_events.push(agentEvent);
    scene.next_sequence = 3;
    const editedScene = structuredClone(scene);
    editedScene.agents[0].pending_events[0].content = "修改后的雾情";
    const deletedScene = structuredClone(editedScene);
    deletedScene.agents[0].pending_events.shift();
    const createdScene = structuredClone(deletedScene);
    createdScene.agents[0].pending_events.push(
      manualEvent(
        "新增潮汐报告",
        "66666666-6666-4666-8666-666666666666",
        3,
      ),
    );
    createdScene.next_sequence = 4;

    let editedBody: unknown;
    let createdBody: unknown;
    vi.spyOn(window, "confirm").mockReturnValue(true);
    await mountOpenedScene(scene, {
      [`PUT /api/scenes/${SCENE_ID}/agents/A/events/${EVENT_ID}`]: (
        init,
      ) => {
        editedBody = JSON.parse(String(init?.body));
        return jsonResponse(editedScene);
      },
      [`DELETE /api/scenes/${SCENE_ID}/agents/A/events/${EVENT_ID}`]:
        jsonResponse(deletedScene),
      [`POST /api/scenes/${SCENE_ID}/agents/A/events`]: (init) => {
        createdBody = JSON.parse(String(init?.body));
        return jsonResponse(createdScene, 201);
      },
    });

    expect(container.textContent).toContain("Agent 事件 · 不可修改");
    expect(container.querySelectorAll(".queue-list textarea")).toHaveLength(1);
    const editArea = container.querySelector(
      ".queue-list textarea",
    ) as HTMLTextAreaElement;
    editArea.value = "修改后的雾情";
    editArea.dispatchEvent(new Event("input", { bubbles: true }));
    await nextTick();
    findButton(container, "保存修改").click();
    await flush();
    expect(editedBody).toEqual({ content: "修改后的雾情" });

    // Scope to the queue so the scene-sidebar "删除" button is not matched.
    const eventDeleteButton = container.querySelector(
      ".queue-list li",
    ) as HTMLElement;
    findButton(eventDeleteButton, "删除").click();
    await flush();
    expect(container.querySelectorAll(".queue-list textarea")).toHaveLength(0);

    const createArea = container.querySelector(
      "#new-event",
    ) as HTMLTextAreaElement;
    createArea.value = "  新增潮汐报告\n";
    createArea.dispatchEvent(new Event("input", { bubbles: true }));
    await nextTick();
    findButton(container, "添加到队尾").click();
    await flush();
    expect(createdBody).toEqual({ content: "  新增潮汐报告\n" });
  });

  it("restores an outer half-round and previews its neutral context", async () => {
    const scene = halfRoundScene();
    scene.model = "gpt-test";
    const preview: ModelRequestPreviewResponse = {
      layer: "outer",
      event_ids: [EVENT_ID],
      context: [
        { role: "system", text: "OUTER A" },
        { role: "user", text: "上一轮外层输入" },
        { role: "assistant", text: "对儿子说：上一轮外层输出" },
        {
          role: "user",
          text:
            "外部事件：\n海面突然起雾。\n\n" +
            "你内心有一个声音：\n先观察风向。\n别急着靠岸。",
        },
      ],
    };
    await mountOpenedScene(scene, {
      [`GET /api/scenes/${SCENE_ID}/agents/A/model-request-preview?layer=outer`]:
        jsonResponse(preview),
    });

    expect(container.textContent).toContain("等待外层人格");
    expect(findButton(container, "生成外层草稿").disabled).toBe(false);
    const details = container.querySelector(
      ".preview-card",
    ) as HTMLDetailsElement;
    details.open = true;
    findButton(details, "加载预览").click();
    await flush();

    const readable = details.querySelector(
      ".readable-context",
    ) as HTMLElement;
    expect(
      [...readable.querySelectorAll("article")].map((article) => ({
        role: article.querySelector("strong")?.textContent,
        text: article.querySelector("pre")?.textContent,
      })),
    ).toEqual(preview.context);
    expect(details.textContent).not.toContain("原始 JSON");
  });

  it("asks before rolling back and returns to the saved outer stage", async () => {
    const scene = completeScene();
    const rolledBack = halfRoundScene();
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    const fetchMock = await mountOpenedScene(scene, {
      [`POST /api/scenes/${SCENE_ID}/rollback`]:
        jsonResponse(rolledBack),
    });

    findButton(container, "回退最近确认").click();
    await flush();

    expect(confirmSpy).toHaveBeenCalledWith(
      expect.stringContaining("Agent A · 外层"),
    );
    expect(fetchMock).toHaveBeenCalledWith(
      `/api/scenes/${SCENE_ID}/rollback`,
      expect.objectContaining({ method: "POST" }),
    );
    expect(container.textContent).toContain("等待外层人格");
    expect(container.querySelectorAll(".timeline-item--outer")).toHaveLength(0);
  });

  it("sends batch event_ids when confirming an inner draft", async () => {
    const firstEvent = manualEvent(
      "第一件事",
      EVENT_ID,
      1,
    );
    const secondEvent = manualEvent(
      "第二件事",
      "77777777-7777-4777-8777-777777777777",
      2,
    );
    const scene = pendingScene();
    scene.agents[0].pending_events = [firstEvent, secondEvent];
    scene.next_sequence = 3;
    const innerBatchDraft: LayerDraftResponse = {
      ...innerDraft(),
      event_ids: [firstEvent.id, secondEvent.id],
    };
    const confirmedInner = halfRoundScene();
    confirmedInner.agents[0].inner_context.turns[0].event_ids = [
      firstEvent.id,
      secondEvent.id,
    ];
    confirmedInner.agents[0].inner_context.turns[0].consumed_events = [
      firstEvent,
      secondEvent,
    ];
    confirmedInner.agents[0].pending_events = [];
    let confirmationBody: unknown;
    await mountOpenedScene(scene, {
      [`POST /api/scenes/${SCENE_ID}/agents/A/inner-drafts`]:
        jsonResponse(innerBatchDraft),
      [`POST /api/scenes/${SCENE_ID}/agents/A/inner-confirmations`]: (
        init,
      ) => {
        confirmationBody = JSON.parse(String(init?.body));
        return jsonResponse(confirmedInner);
      },
    });

    findButton(container, "生成内层草稿").click();
    await flush();
    findButton(container, "确认内层").click();
    await flush();

    expect(confirmationBody).toEqual({
      call_id: INNER_CALL_ID,
      event_ids: [firstEvent.id, secondEvent.id],
      content: "先观察风向。",
      state_token: STATE_TOKEN,
      reasoning: innerDraft().reasoning,
    });
  });

  it("deletes the open scene after confirm and returns to the welcome card", async () => {
    const scene = makeScene();
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    const fetchMock = await mountOpenedScene(scene, {
      [`DELETE /api/scenes/${SCENE_ID}`]: emptyResponse(204),
    });

    expect(
      container.querySelector(".scene-delete-button"),
    ).not.toBeNull();
    findButtonByAriaLabel(container, "删除场景").click();
    await flush();

    expect(confirmSpy).toHaveBeenCalledWith(
      expect.stringContaining(scene.name),
    );
    expect(fetchMock).toHaveBeenCalledWith(
      `/api/scenes/${SCENE_ID}`,
      expect.objectContaining({ method: "DELETE" }),
    );
    expect(container.textContent).toContain("选择或创建一个场景");
    expect(container.querySelector(".scene-delete-button")).toBeNull();
    expect(
      [...container.querySelectorAll(".scene-list li")].some(
        (li) => li.textContent?.includes(scene.name),
      ),
    ).toBe(false);
  });

  it("keeps the scene when the delete confirm is cancelled", async () => {
    const scene = makeScene();
    vi.spyOn(window, "confirm").mockReturnValue(false);
    await mountOpenedScene(scene, {
      [`DELETE /api/scenes/${SCENE_ID}`]: emptyResponse(204),
    });

    findButtonByAriaLabel(container, "删除场景").click();
    await flush();

    expect(container.textContent).toContain(scene.name);
    expect(container.querySelector(".timeline")).not.toBeNull();
  });
});
