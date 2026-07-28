<script setup lang="ts">
import {
  computed,
  nextTick,
  onBeforeUnmount,
  onMounted,
  ref,
  watch,
} from "vue";

import {
  composeSystemPrompt,
  createScene,
  deleteMessage,
  generateMessageDraft,
  getModelRequestPreview,
  getScene,
  listScenes,
  saveScene,
  sendMessage,
} from "./api";
import {
  AGENT_IDS,
  type AgentId,
  type MessageTimelineRecord,
  type MessageDraftUsage,
  type ModelRequest,
  type Scene,
  type SceneSummary,
  type SceneUpdate,
} from "./types";

type ListState = "loading" | "ready" | "error";
type SaveState = "idle" | "saving" | "success" | "error";
interface MessageDraft {
  content: string;
  usage: MessageDraftUsage | null;
}
type PreviewMode = "readable" | "json";

const sceneSummaries = ref<SceneSummary[]>([]);
const currentScene = ref<Scene | null>(null);
const savedScene = ref<Scene | null>(null);
const activeAgentId = ref<AgentId>("A");
const newSceneName = ref("");

const listState = ref<ListState>("loading");
const listError = ref("");
const actionError = ref("");
const createError = ref("");
const openingSceneId = ref<string | null>(null);
const isCreating = ref(false);
const saveState = ref<SaveState>("idle");
const saveError = ref("");
const messageDrafts = ref<Record<AgentId, MessageDraft>>(
  emptyMessageDrafts(),
);
const messageErrors = ref<Record<AgentId, string>>(
  emptyMessageErrors(),
);
const sendingAgentId = ref<AgentId | null>(null);
const deletingMessageId = ref<string | null>(null);
const generatingAgentId = ref<AgentId | null>(null);
const composingAgentId = ref<AgentId | null>(null);
const requestPreviews = ref<Record<AgentId, ModelRequest | null>>(
  emptyRequestPreviews(),
);
const previewErrors = ref<Record<AgentId, string>>(emptyMessageErrors());
const previewingAgentId = ref<AgentId | null>(null);
const previewMode = ref<PreviewMode>("readable");
const timelineScroller = ref<HTMLElement | null>(null);
const isTimelinePinned = ref(true);
const hasNewTimelineUpdate = ref(false);

let listRequestToken = 0;
let summaryMutationVersion = 0;
const TIMELINE_BOTTOM_THRESHOLD = 24;

function cloneScene(scene: Scene): Scene {
  return JSON.parse(JSON.stringify(scene)) as Scene;
}

function emptyMessageDrafts(): Record<AgentId, MessageDraft> {
  return {
    A: {
      content: "",
      usage: null,
    },
    B: {
      content: "",
      usage: null,
    },
    C: {
      content: "",
      usage: null,
    },
  };
}

function emptyMessageErrors(): Record<AgentId, string> {
  return { A: "", B: "", C: "" };
}

function emptyRequestPreviews(): Record<AgentId, ModelRequest | null> {
  return { A: null, B: null, C: null };
}

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message
    ? error.message
    : fallback;
}

function compareSummaries(
  left: SceneSummary,
  right: SceneSummary,
): number {
  if (left.name < right.name) return -1;
  if (left.name > right.name) return 1;
  return left.id.localeCompare(right.id);
}

function mergeSummaries(
  summaries: SceneSummary[],
  newerSummaries: SceneSummary[],
): SceneSummary[] {
  const summariesById = new Map(
    summaries.map((summary) => [summary.id, summary]),
  );
  for (const summary of newerSummaries) {
    summariesById.set(summary.id, summary);
  }
  return [...summariesById.values()].sort(compareSummaries);
}

const isDirty = computed(() => {
  if (currentScene.value === null || savedScene.value === null) {
    return false;
  }
  return (
    JSON.stringify(currentScene.value) !==
    JSON.stringify(savedScene.value)
  );
});

const activeAgent = computed(() =>
  currentScene.value?.agents.find(
    (agent) => agent.id === activeAgentId.value,
  ),
);

const activeMessageDraft = computed(
  () => messageDrafts.value[activeAgentId.value],
);

const hasMessageDrafts = computed(() =>
  AGENT_IDS.some((agentId) => {
    const draft = messageDrafts.value[agentId];
    return draft.content !== "";
  }),
);

const activeDraftHasContent = computed(() => {
  return activeMessageDraft.value.content !== "";
});

const deletableMessageIds = computed(() => {
  const scene = currentScene.value;
  const deletableIds = new Set<string>();
  if (scene === null) {
    return deletableIds;
  }

  const recordsByMessageId = new Map<
    string,
    {
      agentId: AgentId;
      index: number;
      record: MessageTimelineRecord;
    }[]
  >();
  for (const agent of scene.agents) {
    agent.timeline.forEach((record, index) => {
      if (record.type !== "message") {
        return;
      }
      const matches = recordsByMessageId.get(record.message_id) ?? [];
      matches.push({ agentId: agent.id, index, record });
      recordsByMessageId.set(record.message_id, matches);
    });
  }

  for (const [messageId, matches] of recordsByMessageId) {
    if (matches.length !== 2) {
      continue;
    }
    const sent = matches.find(
      ({ record }) => record.direction === "sent",
    );
    const received = matches.find(
      ({ record }) => record.direction === "received",
    );
    if (sent === undefined || received === undefined) {
      continue;
    }
    const sender = scene.agents.find(
      (agent) => agent.id === sent.agentId,
    );
    const recipient = scene.agents.find(
      (agent) => agent.id === received.agentId,
    );
    if (
      sender === undefined ||
      recipient === undefined ||
      sent.agentId === received.agentId ||
      sent.record.counterpart_id !== received.agentId ||
      received.record.counterpart_id !== sent.agentId ||
      !messagePairContentMatches(
        sent.record.content,
        received.record.content,
        sent.agentId,
        received.agentId,
      ) ||
      sent.index !== sender.timeline.length - 1 ||
      received.index !== recipient.timeline.length - 1
    ) {
      continue;
    }
    deletableIds.add(messageId);
  }
  return deletableIds;
});

const editorLocked = computed(
  () =>
    saveState.value === "saving" ||
    sendingAgentId.value !== null ||
    deletingMessageId.value !== null ||
    generatingAgentId.value !== null ||
    composingAgentId.value !== null ||
    previewingAgentId.value !== null ||
    isCreating.value ||
    openingSceneId.value !== null,
);

const canSendMessage = computed(
  () =>
    currentScene.value !== null &&
    !isDirty.value &&
    !editorLocked.value &&
    activeMessageDraft.value.content.trim() !== "",
);

const canGenerateMessageDraft = computed(
  () =>
    currentScene.value !== null &&
    !isDirty.value &&
    !editorLocked.value,
);

const activeTimelineSnapshot = computed(() => ({
  identity: currentScene.value
    ? `${currentScene.value.id}:${activeAgentId.value}`
    : "",
  records:
    activeAgent.value?.timeline
      .map(
        (record) =>
          `message:${record.message_id}:${record.direction}:${record.content}`,
      )
      .join("\u0000") ?? "",
}));

function messagePairContentMatches(
  sentContent: string,
  receivedContent: string,
  senderId: AgentId,
  recipientId: AgentId,
): boolean {
  const sent = sentContent.match(
    new RegExp(
      `^To\\s+${recipientId}\\s*[:：]\\s*(\\S(?:[^\\r\\n]*\\S)?)\\s*$`,
    ),
  );
  const received = receivedContent.match(
    new RegExp(
      `^From\\s+${senderId}\\s*[:：]\\s*(\\S(?:[^\\r\\n]*\\S)?)\\s*$`,
    ),
  );
  return (
    sent !== null &&
    received !== null &&
    sent[1] === received[1]
  );
}

function timelineIsNearBottom(element: HTMLElement): boolean {
  return (
    element.scrollHeight - element.scrollTop - element.clientHeight <=
    TIMELINE_BOTTOM_THRESHOLD
  );
}

function handleTimelineScroll(): void {
  const element = timelineScroller.value;
  if (element === null) {
    return;
  }
  isTimelinePinned.value = timelineIsNearBottom(element);
  if (isTimelinePinned.value) {
    hasNewTimelineUpdate.value = false;
  }
}

async function scrollTimelineToLatest(): Promise<void> {
  await nextTick();
  const element = timelineScroller.value;
  if (element === null) {
    return;
  }
  element.scrollTop = element.scrollHeight;
  isTimelinePinned.value = true;
  hasNewTimelineUpdate.value = false;
}

watch(
  activeTimelineSnapshot,
  (snapshot, previousSnapshot) => {
    if (
      previousSnapshot === undefined ||
      snapshot.identity !== previousSnapshot.identity
    ) {
      void scrollTimelineToLatest();
      return;
    }
    if (snapshot.records === previousSnapshot.records) {
      return;
    }
    if (isTimelinePinned.value) {
      void scrollTimelineToLatest();
    } else {
      hasNewTimelineUpdate.value = true;
    }
  },
  { flush: "post" },
);

function installScene(
  scene: Scene,
  desiredActiveAgentId: AgentId = "A",
  preserveMessageDrafts = false,
): void {
  savedScene.value = cloneScene(scene);
  currentScene.value = cloneScene(scene);
  activeAgentId.value = desiredActiveAgentId;
  if (!preserveMessageDrafts) {
    messageDrafts.value = emptyMessageDrafts();
    messageErrors.value = emptyMessageErrors();
  }
  requestPreviews.value = emptyRequestPreviews();
  previewErrors.value = emptyMessageErrors();
  saveState.value = "idle";
  saveError.value = "";
  actionError.value = "";
}

function upsertSummary(scene: Scene): void {
  summaryMutationVersion += 1;
  const summary = { id: scene.id, name: scene.name };
  sceneSummaries.value = mergeSummaries(
    sceneSummaries.value,
    [summary],
  );
  listState.value = "ready";
  listError.value = "";
}

function confirmDiscardChanges(): boolean {
  if (
    !isDirty.value &&
    !hasMessageDrafts.value
  ) {
    return true;
  }
  return window.confirm(
    "当前场景有未保存的更改或消息草稿。确定要放弃吗？",
  );
}

async function refreshScenes(): Promise<void> {
  const requestToken = ++listRequestToken;
  const mutationVersionAtRequestStart = summaryMutationVersion;
  const summariesAtRequestStart = new Map(
    sceneSummaries.value.map((summary) => [summary.id, summary.name]),
  );
  listState.value = "loading";
  listError.value = "";

  try {
    const summaries = await listScenes();
    if (requestToken !== listRequestToken) {
      return;
    }
    const summariesUpdatedWhileLoading = sceneSummaries.value.filter(
      (summary) =>
        summariesAtRequestStart.get(summary.id) !== summary.name,
    );
    sceneSummaries.value = mergeSummaries(
      summaries,
      summariesUpdatedWhileLoading,
    );
    listState.value = "ready";
  } catch (error) {
    if (requestToken !== listRequestToken) {
      return;
    }
    if (mutationVersionAtRequestStart !== summaryMutationVersion) {
      return;
    }
    listState.value = "error";
    listError.value = errorMessage(error, "无法加载场景列表。");
  }
}

async function openScene(summary: SceneSummary): Promise<void> {
  if (
    summary.id === currentScene.value?.id ||
    openingSceneId.value !== null ||
    isCreating.value ||
    saveState.value === "saving" ||
    sendingAgentId.value !== null ||
    generatingAgentId.value !== null
  ) {
    return;
  }
  if (!confirmDiscardChanges()) {
    return;
  }

  openingSceneId.value = summary.id;
  actionError.value = "";

  try {
    const scene = await getScene(summary.id);
    installScene(scene);
  } catch (error) {
    actionError.value = errorMessage(error, "无法打开场景。");
  } finally {
    openingSceneId.value = null;
  }
}

async function submitNewScene(): Promise<void> {
  createError.value = "";
  const name = newSceneName.value.trim();

  if (!name) {
    createError.value = "请输入场景名称。";
    return;
  }
  if (
    isCreating.value ||
    openingSceneId.value !== null ||
    saveState.value === "saving" ||
    sendingAgentId.value !== null ||
    generatingAgentId.value !== null ||
    !confirmDiscardChanges()
  ) {
    return;
  }

  isCreating.value = true;

  try {
    const scene = await createScene(name);
    upsertSummary(scene);
    installScene(scene);
    newSceneName.value = "";
  } catch (error) {
    createError.value = errorMessage(error, "无法创建场景。");
  } finally {
    isCreating.value = false;
  }
}

function markEdited(): void {
  if (saveState.value === "success" || saveState.value === "error") {
    saveState.value = "idle";
    saveError.value = "";
  }
}

function sceneUpdate(scene: Scene): SceneUpdate {
  return {
    name: scene.name,
    agents: scene.agents.map(
      ({ id, name, persona, desire, fear, memory, system_prompt }) => ({
        id,
        name,
        persona,
        desire,
        fear,
        memory,
        system_prompt,
      }),
    ),
  };
}

async function saveCurrentScene(): Promise<void> {
  const scene = currentScene.value;
  if (
    scene === null ||
    saveState.value === "saving" ||
    sendingAgentId.value !== null ||
    generatingAgentId.value !== null
  ) {
    return;
  }
  const selectedAgentId = activeAgentId.value;

  if (
    !scene.name.trim() ||
    scene.agents.some((agent) => !agent.name.trim())
  ) {
    saveState.value = "error";
    saveError.value = "场景名称和三个 Agent 的显示名均不能为空。";
    return;
  }
  if (scene.agents.some((agent) => !agent.system_prompt.trim())) {
    saveState.value = "error";
    saveError.value = "三个 Agent 的最终系统提示词均不能为空。";
    return;
  }

  saveState.value = "saving";
  saveError.value = "";

  try {
    const saved = await saveScene(scene.id, sceneUpdate(scene));
    upsertSummary(saved);
    installScene(saved, selectedAgentId, true);
    saveState.value = "success";
  } catch (error) {
    saveState.value = "error";
    saveError.value = errorMessage(error, "保存失败，请重试。");
  }
}

function markMessageDraftEdited(): void {
  messageErrors.value[activeAgentId.value] = "";
}

async function generateDraft(): Promise<void> {
  const scene = currentScene.value;
  const senderId = activeAgentId.value;
  if (
    scene === null ||
    isDirty.value ||
    editorLocked.value
  ) {
    return;
  }

  generatingAgentId.value = senderId;
  messageErrors.value[senderId] = "";

  try {
    const generated = await generateMessageDraft(scene.id, senderId);
    messageDrafts.value[senderId] = {
      content: generated.content,
      usage: generated.usage,
    };
  } catch (error) {
    messageErrors.value[senderId] = errorMessage(
      error,
      "草稿生成失败，请重试。",
    );
  } finally {
    generatingAgentId.value = null;
  }
}

async function confirmMessage(): Promise<void> {
  const scene = currentScene.value;
  const senderId = activeAgentId.value;
  const draft = messageDrafts.value[senderId];
  if (
    scene === null ||
    isDirty.value ||
    editorLocked.value ||
    draft.content.trim() === ""
  ) {
    return;
  }

  sendingAgentId.value = senderId;
  messageErrors.value[senderId] = "";

  try {
    const updated = await sendMessage(scene.id, {
      sender_id: senderId,
      content: draft.content,
    });
    installScene(updated, senderId, true);
    messageDrafts.value[senderId] = {
      content: "",
      usage: null,
    };
  } catch (error) {
    messageErrors.value[senderId] = errorMessage(
      error,
      "消息发送失败，请重试。",
    );
  } finally {
    sendingAgentId.value = null;
  }
}

async function confirmDeleteMessage(
  record: MessageTimelineRecord,
): Promise<void> {
  const scene = currentScene.value;
  const viewingAgentId = activeAgentId.value;
  if (
    scene === null ||
    isDirty.value ||
    editorLocked.value ||
    !deletableMessageIds.value.has(record.message_id)
  ) {
    return;
  }

  const senderId =
    record.direction === "sent"
      ? viewingAgentId
      : record.counterpart_id;
  const recipientId =
    record.direction === "sent"
      ? record.counterpart_id
      : viewingAgentId;
  const sender = scene.agents.find((agent) => agent.id === senderId);
  const recipient = scene.agents.find(
    (agent) => agent.id === recipientId,
  );
  const confirmed = window.confirm(
    [
      "确定永久删除这条已确认消息吗？",
      "",
      `${sender?.name ?? senderId}（Agent ${senderId}） → ${recipient?.name ?? recipientId}（Agent ${recipientId}）`,
      record.content,
      "",
      "删除不可撤销，且本次只删除这一条。",
    ].join("\n"),
  );
  if (!confirmed) {
    return;
  }

  deletingMessageId.value = record.message_id;
  actionError.value = "";
  try {
    const updated = await deleteMessage(scene.id, record.message_id);
    installScene(updated, viewingAgentId, true);
  } catch (error) {
    actionError.value = errorMessage(error, "消息删除失败，请重试。");
  } finally {
    deletingMessageId.value = null;
  }
}

async function recomposeActiveSystemPrompt(): Promise<void> {
  const agent = activeAgent.value;
  if (agent === undefined || editorLocked.value) {
    return;
  }
  if (
    !window.confirm(
      "这会用四个拼接素材覆盖当前最终系统提示词。确定继续吗？",
    )
  ) {
    return;
  }

  composingAgentId.value = agent.id;
  saveError.value = "";
  try {
    const candidate = await composeSystemPrompt({
      persona: agent.persona,
      desire: agent.desire,
      fear: agent.fear,
      memory: agent.memory,
    });
    agent.system_prompt = candidate;
    markEdited();
  } catch (error) {
    saveState.value = "error";
    saveError.value = errorMessage(error, "系统提示词拼接失败。");
  } finally {
    composingAgentId.value = null;
  }
}

async function loadRequestPreview(): Promise<void> {
  const scene = currentScene.value;
  const agentId = activeAgentId.value;
  if (scene === null || editorLocked.value) {
    return;
  }

  previewingAgentId.value = agentId;
  previewErrors.value[agentId] = "";
  try {
    requestPreviews.value[agentId] = await getModelRequestPreview(
      scene.id,
      agentId,
    );
    previewMode.value = "readable";
  } catch (error) {
    previewErrors.value[agentId] = errorMessage(
      error,
      "无法加载请求预览。",
    );
  } finally {
    previewingAgentId.value = null;
  }
}

function discardActiveDraft(): void {
  const agentId = activeAgentId.value;
  messageDrafts.value[agentId] = {
    content: "",
    usage: null,
  };
  messageErrors.value[agentId] = "";
}

function prettyJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function readableModelContext(
  request: ModelRequest,
): { label: string; text: string }[] {
  const context: { label: string; text: string }[] = [];
  const system = request.system;
  if (Array.isArray(system)) {
    for (const block of system) {
      if (
        typeof block === "object" &&
        block !== null &&
        "text" in block &&
        typeof block.text === "string"
      ) {
        context.push({ label: "system", text: block.text });
      }
    }
  }

  const messages = request.messages;
  if (!Array.isArray(messages)) {
    return context;
  }
  for (const message of messages) {
    if (
      typeof message !== "object" ||
      message === null ||
      !("role" in message) ||
      typeof message.role !== "string" ||
      !("content" in message) ||
      !Array.isArray(message.content)
    ) {
      continue;
    }
    for (const block of message.content) {
      if (
        typeof block === "object" &&
        block !== null &&
        "text" in block &&
        typeof block.text === "string"
      ) {
        context.push({ label: message.role, text: block.text });
      }
    }
  }
  return context;
}

function handleBeforeUnload(event: BeforeUnloadEvent): void {
  if (
    !isDirty.value &&
    !hasMessageDrafts.value
  ) {
    return;
  }
  event.preventDefault();
  event.returnValue = "";
}

onMounted(() => {
  void refreshScenes();
  window.addEventListener("beforeunload", handleBeforeUnload);
});

onBeforeUnmount(() => {
  window.removeEventListener("beforeunload", handleBeforeUnload);
});
</script>

<template>
  <div class="app-shell">
    <header class="topbar">
      <a class="brand" href="/" aria-label="AI 小镇首页">
        <span class="brand-mark" aria-hidden="true">AT</span>
        <span>
          <strong>AI 小镇</strong>
          <small>三智能体最小实验</small>
        </span>
      </a>
      <p class="milestone">
        <span aria-hidden="true"></span>
        模型草稿 · 人工确认
      </p>
    </header>

    <main class="workspace">
      <aside class="scene-sidebar" aria-labelledby="scene-list-title">
        <div class="sidebar-heading">
          <div>
            <p class="eyebrow">SCENES</p>
            <h1 id="scene-list-title">场景</h1>
          </div>
          <span class="scene-count">{{ sceneSummaries.length }}</span>
        </div>

        <form class="create-form" @submit.prevent="submitNewScene">
          <label for="new-scene-name">创建命名场景</label>
          <div class="create-row">
            <input
              id="new-scene-name"
              v-model="newSceneName"
              name="new-scene-name"
              type="text"
              autocomplete="off"
              placeholder="例如：雨夜港口"
              :disabled="editorLocked"
              @input="createError = ''"
            />
            <button
              class="primary-button create-button"
              type="submit"
              :disabled="editorLocked"
            >
              {{ isCreating ? "创建中…" : "创建" }}
            </button>
          </div>
          <p v-if="createError" class="form-error" role="alert">
            {{ createError }}
          </p>
        </form>

        <div class="scene-list-region">
          <div
            v-if="listState === 'loading'"
            class="sidebar-message"
            role="status"
          >
            <span class="spinner" aria-hidden="true"></span>
            正在加载场景…
          </div>

          <div
            v-else-if="listState === 'error'"
            class="sidebar-message sidebar-message--error"
            role="alert"
          >
            <p>{{ listError }}</p>
            <button
              class="text-button"
              type="button"
              @click="refreshScenes"
            >
              重新加载
            </button>
          </div>

          <p
            v-else-if="sceneSummaries.length === 0"
            class="sidebar-message empty-list"
          >
            还没有场景。先给第一个实验场景起个名字。
          </p>

          <ul v-else class="scene-list">
            <li v-for="summary in sceneSummaries" :key="summary.id">
              <button
                class="scene-list-button"
                :class="{
                  'scene-list-button--active':
                    currentScene?.id === summary.id,
                }"
                type="button"
                :aria-current="
                  currentScene?.id === summary.id ? 'page' : undefined
                "
                :disabled="editorLocked"
                @click="openScene(summary)"
              >
                <span class="scene-list-name">{{ summary.name }}</span>
                <span class="scene-list-meta">
                  <span>{{ summary.id.slice(0, 8) }}</span>
                  <span
                    v-if="openingSceneId === summary.id"
                    class="spinner spinner--small"
                    aria-label="正在打开"
                  ></span>
                  <span
                    v-else-if="currentScene?.id === summary.id"
                    class="active-indicator"
                    aria-label="当前场景"
                  ></span>
                </span>
              </button>
            </li>
          </ul>
        </div>

        <p class="storage-note">
          每个场景独立保存为本机 JSON 文件。已确认消息仅可从双方时间线栈顶逐条删除。
        </p>
      </aside>

      <section class="editor-region" aria-label="场景编辑区">
        <div
          v-if="actionError"
          class="page-notice page-notice--error"
          role="alert"
        >
          <span>{{ actionError }}</span>
          <button
            type="button"
            aria-label="关闭错误提示"
            @click="actionError = ''"
          >
            ×
          </button>
        </div>

        <div v-if="currentScene === null" class="welcome-panel">
          <div class="welcome-illustration" aria-hidden="true">
            <span>A</span>
            <span>B</span>
            <span>C</span>
          </div>
          <p class="eyebrow">FIRST VERTICAL SLICE</p>
          <h2>从一个命名场景开始</h2>
          <p>
            创建新场景，或从左侧打开已有场景。你可以分别编辑三个
            Agent，所有后续修改只会在点击“保存场景”时写入磁盘。
          </p>
        </div>

        <template v-else>
          <header class="editor-header">
            <div class="scene-name-field">
              <label for="scene-name">场景名称</label>
              <input
                id="scene-name"
                v-model="currentScene.name"
                type="text"
                autocomplete="off"
                required
                :disabled="editorLocked"
                @input="markEdited"
              />
            </div>

            <div class="save-controls">
              <div class="save-status" aria-live="polite">
                <span
                  class="save-status-dot"
                  :class="{
                    'save-status-dot--dirty': isDirty,
                    'save-status-dot--success':
                      saveState === 'success' && !isDirty,
                    'save-status-dot--error': saveState === 'error',
                  }"
                  aria-hidden="true"
                ></span>
                <span v-if="saveState === 'saving'">正在保存…</span>
                <span v-else-if="saveState === 'error'">{{
                  saveError
                }}</span>
                <span v-else-if="isDirty">有未保存的更改</span>
                <span v-else-if="saveState === 'success'">保存成功</span>
                <span v-else>当前内容已保存</span>
              </div>
              <button
                class="primary-button save-button"
                type="button"
                :disabled="!isDirty || editorLocked"
                @click="saveCurrentScene"
              >
                {{ saveState === "saving" ? "保存中…" : "保存场景" }}
              </button>
            </div>
          </header>

          <nav class="agent-tabs" aria-label="选择 Agent">
            <button
              v-for="agentId in AGENT_IDS"
              :key="agentId"
              class="agent-tab"
              :class="{
                'agent-tab--active': activeAgentId === agentId,
              }"
              type="button"
              :aria-selected="activeAgentId === agentId"
              role="tab"
              :disabled="editorLocked"
              @click="activeAgentId = agentId"
            >
              <span class="agent-letter">{{ agentId }}</span>
              <span class="agent-tab-copy">
                <small>Agent {{ agentId }}</small>
                <strong>{{
                  currentScene.agents.find(
                    (agent) => agent.id === agentId,
                  )?.name
                }}</strong>
              </span>
            </button>
          </nav>

          <div v-if="activeAgent" class="agent-editor">
            <section class="chat-panel" aria-labelledby="timeline-title">
              <header class="chat-panel-heading">
                <div>
                  <p class="eyebrow">AGENT {{ activeAgent.id }}</p>
                  <h2 id="timeline-title">
                    {{ activeAgent.name || "未命名 Agent" }}
                  </h2>
                  <p>私人时间线 · 仅这位 Agent 可见的已确认消息</p>
                </div>
                <div class="chat-heading-actions">
                  <span class="chat-agent-id">
                    固定身份 <strong>Agent {{ activeAgent.id }}</strong>
                  </span>
                </div>
              </header>

              <div
                ref="timelineScroller"
                class="chat-messages"
                aria-live="polite"
                @scroll="handleTimelineScroll"
              >
                <p
                  v-if="activeAgent.timeline.length === 0"
                  class="timeline-empty"
                >
                  时间线为空
                </p>
                <ol v-else class="chat-message-list">
                  <li
                    v-for="record in activeAgent.timeline"
                    :key="`message-${record.message_id}-${record.direction}`"
                    class="chat-message"
                    :class="`chat-message--${record.direction}`"
                  >
                    <article class="chat-bubble">
                      <p class="chat-message-content">
                        {{ record.content }}
                      </p>
                      <button
                        v-if="deletableMessageIds.has(record.message_id)"
                        class="timeline-delete-button"
                        type="button"
                        :disabled="isDirty || editorLocked"
                        @click="confirmDeleteMessage(record)"
                      >
                        {{
                          deletingMessageId === record.message_id
                            ? "删除中…"
                            : "删除"
                        }}
                      </button>
                    </article>
                  </li>
                </ol>
              </div>

              <button
                v-if="hasNewTimelineUpdate"
                class="new-message-button"
                type="button"
                @click="scrollTimelineToLatest"
              >
                有新消息 · 回到最新
              </button>

              <footer class="chat-composer">
                <div class="message-fields">
                  <label class="message-content" for="message-content">
                    <span>完整消息草稿</span>
                    <textarea
                      id="message-content"
                      v-model="activeMessageDraft.content"
                      rows="3"
                      :disabled="editorLocked"
                      placeholder="To B: 消息正文"
                      @input="markMessageDraftEdited"
                    ></textarea>
                  </label>
                </div>

                <dl
                  v-if="activeMessageDraft.usage"
                  class="usage-metrics"
                  aria-label="本次草稿生成 token 用量"
                >
                  <div>
                    <dt>5 分钟缓存写入</dt>
                    <dd>
                      {{
                        activeMessageDraft.usage
                          .cache_creation_input_tokens
                      }}
                    </dd>
                  </div>
                  <div>
                    <dt>缓存读取</dt>
                    <dd>
                      {{
                        activeMessageDraft.usage.cache_read_input_tokens
                      }}
                    </dd>
                  </div>
                  <div>
                    <dt>未缓存输入</dt>
                    <dd>{{ activeMessageDraft.usage.input_tokens }}</dd>
                  </div>
                  <div>
                    <dt>输出</dt>
                    <dd>{{ activeMessageDraft.usage.output_tokens }}</dd>
                  </div>
                </dl>

                <div class="message-actions">
                  <p v-if="isDirty" class="message-hint">
                    请先保存场景，再生成或确认发送。
                  </p>
                  <p
                    v-else-if="messageErrors[activeAgent.id]"
                    class="form-error message-error"
                    role="alert"
                  >
                    {{ messageErrors[activeAgent.id] }}
                  </p>
                  <p v-else class="message-hint">
                    只有确认后的消息才会进入双方个人时间线。
                  </p>
                  <div class="message-action-buttons">
                    <button
                      v-if="activeDraftHasContent"
                      class="text-button discard-draft-button"
                      type="button"
                      :disabled="editorLocked"
                      @click="discardActiveDraft"
                    >
                      放弃草稿
                    </button>
                    <button
                      class="secondary-button message-generate-button"
                      type="button"
                      :disabled="!canGenerateMessageDraft"
                      @click="generateDraft"
                    >
                      {{
                        generatingAgentId === activeAgent.id
                          ? "生成中…"
                          : activeDraftHasContent
                            ? "重新生成"
                            : "生成草稿"
                      }}
                    </button>
                    <button
                      class="primary-button message-send-button"
                      type="button"
                      :disabled="!canSendMessage"
                      @click="confirmMessage"
                    >
                      {{
                        sendingAgentId === activeAgent.id
                          ? "发送中…"
                          : "确认发送"
                      }}
                    </button>
                  </div>
                </div>
              </footer>
            </section>

            <details class="secondary-panel role-panel">
              <summary>
                <span>
                  <strong>角色设定</strong>
                  <small>显示名、角色提示词、欲望、恐惧与记忆</small>
                </span>
              </summary>
              <div class="secondary-panel-body field-grid">
                <label class="field field--wide" for="agent-name">
                  <span>显示名</span>
                  <small>可编辑；固定 ID 不会随名称改变</small>
                  <input
                    id="agent-name"
                    v-model="activeAgent.name"
                    type="text"
                    autocomplete="off"
                    required
                    :disabled="editorLocked"
                    @input="markEdited"
                  />
                </label>

                <label
                  class="field field--wide system-prompt-field"
                  for="agent-system-prompt"
                >
                  <span>最终系统提示词</span>
                  <small>
                    权威角色提示；保存值会逐字成为 Anthropic 请求中唯一的
                    system 文本
                  </small>
                  <textarea
                    id="agent-system-prompt"
                    v-model="activeAgent.system_prompt"
                    rows="15"
                    :disabled="editorLocked"
                    required
                    @input="markEdited"
                  ></textarea>
                </label>

                <label class="field field--wide" for="agent-persona">
                  <span>人设</span>
                  <small>身份、经历与相对稳定的性格描述</small>
                  <textarea
                    id="agent-persona"
                    v-model="activeAgent.persona"
                    rows="5"
                    :disabled="editorLocked"
                    placeholder="暂时可以留空"
                    @input="markEdited"
                  ></textarea>
                </label>

                <label class="field" for="agent-desire">
                  <span>欲望</span>
                  <small>用自然语言描述 Agent 想要什么</small>
                  <textarea
                    id="agent-desire"
                    v-model="activeAgent.desire"
                    rows="7"
                    :disabled="editorLocked"
                    placeholder="暂时可以留空"
                    @input="markEdited"
                  ></textarea>
                </label>

                <label class="field" for="agent-fear">
                  <span>恐惧</span>
                  <small>用自然语言描述 Agent 害怕什么</small>
                  <textarea
                    id="agent-fear"
                    v-model="activeAgent.fear"
                    rows="7"
                    :disabled="editorLocked"
                    placeholder="暂时可以留空"
                    @input="markEdited"
                  ></textarea>
                </label>

                <label class="field field--wide" for="agent-memory">
                  <span>当前压缩记忆</span>
                  <small>仅供人工编辑；本阶段不会自动更新</small>
                  <textarea
                    id="agent-memory"
                    v-model="activeAgent.memory"
                    rows="7"
                    :disabled="editorLocked"
                    placeholder="暂时可以留空"
                    @input="markEdited"
                  ></textarea>
                </label>

                <button
                  class="secondary-button recompose-button"
                  type="button"
                  :disabled="editorLocked"
                  @click="recomposeActiveSystemPrompt"
                >
                  {{
                    composingAgentId === activeAgent.id
                      ? "拼接中…"
                      : "从槽位重新拼接"
                  }}
                </button>
              </div>
            </details>

            <details class="secondary-panel observability-card">
              <summary>
                <span>
                  <strong>模型请求预览</strong>
                  <small>检查模型实际看到的文本与完整载荷</small>
                </span>
              </summary>
              <div class="secondary-panel-body">
                <div class="preview-actions">
                  <button
                    class="secondary-button preview-button"
                    type="button"
                    :disabled="editorLocked"
                    @click="loadRequestPreview"
                  >
                    {{
                      previewingAgentId === activeAgent.id
                        ? "加载中…"
                        : requestPreviews[activeAgent.id]
                          ? "刷新预览"
                          : "加载预览"
                    }}
                  </button>
                  <div
                    v-if="requestPreviews[activeAgent.id]"
                    class="preview-mode-switch"
                    role="group"
                    aria-label="请求预览模式"
                  >
                    <button
                      type="button"
                      :class="{ active: previewMode === 'readable' }"
                      @click="previewMode = 'readable'"
                    >
                      可读模型上下文
                    </button>
                    <button
                      type="button"
                      :class="{ active: previewMode === 'json' }"
                      @click="previewMode = 'json'"
                    >
                      原始 JSON
                    </button>
                  </div>
                </div>
                <p v-if="isDirty" class="stale-preview-notice">
                  场景有未保存修改；下一次请求预览基于已保存版本，当前显示为旧版本。
                  请先保存再刷新。
                </p>
                <p
                  v-if="previewErrors[activeAgent.id]"
                  class="form-error"
                  role="alert"
                >
                  {{ previewErrors[activeAgent.id] }}
                </p>
                <div class="request-preview">
                  <p
                    v-if="requestPreviews[activeAgent.id] === null"
                    class="request-empty"
                  >
                    尚未加载已保存场景的下一次请求。
                  </p>
                  <div
                    v-else-if="previewMode === 'readable'"
                    class="readable-context"
                  >
                    <article
                      v-for="(block, index) in readableModelContext(
                        requestPreviews[activeAgent.id]!,
                      )"
                      :key="`${block.label}-${index}`"
                    >
                      <strong>{{ block.label }}</strong>
                      <pre>{{ block.text }}</pre>
                    </article>
                  </div>
                  <pre v-else>{{
                    prettyJson(requestPreviews[activeAgent.id])
                  }}</pre>
                </div>
                <p class="observability-note">
                  两种视图来自同一次后端请求快照；可读视图逐字展示其中的
                  system、role 与 text block，原始 JSON 保留传输元数据。
                </p>
              </div>
            </details>
          </div>
        </template>
      </section>
    </main>

  </div>
</template>
