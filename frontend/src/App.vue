<script setup lang="ts">
import {
  computed,
  onBeforeUnmount,
  onMounted,
  ref,
  watch,
} from "vue";

import {
  bindSceneModel,
  confirmLayerDraft,
  createManualEvent,
  createScene,
  deleteManualEvent,
  deleteScene,
  editManualEvent,
  generateLayerDraft,
  getModelOptions,
  getModelRequestPreview,
  getScene,
  listScenes,
  rollbackLatestCall,
  saveScene,
} from "./api";
import {
  AGENT_IDS,
  type Agent,
  type AgentId,
  type ExternalEvent,
  type Layer,
  type LayerDraftResponse,
  type ModelOption,
  type ModelReasoningBlock,
  type ModelRequestPreviewResponse,
  type Scene,
  type SceneSummary,
  type SceneUpdate,
} from "./types";

type ListState = "loading" | "ready" | "error";
const LAYERS: Layer[] = ["inner", "outer"];

interface TimelineItem {
  key: string;
  sequence: number;
  rank: number;
  kind: "event" | "inner" | "outer";
  label: string;
  content: string;
  status?: string;
  input?: string;
  callId?: string;
  reasoning?: ModelReasoningBlock[];
}

const sceneSummaries = ref<SceneSummary[]>([]);
const currentScene = ref<Scene | null>(null);
const savedScene = ref<Scene | null>(null);
const activeAgentId = ref<AgentId>("A");
const newSceneName = ref("");
const newSceneModel = ref("");
const bindingModel = ref("");
const modelOptions = ref<ModelOption[]>([]);
const modelOptionsError = ref("");
const newEventContent = ref<Record<AgentId, string>>({
  A: "",
  B: "",
  C: "",
});
const eventEdits = ref<Record<string, string>>({});
const drafts = ref<Record<AgentId, LayerDraftResponse | null>>({
  A: null,
  B: null,
  C: null,
});

const listState = ref<ListState>("loading");
const listError = ref("");
const actionError = ref("");
const draftErrors = ref<Record<AgentId, string>>({
  A: "",
  B: "",
  C: "",
});
const eventError = ref("");
const saveMessage = ref("");
const openingSceneId = ref<string | null>(null);
const isCreating = ref(false);
const busyAction = ref<string | null>(null);

const previewLayer = ref<Layer>("inner");
const previews = ref<
  Record<string, ModelRequestPreviewResponse | undefined>
>({});
const previewError = ref("");

function cloneScene(scene: Scene): Scene {
  return JSON.parse(JSON.stringify(scene)) as Scene;
}

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message
    ? error.message
    : fallback;
}

function stageFor(agent: Agent): Layer {
  return agent.inner_context.turns.length >
    agent.outer_context.turns.length
    ? "outer"
    : "inner";
}

const activeAgent = computed(() =>
  currentScene.value?.agents.find(
    (agent) => agent.id === activeAgentId.value,
  ),
);

const activeStage = computed<Layer>(() =>
  activeAgent.value ? stageFor(activeAgent.value) : "inner",
);

const activeDraft = computed(
  () => drafts.value[activeAgentId.value],
);

const sceneModelAvailable = computed(() => {
  const model = currentScene.value?.model;
  return (
    model !== null &&
    model !== undefined &&
    modelOptions.value.some((option) => option.model === model)
  );
});

const sceneModelStatus = computed(() => {
  const model = currentScene.value?.model;
  if (model === null || model === undefined) {
    return "此场景尚未绑定模型。绑定后不能更换。";
  }
  if (!sceneModelAvailable.value) {
    return `绑定模型 ${model} 当前不可用。`;
  }
  return "";
});

const isDirty = computed(() => {
  if (currentScene.value === null || savedScene.value === null) {
    return false;
  }
  return (
    JSON.stringify(currentScene.value) !==
    JSON.stringify(savedScene.value)
  );
});

const hasAnyDraft = computed(() =>
  AGENT_IDS.some((agentId) => drafts.value[agentId] !== null),
);

const editorLocked = computed(
  () =>
    busyAction.value !== null ||
    openingSceneId.value !== null ||
    isCreating.value,
);

const canGenerate = computed(() => {
  const agent = activeAgent.value;
  if (
    agent === undefined ||
    !sceneModelAvailable.value ||
    isDirty.value ||
    editorLocked.value
  ) {
    return false;
  }
  return (
    activeStage.value === "outer" ||
    agent.pending_events.length > 0
  );
});

const canConfirm = computed(
  () =>
    activeDraft.value !== null &&
    activeDraft.value.layer === activeStage.value &&
    activeDraft.value.content.trim() !== "" &&
    !isDirty.value &&
    !editorLocked.value,
);

const stageTitle = computed(() => {
  const agent = activeAgent.value;
  if (agent === undefined) {
    return "";
  }
  if (activeStage.value === "outer") {
    return "等待外层人格";
  }
  if (agent.pending_events.length === 0) {
    return "等待外部事件";
  }
  return "等待内层人格";
});

const selectedPreview = computed(
  () =>
    previews.value[
      `${activeAgentId.value}:${previewLayer.value}`
    ],
);

const mixedTimeline = computed<TimelineItem[]>(() => {
  const agent = activeAgent.value;
  if (agent === undefined) {
    return [];
  }

  const items: TimelineItem[] = [];
  for (const turn of agent.inner_context.turns) {
    // A single inner round now consumes the whole queued batch at once, so
    // every consumed event is shown before its shared inner output. Event
    // sequence order alone is enough because the batch only ever contains
    // events that arrived before this turn's call sequence.
    for (const consumed of turn.consumed_events) {
      items.push(eventTimelineItem(consumed, "已处理"));
    }
    items.push({
      key: `inner:${turn.call_id}`,
      sequence: turn.sequence,
      rank: 1,
      kind: "inner",
      label: "内层输出",
      content: turn.output,
      input: turn.input,
      callId: turn.call_id,
      reasoning: turn.reasoning,
    });
  }
  for (const turn of agent.outer_context.turns) {
    items.push({
      key: `outer:${turn.call_id}`,
      sequence: turn.sequence,
      rank: 2,
      kind: "outer",
      label: "外层输出",
      content: turn.output,
      input: turn.input,
      callId: turn.call_id,
      reasoning: turn.reasoning,
    });
  }
  for (const event of agent.pending_events) {
    items.push(eventTimelineItem(event, "队列中"));
  }
  return items.sort(
    (left, right) =>
      left.sequence - right.sequence || left.rank - right.rank,
  );
});

function eventTimelineItem(
  event: ExternalEvent,
  status: string,
): TimelineItem {
  return {
    key: `event:${event.id}`,
    sequence: event.sequence,
    rank: 0,
    kind: "event",
    label: "外部事件",
    content: event.content,
    status,
  };
}

function compareSummaries(
  left: SceneSummary,
  right: SceneSummary,
): number {
  return left.name.localeCompare(right.name) ||
    left.id.localeCompare(right.id);
}

function upsertSummary(scene: Scene): void {
  const byId = new Map(
    sceneSummaries.value.map((summary) => [summary.id, summary]),
  );
  byId.set(scene.id, { id: scene.id, name: scene.name });
  sceneSummaries.value = [...byId.values()].sort(compareSummaries);
}

function syncEventEdits(scene: Scene): void {
  const next: Record<string, string> = {};
  for (const agent of scene.agents) {
    for (const event of agent.pending_events) {
      if (event.kind === "manual") {
        next[event.id] = event.content;
      }
    }
  }
  eventEdits.value = next;
}

function installScene(
  scene: Scene,
  options: {
    activeAgentId?: AgentId;
    preserveDrafts?: boolean;
  } = {},
): void {
  savedScene.value = cloneScene(scene);
  currentScene.value = cloneScene(scene);
  activeAgentId.value = options.activeAgentId ?? activeAgentId.value;
  if (!options.preserveDrafts) {
    drafts.value = { A: null, B: null, C: null };
    draftErrors.value = { A: "", B: "", C: "" };
  }
  syncEventEdits(scene);
  previews.value = {};
  previewError.value = "";
  eventError.value = "";
  saveMessage.value = "";
  actionError.value = "";
  upsertSummary(scene);
  bindingModel.value = modelOptions.value[0]?.model ?? "";
}

function closeScene(): void {
  // Reset every editor/preview surface back to the welcome card.
  savedScene.value = null;
  currentScene.value = null;
  drafts.value = { A: null, B: null, C: null };
  draftErrors.value = { A: "", B: "", C: "" };
  eventEdits.value = {};
  previews.value = {};
  previewError.value = "";
  eventError.value = "";
  saveMessage.value = "";
  actionError.value = "";
  bindingModel.value = modelOptions.value[0]?.model ?? "";
}

function confirmDiscardChanges(): boolean {
  if (!isDirty.value && !hasAnyDraft.value) {
    return true;
  }
  return window.confirm(
    "当前场景有未保存修改或浏览器草稿。确定要放弃吗？",
  );
}

async function refreshScenes(): Promise<void> {
  listState.value = "loading";
  listError.value = "";
  try {
    sceneSummaries.value = (await listScenes()).sort(compareSummaries);
    listState.value = "ready";
  } catch (error) {
    listState.value = "error";
    listError.value = errorMessage(error, "无法加载场景列表。");
  }
}

async function refreshModelOptions(): Promise<void> {
  modelOptionsError.value = "";
  try {
    modelOptions.value = (await getModelOptions()).options;
    if (
      !modelOptions.value.some(
        (option) => option.model === newSceneModel.value,
      )
    ) {
      newSceneModel.value = modelOptions.value[0]?.model ?? "";
    }
    bindingModel.value = modelOptions.value[0]?.model ?? "";
  } catch (error) {
    modelOptions.value = [];
    modelOptionsError.value = errorMessage(
      error,
      "无法加载模型选项。",
    );
  }
}

async function openScene(summary: SceneSummary): Promise<void> {
  if (
    editorLocked.value ||
    summary.id === currentScene.value?.id ||
    !confirmDiscardChanges()
  ) {
    return;
  }
  openingSceneId.value = summary.id;
  actionError.value = "";
  try {
    const scene = await getScene(summary.id);
    activeAgentId.value = "A";
    installScene(scene, { activeAgentId: "A" });
  } catch (error) {
    actionError.value = errorMessage(error, "无法打开场景。");
  } finally {
    openingSceneId.value = null;
  }
}

async function deleteCurrentScene(summary: SceneSummary): Promise<void> {
  const scene = currentScene.value;
  // Only the active scene has untrusted local state to discard.
  if (scene === null || scene.id !== summary.id) {
    return;
  }
  const hasLocalState = isDirty.value || hasAnyDraft.value;
  const message = hasLocalState
    ? `删除场景“${summary.name}”将同时丢弃此页面未保存的修改或草稿。确定删除？`
    : `删除场景“${summary.name}”？此操作不可恢复。`;
  if (
    editorLocked.value ||
    !window.confirm(message)
  ) {
    return;
  }
  busyAction.value = "scene-delete";
  actionError.value = "";
  try {
    await deleteScene(summary.id);
    sceneSummaries.value = sceneSummaries.value.filter(
      (item) => item.id !== summary.id,
    );
    closeScene();
  } catch (error) {
    actionError.value = errorMessage(error, "无法删除场景。");
  } finally {
    busyAction.value = null;
  }
}

async function submitNewScene(): Promise<void> {
  const name = newSceneName.value.trim();
  if (!name) {
    actionError.value = "请输入场景名称。";
    return;
  }
  if (!newSceneModel.value) {
    actionError.value = "请选择模型。";
    return;
  }
  if (editorLocked.value || !confirmDiscardChanges()) {
    return;
  }
  isCreating.value = true;
  actionError.value = "";
  try {
    const scene = await createScene(name, newSceneModel.value);
    newSceneName.value = "";
    activeAgentId.value = "A";
    installScene(scene, { activeAgentId: "A" });
  } catch (error) {
    actionError.value = errorMessage(error, "无法创建场景。");
  } finally {
    isCreating.value = false;
  }
}

async function bindCurrentSceneModel(): Promise<void> {
  const scene = currentScene.value;
  if (
    scene === null ||
    scene.model !== null ||
    !bindingModel.value ||
    isDirty.value ||
    editorLocked.value
  ) {
    return;
  }
  busyAction.value = "bind-model";
  actionError.value = "";
  try {
    const updated = await bindSceneModel(scene.id, bindingModel.value);
    installScene(updated, { activeAgentId: activeAgentId.value });
  } catch (error) {
    actionError.value = errorMessage(error, "无法绑定模型。");
  } finally {
    busyAction.value = null;
  }
}

function sceneUpdate(scene: Scene): SceneUpdate {
  return {
    name: scene.name,
    agents: scene.agents.map((agent) => ({
      id: agent.id,
      name: agent.name,
      inner_context: {
        system_prompt: agent.inner_context.system_prompt,
      },
      outer_context: {
        system_prompt: agent.outer_context.system_prompt,
      },
    })),
  };
}

async function saveCurrentScene(): Promise<void> {
  const scene = currentScene.value;
  if (scene === null || editorLocked.value || !isDirty.value) {
    return;
  }
  busyAction.value = "save";
  saveMessage.value = "";
  try {
    const updated = await saveScene(scene.id, sceneUpdate(scene));
    installScene(updated, {
      activeAgentId: activeAgentId.value,
      preserveDrafts: true,
    });
    saveMessage.value = "已保存";
  } catch (error) {
    saveMessage.value = errorMessage(error, "保存失败。");
  } finally {
    busyAction.value = null;
  }
}

async function addEvent(): Promise<void> {
  const scene = currentScene.value;
  const content = newEventContent.value[activeAgentId.value].trim();
  if (
    scene === null ||
    !content ||
    isDirty.value ||
    editorLocked.value
  ) {
    if (!content) {
      eventError.value = "请输入外部事件。";
    }
    return;
  }
  busyAction.value = "event-create";
  eventError.value = "";
  try {
    const updated = await createManualEvent(
      scene.id,
      activeAgentId.value,
      content,
    );
    newEventContent.value[activeAgentId.value] = "";
    installScene(updated, {
      activeAgentId: activeAgentId.value,
      preserveDrafts: true,
    });
  } catch (error) {
    eventError.value = errorMessage(error, "无法创建外部事件。");
  } finally {
    busyAction.value = null;
  }
}

function updateEventEdit(eventId: string, event: Event): void {
  eventEdits.value[eventId] = (
    event.target as HTMLTextAreaElement
  ).value;
}

async function saveEvent(event: ExternalEvent): Promise<void> {
  const scene = currentScene.value;
  const content = eventEdits.value[event.id]?.trim() ?? "";
  if (
    scene === null ||
    !content ||
    isDirty.value ||
    editorLocked.value
  ) {
    if (!content) {
      eventError.value = "外部事件不能为空。";
    }
    return;
  }
  busyAction.value = `event-edit:${event.id}`;
  eventError.value = "";
  try {
    const updated = await editManualEvent(
      scene.id,
      activeAgentId.value,
      event.id,
      content,
    );
    installScene(updated, {
      activeAgentId: activeAgentId.value,
      preserveDrafts: true,
    });
  } catch (error) {
    eventError.value = errorMessage(error, "无法修改外部事件。");
  } finally {
    busyAction.value = null;
  }
}

async function removeEvent(event: ExternalEvent): Promise<void> {
  const scene = currentScene.value;
  if (
    scene === null ||
    isDirty.value ||
    editorLocked.value ||
    !window.confirm(`删除这条手工事件？\n\n${event.content}`)
  ) {
    return;
  }
  busyAction.value = `event-delete:${event.id}`;
  eventError.value = "";
  try {
    const updated = await deleteManualEvent(
      scene.id,
      activeAgentId.value,
      event.id,
    );
    installScene(updated, {
      activeAgentId: activeAgentId.value,
      preserveDrafts: true,
    });
  } catch (error) {
    eventError.value = errorMessage(error, "无法删除外部事件。");
  } finally {
    busyAction.value = null;
  }
}

async function generateDraft(): Promise<void> {
  const scene = currentScene.value;
  const agentId = activeAgentId.value;
  const layer = activeStage.value;
  if (scene === null || !canGenerate.value) {
    return;
  }
  busyAction.value = `generate:${agentId}:${layer}`;
  draftErrors.value[agentId] = "";
  try {
    // A browser draft can survive a backend restart with a new model list.
    await refreshModelOptions();
    if (!sceneModelAvailable.value) {
      draftErrors.value[agentId] =
        "场景模型当前不可用，未发起生成。";
      return;
    }
    drafts.value[agentId] = await generateLayerDraft(
      scene.id,
      agentId,
      layer,
    );
  } catch (error) {
    // The previous editable draft remains available after an upstream error.
    draftErrors.value[agentId] = errorMessage(
      error,
      `无法生成${layer === "inner" ? "内层" : "外层"}草稿。`,
    );
  } finally {
    busyAction.value = null;
  }
}

function updateDraftContent(event: Event): void {
  const draft = drafts.value[activeAgentId.value];
  if (draft !== null) {
    draft.content = (event.target as HTMLTextAreaElement).value;
    draftErrors.value[activeAgentId.value] = "";
  }
}

function discardDraft(): void {
  drafts.value[activeAgentId.value] = null;
  draftErrors.value[activeAgentId.value] = "";
}

async function confirmDraft(): Promise<void> {
  const scene = currentScene.value;
  const draft = activeDraft.value;
  const agentId = activeAgentId.value;
  if (scene === null || draft === null || !canConfirm.value) {
    return;
  }
  busyAction.value = `confirm:${agentId}:${draft.layer}`;
  draftErrors.value[agentId] = "";
  try {
    const updated = await confirmLayerDraft(
      scene.id,
      agentId,
      draft.layer,
      {
        call_id: draft.call_id,
        event_ids: draft.event_ids,
        content: draft.content,
        state_token: draft.state_token,
        reasoning: activeDraft.value?.reasoning ?? [],
      },
    );
    drafts.value[agentId] = null;
    installScene(updated, {
      activeAgentId: agentId,
      preserveDrafts: true,
    });
  } catch (error) {
    draftErrors.value[agentId] = errorMessage(
      error,
      "确认失败；草稿已保留。",
    );
  } finally {
    busyAction.value = null;
  }
}

async function rollback(): Promise<void> {
  const scene = currentScene.value;
  if (
    scene === null ||
    scene.rollback_stack.length === 0 ||
    isDirty.value ||
    editorLocked.value
  ) {
    return;
  }
  const latest = scene.rollback_stack.at(-1);
  if (
    latest === undefined ||
    !window.confirm(
      `回退全场景最近一次已确认调用？\n\nAgent ${latest.agent_id} · ${
        latest.layer === "inner" ? "内层" : "外层"
      }`,
    )
  ) {
    return;
  }
  busyAction.value = "rollback";
  actionError.value = "";
  try {
    const updated = await rollbackLatestCall(scene.id);
    installScene(updated, {
      activeAgentId: activeAgentId.value,
    });
  } catch (error) {
    actionError.value = errorMessage(error, "无法回退最近调用。");
  } finally {
    busyAction.value = null;
  }
}

async function loadPreview(): Promise<void> {
  const scene = currentScene.value;
  if (
    scene === null ||
    editorLocked.value ||
    !sceneModelAvailable.value
  ) {
    return;
  }
  const key = `${activeAgentId.value}:${previewLayer.value}`;
  busyAction.value = `preview:${key}`;
  previewError.value = "";
  try {
    // Refresh without touching browser-held drafts or confirmation state.
    await refreshModelOptions();
    if (!sceneModelAvailable.value) {
      previewError.value = "场景模型当前不可用，未加载预览。";
      return;
    }
    previews.value[key] = await getModelRequestPreview(
      scene.id,
      activeAgentId.value,
      previewLayer.value,
    );
  } catch (error) {
    previewError.value = errorMessage(error, "无法加载请求预览。");
  } finally {
    busyAction.value = null;
  }
}

function prettyJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function handleBeforeUnload(event: BeforeUnloadEvent): void {
  if (isDirty.value || hasAnyDraft.value) {
    event.preventDefault();
  }
}

watch(
  activeAgentId,
  () => {
    previewLayer.value = activeStage.value;
    previewError.value = "";
    eventError.value = "";
  },
);

watch(activeStage, (stage) => {
  previewLayer.value = stage;
});

onMounted(() => {
  window.addEventListener("beforeunload", handleBeforeUnload);
  void refreshModelOptions();
  void refreshScenes();
});

onBeforeUnmount(() => {
  window.removeEventListener("beforeunload", handleBeforeUnload);
});
</script>

<template>
  <div class="app-shell">
    <header class="topbar">
      <div>
        <p class="eyebrow">TWO-LAYER PERSONA LAB</p>
        <h1>AI 小镇</h1>
      </div>
      <p class="topbar-copy">
        外部事件进入队列，内层先形成判断，外层再决定真正说出的话。
      </p>
    </header>

    <main class="workspace">
      <aside class="scene-sidebar" aria-label="场景列表">
        <form class="new-scene-form" @submit.prevent="submitNewScene">
          <label for="new-scene-name">新场景</label>
          <div>
            <input
              id="new-scene-name"
              v-model="newSceneName"
              type="text"
              placeholder="例如：雨夜车站"
              :disabled="editorLocked"
            />
            <select
              v-model="newSceneModel"
              aria-label="新场景模型"
              :disabled="editorLocked || modelOptions.length === 0"
            >
              <option
                v-for="option in modelOptions"
                :key="option.model"
                :value="option.model"
              >
                {{ option.model }}
              </option>
            </select>
            <button type="submit" :disabled="editorLocked">
              {{ isCreating ? "创建中…" : "创建" }}
            </button>
          </div>
          <p v-if="modelOptionsError" class="inline-error" role="alert">
            {{ modelOptionsError }}
          </p>
        </form>

        <div class="scene-list-wrap">
          <div class="sidebar-heading">
            <h2>本机场景</h2>
            <button
              type="button"
              class="text-button"
              :disabled="editorLocked"
              @click="refreshScenes"
            >
              刷新
            </button>
          </div>
          <p v-if="listState === 'loading'" class="muted">正在加载…</p>
          <div v-else-if="listState === 'error'" class="error-box">
            {{ listError }}
          </div>
          <p v-else-if="sceneSummaries.length === 0" class="muted">
            还没有场景。
          </p>
          <ul v-else class="scene-list">
            <li v-for="summary in sceneSummaries" :key="summary.id">
              <button
                type="button"
                :class="{ active: currentScene?.id === summary.id }"
                :disabled="editorLocked"
                @click="openScene(summary)"
              >
                <strong>{{ summary.name }}</strong>
                <small>{{ summary.id.slice(0, 8) }}</small>
              </button>
              <button
                v-if="currentScene?.id === summary.id"
                type="button"
                class="text-button scene-delete-button"
                :disabled="editorLocked"
                :title="`删除场景“${summary.name}”`"
                aria-label="删除场景"
                @click="deleteCurrentScene(summary)"
              >
                删除
              </button>
            </li>
          </ul>
        </div>

        <p class="storage-note">
          已确认状态写入本机 JSON；未确认草稿只在当前浏览器页面中存在。
        </p>
      </aside>

      <section class="content-region">
        <div v-if="actionError" class="page-error" role="alert">
          <span>{{ actionError }}</span>
          <button
            type="button"
            aria-label="关闭错误"
            @click="actionError = ''"
          >
            ×
          </button>
        </div>

        <div v-if="currentScene === null" class="welcome-card">
          <div class="layer-mark" aria-hidden="true">
            <span>内</span><span>外</span>
          </div>
          <p class="eyebrow">MANUAL STEP-BY-STEP</p>
          <h2>选择或创建一个场景</h2>
          <p>
            每个 Agent 都有彼此隔离的内层与外层完整历史。系统不会自动推进。
          </p>
        </div>

        <template v-else>
          <header class="scene-toolbar">
            <label for="scene-name">
              <span>场景名称</span>
              <input
                id="scene-name"
                v-model="currentScene.name"
                type="text"
                :disabled="editorLocked"
              />
            </label>
            <div class="scene-model">
              <span>场景模型</span>
              <strong>{{ currentScene.model ?? "未绑定" }}</strong>
            </div>
            <div class="toolbar-actions">
              <span
                class="save-state"
                :class="{ dirty: isDirty }"
                aria-live="polite"
              >
                {{
                  saveMessage ||
                  (isDirty ? "有未保存修改" : "已与磁盘同步")
                }}
              </span>
              <button
                type="button"
                class="secondary-button"
                :disabled="
                  currentScene.rollback_stack.length === 0 ||
                  isDirty ||
                  editorLocked
                "
                @click="rollback"
              >
                {{
                  busyAction === "rollback"
                    ? "回退中…"
                    : "回退最近确认"
                }}
              </button>
              <button
                type="button"
                class="primary-button"
                :disabled="!isDirty || editorLocked"
                @click="saveCurrentScene"
              >
                {{ busyAction === "save" ? "保存中…" : "保存设定" }}
              </button>
            </div>
          </header>

          <section
            v-if="sceneModelStatus"
            class="model-status"
            role="status"
          >
            <div>
              <strong>模型调用已停用</strong>
              <p>{{ sceneModelStatus }}</p>
            </div>
            <form
              v-if="currentScene.model === null"
              @submit.prevent="bindCurrentSceneModel"
            >
              <select
                v-model="bindingModel"
                aria-label="绑定场景模型"
                :disabled="editorLocked || modelOptions.length === 0"
              >
                <option
                  v-for="option in modelOptions"
                  :key="option.model"
                  :value="option.model"
                >
                  {{ option.model }}
                </option>
              </select>
              <button
                type="submit"
                class="secondary-button"
                :disabled="editorLocked || isDirty || !bindingModel"
              >
                {{
                  busyAction === "bind-model" ? "绑定中…" : "永久绑定"
                }}
              </button>
            </form>
          </section>

          <nav class="agent-tabs" aria-label="选择 Agent">
            <button
              v-for="agent in currentScene.agents"
              :key="agent.id"
              type="button"
              :class="{ active: activeAgentId === agent.id }"
              :disabled="editorLocked"
              @click="activeAgentId = agent.id"
            >
              <span class="agent-id">{{ agent.id }}</span>
              <span>
                <small>Agent {{ agent.id }}</small>
                <strong>{{ agent.name }}</strong>
              </span>
              <em>
                {{
                  stageFor(agent) === "outer"
                    ? "待外层"
                    : agent.pending_events.length
                      ? `${agent.pending_events.length} 个事件`
                      : "待事件"
                }}
              </em>
            </button>
          </nav>

          <div v-if="activeAgent" class="agent-layout">
            <section class="history-panel">
              <header class="panel-heading">
                <div>
                  <p class="eyebrow">AGENT {{ activeAgent.id }}</p>
                  <h2>{{ activeAgent.name }}</h2>
                  <p>按发生顺序混合展示，只呈现这位 Agent 获得的信息。</p>
                </div>
                <span
                  class="stage-badge"
                  :class="`stage-badge--${activeStage}`"
                >
                  {{ stageTitle }}
                </span>
              </header>

              <div class="timeline" aria-live="polite">
                <p v-if="mixedTimeline.length === 0" class="timeline-empty">
                  尚无外部事件或已确认调用。
                </p>
                <ol v-else>
                  <li
                    v-for="item in mixedTimeline"
                    :key="item.key"
                    class="timeline-item"
                    :class="`timeline-item--${item.kind}`"
                  >
                    <div class="timeline-rail">
                      <span>{{ item.sequence }}</span>
                    </div>
                    <article>
                      <header>
                        <strong>{{ item.label }}</strong>
                        <span v-if="item.status">{{ item.status }}</span>
                      </header>
                      <p>{{ item.content }}</p>
                      <details v-if="item.input" class="call-input">
                        <summary>查看本次实际输入</summary>
                        <pre>{{ item.input }}</pre>
                        <small>call {{ item.callId }}</small>
                      </details>
                      <details
                        v-if="item.reasoning?.length"
                        class="draft-reasoning"
                      >
                        <summary>本次已落盘的 reasoning</summary>
                        <article
                          v-for="(block, index) in item.reasoning"
                          :key="`${block.type}:${index}`"
                        >
                          <strong>{{ block.type }}</strong>
                          <pre>{{ block.text }}</pre>
                        </article>
                      </details>
                    </article>
                  </li>
                </ol>
              </div>

              <footer class="draft-panel">
                <div class="draft-heading">
                  <div>
                    <p class="eyebrow">
                      {{ activeStage === "inner" ? "INNER" : "OUTER" }}
                      STEP
                    </p>
                    <h3>
                      {{
                        activeStage === "inner"
                          ? "内层人格草稿"
                          : "外层人格草稿"
                      }}
                    </h3>
                  </div>
                  <p>
                    {{
                      activeStage === "inner"
                        ? "可使用多行自然文本；确认后一次性消费本 Agent 队列中的全部待处理事件。"
                        : "必须为单行 To X: 正文；确认后才会路由事件。"
                    }}
                  </p>
                </div>

                <textarea
                  id="layer-draft-content"
                  :value="
                    activeDraft?.layer === activeStage
                      ? activeDraft.content
                      : ''
                  "
                  :rows="activeStage === 'inner' ? 6 : 3"
                  :placeholder="
                    activeStage === 'inner'
                      ? '先生成内层草稿'
                      : 'To B: 正文'
                  "
                  :disabled="
                    editorLocked ||
                    activeDraft?.layer !== activeStage
                  "
                  @input="updateDraftContent"
                ></textarea>

                <dl
                  v-if="activeDraft?.layer === activeStage"
                  class="usage-grid"
                >
                  <div>
                    <dt>缓存写入</dt>
                    <dd>
                      {{
                        activeDraft.usage.cache_creation_input_tokens
                      }}
                    </dd>
                  </div>
                  <div>
                    <dt>缓存读取</dt>
                    <dd>{{ activeDraft.usage.cache_read_input_tokens }}</dd>
                  </div>
                  <div>
                    <dt>未缓存输入</dt>
                    <dd>{{ activeDraft.usage.input_tokens }}</dd>
                  </div>
                  <div>
                    <dt>输出</dt>
                    <dd>{{ activeDraft.usage.output_tokens }}</dd>
                  </div>
                </dl>

                <details
                  v-if="
                    activeDraft?.layer === activeStage &&
                    activeDraft.reasoning.length > 0
                  "
                  class="draft-reasoning"
                >
                  <summary>本次临时 reasoning</summary>
                  <article
                    v-for="(block, index) in activeDraft.reasoning"
                    :key="`${block.type}:${index}`"
                  >
                    <strong>{{ block.type }}</strong>
                    <pre>{{ block.text }}</pre>
                  </article>
                  <small>
                    确认后落盘到场景历史、不再回传模型。
                  </small>
                </details>

                <details
                  v-if="activeDraft?.layer === activeStage"
                  class="draft-request"
                >
                  <summary>本次生成的实际请求快照</summary>
                  <pre>{{ prettyJson(activeDraft.request_snapshot) }}</pre>
                </details>

                <p
                  v-if="draftErrors[activeAgent.id]"
                  class="inline-error"
                  role="alert"
                >
                  {{ draftErrors[activeAgent.id] }}
                </p>
                <p v-else-if="!sceneModelAvailable" class="hint">
                  场景模型未绑定或当前不可用，不能预览或生成新草稿；已有有效草稿仍可确认。
                </p>
                <p v-else-if="isDirty" class="hint">
                  请先保存设定，再生成或确认。
                </p>
                <p
                  v-else-if="
                    activeStage === 'inner' &&
                    activeAgent.pending_events.length === 0
                  "
                  class="hint"
                >
                  没有待处理事件，不能启动内层推理。
                </p>
                <p v-else class="hint">
                  生成和重新生成各调用模型一次；确认不调用模型。
                </p>

                <div class="draft-actions">
                  <button
                    v-if="activeDraft?.layer === activeStage"
                    type="button"
                    class="text-button"
                    :disabled="editorLocked"
                    @click="discardDraft"
                  >
                    放弃草稿
                  </button>
                  <button
                    type="button"
                    class="secondary-button"
                    :disabled="!canGenerate"
                    @click="generateDraft"
                  >
                    {{
                      busyAction?.startsWith("generate:")
                        ? "生成中…"
                        : activeDraft?.layer === activeStage
                          ? "重新生成"
                          : `生成${
                              activeStage === "inner" ? "内层" : "外层"
                            }草稿`
                    }}
                  </button>
                  <button
                    type="button"
                    class="primary-button"
                    :disabled="!canConfirm"
                    @click="confirmDraft"
                  >
                    {{
                      busyAction?.startsWith("confirm:")
                        ? "确认中…"
                        : `确认${
                            activeStage === "inner" ? "内层" : "外层"
                          }`
                    }}
                  </button>
                </div>
              </footer>
            </section>

            <aside class="control-column">
              <section class="queue-card">
                <header>
                  <div>
                    <p class="eyebrow">FIFO QUEUE</p>
                    <h3>待处理外部事件</h3>
                  </div>
                  <span>{{ activeAgent.pending_events.length }}</span>
                </header>

                <form @submit.prevent="addEvent">
                  <label for="new-event">给当前 Agent 添加手工事件</label>
                  <textarea
                    id="new-event"
                    v-model="newEventContent[activeAgent.id]"
                    rows="3"
                    placeholder="只进入当前 Agent 的队列"
                    :disabled="editorLocked || isDirty"
                  ></textarea>
                  <button
                    type="submit"
                    class="secondary-button"
                    :disabled="editorLocked || isDirty"
                  >
                    添加到队尾
                  </button>
                </form>

                <p
                  v-if="eventError"
                  class="inline-error"
                  role="alert"
                >
                  {{ eventError }}
                </p>

                <ol v-if="activeAgent.pending_events.length" class="queue-list">
                  <li
                    v-for="(event, index) in activeAgent.pending_events"
                    :key="event.id"
                  >
                    <header>
                      <span>#{{ index + 1 }} {{ index === 0 ? "队首" : "" }}</span>
                      <em>
                        {{
                          event.kind === "manual"
                            ? "手工事件"
                            : "Agent 事件 · 不可修改"
                        }}
                      </em>
                    </header>
                    <template v-if="event.kind === 'manual'">
                      <textarea
                        :value="eventEdits[event.id]"
                        rows="3"
                        :disabled="editorLocked || isDirty"
                        @input="updateEventEdit(event.id, $event)"
                      ></textarea>
                      <div>
                        <button
                          type="button"
                          class="text-button"
                          :disabled="editorLocked || isDirty"
                          @click="removeEvent(event)"
                        >
                          删除
                        </button>
                        <button
                          type="button"
                          class="small-button"
                          :disabled="
                            editorLocked ||
                            isDirty ||
                            eventEdits[event.id] === event.content
                          "
                          @click="saveEvent(event)"
                        >
                          保存修改
                        </button>
                      </div>
                    </template>
                    <p v-else>{{ event.content }}</p>
                  </li>
                </ol>
                <p v-else class="muted">队列为空。</p>
              </section>

              <details class="settings-card">
                <summary>
                  <span>
                    <strong>Agent 与双层提示词</strong>
                    <small>两份完整文本分别保存，不做槽位拼接</small>
                  </span>
                </summary>
                <div class="settings-body">
                  <label for="agent-name">
                    <span>显示名</span>
                    <input
                      id="agent-name"
                      v-model="activeAgent.name"
                      type="text"
                      :disabled="editorLocked"
                    />
                  </label>
                  <label for="inner-system-prompt">
                    <span>内层 system prompt</span>
                    <textarea
                      id="inner-system-prompt"
                      v-model="activeAgent.inner_context.system_prompt"
                      rows="14"
                      :disabled="editorLocked"
                    ></textarea>
                  </label>
                  <label for="outer-system-prompt">
                    <span>外层 system prompt</span>
                    <textarea
                      id="outer-system-prompt"
                      v-model="activeAgent.outer_context.system_prompt"
                      rows="16"
                      :disabled="editorLocked"
                    ></textarea>
                  </label>
                </div>
              </details>

              <details class="preview-card">
                <summary>
                  <span>
                    <strong>模型请求预览</strong>
                    <small>调用前的协议无关完整上下文</small>
                  </span>
                </summary>
                <div class="preview-body">
                  <div class="layer-switch" role="group" aria-label="预览层级">
                    <button
                      v-for="layer in LAYERS"
                      :key="layer"
                      type="button"
                      :class="{ active: previewLayer === layer }"
                      @click="previewLayer = layer"
                    >
                      {{ layer === "inner" ? "内层" : "外层" }}
                    </button>
                  </div>
                  <button
                    type="button"
                    class="secondary-button"
                    :disabled="editorLocked || !sceneModelAvailable"
                    @click="loadPreview"
                  >
                    {{
                      busyAction?.startsWith("preview:")
                        ? "加载中…"
                        : selectedPreview
                          ? "刷新预览"
                          : "加载预览"
                    }}
                  </button>
                  <p v-if="isDirty" class="hint">
                    当前有未保存设定；接口预览仍基于磁盘版本。
                  </p>
                  <p
                    v-if="previewError"
                    class="inline-error"
                    role="alert"
                  >
                    {{ previewError }}
                  </p>

                  <template v-if="selectedPreview">
                    <div class="request-preview">
                      <div class="readable-context">
                        <article
                          v-for="(block, index) in selectedPreview.context"
                          :key="`${block.role}:${index}`"
                        >
                          <strong>{{ block.role }}</strong>
                          <pre>{{ block.text }}</pre>
                        </article>
                      </div>
                    </div>
                  </template>
                </div>
              </details>
            </aside>
          </div>
        </template>
      </section>
    </main>
  </div>
</template>
