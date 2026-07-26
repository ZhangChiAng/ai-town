<script setup lang="ts">
import {
  computed,
  onBeforeUnmount,
  onMounted,
  ref,
} from "vue";

import {
  createScene,
  getScene,
  listScenes,
  saveScene,
  sendMessage,
} from "./api";
import {
  AGENT_IDS,
  type AgentId,
  type Scene,
  type SceneSummary,
  type SceneUpdate,
} from "./types";

type ListState = "loading" | "ready" | "error";
type SaveState = "idle" | "saving" | "success" | "error";
interface MessageDraft {
  recipientId: AgentId | "";
  content: string;
}

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

let listRequestToken = 0;
let summaryMutationVersion = 0;

function cloneScene(scene: Scene): Scene {
  return JSON.parse(JSON.stringify(scene)) as Scene;
}

function emptyMessageDrafts(): Record<AgentId, MessageDraft> {
  return {
    A: { recipientId: "", content: "" },
    B: { recipientId: "", content: "" },
    C: { recipientId: "", content: "" },
  };
}

function emptyMessageErrors(): Record<AgentId, string> {
  return { A: "", B: "", C: "" };
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

const recipientOptions = computed(() =>
  AGENT_IDS.filter((agentId) => agentId !== activeAgentId.value),
);

const hasMessageDrafts = computed(() =>
  AGENT_IDS.some((agentId) => {
    const draft = messageDrafts.value[agentId];
    return draft.recipientId !== "" || draft.content !== "";
  }),
);

const editorLocked = computed(
  () =>
    saveState.value === "saving" ||
    sendingAgentId.value !== null ||
    isCreating.value ||
    openingSceneId.value !== null,
);

const canSendMessage = computed(
  () =>
    currentScene.value !== null &&
    !isDirty.value &&
    !editorLocked.value &&
    activeMessageDraft.value.recipientId !== "" &&
    activeMessageDraft.value.content.trim() !== "",
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
  if (!isDirty.value && !hasMessageDrafts.value) {
    return true;
  }
  return window.confirm(
    "当前场景有未保存的更改或未确认的消息草稿。确定要放弃吗？",
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
    sendingAgentId.value !== null
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
      ({ id, name, persona, desire, fear, memory }) => ({
        id,
        name,
        persona,
        desire,
        fear,
        memory,
      }),
    ),
  };
}

async function saveCurrentScene(): Promise<void> {
  const scene = currentScene.value;
  if (
    scene === null ||
    saveState.value === "saving" ||
    sendingAgentId.value !== null
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

async function confirmMessage(): Promise<void> {
  const scene = currentScene.value;
  const senderId = activeAgentId.value;
  const draft = messageDrafts.value[senderId];
  if (
    scene === null ||
    isDirty.value ||
    sendingAgentId.value !== null ||
    draft.recipientId === "" ||
    draft.content.trim() === ""
  ) {
    return;
  }

  sendingAgentId.value = senderId;
  messageErrors.value[senderId] = "";

  try {
    const updated = await sendMessage(scene.id, {
      sender_id: senderId,
      recipient_id: draft.recipientId,
      content: draft.content.trim(),
    });
    installScene(updated, senderId, true);
    messageDrafts.value[senderId] = {
      recipientId: "",
      content: "",
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

function handleBeforeUnload(event: BeforeUnloadEvent): void {
  if (!isDirty.value && !hasMessageDrafts.value) {
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
        手工消息 · 显式确认
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
          每个场景独立保存为本机 JSON 文件。本阶段不提供删除。
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
            <div class="agent-editor-heading">
              <div>
                <p class="eyebrow">AGENT {{ activeAgent.id }}</p>
                <h2>{{ activeAgent.name || "未命名 Agent" }}</h2>
              </div>
              <p>
                固定身份
                <strong>{{ activeAgent.id }}</strong>
              </p>
            </div>

            <div class="field-grid">
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
            </div>

            <section
              class="message-card"
              aria-labelledby="message-draft-title"
            >
              <div class="message-card-heading">
                <div>
                  <p class="eyebrow">MANUAL MESSAGE</p>
                  <h3 id="message-draft-title">手工消息草稿</h3>
                </div>
                <p>
                  发送者
                  <strong>Agent {{ activeAgent.id }}</strong>
                </p>
              </div>

              <div class="message-fields">
                <label class="message-recipient" for="message-recipient">
                  <span>接收者</span>
                  <select
                    id="message-recipient"
                    v-model="activeMessageDraft.recipientId"
                    :disabled="editorLocked"
                    @change="markMessageDraftEdited"
                  >
                    <option value="">请选择接收者</option>
                    <option
                      v-for="agentId in recipientOptions"
                      :key="agentId"
                      :value="agentId"
                    >
                      Agent {{ agentId }} ·
                      {{
                        currentScene.agents.find(
                          (agent) => agent.id === agentId,
                        )?.name
                      }}
                    </option>
                  </select>
                </label>

                <label class="message-content" for="message-content">
                  <span>正文</span>
                  <textarea
                    id="message-content"
                    v-model="activeMessageDraft.content"
                    rows="4"
                    :disabled="editorLocked"
                    placeholder="手工填写这位 Agent 要发送的内容"
                    @input="markMessageDraftEdited"
                  ></textarea>
                </label>
              </div>

              <div class="message-actions">
                <p v-if="isDirty" class="message-hint">
                  请先保存场景，再确认发送。
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
            </section>

            <section class="timeline-card" aria-labelledby="timeline-title">
              <div class="timeline-heading">
                <p class="eyebrow">TIMELINE</p>
                <h3 id="timeline-title">个人时间线</h3>
                <small>按确认顺序排列，仅当前 Agent 可见的记录</small>
              </div>
              <p
                v-if="activeAgent.timeline.length === 0"
                class="timeline-empty"
              >
                时间线为空
              </p>
              <ol v-else class="timeline-list">
                <li
                  v-for="record in activeAgent.timeline"
                  :key="`${record.message_id}-${record.direction}`"
                >
                  <span class="timeline-direction">
                    {{
                      record.direction === "sent"
                        ? `发送给 ${record.counterpart_id}：`
                        : `${record.counterpart_id}：`
                    }}
                  </span>
                  <span>{{ record.content }}</span>
                </li>
              </ol>
            </section>
          </div>
        </template>
      </section>
    </main>
  </div>
</template>
