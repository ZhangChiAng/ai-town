<script setup lang="ts">
import { onMounted, ref } from "vue";

type ConnectionState = "loading" | "connected" | "error";

interface HealthResponse {
  status: "ok";
}

const connectionState = ref<ConnectionState>("loading");

function isHealthResponse(value: unknown): value is HealthResponse {
  return (
    typeof value === "object" &&
    value !== null &&
    "status" in value &&
    value.status === "ok"
  );
}

async function checkHealth(): Promise<void> {
  connectionState.value = "loading";

  try {
    const response = await fetch("/api/health");

    if (!response.ok) {
      throw new Error(`Health check failed with status ${response.status}`);
    }

    const body: unknown = await response.json();

    if (!isHealthResponse(body)) {
      throw new Error("Health check returned an unexpected response");
    }

    connectionState.value = "connected";
  } catch {
    connectionState.value = "error";
  }
}

onMounted(checkHealth);
</script>

<template>
  <main class="shell">
    <section class="status-card" aria-labelledby="page-title">
      <p class="eyebrow">AI TOWN</p>
      <h1 id="page-title">AI 小镇</h1>
      <p class="intro">最小实验系统正在等待下一阶段。</p>

      <div class="connection" aria-live="polite">
        <div
          v-if="connectionState === 'loading'"
          class="status status--loading"
          role="status"
        >
          <span class="status-dot" aria-hidden="true"></span>
          <span>正在连接后端…</span>
        </div>

        <div
          v-else-if="connectionState === 'connected'"
          class="status status--connected"
          role="status"
        >
          <span class="status-dot" aria-hidden="true"></span>
          <span>后端已连接</span>
        </div>

        <div v-else class="status status--error" role="alert">
          <span class="status-dot" aria-hidden="true"></span>
          <span>无法连接后端</span>
        </div>

        <button
          v-if="connectionState === 'error'"
          class="retry-button"
          type="button"
          @click="checkHealth"
        >
          重新连接
        </button>
      </div>
    </section>
  </main>
</template>
