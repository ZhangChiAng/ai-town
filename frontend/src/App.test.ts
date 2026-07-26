import { createApp, nextTick } from "vue";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App.vue";

let container: HTMLDivElement;
let unmountApp: (() => void) | undefined;

function healthResponse(body: unknown, ok = true): Response {
  return {
    ok,
    status: ok ? 200 : 503,
    json: vi.fn().mockResolvedValue(body),
  } as unknown as Response;
}

function mountApp(): void {
  const app = createApp(App);
  app.mount(container);
  unmountApp = () => app.unmount();
}

async function flushAsyncUpdates(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
  await nextTick();
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
  it("shows the loading state while the health request is pending", () => {
    vi.stubGlobal("fetch", vi.fn(() => new Promise<Response>(() => {})));

    mountApp();

    expect(container.textContent).toContain("正在连接后端…");
  });

  it("shows the connected state after a successful health request", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(healthResponse({ status: "ok" }));
    vi.stubGlobal("fetch", fetchMock);

    mountApp();
    await flushAsyncUpdates();

    expect(fetchMock).toHaveBeenCalledWith("/api/health");
    expect(container.textContent).toContain("后端已连接");
  });

  it("shows the error state when the health request fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));

    mountApp();
    await flushAsyncUpdates();

    expect(container.textContent).toContain("无法连接后端");
    expect(container.querySelector("button")?.textContent?.trim()).toBe(
      "重新连接",
    );
  });
});
