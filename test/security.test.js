import assert from "node:assert/strict";
import test from "node:test";

import worker from "../src/worker.js";

test("shared upstream credentials require Worker authentication", async () => {
  const response = await worker.fetch(
    new Request("https://worker.example/health"),
    { UNLIMITED_SURF_API_KEY: "upstream-secret" },
  );

  assert.equal(response.status, 503);
  assert.equal((await response.json()).error.code, "worker_api_key_required");
});

test("raw proxy strips client credentials and unrelated headers", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    assert.equal(String(url), "https://unlimited.surf/api/usage");
    assert.equal(init.headers.get("authorization"), "Bearer upstream-secret");
    assert.equal(init.headers.get("x-api-key"), null);
    assert.equal(init.headers.get("anthropic-api-key"), null);
    assert.equal(init.headers.get("cookie"), null);
    assert.equal(init.headers.get("x-custom"), null);
    return Response.json({ ok: true });
  };

  try {
    const response = await worker.fetch(
      new Request("https://worker.example/api/usage", {
        headers: {
          "x-api-key": "worker-secret",
          "anthropic-api-key": "worker-secret",
          cookie: "session=sensitive",
          "x-custom": "sensitive",
        },
      }),
      {
        UNLIMITED_SURF_API_KEY: "upstream-secret",
        WORKER_API_KEY: "worker-secret",
      },
    );

    assert.equal(response.status, 200);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("bring-your-own upstream keys are translated to Authorization", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (_url, init) => {
    assert.equal(init.headers.get("authorization"), "Bearer caller-upstream-key");
    assert.equal(init.headers.get("x-api-key"), null);
    return Response.json({ ok: true });
  };

  try {
    const response = await worker.fetch(
      new Request("https://worker.example/api/usage", {
        headers: { "x-api-key": "caller-upstream-key" },
      }),
      {},
    );

    assert.equal(response.status, 200);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("legacy public relay requires an explicit opt-in", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (_url, init) => {
    assert.equal(init.headers.get("authorization"), "Bearer upstream-secret");
    return Response.json({ ok: true });
  };

  try {
    const response = await worker.fetch(
      new Request("https://worker.example/api/usage"),
      {
        ALLOW_UNAUTHENTICATED: "true",
        UNLIMITED_SURF_API_KEY: "upstream-secret",
      },
    );

    assert.equal(response.status, 200);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
