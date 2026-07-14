import { test } from "node:test";
import assert from "node:assert/strict";

import {
  normalizePath,
  chooseUnlimitedRoute,
  buildUnlimitedPayload,
  buildAnthropicUnlimitedPayload,
  responsesToChatBody,
  messagesToText,
  anthropicMessagesToText,
  inputToText,
  contentToText,
  latestUserText,
  hasWebSearchTool,
  reasoningEffort,
  mapUpstreamModel,
  toAnthropicModel,
  anthropicVersionedId,
  providerFromModel,
  fallbackModels,
  parseSseJson,
  openAIStopReason,
  anthropicStopReason,
  usageFromText,
  responseUsageFromText,
  anthropicUsageFromText,
  estimateTokens,
  stripTrailingSlash,
  bytesToBase64,
  looksLikeAnthropicRequest,
  clientApiKey,
  constantTimeEqual,
  validateWorkerApiKey,
  optionalUpstreamApiKey,
  upstreamApiKey,
  upstreamBase,
  serviceInfo,
  agentSetup,
  codexSetup,
  mcpInfo,
} from "../src/worker.js";

// Minimal Request-like stub carrying only what the helpers read: headers + url.
function makeRequest({ headers = {}, url = "https://worker.example.com/v1/models" } = {}) {
  const lower = new Map(
    Object.entries(headers).map(([k, v]) => [k.toLowerCase(), String(v)])
  );
  return {
    url,
    headers: {
      get: (name) => (lower.has(name.toLowerCase()) ? lower.get(name.toLowerCase()) : null),
      has: (name) => lower.has(name.toLowerCase()),
    },
  };
}

test("normalizePath collapses slashes and strips trailing slash", () => {
  assert.equal(normalizePath(""), "/");
  assert.equal(normalizePath(undefined), "/");
  assert.equal(normalizePath("/"), "/");
  assert.equal(normalizePath("/v1//models/"), "/v1/models");
  assert.equal(normalizePath("///"), "/");
  assert.equal(normalizePath("/anthropic///v1/messages/"), "/anthropic/v1/messages");
});

test("chooseUnlimitedRoute picks the correct upstream route", () => {
  assert.equal(chooseUnlimitedRoute({}), "/api/chat");
  assert.equal(chooseUnlimitedRoute({ messages: [{ role: "user", content: "hi" }] }), "/api/chat");
  assert.equal(chooseUnlimitedRoute({ models: ["a", "b"] }), "/api/merge");
  assert.equal(chooseUnlimitedRoute({ models: ["a"] }), "/api/chat");
  assert.equal(chooseUnlimitedRoute({ merge: true }), "/api/merge");
  assert.equal(chooseUnlimitedRoute({ merge_ai: true }), "/api/merge");
  assert.equal(chooseUnlimitedRoute({ query: "weather" }), "/api/search");
  assert.equal(chooseUnlimitedRoute({ web_search: true }), "/api/search");
  assert.equal(chooseUnlimitedRoute({ web_search_options: {} }), "/api/search");
  assert.equal(chooseUnlimitedRoute({ tools: [{ type: "web_search" }] }), "/api/search");
  // merge takes precedence over search
  assert.equal(chooseUnlimitedRoute({ models: ["a", "b"], query: "x" }), "/api/merge");
});

test("buildUnlimitedPayload builds a search payload", () => {
  const payload = buildUnlimitedPayload(
    { query: "what is the weather", model: "gpt-5", effort: "high" },
    "/api/search"
  );
  assert.deepEqual(payload, {
    query: "what is the weather",
    model: "gateway-gpt-5",
    effort: "high",
  });
});

test("buildUnlimitedPayload derives search query from messages/input/prompt", () => {
  assert.equal(
    buildUnlimitedPayload({ messages: [{ role: "user", content: "from msg" }] }, "/api/search").query,
    "from msg"
  );
  assert.equal(buildUnlimitedPayload({ input: "from input" }, "/api/search").query, "from input");
  assert.equal(buildUnlimitedPayload({ prompt: "from prompt" }, "/api/search").query, "from prompt");
  assert.equal(buildUnlimitedPayload({}, "/api/search").query, "");
});

test("buildUnlimitedPayload builds a chat payload with default effort", () => {
  const payload = buildUnlimitedPayload(
    { messages: [{ role: "user", content: "hello" }] },
    "/api/chat"
  );
  assert.equal(payload.message, "user: hello");
  assert.equal(payload.model, "gateway-gpt-5-5");
  assert.equal(payload.effort, "medium");
  assert.equal("models" in payload, false);
});

test("buildUnlimitedPayload maps merge models and prefers explicit message", () => {
  const payload = buildUnlimitedPayload(
    { message: "direct", models: ["gpt-5", "claude-opus-4-7-20260101"] },
    "/api/merge"
  );
  assert.equal(payload.message, "direct");
  assert.deepEqual(payload.models, ["gateway-gpt-5", "gateway-claude-opus-4-7"]);
});

test("buildUnlimitedPayload merge without models is undefined", () => {
  const payload = buildUnlimitedPayload({ message: "hi" }, "/api/merge");
  assert.equal(payload.models, undefined);
});

test("buildAnthropicUnlimitedPayload search + chat", () => {
  const search = buildAnthropicUnlimitedPayload(
    { messages: [{ role: "user", content: "latest news" }] },
    "/api/search"
  );
  assert.equal(search.query, "latest news");

  const chat = buildAnthropicUnlimitedPayload(
    { system: "be nice", messages: [{ role: "user", content: "hi" }] },
    "/api/chat"
  );
  assert.equal(chat.message, "system: be nice\n\nuser: hi");
  assert.equal(chat.model, "gateway-gpt-5-5");
});

test("responsesToChatBody converts instructions/input into messages", () => {
  const body = responsesToChatBody(
    { instructions: "sys", input: "hello", temperature: 0.5 },
    "fallback-model"
  );
  assert.deepEqual(body.messages, [
    { role: "system", content: "sys" },
    { role: "user", content: "hello" },
  ]);
  assert.equal(body.model, "fallback-model");
  assert.equal(body.temperature, 0.5);
});

test("responsesToChatBody keeps explicit model and empty messages", () => {
  const body = responsesToChatBody({ model: "gpt-5" }, "fallback");
  assert.equal(body.model, "gpt-5");
  assert.deepEqual(body.messages, []);
});

test("messagesToText joins role-prefixed messages", () => {
  assert.equal(messagesToText("nope"), "");
  assert.equal(
    messagesToText([
      { role: "system", content: "sys" },
      { content: "no role defaults to user" },
    ]),
    "system: sys\n\nuser: no role defaults to user"
  );
});

test("anthropicMessagesToText includes system and tools", () => {
  const text = anthropicMessagesToText({
    system: "sys",
    tools: [{ name: "search" }],
    messages: [{ role: "user", content: "hi" }],
  });
  assert.match(text, /^system: sys/);
  assert.match(text, /available tools: \[{"name":"search"}\]/);
  assert.match(text, /user: hi$/);
});

test("inputToText handles strings, arrays and message items", () => {
  assert.equal(inputToText(""), "");
  assert.equal(inputToText("plain"), "plain");
  assert.equal(
    inputToText([
      "raw",
      { type: "message", role: "assistant", content: "hi" },
      { role: "user", content: "yo" },
      { type: "input_text", text: "typed" },
    ]),
    "raw\n\nassistant: hi\n\nuser: yo\n\ntyped"
  );
});

test("contentToText handles all supported shapes", () => {
  assert.equal(contentToText(null), "");
  assert.equal(contentToText("str"), "str");
  assert.equal(contentToText(["a", { type: "text", text: "b" }]), "a\nb");
  assert.equal(contentToText({ text: "x" }), "x");
  assert.equal(contentToText({ type: "input_text", text: "y" }), "y");
  assert.equal(
    contentToText({ type: "image_url", image_url: { url: "http://img" } }),
    "[image: http://img]"
  );
  assert.equal(contentToText({ type: "image" }), "[image attached]");
  assert.equal(
    contentToText({ type: "tool_result", tool_use_id: "t1", content: "done" }),
    "[tool_result t1] done"
  );
  assert.equal(
    contentToText({ type: "tool_use", name: "calc", input: { a: 1 } }),
    '[tool_use calc] {"a":1}'
  );
  assert.match(contentToText({ type: "custom", foo: 1 }), /^\[custom\] /);
});

test("latestUserText returns most recent user message", () => {
  assert.equal(latestUserText("nope"), "");
  assert.equal(
    latestUserText([
      { role: "user", content: "first" },
      { role: "assistant", content: "reply" },
      { role: "user", content: "second" },
    ]),
    "second"
  );
  assert.equal(latestUserText([{ role: "assistant", content: "only" }]), "");
});

test("hasWebSearchTool detects search-like tools", () => {
  assert.equal(hasWebSearchTool(null), false);
  assert.equal(hasWebSearchTool([{ type: "code" }]), false);
  assert.equal(hasWebSearchTool([{ type: "web_search" }]), true);
  assert.equal(hasWebSearchTool([{ function: { name: "browser_search" } }]), true);
});

test("reasoningEffort resolves from multiple fields with default", () => {
  assert.equal(reasoningEffort({}), "medium");
  assert.equal(reasoningEffort({ effort: "high" }), "high");
  assert.equal(reasoningEffort({ reasoning_effort: "low" }), "low");
  assert.equal(reasoningEffort({ reasoning: { effort: "minimal" } }), "minimal");
});

test("mapUpstreamModel normalizes provider prefixes", () => {
  assert.equal(mapUpstreamModel(""), "gateway-gpt-5-5");
  assert.equal(mapUpstreamModel(undefined), "gateway-gpt-5-5");
  assert.equal(mapUpstreamModel("gateway-gpt-5"), "gateway-gpt-5");
  assert.equal(mapUpstreamModel("claude-opus-4-7-20260101"), "gateway-claude-opus-4-7");
  assert.equal(mapUpstreamModel("gpt-5"), "gateway-gpt-5");
  assert.equal(mapUpstreamModel("gemini-2.5-pro"), "gateway-google-2.5-pro");
  assert.equal(mapUpstreamModel("mystery-model"), "mystery-model");
});

test("toAnthropicModel and anthropicVersionedId", () => {
  assert.deepEqual(toAnthropicModel({ id: "gateway-claude-opus-4-7", name: "Opus" }), {
    id: "claude-opus-4-7-20260101",
    type: "model",
    display_name: "Opus",
    created_at: "2026-01-01T00:00:00Z",
  });
  // already fully versioned id is left intact
  assert.equal(toAnthropicModel({ id: "claude-opus-4-7-20260101" }).id, "claude-opus-4-7-20260101");
  assert.equal(anthropicVersionedId("claude-sonnet"), "claude-sonnet-20260101");
  assert.equal(anthropicVersionedId("gpt-5"), "gpt-5");
});

test("providerFromModel classifies known providers", () => {
  assert.equal(providerFromModel("claude-opus"), "anthropic");
  assert.equal(providerFromModel("gemini-pro"), "google");
  assert.equal(providerFromModel("gpt-5"), "openai");
  assert.equal(providerFromModel("something-else"), "unlimited.surf");
});

test("fallbackModels returns a non-empty catalog with ids", () => {
  const models = fallbackModels();
  assert.ok(Array.isArray(models) && models.length > 0);
  for (const model of models) {
    assert.ok(model.id && model.name && model.provider);
  }
});

test("parseSseJson parses data payloads and ignores sentinels", () => {
  assert.equal(parseSseJson(""), null);
  assert.equal(parseSseJson("[DONE]"), null);
  assert.equal(parseSseJson("not json"), null);
  assert.deepEqual(parseSseJson('{"delta":"hi"}'), { delta: "hi" });
});

test("openAIStopReason maps upstream finish reasons", () => {
  assert.equal(openAIStopReason(""), "stop");
  assert.equal(openAIStopReason("max_tokens"), "length");
  assert.equal(openAIStopReason("tool_use"), "tool_calls");
  assert.equal(openAIStopReason("end_turn"), "stop");
  assert.equal(openAIStopReason("content_filter"), "content_filter");
});

test("anthropicStopReason maps finish reasons", () => {
  assert.equal(anthropicStopReason(""), "end_turn");
  assert.equal(anthropicStopReason("stop"), "end_turn");
  assert.equal(anthropicStopReason("length"), "max_tokens");
  assert.equal(anthropicStopReason("tool_calls"), "tool_use");
  assert.equal(anthropicStopReason("other"), "other");
});

test("estimateTokens uses ~4 chars per token with a floor of 1", () => {
  assert.equal(estimateTokens(""), 0);
  assert.equal(estimateTokens("a"), 1);
  assert.equal(estimateTokens("abcd"), 1);
  assert.equal(estimateTokens("abcde"), 2);
});

test("usage helpers sum token estimates", () => {
  assert.deepEqual(usageFromText("abcd", "abcdefgh"), {
    prompt_tokens: 1,
    completion_tokens: 2,
    total_tokens: 3,
  });
  assert.deepEqual(responseUsageFromText("abcd", "abcdefgh"), {
    input_tokens: 1,
    output_tokens: 2,
    total_tokens: 3,
  });
  assert.deepEqual(anthropicUsageFromText("abcd", "abcdefgh"), {
    input_tokens: 1,
    output_tokens: 2,
  });
});

test("stripTrailingSlash and upstreamBase", () => {
  assert.equal(stripTrailingSlash("https://x.com///"), "https://x.com");
  assert.equal(stripTrailingSlash(undefined), "");
  assert.equal(upstreamBase({}), "https://unlimited.surf/");
  assert.equal(upstreamBase({ UPSTREAM_BASE_URL: "https://alt.example/" }), "https://alt.example/");
});

test("bytesToBase64 round-trips", () => {
  const bytes = new Uint8Array([72, 101, 108, 108, 111]); // "Hello"
  assert.equal(bytesToBase64(bytes), "SGVsbG8=");
  assert.equal(bytesToBase64(new Uint8Array([])), "");
});

test("looksLikeAnthropicRequest checks anthropic headers", () => {
  assert.equal(looksLikeAnthropicRequest(makeRequest({ headers: {} })), false);
  assert.equal(
    looksLikeAnthropicRequest(makeRequest({ headers: { "anthropic-version": "2023-06-01" } })),
    true
  );
  assert.equal(looksLikeAnthropicRequest(makeRequest({ headers: { "x-api-key": "k" } })), true);
});

test("clientApiKey reads bearer or x-api-key headers", () => {
  assert.equal(clientApiKey(makeRequest({ headers: { authorization: "Bearer  secret " } })), "secret");
  assert.equal(clientApiKey(makeRequest({ headers: { "x-api-key": " k2 " } })), "k2");
  assert.equal(clientApiKey(makeRequest({ headers: { "anthropic-api-key": "k3" } })), "k3");
  assert.equal(clientApiKey(makeRequest({ headers: {} })), "");
});

test("constantTimeEqual compares equal-length strings", () => {
  assert.equal(constantTimeEqual("abc", "abc"), true);
  assert.equal(constantTimeEqual("abc", "abd"), false);
  assert.equal(constantTimeEqual("abc", "abcd"), false);
  assert.equal(constantTimeEqual("", ""), true);
});

test("optionalUpstreamApiKey precedence", () => {
  assert.equal(
    optionalUpstreamApiKey(makeRequest(), { UNLIMITED_SURF_API_KEY: "u" }),
    "u"
  );
  assert.equal(optionalUpstreamApiKey(makeRequest(), { API_KEY: "a" }), "a");
  // WORKER_API_KEY set but no upstream key configured -> empty string
  assert.equal(optionalUpstreamApiKey(makeRequest(), { WORKER_API_KEY: "w" }), "");
  // no config -> falls back to the client key
  assert.equal(
    optionalUpstreamApiKey(makeRequest({ headers: { authorization: "Bearer client" } }), {}),
    "client"
  );
});

test("upstreamApiKey throws when nothing resolves a key", () => {
  assert.equal(upstreamApiKey(makeRequest(), { UNLIMITED_SURF_API_KEY: "u" }), "u");
  assert.throws(
    () => upstreamApiKey(makeRequest(), { WORKER_API_KEY: "w" }),
    /WORKER_API_KEY is enabled/
  );
  assert.throws(() => upstreamApiKey(makeRequest(), {}), /Authorization: Bearer/);
});

test("validateWorkerApiKey enforces the client key when configured", () => {
  assert.equal(validateWorkerApiKey(makeRequest(), {}), null);
  assert.equal(
    validateWorkerApiKey(makeRequest({ headers: { authorization: "Bearer good" } }), { WORKER_API_KEY: "good" }),
    null
  );
  const rejected = validateWorkerApiKey(
    makeRequest({ headers: { authorization: "Bearer bad" } }),
    { WORKER_API_KEY: "good" }
  );
  assert.ok(rejected instanceof Response);
  assert.equal(rejected.status, 401);
});

test("serviceInfo reports ok and route map for the request origin", () => {
  const info = serviceInfo(makeRequest({ url: "https://w.example.com/health" }), {});
  assert.equal(info.ok, true);
  assert.equal(info.upstream, "https://unlimited.surf");
  assert.match(info.routes.openai, /^https:\/\/w\.example\.com\/v1\/chat\/completions/);
});

test("setup/codex/mcp docs embed the request origin", () => {
  const req = makeRequest({ url: "https://w.example.com/v1/setup" });
  assert.match(agentSetup(req), /ANTHROPIC_BASE_URL = "https:\/\/w\.example\.com"/);
  assert.match(codexSetup(req), /base_url = "https:\/\/w\.example\.com\/v1"/);
  const mcp = mcpInfo(req);
  assert.equal(mcp.supported, true);
  assert.equal(mcp.model_endpoint, "https://w.example.com");
});
