import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";
import ts from "typescript";

const policy = {};
const transpile = (path) => ts.transpileModule(readFileSync(new URL(path, import.meta.url), "utf8"), {
  compilerOptions: { module: ts.ModuleKind.CommonJS },
}).outputText;
vm.runInNewContext(transpile("../utils/intent.ts"), { exports: policy });

const examples = policy.INTENT_EXAMPLES.map(({ intent }) =>
  intent === "transcript-only" ? [1, 0, 0] : intent === "cloud-text" ? [0, 1, 0] : [0, 0, 1]);

test("routes each confident intent with consistent cloud and vision flags", () => {
  for (const [embedding, intent, cloud, vision] of [
    [[1, 0, 0], "transcript-only", false, false],
    [[0, 1, 0], "cloud-text", true, false],
    [[0, 0, 1], "cloud-vision", true, true],
  ]) {
    const decision = policy.decideIntent(embedding, examples);
    assert.equal(decision.intent, intent);
    assert.equal(decision.requiresCloud, cloud);
    assert.equal(decision.requiresVision, vision);
    assert.equal(decision.transcriptionOnly, !cloud);
    assert.equal(decision.reason, "matched");
  }
});

test("low similarity and ambiguous cloud/vision or local/cloud matches stay local", () => {
  for (const embedding of [[0.1, 0.2, 0.3], [0, 0.7, 0.69], [0.7, 0.71, 0]]) {
    const decision = policy.decideIntent(embedding, examples);
    assert.equal(decision.requiresCloud, false);
    assert.equal(decision.requiresVision, false);
    assert.equal(decision.reason, "uncertain");
  }
});

test("empty and overlong transcripts stay local rather than being truncated", () => {
  assert.equal(policy.transcriptGuard("  ").reason, "empty");
  assert.equal(policy.transcriptGuard("a".repeat(policy.MAX_TRANSCRIPT_LENGTH + 1)).reason, "too-long");
  assert.equal(policy.transcriptGuard("Explain gravity"), null);
});

test("invalid embeddings cannot produce a cloud decision", () => {
  assert.throws(() => policy.decideIntent([NaN], examples), /Invalid intent embeddings/);
  assert.throws(() => policy.decideIntent([1, 0, 0], []), /Invalid intent embeddings/);
});

function workerSetup(failOnce = false) {
  const messages = [];
  const configurations = [];
  let handler;
  let loads = 0;
  vm.runInNewContext(transpile("../app/workers/intent.worker.ts"), {
    exports: {},
    require: (name) => name.includes("utils/intent") ? policy : {
      env: { backends: { onnx: { wasm: {} } } },
      pipeline: async (...args) => {
        configurations.push(args);
        if (++loads === 1 && failOnce) throw new Error("Download failed");
        return async (texts, options) => {
          assert.equal(options.pooling, "mean");
          assert.equal(options.normalize, true);
          return { tolist: () => Array.isArray(texts) ? examples : [[0, 1, 0]] };
        };
      },
    },
    self: {
      addEventListener: (_, callback) => { handler = callback; },
      postMessage: (message) => messages.push(message),
    },
  });
  return { messages, configurations, send: (text, id = 1) => handler({ data: { id, text } }) };
}

test("worker guards before downloading, uses WASM MiniLM, and caches the model", async () => {
  const worker = workerSetup();
  await worker.send(" ");
  assert.equal(worker.configurations.length, 0);
  await worker.send("Explain gravity", 2);
  await worker.send("Translate hello", 3);
  assert.equal(worker.configurations.length, 1);
  const [task, model, options] = worker.configurations[0];
  assert.equal(task, "feature-extraction");
  assert.equal(model, "onnx-community/all-MiniLM-L6-v2-ONNX");
  assert.equal(options.device, "wasm");
  assert.equal(options.dtype, "q4");
  assert.equal(worker.messages.at(-1).id, 3);
  assert.equal(worker.messages.at(-1).decision.intent, "cloud-text");
});

test("worker reports load failure without a cloud decision and retries later", async () => {
  const worker = workerSetup(true);
  await worker.send("Explain gravity");
  assert.equal(worker.messages.at(-1).status, "error");
  assert.equal(worker.messages.at(-1).decision, undefined);
  await worker.send("Explain gravity", 2);
  assert.equal(worker.messages.at(-1).decision.requiresCloud, true);
  assert.equal(worker.configurations.length, 2);
});

function apiSetup(apiKey = "test-key", model = "test-model") {
  const api = {};
  const calls = [];
  vm.runInNewContext(transpile("../app/api/process-frame/route.ts"), {
    exports: api,
    require: () => ({ NextResponse: { json: (body, options) => ({ body, status: options?.status ?? 200 }) } }),
    process: { env: { NEBIUS_API_KEY: apiKey, NEBIUS_ENDPOINT: "https://example.com/v1", NEBIUS_MODEL: model } },
    console: { error() {} },
    fetch: async (url, options) => {
      calls.push({ url, ...options, payload: JSON.parse(options.body) });
      return { ok: true, json: async () => ({ choices: [{ message: { content: "Answer" } }] }) };
    },
  });
  return { calls, post: (body) => api.POST({ json: async () => body, signal: "request-signal" }) };
}

test("API rejects local, missing, and mismatched routing without calling Nebius", async () => {
  const api = apiSetup();
  for (const body of [
    { image: "data:image/jpeg;base64,AAAA" },
    { transcript: "Just transcribe", routing: { requiresCloud: false, requiresVision: false } },
    { transcript: " ", routing: { requiresCloud: true, requiresVision: false } },
    { transcript: "Look", routing: { requiresCloud: true, requiresVision: true } },
    { transcript: "Explain", routing: { requiresCloud: true, requiresVision: false }, image: "data:image/jpeg;base64,AAAA" },
    { transcript: "Look", routing: { requiresCloud: true, requiresVision: true }, image: "https://example.com/image.jpg" },
  ]) {
    assert.equal((await api.post(body)).status, 400);
  }
  assert.equal(api.calls.length, 0);
});

test("API sends the transcript without images for text reasoning and adds one for vision", async () => {
  const api = apiSetup("test-key", "configured-model");
  await api.post({ transcript: "Explain gravity", routing: { requiresCloud: true, requiresVision: false } });
  assert.equal(api.calls[0].payload.model, "configured-model");
  const textContent = api.calls[0].payload.messages[0].content;
  assert.equal(textContent.length, 1);
  assert.equal(textContent[0].text, "Explain gravity");
  assert.equal(api.calls[0].signal, "request-signal");
  await api.post({ transcript: "Read this sign", routing: { requiresCloud: true, requiresVision: true }, image: "data:image/jpeg;base64,AAAA" });
  const visionContent = api.calls[1].payload.messages[0].content;
  assert.equal(visionContent.length, 2);
  assert.equal(visionContent[0].text, "Read this sign");
  assert.equal(visionContent[1].image_url.url, "data:image/jpeg;base64,AAAA");
});

test("missing cloud credentials never trigger an upstream call", async () => {
  const api = apiSetup("");
  assert.equal((await api.post({ transcript: "Explain gravity", routing: { requiresCloud: true, requiresVision: false } })).status, 503);
  assert.equal(api.calls.length, 0);
});

test("missing or blank cloud model never triggers an upstream call", async () => {
  for (const model of [null, "", "   "]) {
    const api = apiSetup("test-key", model);
    const response = await api.post({ transcript: "Explain gravity", routing: { requiresCloud: true, requiresVision: false } });
    assert.equal(response.status, 503);
    assert.equal(response.body.error, "NEBIUS_MODEL is not configured.");
    assert.equal(api.calls.length, 0);
  }
});

function pageSetup() {
  const slots = [];
  const timers = new Map();
  const messages = [];
  const requests = [];
  const effects = [];
  let hookIndex = 0;
  let dirty = true;
  let tree;
  let worker;
  let now = 0;
  let timerId = 0;
  let captures = 0;
  const track = { readyState: "live", stop() { this.readyState = "ended"; } };
  const stream = { getTracks: () => [track], getVideoTracks: () => [track] };
  const video = { srcObject: null, videoWidth: 1280, videoHeight: 720, play: async () => {} };
  const slot = (create) => slots[hookIndex++] ??= create();
  const react = {
    useRef: (initial) => slot(() => ({ current: initial })),
    useState: (initial) => {
      const state = slot(() => ({ value: initial }));
      return [state.value, (value) => {
        if (!Object.is(state.value, value)) { state.value = value; dirty = true; }
      }];
    },
    useEffect: (effect, dependencies) => {
      const state = slot(() => ({}));
      if (!state.dependencies || dependencies.some((value, index) => !Object.is(value, state.dependencies[index]))) {
        state.dependencies = dependencies;
        effects.push(() => { state.cleanup?.(); state.cleanup = effect(); });
      }
    },
  };
  const element = (type, props) => {
    if (type === "video") props.ref.current = video;
    return { type, props };
  };
  const page = {};
  const source = ts.transpileModule(readFileSync(new URL("../app/page.tsx", import.meta.url), "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
    transformers: { before: [(context) => {
      const visit = (node) => ts.isPropertyAccessExpression(node) && ts.isMetaProperty(node.expression) && node.name.text === "url"
        ? ts.factory.createStringLiteral("file:///app/page.tsx")
        : ts.visitEachChild(node, visit, context);
      return (node) => ts.visitNode(node, visit);
    }] },
  }).outputText;
  vm.runInNewContext(source, {
    exports: page,
    require: (name) => name === "react" ? react : name === "react/jsx-runtime"
      ? { jsx: element, jsxs: element } : name === "@/utils/intent" ? policy : { default() {} },
    URL, AbortController,
    navigator: { mediaDevices: { getUserMedia: async () => stream } },
    document: { createElement: () => ({
      getContext: () => ({ drawImage() {} }),
      toDataURL: () => { captures++; return "data:image/jpeg;base64,AAAA"; },
    }) },
    Worker: class {
      constructor() {
        worker = {
          postMessage(message) { messages.push(message); },
          terminate() {},
        };
        return worker;
      }
    },
    setTimeout: (callback, delay) => {
      timers.set(++timerId, { callback, at: now + delay });
      return timerId;
    },
    clearTimeout: (id) => timers.delete(id),
    fetch: async (_, options) => {
      requests.push(JSON.parse(options.body));
      return { ok: true, json: async () => ({ success: true, analysis: { choices: [{ message: { content: "Answer" } }] } }) };
    },
  });
  const render = () => {
    while (dirty) {
      dirty = false;
      hookIndex = 0;
      tree = page.default();
      effects.splice(0).forEach((effect) => effect());
    }
  };
  const flush = async () => { await new Promise((resolve) => setImmediate(resolve)); render(); };
  const find = (node, type) => {
    if (!node || typeof node !== "object") return undefined;
    if (node.type === type) return node;
    return [node.props?.children].flat().map((child) => find(child, type)).find(Boolean);
  };
  render();
  return {
    messages, requests, track,
    captures: () => captures,
    hasProcessButton: () => Boolean(find(tree, "button")),
    async update(text) {
      find(tree, "textarea").props.onChange({ target: { value: text } });
      render();
      await flush();
    },
    async advance(milliseconds) {
      now += milliseconds;
      for (const [id, timer] of [...timers]) {
        if (timer.at <= now && timers.delete(id)) timer.callback();
      }
      render();
      await flush();
    },
    async reply(decision, id = messages.at(-1).id) {
      await worker.onmessage({ data: { id, status: "result", decision } });
      await flush();
    },
  };
}

test("page automatically debounces updates and does not resubmit identical transcripts", async () => {
  const page = pageSetup();
  assert.equal(page.hasProcessButton(), false);
  await page.update("First draft");
  await page.advance(600);
  await page.update("Latest draft");
  await page.advance(1199);
  assert.equal(page.messages.length, 0);
  await page.advance(1);
  assert.equal(page.messages.length, 1);
  assert.equal(page.messages[0].text, "Latest draft");
  await page.reply(policy.localDecision("matched"));
  await page.advance(5000);
  await page.update("  Latest draft  ");
  await page.advance(5000);
  assert.equal(page.messages.length, 1);
  assert.equal(page.requests.length, 0);
  assert.equal(page.captures(), 0);
});

test("page queues the latest transcript while busy and never escalates stale results", async () => {
  const page = pageSetup();
  const textRoute = policy.decideIntent([0, 1, 0], examples);
  await page.update("Old request");
  await page.advance(1200);
  const oldId = page.messages[0].id;
  await page.update("New request");
  await page.advance(3000);
  assert.equal(page.messages.length, 1);
  await page.reply(textRoute, oldId);
  assert.equal(page.requests.length, 0);
  await page.advance(1200);
  assert.equal(page.messages.length, 2);
  assert.equal(page.messages[1].text, "New request");
  await page.reply(textRoute);
  assert.equal(page.requests.length, 1);
  assert.equal(page.requests[0].transcript, "New request");
  assert.equal(page.requests[0].image, undefined);
  await page.advance(5000);
  assert.equal(page.requests.length, 1);
});

test("automatic visual routing captures one frame and keeps the local camera live", async () => {
  const page = pageSetup();
  await page.update("Describe this scene");
  await page.advance(1200);
  await page.reply(policy.decideIntent([0, 0, 1], examples));
  assert.equal(page.requests.length, 1);
  assert.equal(page.requests[0].image, "data:image/jpeg;base64,AAAA");
  assert.equal(page.captures(), 1);
  assert.equal(page.track.readyState, "live");
  await page.advance(5000);
  assert.equal(page.captures(), 1);
});