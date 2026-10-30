import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";
import ts from "typescript";

const source = ts.transpileModule(
  readFileSync(new URL("../app/workers/asrs.worker.ts", import.meta.url), "utf8"),
  { compilerOptions: { module: ts.ModuleKind.CommonJS } },
).outputText;

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((onResolve, onReject) => {
    resolve = onResolve;
    reject = onReject;
  });
  return { promise, resolve, reject };
}

const settle = () => new Promise((resolve) => setImmediate(resolve));

test("RNNoise readiness bridge signals only after asynchronous initialization", async () => {
  const messages = [];
  const initialization = deferred();
  let RegisteredProcessor;
  class Processor {
    constructor() {
      this.port = { postMessage: (message) => messages.push(message) };
      this.processor = undefined;
      void initialization.promise.then(() => { this.processor = { process() {} }; });
    }
  }
  const registerProcessor = (_, implementation) => { RegisteredProcessor = implementation; };
  const sandbox = { registerProcessor };
  vm.runInNewContext(
    readFileSync(new URL("../public/noise-suppressor/rnnoise-ready.worklet.js", import.meta.url), "utf8"),
    sandbox,
  );
  sandbox.registerProcessor("rnnoise", Processor);
  new RegisteredProcessor({});
  assert.deepEqual(messages, []);
  assert.equal(sandbox.registerProcessor, registerProcessor);
  initialization.resolve();
  await settle();
  assert.deepEqual(messages, ["ready"]);
});

function setup() {
  const loading = deferred();
  const calls = [];
  const configurations = [];
  const messages = [];
  const denoiseRequests = [];
  let denoiser = (audio) => audio;
  let handleMessage;
  let loads = 0;
  let speechRanges;
  const detector = {
    async *run(audio) {
      const lastSpeechIndex = audio.findLastIndex((sample) => Math.abs(sample) > 0.003);
      const ranges = speechRanges ?? (lastSpeechIndex >= 0
        ? [{ start: 0, end: (lastSpeechIndex + 1) / 16 }]
        : []);
      for (const range of ranges) {
        yield { ...range, audio: audio.slice(range.start * 16, range.end * 16) };
      }
    },
  };
  const transcriber = (audio, options) => {
    const pending = deferred();
    calls.push({ audio, options, ...pending });
    return pending.promise;
  };
  vm.runInNewContext(source, {
    exports: {},
    require: (name) => name === "@ricky0123/vad-web" ? {
      NonRealTimeVAD: { new: async () => detector },
    } : ({
      env: { backends: { onnx: { wasm: {} } } },
      pipeline: (task, model, options) => {
        configurations.push({ task, model, options });
        loads++;
        return loading.promise;
      },
    }),
    Float32Array,
    Error,
    self: {
      addEventListener: (_, handler) => { handleMessage = handler; },
      postMessage: (message) => {
        // Stands in for the main thread, which renders RNNoise and replies.
        if (message.status === "denoise-request") {
          denoiseRequests.push(message.audio);
          handleMessage({ data: { command: "denoised", id: message.id, audio: denoiser(message.audio) } });
          return;
        }
        messages.push(message);
      },
    },
  });
  return {
    calls, messages, denoiseRequests, loading, configurations,
    load: () => loading.resolve(transcriber),
    requestLoad: () => handleMessage({ data: { command: "load" } }),
    loads: () => loads,
    chunk: (length, value = 0.1) => handleMessage({
      data: { command: "processChunk", audioChunk: new Float32Array(length).fill(value) },
    }),
    reset: () => handleMessage({ data: { command: "reset" } }),
    exportAudio: () => handleMessage({ data: { command: "exportAudio" } }),
    exportTranscriptionAudio: () => handleMessage({ data: { command: "exportTranscriptionAudio" } }),
    setSpeechRanges: (ranges) => { speechRanges = ranges; },
    setDenoiser: (fn) => { denoiser = fn; },
  };
}

test("waits for context, loads once, and serializes coalesced audio snapshots", async () => {
  const worker = setup();
  worker.chunk(16000);
  assert.equal(worker.loads(), 0);
  worker.chunk(16000);
  worker.chunk(16000);
  await settle();
  assert.equal(worker.loads(), 1);
  worker.load();
  await settle();
  assert.equal(worker.calls.length, 1);
  assert.equal(worker.calls[0].audio.length, 48000);
  worker.chunk(16000);
  worker.chunk(16000);
  assert.equal(worker.calls.length, 1);
  assert.equal(worker.calls[0].audio.length, 48000);
  worker.calls[0].resolve({ text: "first" });
  await settle();
  assert.equal(worker.calls.length, 2);
  assert.equal(worker.calls[1].audio.length, 80000);
  worker.calls[1].resolve({ text: "updated" });
  await settle();
  assert.deepEqual(worker.messages.map((message) => message.text), ["first", "updated"]);
  worker.chunk(100);
  await settle();
  assert.equal(worker.calls.length, 2);
});

test("reset discards in-flight results and processes only new session audio", async () => {
  const worker = setup();
  worker.load();
  worker.chunk(32000, 1);
  await settle();
  worker.reset();
  worker.chunk(32000, 2);
  worker.calls[0].resolve({ text: "stale" });
  await settle();
  assert.equal(worker.messages.length, 0);
  assert.equal(worker.calls.length, 2);
  assert.equal(worker.calls[1].audio.length, 32000);
  assert.ok(worker.calls[1].audio.every((sample) => sample === 2));
  worker.calls[1].resolve({ text: "new session" });
  await settle();
  assert.equal(worker.messages[0].text, "new session");
});

test("reset while loading does not start inference on discarded audio", async () => {
  const worker = setup();
  worker.chunk(32000);
  await settle();
  worker.reset();
  worker.load();
  await settle();
  assert.equal(worker.calls.length, 0);
  assert.equal(worker.messages.length, 0);
});

test("uses the reference WebGPU model and precision with English transcription", async () => {
  const worker = setup();
  worker.load();
  worker.chunk(32000);
  await settle();
  const { model, options } = worker.configurations[0];
  assert.equal(model, "onnx-community/whisper-small.en");
  assert.equal(options.device, "webgpu");
  assert.equal(options.dtype.encoder_model, "fp32");
  assert.equal(options.dtype.decoder_model_merged, "q4");
  assert.equal(worker.calls[0].options.max_new_tokens, 64);
  worker.calls[0].resolve({ text: "hello" });
  await settle();
});

test("loads and warms up once before reporting ready", async () => {
  const worker = setup();
  worker.requestLoad();
  worker.requestLoad();
  assert.equal(worker.loads(), 1);
  worker.load();
  await settle();
  assert.equal(worker.calls.length, 1);
  assert.equal(worker.calls[0].options.max_new_tokens, 1);
  assert.ok(worker.calls[0].audio.every((sample) => sample === 0));
  assert.equal(worker.messages.some((message) => message.status === "ready"), false);
  worker.calls[0].resolve({ text: "" });
  await settle();
  assert.equal(worker.messages.at(-1).status, "ready");
  worker.requestLoad();
  await settle();
  assert.equal(worker.calls.length, 1);
  assert.equal(worker.loads(), 1);
});

test("reports warmup failure and permits retry", async () => {
  const worker = setup();
  worker.load();
  worker.requestLoad();
  await settle();
  worker.calls[0].reject(new Error("WebGPU unavailable"));
  await settle();
  assert.equal(worker.messages.at(-1).error, "WebGPU unavailable");
  worker.requestLoad();
  await settle();
  worker.calls[1].resolve({ text: "" });
  await settle();
  assert.equal(worker.messages.at(-1).status, "ready");
});

test("bounds inference to the latest 30 seconds and continues after the window fills", async () => {
  const worker = setup();
  worker.load();
  worker.chunk(480000, 0.1);
  await settle();
  worker.calls[0].resolve({ text: "first window" });
  await settle();
  worker.chunk(16000, 0.2);
  await settle();
  assert.equal(worker.calls.length, 2);
  assert.equal(worker.calls[1].audio.length, 480000);
  assert.equal(worker.calls[1].audio[0], Math.fround(0.1));
  assert.equal(worker.calls[1].audio.at(-1), Math.fround(0.2));
  worker.calls[1].resolve({ text: "next window" });
  await settle();
  worker.chunk(500000, 0.3);
  await settle();
  assert.equal(worker.calls[2].audio.length, 480000);
  assert.ok(worker.calls[2].audio.every((sample) => sample === Math.fround(0.3)));
  worker.calls[2].resolve({ text: "latest window" });
  await settle();
});

test("silence and low-level noise do not start Whisper", async () => {
  const worker = setup();
  worker.load();
  worker.chunk(32000, 0);
  worker.chunk(32000, 0.0001);
  await settle();
  assert.equal(worker.loads(), 0);
  assert.equal(worker.calls.length, 0);
});

test("audible noise rejected by VAD does not reach Whisper", async () => {
  const worker = setup();
  worker.setSpeechRanges([]);
  worker.load();
  worker.chunk(32000, 0.2);
  await settle();
  assert.equal(worker.loads(), 0);
  assert.equal(worker.calls.length, 0);
  assert.equal(worker.messages.at(-1).status, "no-speech");
});

test("old speech plus new noise does not replace a valid transcript", async () => {
  const worker = setup();
  worker.load();
  worker.setSpeechRanges([{ start: 0, end: 1500 }]);
  worker.chunk(32000);
  await settle();
  worker.calls[0].resolve({ text: "The meeting starts at nine." });
  await settle();
  worker.chunk(16000, 0.2);
  await settle();
  assert.equal(worker.calls.length, 1);
  assert.equal(worker.messages.filter((message) => message.status === "streaming-result").at(-1).text, "The meeting starts at nine.");
  assert.equal(worker.messages.at(-1).status, "no-speech");
});

test("denoises only detected speech and passes the denoised audio to Whisper", async () => {
  const worker = setup();
  worker.setDenoiser((audio) => audio.map((sample) => sample * 0.5));
  worker.load();
  worker.setSpeechRanges([{ start: 500, end: 1000 }]);
  worker.chunk(8000, 0);
  worker.chunk(8000, 0.2);
  worker.chunk(8000, 0);
  worker.chunk(8000, 0);
  await settle();
  assert.equal(worker.denoiseRequests.length, 1);
  assert.equal(worker.denoiseRequests[0].length, 8000);
  assert.ok(worker.denoiseRequests[0].every((sample) => sample === Math.fround(0.2)));
  assert.equal(worker.calls[0].audio.length, 8000);
  assert.ok(worker.calls[0].audio.every((sample) => sample === Math.fround(0.1)));
  worker.exportTranscriptionAudio();
  assert.equal(worker.messages.at(-1).status, "captured-audio");
  assert.deepEqual(worker.messages.at(-1).audio, worker.calls[0].audio);
  worker.reset();
  worker.exportTranscriptionAudio();
  assert.equal(worker.messages.at(-1).audio.length, 0);
  worker.calls[0].resolve({ text: "denoised" });
  await settle();
  assert.equal(worker.messages.some((message) => message.text === "denoised"), false);
});

test("passes only detected speech segments to Whisper", async () => {
  const worker = setup();
  worker.load();
  worker.setSpeechRanges([{ start: 500, end: 1000 }, { start: 1500, end: 2000 }]);
  worker.chunk(8000, 0);
  worker.chunk(8000, 0.1);
  worker.chunk(8000, 0);
  worker.chunk(8000, 0.2);
  await settle();
  const audio = worker.calls[0].audio;
  assert.equal(audio.length, 16000);
  assert.ok(audio.subarray(0, 8000).every((sample) => sample === Math.fround(0.1)));
  assert.ok(audio.subarray(8000).every((sample) => sample === Math.fround(0.2)));
  worker.calls[0].resolve({ text: "two phrases" });
  await settle();
});

test("exports the last 30 seconds of model input before reset", () => {
  const worker = setup();
  worker.chunk(16000, 0.0001);
  worker.chunk(480000, 0.0002);
  worker.exportAudio();
  worker.reset();
  const message = worker.messages[0];
  assert.equal(message.status, "captured-audio");
  assert.equal(message.sampleRate, 16000);
  assert.equal(message.audio.length, 480000);
  assert.ok(message.audio.every((sample) => sample === Math.fround(0.0002)));
  worker.exportAudio();
  assert.equal(worker.messages[1].audio.length, 0);
});

test("pauses do not repeatedly transcribe previously processed speech", async () => {
  const worker = setup();
  worker.load();
  worker.chunk(32000);
  await settle();
  worker.calls[0].resolve({ text: "hello" });
  await settle();
  worker.chunk(16000, 0);
  await settle();
  assert.equal(worker.calls.length, 1);
  worker.chunk(16000);
  await settle();
  assert.equal(worker.calls.length, 2);
  worker.calls[1].resolve({ text: "hello again" });
  await settle();
  assert.equal(worker.messages.at(-1).text, "hello again");
});

test("a stale inference failure does not block the next session", async () => {
  const worker = setup();
  worker.load();
  worker.chunk(32000);
  await settle();
  worker.reset();
  worker.chunk(32000);
  worker.calls[0].reject(new Error("old failure"));
  await settle();
  assert.equal(worker.messages.length, 0);
  assert.equal(worker.calls.length, 2);
  worker.calls[1].resolve({ text: "recovered" });
  await settle();
  assert.equal(worker.messages[0].text, "recovered");
});