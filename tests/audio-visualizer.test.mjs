import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";
import ts from "typescript";

const source = ts.transpileModule(
  readFileSync(new URL("../app/components/AudioVisualizer.tsx", import.meta.url), "utf8"),
  { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } },
).outputText;

function setup(active = false, reducedMotion = false) {
  const frames = new Map();
  const listeners = new Map();
  const strokes = [];
  const bars = [];
  let frameId = 0;
  let cleanup;
  let points = [];
  let disconnected = false;
  let reads = 0;
  let amplitude = 0.2;
  const document = { hidden: false,
    addEventListener: (name, callback) => listeners.set(name, callback),
    removeEventListener: (name) => listeners.delete(name),
  };
  const preference = { matches: reducedMotion,
    addEventListener: (name, callback) => listeners.set(`motion-${name}`, callback),
    removeEventListener: (name) => listeners.delete(`motion-${name}`),
  };
  const context = {
    clearRect() {}, setTransform() {},
    fillRect: (x, y, width, height) => { if (x !== 0 || y !== 0) bars.push(height); },
    createLinearGradient: () => ({ addColorStop() {} }),
    beginPath: () => { points = []; },
    moveTo: (x, y) => points.push([x, y]),
    lineTo: (x, y) => points.push([x, y]),
    stroke: () => strokes.push(points),
  };
  const canvas = { width: 0, height: 0,
    getContext: () => context,
    getBoundingClientRect: () => ({ width: 320, height: 160 }),
  };
  const analyser = {
    fftSize: 2048, frequencyBinCount: 1024, context: { sampleRate: 16000 },
    getFloatTimeDomainData(data) {
      reads++;
      for (let index = 0; index < data.length; index++) data[index] = Math.sin(index / 20) * amplitude;
    },
    getByteFrequencyData: (data) => data.fill(amplitude ? 192 : 0),
  };
  const component = {};
  vm.runInNewContext(source, {
    exports: component,
    require: (name) => name === "react" ? {
      useRef: () => ({ current: canvas }),
      useEffect: (effect) => { cleanup = effect(); },
    } : { jsx: (type, props) => ({ type, props }) },
    document, devicePixelRatio: 3,
    matchMedia: () => preference,
    requestAnimationFrame: (callback) => { frames.set(++frameId, callback); return frameId; },
    cancelAnimationFrame: (id) => frames.delete(id),
    ResizeObserver: class { observe() {} disconnect() { disconnected = true; } },
  });
  const element = component.default({ analyser: active ? analyser : null });
  return {
    canvas, element, frames, listeners, strokes, bars, cleanup,
    reads: () => reads,
    disconnected: () => disconnected,
    silence: () => { amplitude = 0; },
    visibility(hidden) { document.hidden = hidden; listeners.get("visibilitychange")(); },
    frame(time) {
      const [id, callback] = frames.entries().next().value;
      frames.delete(id);
      callback(time);
    },
  };
}

test("idle visualizer renders a stable high-DPI baseline without an animation loop", () => {
  const view = setup();
  assert.equal(view.canvas.width, 640);
  assert.equal(view.canvas.height, 320);
  assert.equal(view.frames.size, 0);
  assert.ok(view.strokes.at(-1).every(([, height]) => height === 64));
  assert.equal(view.element.props["aria-label"], "Microphone visualizer idle");
  view.cleanup();
  assert.equal(view.listeners.size, 0);
  assert.equal(view.disconnected(), true);
});

test("live waveform and spectrum react to audio and settle flat on silence", () => {
  const view = setup(true);
  assert.equal(view.frames.size, 1);
  assert.ok(view.strokes.at(-1).some(([, height]) => Math.abs(height - 64) > 5));
  assert.ok(Math.max(...view.bars) > 2);
  view.silence();
  view.bars.length = 0;
  view.frame(64);
  assert.ok(view.strokes.at(-1).every(([, height]) => height === 64));
  assert.equal(Math.max(...view.bars), 2);
  view.cleanup();
  assert.equal(view.frames.size, 0);
});

test("hidden tabs pause rendering and visible tabs resume a single loop", () => {
  const view = setup(true);
  view.visibility(true);
  assert.equal(view.frames.size, 0);
  view.visibility(false);
  view.visibility(false);
  assert.equal(view.frames.size, 1);
  view.frame(64);
  assert.equal(view.frames.size, 1);
  view.cleanup();
  assert.equal(view.frames.size, 0);
  assert.equal(view.listeners.size, 0);
});

test("reduced motion lowers the audio refresh rate", () => {
  const view = setup(true, true);
  const initialReads = view.reads();
  view.frame(64);
  assert.equal(view.reads(), initialReads);
  view.frame(112);
  assert.equal(view.reads(), initialReads + 1);
  view.cleanup();
});