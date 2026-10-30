import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";
import ts from "typescript";

const exports = {};
vm.runInNewContext(ts.transpileModule(
  readFileSync(new URL("../utils/wav.ts", import.meta.url), "utf8"),
  { compilerOptions: { module: ts.ModuleKind.CommonJS } },
).outputText, { exports, Blob });

test("encodes mono PCM with the correct rate, duration, and sample order", async () => {
  const samples = new Float32Array(16000);
  samples.set([-2, -1, -0.5, 0, 0.5, 1, 2]);
  const blob = exports.encodeWav(samples, 16000);
  const buffer = await blob.arrayBuffer();
  const view = new DataView(buffer);
  assert.equal(blob.type, "audio/wav");
  assert.equal(new TextDecoder().decode(buffer.slice(0, 4)), "RIFF");
  assert.equal(new TextDecoder().decode(buffer.slice(8, 12)), "WAVE");
  assert.equal(view.getUint32(4, true), buffer.byteLength - 8);
  assert.equal(view.getUint16(20, true), 1);
  assert.equal(view.getUint16(22, true), 1);
  assert.equal(view.getUint32(24, true), 16000);
  assert.equal(view.getUint32(28, true), 32000);
  assert.equal(view.getUint16(32, true), 2);
  assert.equal(view.getUint16(34, true), 16);
  assert.equal(view.getUint32(40, true), 32000);
  assert.deepEqual(Array.from({ length: 7 }, (_, index) => view.getInt16(44 + index * 2, true)),
    [-32768, -32768, -16384, 0, 16384, 32767, 32767]);
});