// Copies the onnxruntime-web WASM runtime used by transformers.js into public/ so it is
// served from our own origin instead of a CDN (jsDelivr truncates files this large).
import { createRequire } from "node:module";
import { copyFileSync, mkdirSync, realpathSync } from "node:fs";
import path from "node:path";

const require = createRequire(import.meta.url);
// onnxruntime-web is a sibling of transformers in the pnpm store (or hoisted by other managers).
const entry = realpathSync(require.resolve("@huggingface/transformers"));
const nodeModules = entry.slice(0, entry.lastIndexOf("node_modules") + "node_modules".length);
const ortDist = path.join(nodeModules, "onnxruntime-web", "dist");
const outDir = path.join(process.cwd(), "public", "ort");
mkdirSync(outDir, { recursive: true });
for (const file of ["ort-wasm-simd-threaded.asyncify.mjs", "ort-wasm-simd-threaded.asyncify.wasm"]) {
  copyFileSync(path.join(ortDist, file), path.join(outDir, file));
}
console.log(`Copied ONNX Runtime WASM to ${outDir}`);
