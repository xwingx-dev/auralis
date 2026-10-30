import { env, pipeline, type FeatureExtractionPipeline } from "@huggingface/transformers";
import { decideIntent, INTENT_EXAMPLES, transcriptGuard } from "../../utils/intent";

env.allowLocalModels = false;
if (env.backends.onnx.wasm) {
  env.backends.onnx.wasm.numThreads = 1;
  // Serve the ONNX Runtime WASM from our own origin instead of the jsDelivr default.
  env.backends.onnx.wasm.wasmPaths = {
    mjs: `${self.location.origin}/ort/ort-wasm-simd-threaded.asyncify.mjs`,
    wasm: `${self.location.origin}/ort/ort-wasm-simd-threaded.asyncify.wasm`,
  };
}

let classifier: Promise<{ extractor: FeatureExtractionPipeline; examples: number[][] }> | null = null;

function loadClassifier() {
  classifier ??= (async () => {
    const extractor = await pipeline("feature-extraction", "onnx-community/all-MiniLM-L6-v2-ONNX", {
      device: "wasm",
      dtype: "q4",
    });
    const embeddings = await extractor(INTENT_EXAMPLES.map((example) => example.text), {
      pooling: "mean", normalize: true,
    });
    return { extractor, examples: embeddings.tolist() as number[][] };
  })().catch((error: unknown) => {
    classifier = null;
    throw error;
  });
  return classifier;
}

let busy = false;

self.addEventListener("message", async (event: MessageEvent<{ id: number; text: string }>) => {
  const { id, text } = event.data;
  if (busy) {
    self.postMessage({ id, status: "error", error: "The classifier is busy." });
    return;
  }
  busy = true;
  try {
    const guarded = transcriptGuard(text);
    if (guarded) {
      self.postMessage({ id, status: "result", decision: guarded });
      return;
    }
    self.postMessage({ id, status: "loading" });
    const { extractor, examples } = await loadClassifier();
    const output = await extractor(text.trim(), { pooling: "mean", normalize: true });
    const decision = decideIntent((output.tolist() as number[][])[0], examples);
    self.postMessage({ id, status: "result", decision });
  } catch (error) {
    self.postMessage({ id, status: "error", error: error instanceof Error ? error.message : "Intent classification failed." });
  } finally {
    busy = false;
  }
});