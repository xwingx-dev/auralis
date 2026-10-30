import { pipeline, env, type AutomaticSpeechRecognitionPipeline, type ProgressCallback } from "@huggingface/transformers";
import { NonRealTimeVAD } from "@ricky0123/vad-web";

// Model weights are fetched from the Hugging Face Hub at runtime.
env.allowLocalModels = false;

// Whisper expects 16 kHz mono input.
const WHISPER_SAMPLING_RATE = 16_000;
// Rolling window: only the latest 30 seconds are kept.
const MAX_SAMPLES = WHISPER_SAMPLING_RATE * 30;
// Minimum buffered audio before the first transcription.
const MIN_SAMPLES = WHISPER_SAMPLING_RATE * 2;
// Minimum new audio required between transcriptions.
const UPDATE_SAMPLES = WHISPER_SAMPLING_RATE;

// Silero VAD gates Whisper so silence and noise do not produce text.
class SpeechDetector {
  static instance: Promise<NonRealTimeVAD> | null = null;

  static getInstance() {
    this.instance ??= NonRealTimeVAD.new({
      // CDN asset must match the installed @ricky0123/vad-web version.
      modelURL: "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.31/dist/silero_vad_legacy.onnx",
      positiveSpeechThreshold: 0.6,
      negativeSpeechThreshold: 0.35,
      minSpeechMs: 400,
      preSpeechPadMs: 300,
      redemptionMs: 400,
      ortConfig: (ort) => {
        // Single-threaded WASM avoids the cross-origin isolation headers multi-threading requires.
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`;
      },
    }).catch((error: unknown) => {
      this.instance = null;
      throw error;
    });
    return this.instance;
  }
}

class PipelineSingleton {
  static task = "automatic-speech-recognition" as const;
  // WebGPU with FP32 encoder and Q4 decoder, matching the Hugging Face reference.
  //static model = "onnx-community/whisper-base";
  static model = "onnx-community/whisper-small.en";
  static instance: Promise<AutomaticSpeechRecognitionPipeline> | null = null;

  static getInstance(progress_callback?: ProgressCallback) {
    if (this.instance === null) {
      this.instance = pipeline(this.task, this.model, {
        device: "webgpu",
        dtype: {
          encoder_model: "fp32",
          decoder_model_merged: "q4",
        },
        progress_callback,
      }).then((transcriber) => transcriber as AutomaticSpeechRecognitionPipeline).catch((error: unknown) => {
        this.instance = null;
        throw error;
      });
    }
    return this.instance;
  }
}

// Rolling window of 16 kHz mono samples awaiting transcription.
let masterAudioBuffer = new Float32Array(0);
let lastTranscriptionAudio: Float32Array<ArrayBuffer> | null = null;
// Total samples received this session; keeps counting after older audio is trimmed.
let receivedSamples = 0;
// Stream position up to which audio has already been evaluated.
let processedSamples = 0;
// Incremented on reset so stale async work can detect that it was discarded.
let generation = 0;
// Ensures only one processing pass runs at a time.
let isProcessing = false;
let isLoadingModel = false;
let isModelReady = false;
// Denoise requests awaiting the main thread, keyed by request id.
const pendingDenoises = new Map<number, (audio: Float32Array<ArrayBuffer>) => void>();
let nextDenoiseId = 0;

// Idempotent: returns immediately while a load is already in progress.
async function loadModel() {
  if (isLoadingModel) return;
  isLoadingModel = true;
  try {
    if (!isModelReady) {
      self.postMessage({ status: "loading", message: "Loading Whisper Small English..." });
      const transcriber = await PipelineSingleton.getInstance((progress) => {
        if (progress.status === "progress") {
          self.postMessage({
            status: "loading",
            percentage: Math.round(progress.progress || 0),
            file: progress.file,
          });
        }
      });
      self.postMessage({ status: "loading", message: "Loading speech detector..." });
      await SpeechDetector.getInstance();
      // Compiles WebGPU shaders so the first real transcription is not delayed.
      self.postMessage({ status: "loading", message: "Warming up WebGPU..." });
      await transcriber(new Float32Array(WHISPER_SAMPLING_RATE), { max_new_tokens: 1 });
      isModelReady = true;
    }
    self.postMessage({ status: "ready" });
  } catch (error: unknown) {
    self.postMessage({
      status: "error",
      error: error instanceof Error ? error.message : String(error),
    });
  } finally {
    isLoadingModel = false;
  }
}

// Noise suppression needs an AudioWorklet, which only the main thread provides.
function suppressNoise(audio: Float32Array<ArrayBuffer>) {
  const id = ++nextDenoiseId;
  return new Promise<Float32Array<ArrayBuffer>>((resolve) => {
    pendingDenoises.set(id, resolve);
    self.postMessage({ status: "denoise-request", id, audio }, { transfer: [audio.buffer] });
  });
}

// Runs passes until no unprocessed audio remains; concurrent calls return immediately.
async function processPendingAudio() {
  if (isProcessing) return;
  isProcessing = true;
  let activeGeneration = generation;
  try {
    while (masterAudioBuffer.length >= MIN_SAMPLES &&
      receivedSamples - processedSamples >= UPDATE_SAMPLES) {
      activeGeneration = generation;
      const detector = await SpeechDetector.getInstance();
      if (generation !== activeGeneration) continue;

      // Audio arriving during this pass is picked up by the next loop iteration.
      const snapshot = masterAudioBuffer;
      const snapshotStart = receivedSamples - snapshot.length;
      const previousProcessedSamples = processedSamples;
      processedSamples = receivedSamples;
      const segments: Float32Array[] = [];
      let hasNewSpeech = false;
      for await (const segment of detector.run(snapshot, WHISPER_SAMPLING_RATE)) {
        segments.push(segment.audio);
        // segment.end is in milliseconds, relative to the start of the snapshot.
        if (snapshotStart + Math.round(segment.end * WHISPER_SAMPLING_RATE / 1000) > previousProcessedSamples) {
          hasNewSpeech = true;
        }
      }
      if (generation !== activeGeneration) continue;
      // Only previously transcribed speech was found; the current transcript is kept.
      if (!hasNewSpeech) {
        self.postMessage({ status: "no-speech" });
        continue;
      }

      const speech = new Float32Array(segments.reduce((length, segment) => length + segment.length, 0));
      let offset = 0;
      for (const segment of segments) {
        speech.set(segment, offset);
        offset += segment.length;
      }
      const denoised = await suppressNoise(speech);
      if (generation !== activeGeneration) continue;

      const transcriber = await PipelineSingleton.getInstance((progress) => {
        if (generation === activeGeneration && progress.status === "progress") {
          self.postMessage({
            status: "loading",
            percentage: Math.round(progress.progress || 0),
            file: progress.file,
          });
        }
      });
      if (generation !== activeGeneration) continue;

      lastTranscriptionAudio = denoised.slice();
      const result = await transcriber(denoised, {
        //  language: "english",
        //  task: "transcribe",
        max_new_tokens: 64,
        return_timestamps: false,
      });

      if (generation === activeGeneration) {
        self.postMessage({ status: "streaming-result", text: result.text });
      }
    }
  } catch (error: unknown) {
    if (generation === activeGeneration) {
      self.postMessage({
        status: "error",
        error: error instanceof Error ? error.message : String(error),
      });
    }
  } finally {
    isProcessing = false;
    // A reset during this pass may have left audio waiting; restart the loop.
    if (generation !== activeGeneration) void processPendingAudio();
  }
}

self.addEventListener("message", (event: MessageEvent) => {
  const { command, audioChunk, id, audio } = event.data;

  if (command === "load") {
    void loadModel();
    return;
  }

  // Copies the rolling window for playback; the copy is transferred, not shared.
  if (command === "exportAudio") {
    const audio = masterAudioBuffer.slice();
    self.postMessage({ status: "captured-audio", audio, sampleRate: WHISPER_SAMPLING_RATE }, { transfer: [audio.buffer] });
    return;
  }

  if (command === "exportTranscriptionAudio") {
    const audio = (lastTranscriptionAudio ?? masterAudioBuffer).slice();
    self.postMessage({ status: "captured-audio", audio, sampleRate: WHISPER_SAMPLING_RATE }, { transfer: [audio.buffer] });
    return;
  }

  // Clears audio and invalidates in-flight results; the loaded model is kept.
  if (command === "reset") {
    generation++;
    masterAudioBuffer = new Float32Array(0);
    lastTranscriptionAudio = null;
    receivedSamples = 0;
    processedSamples = 0;
    return;
  }

  if (command === "denoised") {
    const resolveDenoise = pendingDenoises.get(id);
    pendingDenoises.delete(id);
    resolveDenoise?.(audio);
    return;
  }

  if (command !== "processChunk" || !(audioChunk instanceof Float32Array) || !audioChunk.length) return;

  // A single NaN or Infinity sample invalidates the whole chunk.
  for (const sample of audioChunk) {
    if (!Number.isFinite(sample)) return;
  }
  receivedSamples += audioChunk.length;

  // Trims the oldest audio so the buffer never exceeds the 30-second window.
  const retainedChunk = audioChunk.subarray(-MAX_SAMPLES);
  const retainedPrevious = masterAudioBuffer.subarray(Math.max(0, masterAudioBuffer.length - (MAX_SAMPLES - retainedChunk.length)));
  const newBuffer = new Float32Array(retainedPrevious.length + retainedChunk.length);
  newBuffer.set(retainedPrevious);
  newBuffer.set(retainedChunk, retainedPrevious.length);
  masterAudioBuffer = newBuffer;
  void processPendingAudio();
});