// RNNoise only operates at 48 kHz; the ASR pipeline uses 16 kHz.
const RNNOISE_SAMPLE_RATE = 48_000;
// Trailing silence lets the RNNoise frame buffer flush the final speech.
const FLUSH_SECONDS = 0.1;

let wasmBinary: Promise<ArrayBuffer> | null = null;

// Renders mono audio through RNNoise. Requires AudioWorklet, so it must run on the main thread.
export async function suppressNoise(audio: Float32Array<ArrayBuffer>, sampleRate: number): Promise<Float32Array<ArrayBuffer>> {
  const { loadRnnoise, RnnoiseWorkletNode } = await import("@sapphi-red/web-noise-suppressor");
  wasmBinary ??= loadRnnoise({
    url: "/noise-suppressor/rnnoise.wasm",
    simdUrl: "/noise-suppressor/rnnoise_simd.wasm",
  }).catch((error: unknown) => {
    wasmBinary = null;
    throw error;
  });
  const binary = await wasmBinary;

  const length = audio.length + Math.round(FLUSH_SECONDS * sampleRate);
  const input = new AudioBuffer({ numberOfChannels: 1, length, sampleRate });
  input.copyToChannel(audio, 0);

  // The browser resamples the 16 kHz input up to the 48 kHz render rate.
  const upsampler = new OfflineAudioContext(1, Math.ceil(length * RNNOISE_SAMPLE_RATE / sampleRate), RNNOISE_SAMPLE_RATE);
  const scripts = await Promise.all([
    "/noise-suppressor/rnnoise-ready.worklet.js",
    "/noise-suppressor/rnnoiseWorklet.js",
  ].map(async (url) => {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`RNNoise worklet load failed: ${response.status}`);
    return response.text();
  }));
  const workletUrl = URL.createObjectURL(new Blob(scripts, { type: "text/javascript" }));
  try {
    await upsampler.audioWorklet.addModule(workletUrl);
  } finally {
    URL.revokeObjectURL(workletUrl);
  }
  // The library types declare AudioContext, but the node works with any BaseAudioContext.
  const node = new RnnoiseWorkletNode(upsampler as unknown as AudioContext, { maxChannels: 1, wasmBinary: binary });
  try {
    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => {
        node.port.onmessage = null;
        node.onprocessorerror = null;
        reject(new Error("RNNoise initialization timed out."));
      }, 10_000);
      const finish = (error?: Error) => {
        clearTimeout(timeout);
        node.port.onmessage = null;
        node.onprocessorerror = null;
        if (error) reject(error);
        else resolve();
      };
      node.port.onmessage = (event: MessageEvent) => {
        if (event.data === "ready") finish();
      };
      node.onprocessorerror = () => finish(new Error("RNNoise initialization failed."));
    });
    const source = upsampler.createBufferSource();
    source.buffer = input;
    source.connect(node).connect(upsampler.destination);
    source.start();
    const denoised = await upsampler.startRendering();

    // Resampling back to 16 kHz uses the browser's anti-aliasing filter.
    const downsampler = new OfflineAudioContext(1, length, sampleRate);
    const resampled = downsampler.createBufferSource();
    resampled.buffer = denoised;
    resampled.connect(downsampler.destination);
    resampled.start();
    const output = await downsampler.startRendering();
    return output.getChannelData(0).slice();
  } finally {
    node.destroy();
    node.disconnect();
    node.port.close();
  }
}
