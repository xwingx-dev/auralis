"use client";

import { useEffect, useRef, useState } from "react";
import { encodeWav } from "@/utils/wav";
import { suppressNoise } from "@/utils/denoise";
import AudioVisualizer from "./AudioVisualizer";

// Mutable capture resources; generation invalidates async setup that is still pending.
type AudioCapture = {
  context: AudioContext | null;
  stream: MediaStream | null;
  processor: AudioWorkletNode | null;
  source: MediaStreamAudioSourceNode | null;
  analyser: AnalyserNode | null;
  generation: number;
};

function releaseCapture(capture: AudioCapture) {
  // Cancels any startStreaming call still awaiting permission or setup.
  capture.generation++;
  if (capture.processor) {
    capture.processor.port.onmessage = null;
    capture.processor.port.close();
    capture.processor.disconnect();
  }
  capture.source?.disconnect();
  capture.analyser?.disconnect();
  capture.stream?.getTracks().forEach((track) => track.stop());
  if (capture.context && capture.context.state !== "closed") {
    void capture.context.close().catch(() => {});
  }
  capture.processor = null;
  capture.source = null;
  capture.analyser = null;
  capture.stream = null;
  capture.context = null;
}

export default function LiveAudioTranscriber({ onTranscriptChange }: { onTranscriptChange?: (text: string) => void }) {
  const onTranscriptChangeRef = useRef(onTranscriptChange);
  useEffect(() => {
    onTranscriptChangeRef.current = onTranscriptChange;
  }, [onTranscriptChange]);
  const workerRef = useRef<Worker | null>(null);
  const captureRef = useRef<AudioCapture>({
    context: null,
    stream: null,
    processor: null,
    source: null,
    analyser: null,
    generation: 0,
  });

  const [transcript, setTranscript] = useState<string>("");
  const [isLive, setIsLive] = useState<boolean>(false);
  const [status, setStatus] = useState<string>("Ready");
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isModelReady, setIsModelReady] = useState(false);
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [audioAnalyser, setAudioAnalyser] = useState<AnalyserNode | null>(null);

  useEffect(() => {
    const capture = captureRef.current;
    // Tracks the object URL so it can be revoked on replacement and unmount.
    let playbackUrl: string | null = null;
    // VAD and Whisper run off the main thread to keep the interface responsive.
    workerRef.current = new Worker(
      new URL("../workers/asrs.worker.ts", import.meta.url),
      { type: "module" }
    );

    workerRef.current.onmessage = (event: MessageEvent) => {
      const { status, text, percentage, file, error } = event.data;
      if (status === "captured-audio") {
        // Playback exposes the last Whisper input, not necessarily the full recording.
        if (playbackUrl) URL.revokeObjectURL(playbackUrl);
        playbackUrl = event.data.audio.length
          ? URL.createObjectURL(encodeWav(event.data.audio, event.data.sampleRate))
          : null;
        setAudioUrl(playbackUrl);
        return;
      }
      if (status === "denoise-request") {
        // VAD has already selected speech; RNNoise needs Web Audio on the main thread.
        const { id, audio } = event.data;
        void suppressNoise(audio, 16000)
          .catch((failure: unknown) => {
            // Falls back to unsuppressed speech so transcription keeps running.
            console.error("Noise suppression failed:", failure);
            return audio;
          })
          .then((denoised: Float32Array<ArrayBuffer>) => {
            // The id resumes the worker pass waiting for this specific audio buffer.
            workerRef.current?.postMessage({ command: "denoised", id, audio: denoised }, { transfer: [denoised.buffer] });
          });
        return;
      }
      if (status === "loading") {
        setStatus(event.data.message ?? `Downloading Model: ${percentage}% (${String(file).substring(0, 15)}...)`);
      } else if (status === "ready") {
        setIsModelReady(true);
        setIsModelLoading(false);
        setStatus("Ready");
      } else if (status === "streaming-result" && capture.stream) {
        // Each result replaces the rolling-window hypothesis rather than appending text.
        setTranscript(text);
        onTranscriptChangeRef.current?.(text);
        setStatus("Listening...");
      } else if (status === "no-speech" && capture.stream) {
        setStatus("Waiting for speech...");
      } else if (status === "error") {
        setIsModelLoading(false);
        setStatus(`Error: ${error}`);
      }
    };

    workerRef.current.onerror = (event) => {
      releaseCapture(capture);
      setAudioAnalyser(null);
      setIsLive(false);
      setIsModelLoading(false);
      setIsModelReady(false);
      setStatus(`Worker error: ${event.message}`);
    };

    return () => {
      releaseCapture(capture);
      workerRef.current?.terminate();
      if (playbackUrl) URL.revokeObjectURL(playbackUrl);
    };
  }, []);

  const loadModel = () => {
    if (!("gpu" in navigator)) {
      // The worker targets WebGPU only and has no WASM fallback.
      setStatus("WebGPU is unavailable. Use a WebGPU-enabled browser on HTTPS or localhost.");
      return;
    }
    setIsModelLoading(true);
    setStatus("Loading Whisper Small English...");
    workerRef.current?.postMessage({ command: "load" });
  };

  const startStreaming = async () => {
    const capture = captureRef.current;
    if (!isModelReady || capture.stream || capture.context) return;
    // Identifies this start attempt; a stop or a newer start makes it stale.
    const generation = ++capture.generation;
    setAudioUrl(null);
    setTranscript("");
    onTranscriptChangeRef.current?.("");
    setIsLive(true);
    try {
      setStatus("Initializing Microphone...");
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          // Avoid applying browser suppression and RNNoise to the same speech.
          noiseSuppression: false,
          autoGainControl: true
        }
      });
      if (capture.generation !== generation) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      capture.stream = stream;

      // The context resamples microphone input to Whisper's 16 kHz before capture.
      const audioContext = new AudioContext({ sampleRate: 16000 });
      capture.context = audioContext;
      if (audioContext.sampleRate !== 16000) {
        throw new Error("The browser could not create a 16 kHz audio context.");
      }
      // Public assets use root-relative URLs; the processor name is registered by this script.
      await audioContext.audioWorklet.addModule("/worklets/audio-capture.worklet.js");
      if (capture.generation !== generation) return;

      const source = audioContext.createMediaStreamSource(stream);
      capture.source = source;
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.78;
      capture.analyser = analyser;
      source.connect(analyser);
      const processor = new AudioWorkletNode(audioContext, "audio-capture", {
        channelCount: 1,
        channelCountMode: "explicit",
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
      });
      capture.processor = processor;

      // Chunks are transferred to the worker, so no copy is made.
      processor.port.onmessage = (event: MessageEvent<Float32Array>) => {
        if (capture.generation !== generation) return;
        workerRef.current?.postMessage({
          command: "processChunk",
          audioChunk: event.data,
        }, [event.data.buffer]);
      };

      source.connect(processor);
      // The destination keeps rendering active; the capture processor outputs silence.
      processor.connect(audioContext.destination);
      await audioContext.resume();
      if (capture.generation !== generation) return;
      setAudioAnalyser(analyser);
      setStatus("Listening...");
    } catch (error) {
      if (capture.generation !== generation) return;
      releaseCapture(capture);
      setAudioAnalyser(null);
      setIsLive(false);
      setStatus(error instanceof Error ? `Audio capture failed: ${error.message}` : "Audio capture failed.");
    }
  };

  const stopStreaming = () => {
    setAudioAnalyser(null);
    setIsLive(false);
    setStatus("Stopped");
    releaseCapture(captureRef.current);
    // Export is queued before reset so the worker copies the audio before clearing it.
    workerRef.current?.postMessage({ command: "exportTranscriptionAudio" });
    workerRef.current?.postMessage({ command: "reset" });
  };

  const chips = [
    ["Speech Analysis", "M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3Zm-7 9a7 7 0 0 0 14 0M12 19v3"],
    ["Visual Reasoning", "M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Zm10-3a3 3 0 1 0 0 6 3 3 0 0 0 0-6Z"],
    ["Routing Decision", "M5 5l5 5m4 4 5 5M19 5l-5 5m-4 4-5 5"],
  ];
  const buttonBase = "rounded-xl px-6 py-2.5 font-semibold text-white disabled:cursor-wait disabled:opacity-50";

  return (
    <div className="min-w-0 space-y-4 rounded-2xl border border-line bg-surface/80 p-5 shadow-[0_0_24px_rgba(29,107,255,0.08)]">
      <div className="flex flex-wrap items-center gap-3">
        <span aria-hidden className="flex h-7 items-end gap-1">
          {[3, 6, 4, 7, 3].map((h, i) => (
            <span key={i} className="w-1 rounded-full bg-brand-cyan" style={{ height: `${h * 4}px` }} />
          ))}
        </span>
        <h2 className="text-xl font-semibold text-white">Auralis Engine</h2>
        <span className="rounded-full border border-brand-mint/40 bg-brand-mint/10 px-3 py-1 text-xs font-medium text-brand-mint">
          Local Processing (WebGPU / WASM)
        </span>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        {chips.map(([label, icon]) => (
          <div key={label} className="flex items-center gap-3 rounded-xl border border-line bg-surface-2/40 p-3">
            <span className="flex size-10 shrink-0 items-center justify-center rounded-full bg-surface-2 text-brand-cyan">
              <svg viewBox="0 0 24 24" className="size-5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden><path d={icon} /></svg>
            </span>
            <div className="min-w-0 text-sm">
              <p className="truncate font-medium text-white">{label}</p>
              <p className="flex items-center gap-1.5 text-xs text-brand-mint"><span className="size-1.5 rounded-full bg-brand-mint" />On Device</p>
            </div>
          </div>
        ))}
      </div>

      <div className="rounded-xl border border-line bg-surface-2/40 p-3">
        <div className="mb-2 flex items-center justify-between gap-2 text-sm">
          <span className="text-white">Audio Stream (local)</span>
          <span className={`rounded-full border border-brand-mint/40 bg-brand-mint/10 px-3 py-0.5 text-xs text-brand-mint ${isLive ? "animate-pulse" : ""}`}>
            {isLive ? "Processing..." : "Idle"}
          </span>
        </div>
        <AudioVisualizer analyser={audioAnalyser} />
      </div>

      <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-line bg-gradient-to-r from-brand-mint/30 via-brand-primary/30 to-brand-violet/20 p-4">
        <div className="flex min-w-0 items-center gap-4">
          <span className={`flex size-12 shrink-0 items-center justify-center rounded-full border-2 border-brand-mint/70 ${isLive ? "animate-pulse" : ""}`}>
            <span className="size-4 rounded-full bg-brand-mint" />
          </span>
          <div className="min-w-0">
            <p className="text-lg font-semibold text-white">{isLive ? "Listening" : "Dictation"}</p>
            <p role="status" className="break-words text-sm text-brand-muted">{status}</p>
          </div>
        </div>
        {!isModelReady ? (
          <button onClick={loadModel} disabled={isModelLoading} className={`${buttonBase} bg-brand-primary hover:bg-brand-primary/80`}>
            {isModelLoading ? "Loading Model..." : "Load Whisper Small English"}
          </button>
        ) : !isLive ? (
          <button onClick={startStreaming} className={`${buttonBase} bg-brand-mint/80 hover:bg-brand-mint`}>
            Turn On Dictation
          </button>
        ) : (
          <button onClick={stopStreaming} className={`${buttonBase} flex items-center gap-2 border border-brand-danger bg-brand-danger/20 hover:bg-brand-danger/40`}>
            <span className="size-3 rounded-sm bg-brand-danger" />Stop
          </button>
        )}
      </div>

      <div className="min-h-[96px] rounded-xl border border-line bg-surface-2/40 p-4">
        <p className="whitespace-pre-wrap text-sm leading-relaxed text-white">
          {transcript || <span className="italic text-brand-muted">Speak now... text will sync perfectly.</span>}
        </p>
      </div>
      {!isLive && audioUrl && (
        <audio
          controls
          src={audioUrl}
          aria-label="Last transcription input audio"
          className="w-full"
        />
      )}
    </div>
  );
}
