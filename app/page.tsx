"use client";
import { useEffect, useRef, useState } from "react";
import LiveAudioTranscriber from "./components/AudioTranscriberStreaming";
import { MAX_TRANSCRIPT_LENGTH, type RoutingDecision } from "@/utils/intent";

type ProcessFrameResponse =
  | { success: true; analysis: { choices?: { message?: { content?: string | null } }[] } }
  | { success: false; error?: string };

type IntentMessage = {
  id: number;
  status: "loading" | "result" | "error";
  decision?: RoutingDecision;
  error?: string;
};

const INTENT_SETTLE_MS = 1200;

export default function CameraPipeline() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const workerRef = useRef<Worker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const cameraReadyRef = useRef<Promise<void> | null>(null);
  const controllerRef = useRef<AbortController | null>(null);
  const requestId = useRef(0);
  const transcriptRef = useRef("");
  const busyRef = useRef(false);
  const lastSubmittedRef = useRef<string | null>(null);
  const [transcript, setTranscript] = useState("");
  const [decision, setDecision] = useState<RoutingDecision | null>(null);
  const [analysis, setAnalysis] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("Ready");
  const [error, setError] = useState<string | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    const video = videoRef.current;
    let stream: MediaStream | null = null;
    const releaseCamera = () => {
      stream?.getTracks().forEach((track) => track.stop());
      if (streamRef.current === stream) streamRef.current = null;
      if (video?.srcObject === stream) video.srcObject = null;
    };

    cameraReadyRef.current = (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (!active) {
          releaseCamera();
          return;
        }
        if (!video) throw new Error("Camera preview is unavailable.");
        streamRef.current = stream;
        video.srcObject = stream;
        await video.play();
      } catch (failure) {
        releaseCamera();
        if (active) setCameraError(failure instanceof Error ? failure.message : "Unable to access camera.");
      }
    })();

    return () => {
      active = false;
      releaseCamera();
    };
  }, []);

  useEffect(() => {
    let active = true;
    const worker = new Worker(new URL("./workers/intent.worker.ts", import.meta.url), { type: "module" });
    workerRef.current = worker;

    worker.onmessage = async (event: MessageEvent<IntentMessage>) => {
      const message = event.data;
      if (message.status === "loading") {
        if (message.id === requestId.current) setStatus("Classifying locally with MiniLM...");
        return;
      }
      try {
        if (!active || message.id !== requestId.current) return;
        if (message.status === "error" || !message.decision) {
          throw new Error(message.error ?? "Intent classification failed. No cloud request was sent.");
        }
        const route = message.decision;
        setDecision(route);
        if (!route.requiresCloud) {
          setStatus(route.reason === "matched" ? "Transcript only" : "Kept local: intent uncertain or unsupported");
          return;
        }
        const text = transcriptRef.current.trim();
        const controller = new AbortController();
        controllerRef.current = controller;
        let image: string | undefined;
        if (route.requiresVision) {
          setStatus("Capturing visual context...");
          await cameraReadyRef.current;
          if (!active || message.id !== requestId.current) return;
          const video = videoRef.current;
          if (!video || !streamRef.current?.getVideoTracks().some((track) => track.readyState === "live")) {
            throw new Error("Camera is unavailable. Visual requests require a live camera stream.");
          }
          if (!video.videoWidth || !video.videoHeight) throw new Error("Camera frame is not ready.");
          const canvas = document.createElement("canvas");
          const scale = Math.min(1, 960 / Math.max(video.videoWidth, video.videoHeight));
          canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
          canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
          const context = canvas.getContext("2d");
          if (!context) throw new Error("Unable to capture camera frame.");
          context.drawImage(video, 0, 0, canvas.width, canvas.height);
          image = canvas.toDataURL("image/jpeg", 0.75);
        }
        if (!active || message.id !== requestId.current) return;
        setStatus(route.requiresVision ? "Reasoning with visual context..." : "Reasoning with text...");
        const response = await fetch("/api/process-frame", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ transcript: text, routing: route, ...(image ? { image } : {}) }),
          signal: controller.signal,
        });
        const data: ProcessFrameResponse = await response.json();
        if (!response.ok || !data.success) {
          throw new Error(!data.success ? data.error ?? "Cloud reasoning failed." : "Cloud request failed.");
        }
        const content = data.analysis?.choices?.[0]?.message?.content;
        if (typeof content !== "string" || !content.trim()) throw new Error("No response returned.");
        if (active && message.id === requestId.current) {
          setAnalysis(content);
          setStatus("Complete");
        }
      } catch (failure) {
        if (active && message.id === requestId.current) {
          setError(failure instanceof Error ? failure.message : "Unable to process request.");
          setStatus("Request failed");
        }
      } finally {
        busyRef.current = false;
        if (active) setLoading(false);
      }
    };
    worker.onerror = (event) => {
      requestId.current++;
      controllerRef.current?.abort();
      busyRef.current = false;
      setLoading(false);
      setError(`Classifier unavailable: ${event.message}. No further cloud requests will be sent.`);
      setStatus("Classifier unavailable");
      workerRef.current = null;
      worker.terminate();
    };
    return () => {
      active = false;
      controllerRef.current?.abort();
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  useEffect(() => {
    const text = transcript.trim();
    if (!text || loading || !workerRef.current || lastSubmittedRef.current === text) return;
    const timer = setTimeout(() => {
      if (busyRef.current || !workerRef.current || transcriptRef.current !== text) return;
      busyRef.current = true;
      lastSubmittedRef.current = text;
      setLoading(true);
      setStatus("Classifying locally with MiniLM...");
      workerRef.current.postMessage({ id: ++requestId.current, text });
    }, INTENT_SETTLE_MS);
    return () => clearTimeout(timer);
  }, [transcript, loading]);

  const updateTranscript = (text: string) => {
    setTranscript(text);
    const normalized = text.trim();
    if (normalized === transcriptRef.current) return;
    transcriptRef.current = normalized;
    lastSubmittedRef.current = null;
    requestId.current++;
    controllerRef.current?.abort();
    setDecision(null);
    setAnalysis(null);
    setError(null);
    setStatus(normalized ? "Waiting for transcript to settle..." : "Ready");
  };

  const card = "rounded-2xl border border-line bg-surface/80 p-5 shadow-[0_0_24px_rgba(29,107,255,0.08)]";
  const yesNo = (value: boolean) => (value ? "Yes" : "No");
  const rows = decision
    ? [
        ["Transcription only", yesNo(decision.transcriptionOnly)],
        ["Cloud reasoning", yesNo(decision.requiresCloud)],
        ["Visual context", yesNo(decision.requiresVision)],
      ]
    : [];

  return (
    <main className="grid w-full items-start gap-5 p-5 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
      <div className="min-w-0 space-y-5">
        <section aria-label="Local camera preview" className="min-w-0 space-y-2">
          <div className="relative overflow-hidden rounded-2xl border border-line bg-black">
            <span className="absolute top-3 left-3 z-10 rounded-lg bg-black/50 px-3 py-1.5 text-sm font-medium backdrop-blur">Live Video</span>
            <video ref={videoRef} autoPlay playsInline muted aria-label="Live camera feed" className="aspect-video w-full bg-black object-cover" />
          </div>
          {cameraError && <p role="alert" className="break-words text-white">{cameraError}</p>}
        </section>
        <LiveAudioTranscriber onTranscriptChange={updateTranscript} />
      </div>
      <div className="min-w-0 space-y-5">
        <section aria-labelledby="request-heading" aria-busy={loading} className={`${card} min-w-0 space-y-3`}>
          <div className="flex items-center justify-between gap-3">
            <h1 id="request-heading" className="text-xl font-semibold text-white">Request</h1>
            <span className="rounded-full border border-line bg-surface-2 px-3 py-1 text-xs text-brand-muted">Live Transcription</span>
          </div>
          <textarea aria-label="Request transcript" value={transcript} onChange={(event) => updateTranscript(event.target.value)} maxLength={MAX_TRANSCRIPT_LENGTH} rows={5} placeholder="Describe what is happening in this scene." className="w-full resize-y rounded-xl border border-line bg-surface-2/60 p-4 text-lg text-white outline-none placeholder:text-brand-muted/60 focus:border-brand-cyan focus:ring-1 focus:ring-brand-cyan" />
          {error && <p role="alert" className="break-words text-white">{error}</p>}
          {analysis && <p className="whitespace-pre-wrap break-words">{analysis}</p>}
        </section>
        <section aria-label="Local decision" className={`${card} space-y-4`}>
          <div className="flex flex-wrap items-center gap-3">
            <h2 className="text-xl font-semibold text-white">Local Decision</h2>
            {decision && !decision.requiresCloud && (
              <span className="rounded-full border border-brand-mint/40 bg-brand-mint/10 px-3 py-1 text-sm text-brand-mint">No Escalation Needed</span>
            )}
          </div>
          <p role="status" className="font-medium text-brand-mint">{status}</p>
          {decision && (
            <dl className="divide-y divide-line overflow-hidden rounded-xl border border-line bg-surface-2/40">
              {rows.map(([label, value]) => (
                <div key={label} className="flex items-center justify-between px-5 py-4">
                  <dt className="text-white">{label}</dt>
                  <dd className={value === "Yes" ? "text-brand-mint" : "text-brand-muted"}>{value}</dd>
                </div>
              ))}
            </dl>
          )}
        </section>
        <section aria-label="Why no escalation" className={`${card} space-y-4`}>
          <h2 className="text-xl font-semibold text-white">Why No Escalation?</h2>
          <p className="leading-relaxed text-brand-muted">The local engine detected low intent confidence and no need for visual context. Your audio is processed in the browser using WebAssembly, saving bandwidth, reducing latency, and cutting cloud costs.</p>
          <div className="grid grid-cols-3 divide-x divide-line border-t border-line pt-4 text-center text-sm text-brand-muted">
            <div className="px-2"><p className="font-semibold text-brand-mint">Faster</p><p>No round-trip for simple queries</p></div>
            <div className="px-2"><p className="font-semibold text-brand-cyan">Lower Cost</p><p>Only escalate when needed</p></div>
            <div className="px-2"><p className="font-semibold text-brand-violet">More Private</p><p>Keeps your data on your device longer</p></div>
          </div>
        </section>
      </div>
    </main>
  );
}
