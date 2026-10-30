"use client";

import { useEffect, useRef } from "react";

export default function AudioVisualizer({ analyser }: { analyser: AnalyserNode | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;

    const motionPreference = matchMedia("(prefers-reduced-motion: reduce)");
    const waveform = new Float32Array(analyser?.fftSize ?? 2048);
    const frequencies = new Uint8Array(analyser?.frequencyBinCount ?? 1024);
    let animation = 0;
    let previousFrame = 0;
    let width = 0;
    let height = 0;

    const paint = () => {
      if (!width || !height) return;
      if (analyser) {
        analyser.getFloatTimeDomainData(waveform);
        analyser.getByteFrequencyData(frequencies);
      }
      const rms = Math.sqrt(waveform.reduce((sum, sample) => sum + sample * sample, 0) / waveform.length);
      const energy = Math.min(1, rms * 6);
      context.clearRect(0, 0, width, height);
      context.fillStyle = "#061634";
      context.fillRect(0, 0, width, height);

      const horizon = height * 0.62;
      context.lineWidth = 1;
      context.strokeStyle = "#12376e";
      context.beginPath();
      for (let column = -6; column <= 6; column++) {
        context.moveTo(width / 2 + column * width * 0.055, horizon);
        context.lineTo(width / 2 + column * width * 0.22, height);
      }
      for (const fraction of [0, 0.12, 0.3, 0.58, 1]) {
        const gridY = horizon + (height - horizon) * fraction;
        context.moveTo(0, gridY);
        context.lineTo(width, gridY);
      }
      context.stroke();

      const spectrum = context.createLinearGradient(0, height, 0, height * 0.2);
      spectrum.addColorStop(0, "#7c3aed");
      spectrum.addColorStop(0.55, "#facc15");
      spectrum.addColorStop(1, "#22d3ee");
      context.fillStyle = spectrum;
      const bands = 48;
      const gap = 3;
      const barWidth = Math.max(1, (width - 32 - gap * (bands - 1)) / bands);
      const minBin = analyser ? Math.max(1, Math.floor(70 * analyser.fftSize / analyser.context.sampleRate)) : 1;
      const maxBin = analyser ? Math.min(frequencies.length, Math.ceil(7500 * analyser.fftSize / analyser.context.sampleRate)) : frequencies.length;
      for (let band = 0; band < bands; band++) {
        const start = Math.floor(minBin * (maxBin / minBin) ** (band / bands));
        const end = Math.max(start + 1, Math.floor(minBin * (maxBin / minBin) ** ((band + 1) / bands)));
        let peak = 0;
        for (let bin = start; bin < end; bin++) peak = Math.max(peak, frequencies[bin] ?? 0);
        const barHeight = Math.max(2, (peak / 255) ** 1.6 * height * 0.58 * Math.min(1, energy * 3));
        const barX = 16 + band * (barWidth + gap);
        context.globalAlpha = 0.28 + energy * 0.6;
        context.fillRect(barX, height - 14 - barHeight, barWidth, barHeight);
        if (barHeight > 4) {
          context.globalAlpha = 0.85;
          context.fillRect(barX, height - 17 - barHeight, barWidth, 2);
        }
      }
      context.globalAlpha = 1;

      const line = context.createLinearGradient(0, 0, width, 0);
      line.addColorStop(0, "#1d6bff");
      line.addColorStop(0.5, "#22d3ee");
      line.addColorStop(1, "#facc15");
      context.strokeStyle = line;
      context.lineWidth = 1.8;
      context.lineJoin = "round";
      context.shadowColor = "#22d3ee";
      context.shadowBlur = motionPreference.matches ? 0 : 5 + energy * 12;
      context.beginPath();
      const center = height * 0.4;
      const gain = Math.min(4, 0.18 / Math.max(rms, 0.045));
      const points = Math.max(2, Math.floor(width));
      for (let point = 0; point < points; point++) {
        const sample = waveform[Math.floor(point * waveform.length / points)];
        const envelope = Math.sin(Math.PI * point / (points - 1)) ** 0.5;
        const lineX = point * width / (points - 1);
        const lineY = center + Math.max(-1, Math.min(1, sample * gain)) * height * 0.29 * envelope;
        if (point === 0) context.moveTo(lineX, lineY);
        else context.lineTo(lineX, lineY);
      }
      context.stroke();
      context.shadowBlur = 0;
    };

    const animate = (time: number) => {
      if (document.hidden) return;
      if (time - previousFrame >= (motionPreference.matches ? 100 : 32)) {
        paint();
        previousFrame = time;
      }
      if (analyser) animation = requestAnimationFrame(animate);
    };

    const resize = () => {
      const bounds = canvas.getBoundingClientRect();
      width = bounds.width;
      height = bounds.height;
      const ratio = Math.min(devicePixelRatio || 1, 2);
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      paint();
    };
    const onVisibilityChange = () => {
      cancelAnimationFrame(animation);
      if (!document.hidden && analyser) animation = requestAnimationFrame(animate);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);
    resize();
    if (analyser && !document.hidden) animation = requestAnimationFrame(animate);
    document.addEventListener("visibilitychange", onVisibilityChange);
    motionPreference.addEventListener("change", paint);

    return () => {
      cancelAnimationFrame(animation);
      observer.disconnect();
      document.removeEventListener("visibilitychange", onVisibilityChange);
      motionPreference.removeEventListener("change", paint);
    };
  }, [analyser]);

  return (
    <canvas
      ref={canvasRef}
      role="img"
      aria-label={analyser ? "Live microphone waveform and frequency spectrum" : "Microphone visualizer idle"}
      className="block h-40 w-full rounded-lg bg-surface"
    />
  );
}