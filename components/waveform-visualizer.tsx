"use client";

import { useRef, useEffect, useCallback } from "react";

/**
 * WaveformVisualizer
 *
 * Renders a real-time audio waveform on a <canvas>.
 * - When `analyser` is provided (live recording), it draws a live waveform.
 * - When `audioBuffer` is provided (uploaded file), it draws the static waveform.
 */

interface WaveformVisualizerProps {
  analyser?: AnalyserNode | null;
  audioBuffer?: AudioBuffer | null;
  isRecording?: boolean;
  height?: number;
}

export function WaveformVisualizer({
  analyser,
  audioBuffer,
  isRecording = false,
  height = 120,
}: WaveformVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);

  // Draw live waveform from AnalyserNode
  const drawLive = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !analyser) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const bufferLength = analyser.fftSize;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      animationRef.current = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(dataArray);

      const { width, height: h } = canvas;
      ctx.clearRect(0, 0, width, h);

      // Background glow effect
      const gradient = ctx.createLinearGradient(0, 0, width, 0);
      gradient.addColorStop(0, "rgba(96, 165, 250, 0.05)");
      gradient.addColorStop(0.5, "rgba(96, 165, 250, 0.1)");
      gradient.addColorStop(1, "rgba(96, 165, 250, 0.05)");
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, h);

      // Draw waveform
      ctx.lineWidth = 2;
      ctx.strokeStyle = isRecording
        ? "rgba(239, 68, 68, 0.9)"
        : "rgba(96, 165, 250, 0.8)";
      ctx.beginPath();

      const sliceWidth = width / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * h) / 2;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
        x += sliceWidth;
      }

      ctx.lineTo(width, h / 2);
      ctx.stroke();

      // Draw center line
      ctx.strokeStyle = "rgba(148, 163, 184, 0.15)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(0, h / 2);
      ctx.lineTo(width, h / 2);
      ctx.stroke();
      ctx.setLineDash([]);
    };

    draw();
  }, [analyser, isRecording]);

  // Draw static waveform from AudioBuffer
  const drawStatic = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !audioBuffer) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { width, height: h } = canvas;
    ctx.clearRect(0, 0, width, h);

    // Get mono channel data
    const data = audioBuffer.getChannelData(0);
    const step = Math.ceil(data.length / width);
    const amp = h / 2;

    // Background
    const gradient = ctx.createLinearGradient(0, 0, width, 0);
    gradient.addColorStop(0, "rgba(96, 165, 250, 0.03)");
    gradient.addColorStop(0.5, "rgba(96, 165, 250, 0.06)");
    gradient.addColorStop(1, "rgba(96, 165, 250, 0.03)");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, h);

    // Draw waveform bars
    for (let i = 0; i < width; i++) {
      let min = 1.0;
      let max = -1.0;
      for (let j = 0; j < step; j++) {
        const datum = data[i * step + j] || 0;
        if (datum < min) min = datum;
        if (datum > max) max = datum;
      }

      const barHeight = Math.max((max - min) * amp, 1);
      const y = (1 + min) * amp;

      // Color gradient based on amplitude
      const intensity = Math.min(barHeight / amp, 1);
      const r = Math.round(96 + intensity * 60);
      const g = Math.round(165 - intensity * 30);
      const b = 250;
      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.4 + intensity * 0.5})`;
      ctx.fillRect(i, y, 1, barHeight);
    }

    // Center line
    ctx.strokeStyle = "rgba(148, 163, 184, 0.15)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(width, h / 2);
    ctx.stroke();
    ctx.setLineDash([]);
  }, [audioBuffer]);

  // Draw idle state
  const drawIdle = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { width, height: h } = canvas;
    ctx.clearRect(0, 0, width, h);

    // Center line
    ctx.strokeStyle = "rgba(148, 163, 184, 0.2)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(width, h / 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Subtle sine wave
    ctx.strokeStyle = "rgba(96, 165, 250, 0.15)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([]);
    ctx.beginPath();
    for (let x = 0; x < width; x++) {
      const y = h / 2 + Math.sin(x * 0.02) * 4;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }, []);

  useEffect(() => {
    // Resize canvas to match container
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = height * window.devicePixelRatio;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }

    if (analyser) {
      drawLive();
    } else if (audioBuffer) {
      drawStatic();
    } else {
      drawIdle();
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [analyser, audioBuffer, drawLive, drawStatic, drawIdle, height]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full rounded-lg"
      style={{ height: `${height}px` }}
      aria-label="Audio waveform visualization"
    />
  );
}
