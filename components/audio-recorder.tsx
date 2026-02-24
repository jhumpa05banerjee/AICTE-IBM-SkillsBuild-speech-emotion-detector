"use client";

import { useState, useRef, useCallback } from "react";
import { Mic, Square, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { WaveformVisualizer } from "@/components/waveform-visualizer";

/**
 * AudioRecorder
 *
 * Records audio from the user's microphone using the MediaRecorder API.
 * Provides a live waveform visualization during recording and returns
 * the recorded audio as an AudioBuffer for feature extraction.
 */

interface AudioRecorderProps {
  onRecordingComplete: (audioBuffer: AudioBuffer) => void;
  isProcessing: boolean;
}

export function AudioRecorder({
  onRecordingComplete,
  isProcessing,
}: AudioRecorderProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [analyserNode, setAnalyserNode] = useState<AnalyserNode | null>(null);
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Start recording
  const startRecording = useCallback(async () => {
    setError(null);
    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100,
        },
      });

      // Set up AudioContext and AnalyserNode for visualization
      const audioCtx = new AudioContext();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);

      audioContextRef.current = audioCtx;
      setAnalyserNode(analyser);

      // Set up MediaRecorder
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
          ? "audio/webm;codecs=opus"
          : "audio/webm",
      });

      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = async () => {
        // Stop all tracks
        stream.getTracks().forEach((track) => track.stop());

        // Decode recorded audio
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        try {
          const arrayBuffer = await blob.arrayBuffer();
          const decodeCtx = new AudioContext();
          const audioBuffer = await decodeCtx.decodeAudioData(arrayBuffer);
          await decodeCtx.close();
          onRecordingComplete(audioBuffer);
        } catch {
          setError("Failed to decode recorded audio. Please try again.");
        }
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start(100); // collect data every 100ms
      setIsRecording(true);
      setRecordingDuration(0);

      // Duration timer
      timerRef.current = setInterval(() => {
        setRecordingDuration((prev) => prev + 1);
      }, 1000);
    } catch {
      setError(
        "Microphone access denied. Please allow microphone access in your browser settings."
      );
    }
  }, [onRecordingComplete]);

  // Stop recording
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setIsRecording(false);
    setAnalyserNode(null);
  }, []);

  // Format duration as MM:SS
  const formatDuration = (seconds: number): string => {
    const m = Math.floor(seconds / 60)
      .toString()
      .padStart(2, "0");
    const s = (seconds % 60).toString().padStart(2, "0");
    return `${m}:${s}`;
  };

  return (
    <div className="flex flex-col gap-6">
      {/* Waveform visualization */}
      <div className="rounded-xl border border-border/50 bg-secondary/30 p-4">
        <WaveformVisualizer
          analyser={analyserNode}
          isRecording={isRecording}
          height={100}
        />
      </div>

      {/* Recording controls */}
      <div className="flex flex-col items-center gap-4">
        {isRecording && (
          <div className="flex items-center gap-3">
            <span className="relative flex h-3 w-3">
              <span className="absolute inline-flex h-full w-full rounded-full bg-destructive opacity-75 animate-pulse-ring" />
              <span className="relative inline-flex h-3 w-3 rounded-full bg-destructive" />
            </span>
            <span className="font-mono text-sm text-muted-foreground">
              Recording {formatDuration(recordingDuration)}
            </span>
          </div>
        )}

        <div className="flex items-center gap-3">
          {!isRecording ? (
            <Button
              onClick={startRecording}
              disabled={isProcessing}
              size="lg"
              className="gap-2 bg-primary text-primary-foreground hover:bg-primary/80"
            >
              {isProcessing ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                <Mic className="h-5 w-5" />
              )}
              {isProcessing ? "Analyzing..." : "Start Recording"}
            </Button>
          ) : (
            <Button
              onClick={stopRecording}
              size="lg"
              variant="destructive"
              className="gap-2"
            >
              <Square className="h-4 w-4" />
              Stop Recording
            </Button>
          )}
        </div>

        <p className="text-xs text-muted-foreground text-center max-w-xs">
          Click to record your voice. Speak naturally for 3-10 seconds for best results.
        </p>
      </div>

      {/* Error display */}
      {error && (
        <div className="rounded-lg border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          {error}
        </div>
      )}
    </div>
  );
}
