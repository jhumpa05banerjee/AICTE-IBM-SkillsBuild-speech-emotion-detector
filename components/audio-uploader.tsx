"use client";

import { useState, useRef, useCallback } from "react";
import { Upload, FileAudio, X, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { WaveformVisualizer } from "@/components/waveform-visualizer";
import { decodeAudioFile } from "@/lib/audio-analyzer";

/**
 * AudioUploader
 *
 * Allows users to upload an audio file (WAV, MP3, OGG, FLAC, etc.)
 * for emotion analysis. Displays the file waveform after upload
 * and passes the decoded AudioBuffer to the parent for processing.
 */

interface AudioUploaderProps {
  onFileLoaded: (audioBuffer: AudioBuffer) => void;
  isProcessing: boolean;
}

const ACCEPTED_TYPES = [
  "audio/wav",
  "audio/wave",
  "audio/x-wav",
  "audio/mp3",
  "audio/mpeg",
  "audio/ogg",
  "audio/flac",
  "audio/webm",
  "audio/aac",
  "audio/mp4",
];

export function AudioUploader({
  onFileLoaded,
  isProcessing,
}: AudioUploaderProps) {
  const [file, setFile] = useState<File | null>(null);
  const [audioBuffer, setAudioBuffer] = useState<AudioBuffer | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDecoding, setIsDecoding] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Process the selected/dropped file
  const processFile = useCallback(
    async (selectedFile: File) => {
      setError(null);
      setIsDecoding(true);

      // Validate file type
      if (
        !ACCEPTED_TYPES.includes(selectedFile.type) &&
        !selectedFile.name.match(/\.(wav|mp3|ogg|flac|webm|aac|m4a)$/i)
      ) {
        setError(
          "Unsupported file format. Please upload a WAV, MP3, OGG, FLAC, or WebM file."
        );
        setIsDecoding(false);
        return;
      }

      // Validate file size (max 50MB)
      if (selectedFile.size > 50 * 1024 * 1024) {
        setError("File is too large. Please upload a file under 50MB.");
        setIsDecoding(false);
        return;
      }

      try {
        const buffer = await decodeAudioFile(selectedFile);
        setFile(selectedFile);
        setAudioBuffer(buffer);
        onFileLoaded(buffer);
      } catch {
        setError("Failed to decode audio file. Please try a different format.");
      } finally {
        setIsDecoding(false);
      }
    },
    [onFileLoaded]
  );

  // Drag & Drop handlers
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile) {
        processFile(droppedFile);
      }
    },
    [processFile]
  );

  // File input change
  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const selectedFile = e.target.files?.[0];
      if (selectedFile) {
        processFile(selectedFile);
      }
    },
    [processFile]
  );

  // Clear file
  const clearFile = useCallback(() => {
    setFile(null);
    setAudioBuffer(null);
    setError(null);
    if (inputRef.current) {
      inputRef.current.value = "";
    }
  }, []);

  // Format file size
  const formatSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="flex flex-col gap-6">
      {/* Drop zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        aria-label="Upload audio file"
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
        }}
        className={`
          flex flex-col items-center justify-center gap-4 rounded-xl border-2 border-dashed
          p-8 transition-all cursor-pointer
          ${
            isDragging
              ? "border-primary bg-primary/10"
              : "border-border/50 bg-secondary/20 hover:border-primary/50 hover:bg-secondary/30"
          }
          ${isDecoding ? "pointer-events-none opacity-60" : ""}
        `}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".wav,.mp3,.ogg,.flac,.webm,.aac,.m4a"
          onChange={handleFileChange}
          className="hidden"
          aria-hidden="true"
        />

        {isDecoding ? (
          <>
            <Loader2 className="h-10 w-10 text-primary animate-spin" />
            <p className="text-sm text-muted-foreground">Decoding audio...</p>
          </>
        ) : (
          <>
            <div className="rounded-full bg-primary/10 p-3">
              <Upload className="h-6 w-6 text-primary" />
            </div>
            <div className="text-center">
              <p className="text-sm font-medium text-foreground">
                Drop audio file here or click to browse
              </p>
              <p className="text-xs text-muted-foreground mt-1">
                WAV, MP3, OGG, FLAC, WebM (max 50MB)
              </p>
            </div>
          </>
        )}
      </div>

      {/* File info & waveform */}
      {file && audioBuffer && (
        <div className="flex flex-col gap-4">
          {/* File info */}
          <div className="flex items-center gap-3 rounded-lg border border-border/50 bg-secondary/30 px-4 py-3">
            <FileAudio className="h-5 w-5 text-primary shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-foreground truncate">
                {file.name}
              </p>
              <p className="text-xs text-muted-foreground">
                {formatSize(file.size)} &middot;{" "}
                {audioBuffer.duration.toFixed(1)}s &middot;{" "}
                {audioBuffer.sampleRate}Hz
              </p>
            </div>
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={(e) => {
                e.stopPropagation();
                clearFile();
              }}
              disabled={isProcessing}
              aria-label="Remove file"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>

          {/* Waveform */}
          <div className="rounded-xl border border-border/50 bg-secondary/30 p-4">
            <WaveformVisualizer audioBuffer={audioBuffer} height={100} />
          </div>
        </div>
      )}

      {/* Error display */}
      {error && (
        <div className="rounded-lg border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          {error}
        </div>
      )}
    </div>
  );
}
