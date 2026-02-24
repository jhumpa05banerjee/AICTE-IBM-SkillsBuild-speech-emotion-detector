"use client";

import { useState, useCallback, useEffect } from "react";
import { AudioWaveform, Brain, Mic, Upload } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AudioRecorder } from "@/components/audio-recorder";
import { AudioUploader } from "@/components/audio-uploader";
import { EmotionResults } from "@/components/emotion-results";
import { extractFeatures } from "@/lib/audio-analyzer";
import {
  classifyEmotion,
  loadModel,
  type ClassificationResult,
} from "@/lib/emotion-classifier";

/**
 * SpeechEmotionDetector
 *
 * Main application component that orchestrates:
 *  1. Audio input (recording or file upload)
 *  2. Feature extraction (MFCC, chroma, mel)
 *  3. Emotion classification
 *  4. Results display
 */

export function SpeechEmotionDetector() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [analysisSource, setAnalysisSource] = useState<string | null>(null);

  // Attempt to load trained model on mount
  useEffect(() => {
    loadModel().then((loaded) => {
      if (loaded) {
        console.log("[v0] Using trained MLP model for classification");
      } else {
        console.log("[v0] No trained model found, using heuristic classifier");
      }
    });
  }, []);

  /**
   * Process audio buffer through the feature extraction + classification pipeline.
   * This mirrors the Python pipeline: extract_feature() -> scaler.transform() -> model.predict_proba()
   */
  const processAudio = useCallback(
    async (audioBuffer: AudioBuffer, source: string) => {
      setIsProcessing(true);
      setResult(null);
      setAnalysisSource(source);

      try {
        // Small delay for UI feedback
        await new Promise((resolve) => setTimeout(resolve, 300));

        // Step 1: Extract audio features (MFCC, Chroma, Mel + prosodic features)
        const features = extractFeatures(audioBuffer);

        // Debug: log key features so we can verify they vary across inputs
        console.log("[v0] Audio features:", {
          rmsEnergy: features.rmsEnergy.toFixed(5),
          spectralCentroid: features.spectralCentroid.toFixed(1),
          zcr: features.zeroCrossingRate.toFixed(5),
          speechRate: features.speechRate.toFixed(2),
          mfccMean: (features.mfcc.reduce((a, b) => a + b, 0) / features.mfcc.length).toFixed(3),
          mfccVar: (features.mfcc.reduce((a, b) => a + (b - features.mfcc.reduce((c, d) => c + d, 0) / features.mfcc.length) ** 2, 0) / features.mfcc.length).toFixed(3),
          melLow: features.mel.slice(0, 20).reduce((a, b) => a + b, 0).toFixed(4),
          melHigh: features.mel.slice(80).reduce((a, b) => a + b, 0).toFixed(4),
        });

        // Step 2: Classify emotion using the extracted features
        const classification = classifyEmotion(features);

        // Debug: log classification results
        console.log("[v0] Classification:", {
          mode: classification.mode,
          dominant: classification.dominantEmotion,
          confidence: (classification.confidence * 100).toFixed(1) + "%",
          all: classification.predictions.map(p => `${p.emotion}: ${(p.probability * 100).toFixed(1)}%`).join(", "),
        });

        // Step 3: Display results
        setResult(classification);
      } catch (err) {
        console.error("Analysis failed:", err);
      } finally {
        setIsProcessing(false);
      }
    },
    []
  );

  // Handle recording complete
  const handleRecordingComplete = useCallback(
    (audioBuffer: AudioBuffer) => {
      processAudio(audioBuffer, "microphone");
    },
    [processAudio]
  );

  // Handle file upload
  const handleFileLoaded = useCallback(
    (audioBuffer: AudioBuffer) => {
      processAudio(audioBuffer, "file");
    },
    [processAudio]
  );

  return (
    <div className="flex flex-col gap-8">
      {/* Header */}
      <header className="flex flex-col items-center gap-4 text-center">
        <div className="flex items-center gap-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary/15 border border-primary/20">
            <AudioWaveform className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-foreground text-balance">
              Speech Emotion Detection
            </h1>
          </div>
        </div>
        <p className="text-sm text-muted-foreground max-w-lg text-pretty leading-relaxed">
          Record your voice or upload an audio file to detect emotions using
          machine learning. The model analyzes MFCC, chroma, and mel spectrogram
          features to classify speech into emotional categories.
        </p>
      </header>

      {/* Input tabs */}
      <Tabs defaultValue="record" className="w-full">
        <TabsList className="w-full grid grid-cols-2 bg-secondary/50">
          <TabsTrigger value="record" className="gap-2">
            <Mic className="h-4 w-4" />
            <span className="hidden sm:inline">Record Audio</span>
            <span className="sm:hidden">Record</span>
          </TabsTrigger>
          <TabsTrigger value="upload" className="gap-2">
            <Upload className="h-4 w-4" />
            <span className="hidden sm:inline">Upload File</span>
            <span className="sm:hidden">Upload</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="record" className="mt-6">
          <AudioRecorder
            onRecordingComplete={handleRecordingComplete}
            isProcessing={isProcessing}
          />
        </TabsContent>

        <TabsContent value="upload" className="mt-6">
          <AudioUploader
            onFileLoaded={handleFileLoaded}
            isProcessing={isProcessing}
          />
        </TabsContent>
      </Tabs>

      {/* Processing indicator */}
      {isProcessing && (
        <div className="flex flex-col items-center gap-3 py-8">
          <div className="relative">
            <Brain className="h-8 w-8 text-primary animate-pulse" />
            <div className="absolute inset-0 rounded-full bg-primary/20 animate-ping" />
          </div>
          <div className="text-center">
            <p className="text-sm font-medium text-foreground">
              Analyzing speech patterns...
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              Extracting features and classifying emotions
            </p>
          </div>
        </div>
      )}

      {/* Results */}
      {result && !isProcessing && (
        <section aria-label="Emotion analysis results">
          <div className="flex items-center gap-2 mb-4">
            <Brain className="h-4 w-4 text-primary" />
            <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-widest">
              Analysis Results
              {analysisSource && (
                <span className="ml-2 text-xs normal-case font-normal">
                  (from {analysisSource})
                </span>
              )}
            </h2>
          </div>
          <EmotionResults result={result} />
        </section>
      )}

      {/* Feature info footer */}
      {!result && !isProcessing && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-4">
          {[
            {
              title: "MFCC",
              desc: "40 Mel-Frequency Cepstral Coefficients capturing vocal characteristics",
            },
            {
              title: "Chroma",
              desc: "12-dimensional pitch class profile for harmonic content",
            },
            {
              title: "Mel Spectrogram",
              desc: "128-band energy distribution across perceptual frequency scale",
            },
          ].map((feature) => (
            <div
              key={feature.title}
              className="rounded-lg border border-border/40 bg-secondary/20 p-4"
            >
              <h3 className="text-xs font-semibold uppercase tracking-wider text-primary mb-1">
                {feature.title}
              </h3>
              <p className="text-xs text-muted-foreground leading-relaxed">
                {feature.desc}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
