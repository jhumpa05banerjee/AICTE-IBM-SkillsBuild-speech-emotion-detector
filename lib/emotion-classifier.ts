/**
 * Emotion Classifier
 *
 * Two-mode classifier for speech emotion detection:
 *
 * MODE 1 — Trained Model (when public/model.json exists):
 *   Loads the exported sklearn MLPClassifier weights and runs a proper
 *   neural network forward pass: StandardScaler -> ReLU MLP -> Softmax.
 *   This produces the same results as the Python model.
 *
 * MODE 2 — Heuristic Fallback (when no model.json):
 *   Uses a multi-feature scoring approach based on acoustic correlates
 *   from speech emotion research. Each emotion has scoring functions
 *   that evaluate how well the input matches expected acoustic patterns.
 *   This avoids the prototype-distance problem where one emotion always
 *   dominates because mid-range feature values favor a single prototype.
 *
 * Observed emotions (matching train.py): calm, happy, fearful, disgust
 */

import type { AudioFeatures } from "./audio-analyzer";

// ======================================================
// Types
// ======================================================

export interface EmotionPrediction {
  emotion: string;
  probability: number;
  color: string;
}

export interface ClassificationResult {
  predictions: EmotionPrediction[];
  dominantEmotion: string;
  confidence: number;
  mode: "model" | "heuristic";
}

// ======================================================
// Emotion configuration
// ======================================================

const EMOTION_CONFIG: Record<string, { color: string; label: string }> = {
  calm: { color: "#60a5fa", label: "Calm" },
  happy: { color: "#facc15", label: "Happy" },
  fearful: { color: "#c084fc", label: "Fearful" },
  disgust: { color: "#4ade80", label: "Disgust" },
};

export const EMOTIONS = Object.keys(EMOTION_CONFIG);

export function getEmotionColor(emotion: string): string {
  return EMOTION_CONFIG[emotion]?.color ?? "#94a3b8";
}

export function getEmotionLabel(emotion: string): string {
  return EMOTION_CONFIG[emotion]?.label ?? emotion;
}

// ======================================================
// Model data types (matches JSON export from train_and_export.py)
// ======================================================

interface ModelLayer {
  weights: number[][];
  biases: number[];
}

interface ModelData {
  scaler: {
    mean: number[];
    scale: number[];
  };
  layers: ModelLayer[];
  labels: string[];
  activation: string;
  metadata: {
    accuracy: number;
    n_features: number;
  };
}

// ======================================================
// Model loading (singleton)
// ======================================================

let modelData: ModelData | null = null;
let modelLoadAttempted = false;

export async function loadModel(): Promise<boolean> {
  if (modelData) return true;
  if (modelLoadAttempted) return false;

  modelLoadAttempted = true;

  try {
    const response = await fetch("/model.json");
    if (!response.ok) {
      console.warn("No trained model found at /model.json - using heuristic fallback");
      return false;
    }

    const data = await response.json();

    if (!data.scaler || !data.layers || !data.labels) {
      console.warn("Invalid model.json structure - using heuristic fallback");
      return false;
    }

    modelData = data as ModelData;
    console.log(
      `Trained model loaded: ${data.labels.length} classes, ` +
      `${data.layers.length} layers, accuracy: ${(data.metadata?.accuracy * 100).toFixed(1)}%`
    );
    return true;
  } catch {
    console.warn("Failed to load model.json - using heuristic fallback");
    return false;
  }
}

export function isModelLoaded(): boolean {
  return modelData !== null;
}

// ======================================================
// MODE 1: MLP Forward Pass (trained model)
// ======================================================

function classifyWithModel(features: AudioFeatures): ClassificationResult {
  const model = modelData!;

  // Build 180-d feature vector: MFCC(40) + chroma(12) + mel(128)
  const featureVector: number[] = [
    ...features.mfcc,
    ...features.chroma,
    ...features.mel,
  ];

  // Pad or truncate to match model input size
  const expectedFeatures = model.scaler.mean.length;
  while (featureVector.length < expectedFeatures) {
    featureVector.push(0);
  }
  if (featureVector.length > expectedFeatures) {
    featureVector.length = expectedFeatures;
  }

  // StandardScaler transform
  let x = featureVector.map(
    (v, i) => (v - model.scaler.mean[i]) / (model.scaler.scale[i] || 1)
  );

  // Forward pass through hidden layers with ReLU
  for (let l = 0; l < model.layers.length - 1; l++) {
    const layer = model.layers[l];
    const output = new Array(layer.biases.length).fill(0);
    for (let j = 0; j < layer.biases.length; j++) {
      let sum = layer.biases[j];
      for (let i = 0; i < x.length; i++) {
        sum += x[i] * layer.weights[i][j];
      }
      output[j] = Math.max(0, sum); // ReLU
    }
    x = output;
  }

  // Output layer (linear)
  const lastLayer = model.layers[model.layers.length - 1];
  const logits = new Array(lastLayer.biases.length).fill(0);
  for (let j = 0; j < lastLayer.biases.length; j++) {
    let sum = lastLayer.biases[j];
    for (let i = 0; i < x.length; i++) {
      sum += x[i] * lastLayer.weights[i][j];
    }
    logits[j] = sum;
  }

  // Softmax
  const probs = softmaxArray(logits);

  const predictions: EmotionPrediction[] = model.labels.map((label, i) => ({
    emotion: label,
    probability: probs[i],
    color: getEmotionColor(label),
  })).sort((a, b) => b.probability - a.probability);

  return {
    predictions,
    dominantEmotion: predictions[0].emotion,
    confidence: predictions[0].probability,
    mode: "model",
  };
}

// ======================================================
// MODE 2: Heuristic Fallback (rule-based scoring)
// ======================================================

/**
 * Rule-based emotion classifier using acoustic correlates from speech research.
 *
 * Instead of prototype-distance (which caused the "always fearful" bug because
 * mid-range features had minimum distance to fearful), this uses independent
 * scoring functions per emotion. Each function evaluates multiple acoustic
 * features using Gaussian membership functions centered on empirically
 * determined values.
 *
 * The key fix: each emotion has its OWN scoring logic that responds to
 * different feature combinations. This prevents any single emotion from
 * dominating across all input types.
 *
 * References:
 * - Scherer (2003) "Vocal communication of emotion"
 * - Juslin & Laukka (2003) "Communication of emotions in vocal expression"
 * - Banse & Scherer (1996) "Acoustic profiles in vocal emotion expression"
 */

function classifyWithHeuristic(features: AudioFeatures): ClassificationResult {
  // Step 1: Extract and normalize acoustic indicators
  const energy = features.rmsEnergy;
  const centroid = features.spectralCentroid;
  const zcr = features.zeroCrossingRate;
  const speechRate = features.speechRate;

  // Compute derived features from the raw MFCC/chroma/mel vectors
  const mfccMean = mean(features.mfcc);
  const mfccVar = variance(features.mfcc);
  const mfccRange = range(features.mfcc);

  const chromaMax = Math.max(...features.chroma);
  const chromaVar = variance(features.chroma);
  const chromaFlat = flatness(features.chroma);

  const melLow = mean(features.mel.slice(0, 20));    // low freq energy
  const melMid = mean(features.mel.slice(20, 80));    // mid freq energy
  const melHigh = mean(features.mel.slice(80));       // high freq energy
  const melTotal = melLow + melMid + melHigh + 1e-10;
  const melLowRatio = melLow / melTotal;
  const melHighRatio = melHigh / melTotal;
  const melSkew = (melHigh - melLow) / melTotal;

  // Step 2: Score each emotion using Gaussian membership functions
  // Each emotion is scored based on how well the input matches
  // its expected acoustic profile across INDEPENDENT feature axes

  const scores: Record<string, number> = {};

  // ---- CALM ----
  // Low energy, low spectral centroid, slow rate, low ZCR, high low-freq ratio
  // Calm speech is soft, slow, regular, with energy concentrated in lower frequencies
  scores.calm =
    gaussian(energy, 0.02, 0.025) * 2.5 +      // soft voice
    gaussian(centroid, 1200, 600) * 2.0 +        // low brightness
    gaussian(zcr, 0.03, 0.02) * 1.5 +            // smooth signal
    gaussian(speechRate, 3.0, 2.0) * 1.5 +        // slow speaking
    gaussian(melLowRatio, 0.55, 0.15) * 2.0 +    // energy in low freq
    gaussian(mfccVar, 100, 80) * 1.0 +            // low spectral variation
    gaussian(chromaFlat, 0.7, 0.2) * 0.8 +        // regular harmonics
    gaussian(melSkew, -0.2, 0.2) * 1.0;           // bass-heavy

  // ---- HAPPY ----
  // High energy, high centroid, fast rate, moderate-high ZCR, wide MFCC range
  // Happy speech is loud, bright, fast, with wide spectral variation
  scores.happy =
    gaussian(energy, 0.12, 0.08) * 2.5 +          // loud voice
    gaussian(centroid, 3000, 1000) * 2.0 +         // bright/high-pitched
    gaussian(zcr, 0.07, 0.03) * 1.2 +             // energetic signal
    gaussian(speechRate, 8.0, 3.0) * 1.8 +         // fast speaking
    gaussian(melHighRatio, 0.2, 0.1) * 1.5 +       // energy in high freq
    gaussian(mfccRange, 60, 30) * 1.5 +            // wide spectral range
    gaussian(chromaMax, 0.8, 0.2) * 1.0 +          // strong pitch
    gaussian(melSkew, 0.1, 0.15) * 0.8;            // treble presence

  // ---- FEARFUL ----
  // Moderate energy, high centroid, moderate-fast rate, HIGH ZCR, high MFCC variance
  // Fearful speech is tense, breathy, with irregular patterns and noise
  scores.fearful =
    gaussian(energy, 0.05, 0.04) * 1.5 +           // moderate-soft voice
    gaussian(centroid, 2800, 800) * 1.8 +           // high brightness (tension)
    gaussian(zcr, 0.09, 0.03) * 2.5 +              // high noise (breathiness)
    gaussian(speechRate, 6.0, 2.5) * 1.2 +          // moderate-fast rate
    gaussian(mfccVar, 400, 200) * 2.0 +             // high spectral irregularity
    gaussian(chromaVar, 0.06, 0.03) * 1.5 +         // irregular pitch
    gaussian(melHighRatio, 0.25, 0.1) * 1.5 +       // breathy high-freq
    gaussian(melSkew, 0.2, 0.15) * 0.8;             // treble-heavy

  // ---- DISGUST ----
  // Low-moderate energy, low centroid, slow rate, moderate ZCR, guttural
  // Disgust speech is low-pitched, rough, slow, with energy in low-mid frequencies
  scores.disgust =
    gaussian(energy, 0.04, 0.035) * 1.8 +          // low-moderate energy
    gaussian(centroid, 1500, 500) * 2.2 +           // low brightness (guttural)
    gaussian(zcr, 0.05, 0.025) * 1.8 +             // moderate roughness
    gaussian(speechRate, 3.5, 2.0) * 1.5 +          // slow speaking
    gaussian(melLowRatio, 0.45, 0.15) * 1.5 +      // low-freq energy
    gaussian(mfccMean, -5, 15) * 1.0 +              // negative MFCC mean (dark timbre)
    gaussian(chromaFlat, 0.5, 0.2) * 1.0 +          // moderate harmonic regularity
    gaussian(melSkew, -0.1, 0.2) * 0.8;             // slight bass emphasis

  // Step 3: Add small base score to prevent zero probabilities
  for (const emotion of EMOTIONS) {
    scores[emotion] = (scores[emotion] || 0) + 0.5;
  }

  // Step 4: Convert scores to probabilities via softmax with temperature
  // Higher temperature = more spread out probabilities
  const temperature = 0.8;
  const logits: Record<string, number> = {};
  for (const emotion of EMOTIONS) {
    logits[emotion] = scores[emotion] / temperature;
  }

  const probs = softmaxObj(logits);

  // Step 5: Build sorted predictions
  const predictions: EmotionPrediction[] = EMOTIONS.map((emotion) => ({
    emotion,
    probability: probs[emotion],
    color: getEmotionColor(emotion),
  })).sort((a, b) => b.probability - a.probability);

  return {
    predictions,
    dominantEmotion: predictions[0].emotion,
    confidence: predictions[0].probability,
    mode: "heuristic",
  };
}

// ======================================================
// Scoring helper functions
// ======================================================

/**
 * Gaussian membership function.
 * Returns a value in (0, 1] indicating how close `x` is to `center`.
 * `sigma` controls the width: larger sigma = more tolerant.
 */
function gaussian(x: number, center: number, sigma: number): number {
  const diff = x - center;
  return Math.exp(-(diff * diff) / (2 * sigma * sigma));
}

/** Arithmetic mean of an array */
function mean(arr: number[]): number {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

/** Variance of an array */
function variance(arr: number[]): number {
  if (arr.length === 0) return 0;
  const m = mean(arr);
  return arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length;
}

/** Range (max - min) of an array */
function range(arr: number[]): number {
  if (arr.length === 0) return 0;
  return Math.max(...arr) - Math.min(...arr);
}

/** Spectral flatness: geometric mean / arithmetic mean. High = noise-like, low = tonal */
function flatness(arr: number[]): number {
  if (arr.length === 0) return 0;
  const absArr = arr.map((v) => Math.abs(v) + 1e-10);
  const logSum = absArr.reduce((a, b) => a + Math.log(b), 0);
  const geometricMean = Math.exp(logSum / absArr.length);
  const arithmeticMean = absArr.reduce((a, b) => a + b, 0) / absArr.length;
  return arithmeticMean > 0 ? geometricMean / arithmeticMean : 0;
}

// ======================================================
// Main classify function
// ======================================================

export function classifyEmotion(features: AudioFeatures): ClassificationResult {
  if (modelData) {
    return classifyWithModel(features);
  }
  return classifyWithHeuristic(features);
}

// ======================================================
// Math utilities
// ======================================================

function softmaxArray(logits: number[]): number[] {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - maxLogit));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sumExp);
}

function softmaxObj(logits: Record<string, number>): Record<string, number> {
  const keys = Object.keys(logits);
  const maxLogit = Math.max(...keys.map((k) => logits[k]));

  const exps: Record<string, number> = {};
  let sumExp = 0;
  for (const k of keys) {
    exps[k] = Math.exp(logits[k] - maxLogit);
    sumExp += exps[k];
  }

  const result: Record<string, number> = {};
  for (const k of keys) {
    result[k] = exps[k] / sumExp;
  }
  return result;
}
