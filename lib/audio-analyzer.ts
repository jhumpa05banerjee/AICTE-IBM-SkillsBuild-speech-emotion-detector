/**
 * Audio Feature Extraction Engine
 *
 * Extracts MFCC-like features from audio using the Web Audio API.
 * This mirrors the Python librosa feature extraction pipeline used in train.py:
 *   - MFCC (Mel-Frequency Cepstral Coefficients)
 *   - Chroma features (pitch class profile)
 *   - Mel spectrogram energy
 *
 * Since we cannot run librosa in the browser, we use the Web Audio API's
 * AnalyserNode and manual DCT / filterbank computations.
 */

// ---- helpers ----

/** Convert Hz to Mel scale */
function hzToMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}

/** Convert Mel back to Hz */
function melToHz(mel: number): number {
  return 700 * (Math.pow(10, mel / 2595) - 1);
}

/**
 * Build a triangular Mel filterbank matrix.
 * Returns an array of `numFilters` arrays, each of length `fftSize/2+1`.
 */
function melFilterbank(
  numFilters: number,
  fftSize: number,
  sampleRate: number
): number[][] {
  const numBins = fftSize / 2 + 1;
  const lowMel = hzToMel(0);
  const highMel = hzToMel(sampleRate / 2);

  // Equally spaced points in Mel space
  const melPoints: number[] = [];
  for (let i = 0; i <= numFilters + 1; i++) {
    melPoints.push(lowMel + (i * (highMel - lowMel)) / (numFilters + 1));
  }
  const binPoints = melPoints.map((m) =>
    Math.floor(((fftSize + 1) * melToHz(m)) / sampleRate)
  );

  const filters: number[][] = [];
  for (let i = 0; i < numFilters; i++) {
    const row = new Array(numBins).fill(0);
    for (let j = binPoints[i]; j < binPoints[i + 1]; j++) {
      row[j] = (j - binPoints[i]) / (binPoints[i + 1] - binPoints[i]);
    }
    for (let j = binPoints[i + 1]; j < binPoints[i + 2]; j++) {
      row[j] = (binPoints[i + 2] - j) / (binPoints[i + 2] - binPoints[i + 1]);
    }
    filters.push(row);
  }
  return filters;
}

/**
 * Compute a Type-II DCT (used for MFCC).
 * Takes log-mel energies of length N, returns `numCoeffs` cepstral coefficients.
 */
function dctII(input: number[], numCoeffs: number): number[] {
  const N = input.length;
  const result: number[] = [];
  for (let k = 0; k < numCoeffs; k++) {
    let sum = 0;
    for (let n = 0; n < N; n++) {
      sum += input[n] * Math.cos((Math.PI * k * (2 * n + 1)) / (2 * N));
    }
    result.push(sum);
  }
  return result;
}

// ---- public types ----

export interface AudioFeatures {
  /** Mean MFCC vector (length 40) */
  mfcc: number[];
  /** Mean chroma vector (length 12) */
  chroma: number[];
  /** Mean mel energy vector (length 128) */
  mel: number[];
  /** RMS energy of the signal */
  rmsEnergy: number;
  /** Spectral centroid (brightness) */
  spectralCentroid: number;
  /** Zero-crossing rate */
  zeroCrossingRate: number;
  /** Speech rate proxy (number of energy peaks per second) */
  speechRate: number;
  /** Raw audio samples (mono, float32) */
  rawSamples: Float32Array;
  /** Sample rate */
  sampleRate: number;
}

// ---- main extraction function ----

/**
 * Extract audio features from a decoded AudioBuffer.
 * Mirrors the Python `extract_feature(file_name, mfcc=True, chroma=True, mel=True)`.
 */
export function extractFeatures(audioBuffer: AudioBuffer): AudioFeatures {
  // Step 1: Convert to mono
  const samples =
    audioBuffer.numberOfChannels === 1
      ? audioBuffer.getChannelData(0)
      : mixToMono(audioBuffer);
  const sr = audioBuffer.sampleRate;

  // Step 2: STFT parameters (match librosa defaults)
  const fftSize = 2048;
  const hopLength = 512;
  const numFrames = Math.floor((samples.length - fftSize) / hopLength) + 1;

  // Step 3: Compute power spectrum for each frame
  const powerSpectra: number[][] = [];
  for (let i = 0; i < numFrames; i++) {
    const start = i * hopLength;
    const frame = samples.slice(start, start + fftSize);
    const spectrum = computePowerSpectrum(frame, fftSize);
    powerSpectra.push(spectrum);
  }

  // Step 4: Mel filterbank
  const numMelFilters = 128;
  const filters = melFilterbank(numMelFilters, fftSize, sr);

  // Step 5: Apply filterbank to each frame -> mel spectrogram
  const melSpectrogram: number[][] = powerSpectra.map((spectrum) =>
    filters.map((filter) => {
      let energy = 0;
      for (let j = 0; j < filter.length && j < spectrum.length; j++) {
        energy += filter[j] * spectrum[j];
      }
      return Math.max(energy, 1e-10); // prevent log(0)
    })
  );

  // Step 6: MFCC — DCT of log mel energies
  const numMfcc = 40;
  const mfccFrames: number[][] = melSpectrogram.map((melEnergies) => {
    const logMel = melEnergies.map((e) => Math.log(e));
    return dctII(logMel, numMfcc);
  });

  // Step 7: Mean MFCC across frames
  const mfcc = meanAcrossFrames(mfccFrames, numMfcc);

  // Step 8: Chroma features (simplified pitch class profile)
  const numChroma = 12;
  const chromaFrames: number[][] = powerSpectra.map((spectrum) => {
    const chroma = new Array(numChroma).fill(0);
    for (let bin = 1; bin < spectrum.length; bin++) {
      const freq = (bin * sr) / fftSize;
      if (freq > 0) {
        const pitchClass =
          Math.round(12 * Math.log2(freq / 440) + 69) % 12;
        const idx = ((pitchClass % 12) + 12) % 12;
        chroma[idx] += spectrum[bin];
      }
    }
    // Normalize
    const maxVal = Math.max(...chroma, 1e-10);
    return chroma.map((c) => c / maxVal);
  });
  const chroma = meanAcrossFrames(chromaFrames, numChroma);

  // Step 9: Mean mel energy
  const mel = meanAcrossFrames(melSpectrogram, numMelFilters);

  // Step 10: Additional features for emotion analysis
  const rmsEnergy = computeRMS(samples);
  const spectralCentroid = computeSpectralCentroid(powerSpectra, sr, fftSize);
  const zeroCrossingRate = computeZCR(samples);
  const speechRate = estimateSpeechRate(samples, sr);

  return {
    mfcc,
    chroma,
    mel,
    rmsEnergy,
    spectralCentroid,
    zeroCrossingRate,
    speechRate,
    rawSamples: new Float32Array(samples),
    sampleRate: sr,
  };
}

// ---- utility functions ----

function mixToMono(buffer: AudioBuffer): Float32Array {
  const length = buffer.length;
  const mixed = new Float32Array(length);
  const numChannels = buffer.numberOfChannels;
  for (let ch = 0; ch < numChannels; ch++) {
    const channelData = buffer.getChannelData(ch);
    for (let i = 0; i < length; i++) {
      mixed[i] += channelData[i] / numChannels;
    }
  }
  return mixed;
}

function computePowerSpectrum(
  frame: Float32Array,
  fftSize: number
): number[] {
  // Apply Hann window
  const windowed = new Float32Array(fftSize);
  for (let i = 0; i < fftSize; i++) {
    const w = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (fftSize - 1)));
    windowed[i] = i < frame.length ? frame[i] * w : 0;
  }

  // Simple DFT for the positive frequencies (fftSize/2+1 bins)
  // For performance we use a Cooley-Tukey FFT
  const { real, imag } = fft(windowed);
  const numBins = fftSize / 2 + 1;
  const power: number[] = [];
  for (let i = 0; i < numBins; i++) {
    power.push(real[i] * real[i] + imag[i] * imag[i]);
  }
  return power;
}

/** Radix-2 Cooley-Tukey FFT (in-place, iterative) */
function fft(input: Float32Array): { real: number[]; imag: number[] } {
  const N = input.length;
  const real = Array.from(input) as number[];
  const imag = new Array(N).fill(0) as number[];

  // Bit-reversal permutation
  let j = 0;
  for (let i = 0; i < N; i++) {
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
    let m = N >> 1;
    while (m >= 1 && j >= m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }

  // FFT butterfly
  for (let size = 2; size <= N; size *= 2) {
    const halfSize = size / 2;
    const step = (2 * Math.PI) / size;
    for (let i = 0; i < N; i += size) {
      for (let k = 0; k < halfSize; k++) {
        const angle = -step * k;
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        const tReal = cos * real[i + k + halfSize] - sin * imag[i + k + halfSize];
        const tImag = sin * real[i + k + halfSize] + cos * imag[i + k + halfSize];
        real[i + k + halfSize] = real[i + k] - tReal;
        imag[i + k + halfSize] = imag[i + k] - tImag;
        real[i + k] += tReal;
        imag[i + k] += tImag;
      }
    }
  }

  return { real, imag };
}

function meanAcrossFrames(frames: number[][], dim: number): number[] {
  const result = new Array(dim).fill(0);
  if (frames.length === 0) return result;
  for (const frame of frames) {
    for (let i = 0; i < dim; i++) {
      result[i] += frame[i] || 0;
    }
  }
  return result.map((v) => v / frames.length);
}

function computeRMS(samples: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) {
    sum += samples[i] * samples[i];
  }
  return Math.sqrt(sum / samples.length);
}

function computeSpectralCentroid(
  powerSpectra: number[][],
  sr: number,
  fftSize: number
): number {
  let totalCentroid = 0;
  for (const spectrum of powerSpectra) {
    let weightedSum = 0;
    let totalEnergy = 0;
    for (let i = 0; i < spectrum.length; i++) {
      const freq = (i * sr) / fftSize;
      weightedSum += freq * spectrum[i];
      totalEnergy += spectrum[i];
    }
    totalCentroid += totalEnergy > 0 ? weightedSum / totalEnergy : 0;
  }
  return powerSpectra.length > 0 ? totalCentroid / powerSpectra.length : 0;
}

function computeZCR(samples: Float32Array): number {
  let crossings = 0;
  for (let i = 1; i < samples.length; i++) {
    if ((samples[i] >= 0 && samples[i - 1] < 0) ||
        (samples[i] < 0 && samples[i - 1] >= 0)) {
      crossings++;
    }
  }
  return crossings / samples.length;
}

function estimateSpeechRate(samples: Float32Array, sr: number): number {
  // Estimate speech rate by counting energy peaks (syllable-like)
  const frameSize = Math.floor(sr * 0.025); // 25ms frames
  const hopSize = Math.floor(sr * 0.01); // 10ms hop
  const energies: number[] = [];
  for (let i = 0; i + frameSize < samples.length; i += hopSize) {
    let e = 0;
    for (let j = 0; j < frameSize; j++) {
      e += samples[i + j] * samples[i + j];
    }
    energies.push(e / frameSize);
  }

  // Count peaks above mean energy
  const meanEnergy = energies.reduce((a, b) => a + b, 0) / energies.length;
  let peaks = 0;
  for (let i = 1; i < energies.length - 1; i++) {
    if (energies[i] > meanEnergy && energies[i] > energies[i - 1] && energies[i] > energies[i + 1]) {
      peaks++;
    }
  }

  const durationSec = samples.length / sr;
  return durationSec > 0 ? peaks / durationSec : 0;
}

/**
 * Decode an audio file (Blob or File) into an AudioBuffer via OfflineAudioContext.
 */
export async function decodeAudioFile(file: Blob): Promise<AudioBuffer> {
  const arrayBuffer = await file.arrayBuffer();
  const audioContext = new AudioContext();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
  await audioContext.close();
  return audioBuffer;
}
