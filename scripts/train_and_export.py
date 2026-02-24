"""
Speech Emotion Detection - Model Training & Export Script

This script:
1. Trains an MLPClassifier on the RAVDESS audio dataset with data augmentation
2. Uses GridSearch/cross-validation for better hyperparameter selection
3. Exports the complete model (weights, biases, scaler, labels) as JSON
4. The JSON file is loaded by the Next.js web app for browser-side inference

Usage:
  python scripts/train_and_export.py

Requirements:
  pip install librosa soundfile scikit-learn numpy joblib

  RAVDESS dataset structure:
    assets/Actor_01/*.wav
    assets/Actor_02/*.wav
    ...

Output:
  - public/model.json  (for web app inference)
  - emotion_model.pkl  (sklearn model backup)
  - scaler.pkl         (sklearn scaler backup)
  - label_encoder.pkl  (sklearn encoder backup)
"""

import librosa
import soundfile
import os
import glob
import numpy as np
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')


# ======================================================
# STEP 1: Feature Extraction (matches web app's audio-analyzer.ts)
# ======================================================

def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    """
    Extract audio features.
    Returns: [40 MFCCs] + [12 chroma bins] + [128 mel bands] = 180 features
    """
    try:
        with soundfile.SoundFile(file_name) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate

            if len(X.shape) > 1:
                X = np.mean(X, axis=1)

            result = np.array([])

            if mfcc:
                mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
                mfccs = np.mean(mfccs.T, axis=0)
                result = np.hstack((result, mfccs))

            if chroma:
                stft = np.abs(librosa.stft(X))
                chroma_feat = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
                chroma_feat = np.mean(chroma_feat.T, axis=0)
                result = np.hstack((result, chroma_feat))

            if mel:
                mel_feat = librosa.feature.melspectrogram(y=X, sr=sample_rate)
                mel_feat = np.mean(mel_feat.T, axis=0)
                result = np.hstack((result, mel_feat))

        return result

    except Exception as e:
        print(f"  Warning: Could not process {file_name}: {e}")
        return None


def extract_feature_from_signal(X, sample_rate, mfcc=True, chroma=True, mel=True):
    """Extract features from a raw signal array (for augmented data)."""
    result = np.array([])

    if mfcc:
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
        mfccs = np.mean(mfccs.T, axis=0)
        result = np.hstack((result, mfccs))

    if chroma:
        stft = np.abs(librosa.stft(X))
        chroma_feat = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        chroma_feat = np.mean(chroma_feat.T, axis=0)
        result = np.hstack((result, chroma_feat))

    if mel:
        mel_feat = librosa.feature.melspectrogram(y=X, sr=sample_rate)
        mel_feat = np.mean(mel_feat.T, axis=0)
        result = np.hstack((result, mel_feat))

    return result


# ======================================================
# STEP 2: Data Augmentation
# ======================================================

def augment_audio(X, sr):
    """
    Generate augmented versions of an audio signal.
    Returns list of (augmented_signal, augmentation_name) tuples.
    """
    augmented = []

    # Noise injection
    noise = np.random.randn(len(X)) * 0.005
    augmented.append((X + noise, "noise"))

    # Time stretch (slight speed up and slow down)
    try:
        stretched_fast = librosa.effects.time_stretch(X, rate=1.1)
        augmented.append((stretched_fast, "stretch_fast"))

        stretched_slow = librosa.effects.time_stretch(X, rate=0.9)
        augmented.append((stretched_slow, "stretch_slow"))
    except Exception:
        pass

    # Pitch shift
    try:
        pitched_up = librosa.effects.pitch_shift(X, sr=sr, n_steps=1.5)
        augmented.append((pitched_up, "pitch_up"))

        pitched_down = librosa.effects.pitch_shift(X, sr=sr, n_steps=-1.5)
        augmented.append((pitched_down, "pitch_down"))
    except Exception:
        pass

    return augmented


# ======================================================
# STEP 3: Emotion Labels (RAVDESS coding)
# ======================================================

emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

observed_emotions = ['calm', 'happy', 'fearful', 'disgust']


# ======================================================
# STEP 4: Load Dataset with Augmentation
# ======================================================

def load_data(test_size=0.20, augment=True):
    """Load RAVDESS, extract features, optionally augment, split train/test."""
    x, y = [], []
    files_processed = 0
    files_skipped = 0

    all_files = glob.glob(os.path.join("assets", "Actor_*", "*.wav"))
    print(f"  Found {len(all_files)} total WAV files")

    for file in all_files:
        file_name = os.path.basename(file)
        parts = file_name.split("-")
        if len(parts) < 3:
            files_skipped += 1
            continue

        emotion = emotions.get(parts[2])
        if emotion not in observed_emotions:
            files_skipped += 1
            continue

        # Original features
        feature = extract_feature(file)
        if feature is not None:
            x.append(feature)
            y.append(emotion)
            files_processed += 1

            # Augmented features
            if augment:
                try:
                    with soundfile.SoundFile(file) as sf:
                        signal = sf.read(dtype="float32")
                        sr = sf.samplerate
                        if len(signal.shape) > 1:
                            signal = np.mean(signal, axis=1)

                    for aug_signal, aug_name in augment_audio(signal, sr):
                        aug_feature = extract_feature_from_signal(aug_signal, sr)
                        if aug_feature is not None and len(aug_feature) == len(feature):
                            x.append(aug_feature)
                            y.append(emotion)
                except Exception:
                    pass

    x = np.array(x)
    y = np.array(y)

    print(f"  Files processed: {files_processed}")
    print(f"  Files skipped (wrong emotion): {files_skipped}")
    print(f"  Total samples (with augmentation): {len(x)}")
    print(f"  Emotion distribution:")
    for e in observed_emotions:
        count = np.sum(y == e)
        print(f"    {e}: {count} ({count/len(y)*100:.1f}%)")

    return train_test_split(
        x, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )


# ======================================================
# STEP 5: Train Model
# ======================================================

print("=" * 60)
print("Speech Emotion Detection - Training Pipeline")
print("=" * 60)
print()

print("[1/5] Loading and extracting features from RAVDESS dataset...")
x_train, x_test, y_train, y_test = load_data(augment=True)
print(f"\n  Train samples: {x_train.shape[0]}")
print(f"  Test samples:  {x_test.shape[0]}")
print(f"  Feature dims:  {x_train.shape[1]}")
print()

print("[2/5] Encoding labels...")
encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_test_enc = encoder.transform(y_test)
print(f"  Classes: {list(encoder.classes_)}")
print()

print("[3/5] Scaling features (StandardScaler)...")
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
print(f"  Mean range: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
print(f"  Scale range: [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")
print()

print("[4/5] Training MLP Classifier with hyperparameter search...")

# Define parameter grid for search
param_grid = {
    'hidden_layer_sizes': [(256, 128), (256, 128, 64), (512, 256)],
    'alpha': [0.0001, 0.001],
    'learning_rate': ['adaptive'],
    'batch_size': [32, 64],
}

base_model = MLPClassifier(
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.15,
)

# Use GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    base_model,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    refit=True
)

grid_search.fit(x_train_scaled, y_train_enc)

model = grid_search.best_estimator_

print(f"\n  Best parameters: {grid_search.best_params_}")
print(f"  Best CV accuracy: {grid_search.best_score_ * 100:.2f}%")

# Evaluate on test set
y_pred = model.predict(x_test_scaled)
accuracy = accuracy_score(y_test_enc, y_pred)
print(f"  Test accuracy: {accuracy * 100:.2f}%")
print()
print("  Classification Report:")
print(classification_report(y_test_enc, y_pred, target_names=encoder.classes_))
print()


# ======================================================
# STEP 6: Export Model for Web Inference
# ======================================================

print("[5/5] Exporting model for web inference...")

# Save sklearn objects
joblib.dump(model, "emotion_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "label_encoder.pkl")
print("  Saved: emotion_model.pkl, scaler.pkl, label_encoder.pkl")

# Build JSON export for the web app's MLP forward pass
model_export = {
    "scaler": {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    },
    "layers": [],
    "labels": encoder.classes_.tolist(),
    "activation": "relu",
    "metadata": {
        "accuracy": round(float(accuracy), 4),
        "n_features": int(x_train.shape[1]),
        "n_samples_train": int(x_train.shape[0]),
        "n_samples_test": int(x_test.shape[0]),
        "hidden_layer_sizes": list(model.hidden_layer_sizes),
        "best_params": grid_search.best_params_,
        "cv_accuracy": round(float(grid_search.best_score_), 4),
        "observed_emotions": observed_emotions
    }
}

# Export layers (weights + biases)
for i, (w, b) in enumerate(zip(model.coefs_, model.intercepts_)):
    model_export["layers"].append({
        "weights": w.tolist(),
        "biases": b.tolist()
    })
    print(f"  Layer {i}: weights {w.shape}, biases {b.shape}")

# Save JSON
os.makedirs("public", exist_ok=True)
output_path = os.path.join("public", "model.json")
with open(output_path, "w") as f:
    json.dump(model_export, f)

file_size_kb = os.path.getsize(output_path) / 1024
print(f"  Saved: {output_path} ({file_size_kb:.1f} KB)")

print()
print("=" * 60)
print("Training complete!")
print(f"  Test Accuracy:  {accuracy * 100:.2f}%")
print(f"  CV Accuracy:    {grid_search.best_score_ * 100:.2f}%")
print(f"  Model exported: {output_path}")
print()
print("To use in the web app:")
print("  1. Copy public/model.json to your Next.js project's public/ folder")
print("  2. The app will automatically detect and load the trained model")
print("  3. If model.json is missing, the app uses a heuristic classifier")
print("=" * 60)
