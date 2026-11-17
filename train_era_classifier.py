"""
JavenEQ Era Classifier — Training Script (IMPROVED)
Better file finding, case-insensitive matching, detailed debugging.

Usage:
    python train_era_classifier.py
"""

import os
import json
import numpy as np
import librosa
from scipy import signal
import pickle
from sklearn.ensemble import RandomForestClassifier
import glob


class AudioFeatureExtractor:
    """Extracts DSP features from audio for era classification."""
    
    @staticmethod
    def extract(audio_path: str) -> dict:
        """Extract features from an audio file."""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            
            # Compute STFT
            stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
            avg = np.mean(stft, axis=1)
            
            # Helper: energy in frequency band
            def band_energy(lo, hi):
                mask = (freqs >= lo) & (freqs < hi)
                return float(np.mean(avg[mask])) if mask.any() else 0.0
            
            # Extract features
            rolloff = float(librosa.feature.spectral_rolloff(
                y=y, sr=sr, roll_percent=0.85).mean())
            
            high_e = band_energy(8000, 20000)
            total = float(np.mean(avg)) + 1e-8
            high_e_ratio = high_e / total
            
            highmid_e = band_energy(4000, 8000) / total
            
            frame_rms = [np.sqrt(np.mean(y[i:i+4096]**2))
                        for i in range(0, len(y)-4096, 2048)]
            dynamic_range = (np.percentile(frame_rms, 95) /
                           (np.percentile(frame_rms, 5) + 1e-8))
            
            flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
            centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
            
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=3)
            mfcc_means = [float(np.mean(mfcc[i])) for i in range(3)]
            
            bass_e = band_energy(80, 250) / total
            subbass_e = band_energy(20, 80) / total
            
            return {
                "rolloff": rolloff,
                "high_e_ratio": high_e_ratio,
                "highmid_e": highmid_e,
                "dynamic_range": dynamic_range,
                "flatness": flatness,
                "centroid": centroid,
                "zcr": zcr,
                "mfcc_0": mfcc_means[0],
                "mfcc_1": mfcc_means[1],
                "mfcc_2": mfcc_means[2],
                "bass_e": bass_e,
                "subbass_e": subbass_e,
            }
        except Exception as e:
            print(f"  [Error] Failed to extract features: {e}")
            return None


def find_audio_file(filename: str, search_root: str = ".") -> str:
    """
    Intelligently find audio file.
    Tries multiple strategies:
    1. Exact path match
    2. Case-insensitive filename match
    3. Recursive search
    """
    
    # Strategy 1: Exact match
    if os.path.exists(filename):
        return filename
    
    # Strategy 2: Look in data/songs/ if not already specified
    if not filename.startswith("data/"):
        data_songs_path = f"data/songs/{filename}"
        if os.path.exists(data_songs_path):
            return data_songs_path
    
    # Strategy 3: Case-insensitive search
    filename_lower = os.path.basename(filename).lower()
    for root, dirs, files in os.walk(search_root):
        for file in files:
            if file.lower() == filename_lower:
                return os.path.join(root, file)
    
    # Strategy 4: Partial name match (if filename has special chars)
    # Try to find any file containing the base name
    base_name = os.path.splitext(os.path.basename(filename))[0]
    for root, dirs, files in os.walk(search_root):
        for file in files:
            if base_name.lower() in file.lower():
                # Make sure it's an audio file
                if file.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a')):
                    return os.path.join(root, file)
    
    return None


def list_available_audio_files(search_root: str = ".") -> list:
    """Find all audio files in project."""
    audio_extensions = ('.mp3', '.wav', '.flac', '.ogg', '.m4a')
    found_files = []
    
    for root, dirs, files in os.walk(search_root):
        for file in files:
            if file.lower().endswith(audio_extensions):
                full_path = os.path.join(root, file)
                found_files.append(full_path)
    
    return sorted(found_files)


def train_classifier():
    """Main training function."""
    
    print("\n" + "="*60)
    print("JavenEQ Era Classifier — Training Script")
    print("="*60)
    
    # Step 1: Load labels
    if not os.path.exists("era_labels.json"):
        print("\n[Error] era_labels.json not found!")
        print("\nCreate era_labels.json with format:")
        print(json.dumps({
            "data/songs/song1.wav": "1970s_1980s",
            "data/songs/song2.wav": "1930s_1950s",
        }, indent=2))
        return False
    
    with open("era_labels.json", "r") as f:
        labels_dict = json.load(f)
    
    print(f"\n[Step 1] Loaded era_labels.json")
    print(f"  Labeled files: {len(labels_dict)}")
    print(f"  Eras: {set(labels_dict.values())}")
    
    # Step 2: Find audio files
    print(f"\n[Step 2] Searching for audio files...")
    print(f"  Working directory: {os.getcwd()}")
    
    # First, list what's available
    available_files = list_available_audio_files()
    print(f"\n  Available audio files in project:")
    for i, f in enumerate(available_files[:10], 1):
        print(f"    {i}. {f}")
    if len(available_files) > 10:
        print(f"    ... and {len(available_files) - 10} more")
    
    # Step 3: Match labeled files to actual files
    print(f"\n[Step 3] Matching labeled files to actual files...")
    X = []
    y = []
    feature_names = None
    matched_files = []
    
    for labeled_filename, era in labels_dict.items():
        print(f"\n  Looking for: {labeled_filename}")
        print(f"    Era label: {era}")
        
        # Try to find the file
        audio_path = find_audio_file(labeled_filename)
        
        if audio_path is None:
            print(f"    ❌ NOT FOUND (tried multiple strategies)")
            print(f"\n    Hint: Your labeled filename might not match actual file.")
            print(f"    Try one of these instead:")
            base_name = os.path.splitext(os.path.basename(labeled_filename))[0]
            matches = [f for f in available_files if base_name.lower() in f.lower()]
            if matches:
                for m in matches[:3]:
                    print(f"      - {m}")
            else:
                print(f"      (No close matches found)")
            continue
        
        print(f"    ✅ FOUND: {audio_path}")
        
        # Extract features
        print(f"    Extracting features...")
        features = AudioFeatureExtractor.extract(audio_path)
        
        if features is None:
            print(f"    ❌ Feature extraction failed")
            continue
        
        if feature_names is None:
            feature_names = list(features.keys())
        
        X.append(list(features.values()))
        y.append(era)
        matched_files.append(audio_path)
        print(f"    ✅ Features extracted successfully")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n[Step 4] Training Summary")
    print(f"  ✅ Successfully matched: {len(matched_files)} files")
    print(f"  ❌ Failed to find: {len(labels_dict) - len(matched_files)} files")
    
    if len(X) < 3:
        print(f"\n[Error] Need at least 3 labeled audio files, but only found {len(X)}!")
        print(f"\nDo your filenames in era_labels.json match your actual files?")
        print(f"Use the suggestions above to fix the filenames.\n")
        return False
    
    print(f"\n  Training samples: {len(X)}")
    print(f"  Features per sample: {len(feature_names)}")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Step 5: Train Random Forest
    print(f"\n[Step 5] Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    
    # Feature importance
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]
    
    print(f"  ✅ Model trained!\n")
    print(f"  Feature Importance (top 6):")
    for idx in sorted_idx[:6]:
        print(f"    {feature_names[idx]:<15}: {importance[idx]:.4f}")
    
    # Step 6: Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/era_classifier.pkl"
    
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_names": feature_names,
            "classes": model.classes_,
        }, f)
    
    print(f"\n[Step 6] ✅ Model saved to: {model_path}")
    print(f"  Classes: {list(model.classes_)}")
    
    # Cross-validation
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(model, X, y, cv=min(3, len(X)))
    print(f"\n[Step 7] Cross-validation Accuracy: {cv_scores.mean():.1%} (±{cv_scores.std():.1%})")
    
    print(f"\n" + "="*60)
    print(f"✅ Training Complete!")
    print(f"="*60)
    print(f"\nNext steps:")
    print(f"  1. Replace ai_eq_engine.py with ai_eq_engine_v2.py")
    print(f"  2. Run your app: python app.py")
    print(f"  3. Era detection will now use ML!\n")
    
    return True


if __name__ == "__main__":
    success = train_classifier()