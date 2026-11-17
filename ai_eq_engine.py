import numpy as np
import librosa
import pickle  #For loading the AI model
from eq_profile import EQProfile

class AIEQEngine:
    """
    The AI-powered EQ analysis engine.
    Detects frequency characteristics, vocals, instruments, and ERA using
    a Scikit-Learn Random Forest classifier.
    Matches the AIEQEngine class in the class diagram.
    """

    def __init__(self):
        self.analysis_results  = {}
        self.detected_era      = None
        self.era_profile       = None
        self.instrument_levels = {}   # Stores measured energy per instrument
        
        # <-- NEW AI CODE: Load the era classifier
        self.era_model = None
        self._load_era_model()

    # ------------------------------------------------------------------
    # <-- NEW AI CODE: Load the Scikit-Learn model
    # ------------------------------------------------------------------
    def _load_era_model(self):
        """Load the pre-trained Random Forest classifier from disk."""
        try:
            with open('era_classifier.pkl', 'rb') as f:
                self.era_model = pickle.load(f)
            print("[AIEQEngine] ✅ Era classifier loaded successfully (Scikit-Learn AI).")
        except FileNotFoundError:
            print("[AIEQEngine] ⚠️  era_classifier.pkl not found. Falling back to rule-based detection.")
            self.era_model = None
        except Exception as e:
            print(f"[AIEQEngine] ⚠️  Error loading AI model: {e}. Falling back to rules.")
            self.era_model = None

    # ------------------------------------------------------------------
    # analyzeAudio (unchanged)
    # ------------------------------------------------------------------
    def analyzeAudio(self, audio_file):
        print("\n[AIEQEngine] Starting audio analysis...")
        mono = audio_file.get_mono()
        sr   = audio_file.sampleRate
        freq_info  = self.detectFreq(mono, sr)
        vocal_info = self.detectVocals(mono, sr)
        instr_info = self.detectInstruments(mono, sr)
        era_info   = self.detectEra(mono, sr, freq_info)
        timeline_info = self.detectTimelineChanges(mono, sr)  # <-- This now returns a dict
        
        self.analysis_results = {
            "frequencies": freq_info,
            "vocals":      vocal_info,
            "instruments": instr_info,
            "era":         era_info,
            "timeline_top": timeline_info["top"],   # <-- For the graph
            "timeline_all": timeline_info["all"],   # <-- For the report
        }
        print("\n[AIEQEngine] Analysis complete.")
        self._printSummary()
        return self.analysis_results

    # ------------------------------------------------------------------
    # _measureInstrumentLevels (unchanged)
    # ------------------------------------------------------------------
    def _measureInstrumentLevels(self, mono_audio, sample_rate):
        """
        Measures the actual energy level of each instrument's
        primary frequency range.

        Returns a dict of instrument → energy ratio (0.0 to 1.0)
        so generateEQCurve() can boost quiet instruments more.
        """
        freqs, psd = __import__('scipy').signal.welch(mono_audio,
                                                       sample_rate, nperseg=4096)
        total = np.sum(psd) + 1e-12

        def energy(lo, hi):
            mask = (freqs >= lo) & (freqs < hi)
            return float(np.sum(psd[mask])) / total if mask.any() else 0.0

        levels = {
            "kick":    energy(60,   100),   # Kick drum fundamental
            "bass":    energy(80,   250),   # Bass guitar body
            "guitar":  energy(500,  5000),  # Guitar range
            "piano":   energy(200,  5000),  # Piano range
            "brass":   energy(300,  2500),  # Brass/winds
            "cymbal":  energy(8000, 16000), # Cymbals/hats
            "vocal":   energy(300,  4000),  # Vocal range
        }
        return levels

    def _adaptiveGain(self, instrument: str, levels: dict,
                       base_gain: float, target: float = 0.15) -> float:
        """
        Calculates adaptive gain for an instrument.

        If instrument energy < target  → boost more (up to 2x base gain)
        If instrument energy >= target → boost less (down to 0.5x base gain)

        Args:
            instrument : key in levels dict
            levels     : measured energy dict
            base_gain  : the default gain in dB
            target     : target energy ratio (default 15%)

        Returns adjusted gain in dB.
        """
        level = levels.get(instrument, target)
        if level <= 0:
            return base_gain

        ratio = target / (level + 1e-8)
        # Clamp multiplier between 0.5 and 2.0
        multiplier = np.clip(ratio, 0.5, 2.0)
        adjusted = round(base_gain * multiplier, 1)
        return adjusted

    # ------------------------------------------------------------------
    # generateEQCurve (unchanged - still uses your brilliant adaptive logic)
    # ------------------------------------------------------------------
    def generateEQCurve(self, analysis_results: dict = None,
                         mono_audio=None, sample_rate=None) -> EQProfile:
        """
        Generates an adaptive EQ curve:
        Layer 1: Era-based corrections
        Layer 2: Adaptive instrument enhancement
                 (quiet instruments boosted more, loud ones less)
        Layer 3: Vocal enhancement
        Layer 4: Dynamic corrections
        """
        if analysis_results is None:
            analysis_results = self.analysis_results
        if not analysis_results:
            print("[AIEQEngine] No analysis results. Run analyzeAudio() first.")
            return None

        era         = analysis_results.get("era", {})
        freq        = analysis_results.get("frequencies", {})
        vocals      = analysis_results.get("vocals", {})
        instruments = analysis_results.get("instruments", {})
        era_label   = era.get("era_label", "")
        bands       = freq.get("band_energy", {})

        print(f"\n[AIEQEngine] Generating adaptive EQ curve for: {era_label}")
        profile = EQProfile(era=era_label, genre="vintage")

        # Measure instrument levels for adaptive boosting
        if mono_audio is not None and sample_rate is not None:
            levels = self._measureInstrumentLevels(mono_audio, sample_rate)
            print(f"\n  [Adaptive] Measured instrument energy levels:")
            for k, v in levels.items():
                bar  = "█" * int(v * 200)
                flag = "🔇 quiet" if v < 0.05 else ("🔊 loud" if v > 0.20 else "✅ normal")
                print(f"    {k:<10}: {v:.4f} ({v*100:.1f}%)  {bar}  {flag}")
        else:
            # Fallback: use band energy from freq analysis
            levels = {
                "kick":   bands.get("sub_bass_20_80hz",  10) / 100,
                "bass":   bands.get("bass_80_250hz",     20) / 100,
                "guitar": bands.get("mid_500_2000hz",    20) / 100,
                "cymbal": bands.get("high_6000_20000hz", 5)  / 100,
                "vocal":  bands.get("mid_500_2000hz",    20) / 100,
            }

        # ==============================================================
        # LAYER 1: ERA-BASED CORRECTIONS
        # ==============================================================
        print("\n  [Layer 1] Era-based corrections...")

        if "1930s" in era_label:
            profile.addBand(freq=60,    gain=-3.0, Q=0.7, type="lowshelf")
            profile.addBand(freq=350,   gain=-1.5, Q=1.0, type="peak")
            profile.addBand(freq=4000,  gain=+2.5, Q=1.2, type="peak")
            profile.addBand(freq=10000, gain=+4.0, Q=0.7, type="highshelf")
            print("    + 1930s: Rumble cut, mud cut, presence + air boost")

        elif "1960s" in era_label:
            profile.addBand(freq=80,    gain=-1.5, Q=0.7, type="lowshelf")
            profile.addBand(freq=300,   gain=-1.5, Q=1.0, type="peak")
            profile.addBand(freq=3000,  gain=+2.0, Q=1.2, type="peak")
            profile.addBand(freq=10000, gain=+3.0, Q=0.7, type="highshelf")
            print("    + 1960s: Light rumble cut, presence + air boost")

        elif "1970s" in era_label:
            profile.addBand(freq=50,    gain=+1.5, Q=0.8, type="lowshelf")
            profile.addBand(freq=3000,  gain=-1.0, Q=1.5, type="peak")
            profile.addBand(freq=14000, gain=+1.0, Q=0.7, type="highshelf")
            print("    + 1970s: Warmth, harshness cut, very gentle air boost")

        else:
            profile.addBand(freq=100,   gain=-1.5, Q=0.7, type="lowshelf")
            profile.addBand(freq=10000, gain=+2.5, Q=0.7, type="highshelf")
            print("    + Generic vintage restoration")

        # ==============================================================
        # LAYER 2: ADAPTIVE INSTRUMENT ENHANCEMENT
        # ==============================================================
        print("\n  [Layer 2] Adaptive instrument enhancement...")

        detected = instruments.get("instruments_detected", {})
        bass_pct = bands.get("bass_80_250hz", 0)

        # --- DRUMS / PERCUSSION ---
        if "drums_percussion" in detected:
            confidence = detected["drums_percussion"]["confidence"]
            conf_mult  = 1.0 if confidence == "High" else 0.7

            kick_gain  = self._adaptiveGain("kick",   levels, 2.5, target=0.08)
            snare_gain = self._adaptiveGain("cymbal", levels, 1.5, target=0.10)

            if bass_pct < 40:
                profile.addBand(freq=70,   gain=round(kick_gain*conf_mult,1),  Q=1.2, type="peak")
                print(f"    + Kick @70Hz: {round(kick_gain*conf_mult,1):+.1f}dB (adaptive, level={levels.get('kick',0):.3f})")
            profile.addBand(freq=2500, gain=round(snare_gain*conf_mult,1), Q=2.0, type="peak")
            print(f"    + Snare @2.5kHz: {round(snare_gain*conf_mult,1):+.1f}dB [{confidence}]")

        # --- BASS GUITAR ---
        if "bass" in detected:
            confidence  = detected["bass"]["confidence"]
            conf_mult   = 1.0 if confidence == "High" else 0.7

            bass_fund   = self._adaptiveGain("bass", levels, 2.5, target=0.12)
            bass_def    = self._adaptiveGain("bass", levels, 1.0, target=0.12)
            mud_cut     = -1.0

            profile.addBand(freq=120, gain=round(bass_fund*conf_mult,1), Q=1.0, type="peak")
            profile.addBand(freq=220, gain=mud_cut,                       Q=1.5, type="peak")
            profile.addBand(freq=800, gain=round(bass_def*conf_mult,1),  Q=2.0, type="peak")
            print(f"    + Bass [{confidence}]: Fundamental {round(bass_fund*conf_mult,1):+.1f}dB@120Hz "
                  f"(adaptive, level={levels.get('bass',0):.3f}), "
                  f"mud {mud_cut:+.1f}dB@220Hz, "
                  f"def {round(bass_def*conf_mult,1):+.1f}dB@800Hz")

        # --- GUITAR / STRINGS ---
        if "guitar_strings" in detected:
            confidence   = detected["guitar_strings"]["confidence"]
            conf_mult    = 1.0 if confidence == "High" else 0.7

            gtr_body     = self._adaptiveGain("guitar", levels, 1.5, target=0.15)
            gtr_bite     = self._adaptiveGain("guitar", levels, 2.0, target=0.10)
            gtr_sparkle  = self._adaptiveGain("cymbal", levels, 2.0, target=0.05)

            profile.addBand(freq=500,  gain=round(gtr_body   *conf_mult,1), Q=2.0, type="peak")
            profile.addBand(freq=4000, gain=round(gtr_bite   *conf_mult,1), Q=1.5, type="peak")
            profile.addBand(freq=9000, gain=round(gtr_sparkle*conf_mult,1), Q=1.0, type="peak")
            print(f"    + Guitar [{confidence}]: body {round(gtr_body*conf_mult,1):+.1f}dB@500Hz, "
                  f"bite {round(gtr_bite*conf_mult,1):+.1f}dB@4kHz (adaptive), "
                  f"sparkle {round(gtr_sparkle*conf_mult,1):+.1f}dB@9kHz")

        # --- PIANO / KEYS ---
        if "piano_keys" in detected:
            confidence = detected["piano_keys"]["confidence"]
            conf_mult  = 1.0 if confidence == "High" else 0.7

            pno_warmth  = self._adaptiveGain("piano", levels, 2.0, target=0.12)
            pno_clarity = self._adaptiveGain("piano", levels, 2.0, target=0.08)

            profile.addBand(freq=300,  gain=round(pno_warmth *conf_mult,1), Q=1.5, type="peak")
            profile.addBand(freq=2500, gain=round(pno_clarity*conf_mult,1), Q=2.0, type="peak")
            print(f"    + Piano [{confidence}]: warmth {round(pno_warmth*conf_mult,1):+.1f}dB@300Hz, "
                  f"clarity {round(pno_clarity*conf_mult,1):+.1f}dB@2.5kHz (adaptive)")

        # --- BRASS / WINDS ---
        if "brass_winds" in detected:
            confidence  = detected["brass_winds"]["confidence"]
            conf_mult   = 1.0 if confidence == "High" else 0.7

            brass_body  = self._adaptiveGain("brass", levels, 2.0, target=0.12)
            brass_pres  = self._adaptiveGain("brass", levels, 2.0, target=0.10)

            profile.addBand(freq=400,  gain=round(brass_body*conf_mult,1), Q=1.5, type="peak")
            profile.addBand(freq=2000, gain=round(brass_pres*conf_mult,1), Q=2.0, type="peak")
            print(f"    + Brass [{confidence}]: body {round(brass_body*conf_mult,1):+.1f}dB@400Hz, "
                  f"presence {round(brass_pres*conf_mult,1):+.1f}dB@2kHz (adaptive)")

        # --- CYMBALS / HI-HATS ---
        if "cymbals_hihats" in detected:
            confidence  = detected["cymbals_hihats"]["confidence"]
            conf_mult   = 1.0 if confidence == "High" else 0.7

            cym_sparkle = self._adaptiveGain("cymbal", levels, 0.5, target=0.20)
            cym_air     = self._adaptiveGain("cymbal", levels, 0.3, target=0.20)

            profile.addBand(freq=10000, gain=round(cym_sparkle*conf_mult,1), Q=0.8, type="peak")
            profile.addBand(freq=14000, gain=round(cym_air    *conf_mult,1), Q=0.7, type="highshelf")
            print(f"    + Cymbals [{confidence}]: sparkle {round(cym_sparkle*conf_mult,1):+.1f}dB@10kHz, "
                  f"air {round(cym_air*conf_mult,1):+.1f}dB@14kHz (adaptive, "
                  f"level={levels.get('cymbal',0):.3f})")

        # ==============================================================
        # LAYER 3: VOCAL ENHANCEMENT
        # ==============================================================
        print("\n  [Layer 3] Vocal enhancement...")
        vocal_presence = vocals.get("vocal_presence", "")

        if "Strong" in vocal_presence or "Moderate" in vocal_presence:
            voc_body    = self._adaptiveGain("vocal", levels, 2.5, target=0.12)
            voc_warmth  = self._adaptiveGain("vocal", levels, 1.5, target=0.10)
            voc_clarity = self._adaptiveGain("vocal", levels, 1.5, target=0.10)
            voc_pres    = self._adaptiveGain("vocal", levels, 1.0, target=0.08)

            profile.addBand(freq=300,   gain=voc_body,    Q=1.5, type="peak")
            profile.addBand(freq=600,   gain=voc_warmth,  Q=2.0, type="peak")
            profile.addBand(freq=1500,  gain=voc_clarity, Q=2.0, type="peak")
            profile.addBand(freq=3000,  gain=voc_pres,    Q=2.0, type="peak")
            profile.addBand(freq=7000,  gain=-4.0,        Q=2.0, type="peak")
            profile.addBand(freq=12000, gain=+1.5,        Q=0.8, type="peak")

            print(f"    + Vocal body: {voc_body:+.1f}dB@300Hz (adaptive, level={levels.get('vocal',0):.3f})")
            print(f"    + Vocal warmth: {voc_warmth:+.1f}dB@600Hz")
            print(f"    + Vocal clarity: {voc_clarity:+.1f}dB@1.5kHz")
            print(f"    + Vocal presence: {voc_pres:+.1f}dB@3kHz")
            print(f"    + Sibilance control: -4.0dB@7kHz")
            print(f"    + Vocal air: +1.5dB@12kHz")
        else:
            print("    - No vocals detected, skipping")

        # ==============================================================
        # LAYER 4: DYNAMIC CORRECTIONS
        # ==============================================================
        print("\n  [Layer 4] Dynamic corrections...")
        high_pct = bands.get("high_6000_20000hz", 0)
        rolloff  = freq.get("spectral_rolloff_hz", 10000)

        if bass_pct > 60:
            profile.addBand(freq=160, gain=-1.5, Q=1.0, type="peak")
            print(f"    + Bass heavy ({bass_pct:.1f}%): -1.5dB@160Hz")
        if high_pct < 2.0:
            profile.addBand(freq=12000, gain=+1.5, Q=0.7, type="highshelf")
            print(f"    + Highs low ({high_pct:.1f}%): +1.5dB@12kHz shelf")
        if rolloff < 5000:
            profile.addBand(freq=6000, gain=+2.0, Q=0.8, type="peak")
            print(f"    + Low rolloff ({rolloff:.0f}Hz): +2.0dB@6kHz")

        print(f"\n[AIEQEngine] Adaptive EQ curve — {len(profile.bands)} bands total.")
        print(f"  Layer 1 (Era)        : ✅")
        print(f"  Layer 2 (Instruments): ✅ {len(detected)} instrument(s) — adaptive gains applied")
        print(f"  Layer 3 (Vocals)     : {'✅' if 'Strong' in vocal_presence or 'Moderate' in vocal_presence else '⏭  skipped'}")
        print(f"  Layer 4 (Dynamic)    : ✅")

        return profile

    # ------------------------------------------------------------------
    # detectEra  <-- UPDATED WITH AI + FALLBACK
    # ------------------------------------------------------------------
    def detectEra(self, mono_audio, sample_rate, freq_info=None):
        print("[AIEQEngine] Detecting recording era...")
        
        # 1. Extract the 4 features used by the AI model
        rolloff = float(librosa.feature.spectral_rolloff(
            y=mono_audio, sr=sample_rate, roll_percent=0.85).mean())
        
        stft = np.abs(librosa.stft(mono_audio, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
        avg = np.mean(stft, axis=1)
        total = float(np.mean(avg)) + 1e-8
        
        def band_energy(lo, hi):
            mask = (freqs >= lo) & (freqs < hi)
            return float(np.mean(avg[mask])) / total if mask.any() else 0.0
        
        high_e = band_energy(8000, 20000)
        
        frame_rms = [np.sqrt(np.mean(mono_audio[i:i+4096]**2)) 
                     for i in range(0, len(mono_audio)-4096, 2048)]
        dynamic_range = float(np.percentile(frame_rms, 95) / 
                              (np.percentile(frame_rms, 5) + 1e-8))
        
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=mono_audio)))
        
        features = np.array([[rolloff, high_e, dynamic_range, flatness]])
        
        # 2. Define the era profiles dictionary (unchanged)
        era_profiles = {
            "1930s_1950s": {
                "era_label": "1930s – 1950s (Early Analog)",
                "noise_threshold": 0.08, "normalize_target": 0.12,
                "eq_hints": [
                    "Boost highs above 8kHz (+4 to +6 dB) to restore air",
                    "Cut sub-bass below 60Hz (heavy rumble)",
                    "Boost presence at 3-5kHz (+3 dB) for clarity",
                ],
                "description": "Heavy tape hiss, very limited high frequency, mono recording typical of this era."
            },
            "1960s_1970s": {
                "era_label": "1960s – 1970s (Classic Rock / Soul Era)",
                "noise_threshold": 0.04, "normalize_target": 0.11,
                "eq_hints": [
                    "Boost highs above 10kHz (+2 to +4 dB) to add air",
                    "Cut low-mids at 300-400Hz to reduce muddiness",
                    "Boost presence at 2-4kHz (+2 dB) for vocal clarity",
                ],
                "description": "Moderate tape noise, early stereo mixing, some high-frequency rolloff typical of this era."
            },
            "1970s_1980s": {
                "era_label": "1970s – 1980s (Late Analog / Early Digital)",
                "noise_threshold": 0.05, "normalize_target": 0.10,
                "eq_hints": [
                    "Cut harsh upper-mids at 2-4kHz (-1 to -2 dB)",
                    "Boost sub-bass at 40-60Hz (+2 dB) for warmth",
                    "Cut low-mids at 200-300Hz to reduce muddiness",
                    "Apply soft high-shelf boost above 12kHz for air",
                ],
                "description": "Transition from analog to digital, some compression artifacts and cassette-era coloring."
            },
        }

        era_map = {0: "1930s_1950s", 1: "1960s_1970s", 2: "1970s_1980s"}
        detected_era = "1960s_1970s"  # default fallback
        scores = {"1930s_1950s": 0, "1960s_1970s": 0, "1970s_1980s": 0}
        
        # 3. If AI model exists, use it!
        if self.era_model is not None:
            try:
                prediction = self.era_model.predict(features)[0]
                proba = self.era_model.predict_proba(features)[0]
                
                detected_era = era_map[prediction]
                scores = {k: float(proba[i]) for i, k in enumerate(era_map.values())}
                
                print(f"  🤖 AI Prediction: {detected_era} (Confidence: {max(proba):.2f})")
            except Exception as e:
                print(f"  ⚠️ AI inference failed ({e}). Falling back to rules.")
                detected_era = None  # force fallback
            
        # 4. Fallback to original rule-based logic if AI failed or doesn't exist
        if detected_era is None or self.era_model is None:
            print("  ⚙️ Using rule-based detection fallback...")
            scores = {"1930s_1950s": 0, "1960s_1970s": 0, "1970s_1980s": 0}
            
            if rolloff < 6000:    scores["1930s_1950s"] += 3
            elif rolloff < 10000: scores["1960s_1970s"] += 3
            else:                 scores["1970s_1980s"] += 3
                
            if high_e < 0.01:     scores["1930s_1950s"] += 2
            elif high_e < 0.05:   scores["1960s_1970s"] += 2
            else:                 scores["1970s_1980s"] += 2
                
            if dynamic_range > 15:  scores["1930s_1950s"] += 1
            elif dynamic_range > 8: scores["1960s_1970s"] += 1
            else:                   scores["1970s_1980s"] += 2
                
            if flatness > 0.01:     scores["1970s_1980s"] += 1
            elif flatness < 0.003:  scores["1930s_1950s"] += 1
            else:                   scores["1960s_1970s"] += 1
                
            if high_e > 0.15:       scores["1970s_1980s"] += 1
                
            detected_era = max(scores, key=scores.get)
            print(f"  ⚙️ Rules picked: {detected_era}")

        # 5. Set the profile and return results (identical to original)
        self.detected_era = detected_era
        self.era_profile = era_profiles[detected_era]
        
        return {
            "detected_era":    detected_era,
            "era_label":       self.era_profile["era_label"],
            "description":     self.era_profile["description"],
            "confidence_scores": scores,
            "spectral_rolloff":  round(rolloff, 1),
            "dynamic_range":     round(dynamic_range, 2),
            "spectral_flatness": round(flatness, 5),
            "noise_threshold_recommended": self.era_profile["noise_threshold"],
            "eq_hints": self.era_profile["eq_hints"],
        }

    def getEraProfile(self):
        return self.era_profile

    # ------------------------------------------------------------------
    # detectFreq (unchanged)
    # ------------------------------------------------------------------
    def detectFreq(self, mono_audio, sample_rate):
        print("[AIEQEngine] Detecting frequency characteristics...")
        stft  = np.abs(librosa.stft(mono_audio, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
        avg   = np.mean(stft, axis=1)
        dominant_freq = freqs[np.argmax(avg)]

        def band_energy(lo, hi):
            mask = (freqs >= lo) & (freqs < hi)
            return float(np.mean(avg[mask])) if mask.any() else 0.0

        sub_bass = band_energy(20,   80)
        bass     = band_energy(80,   250)
        low_mid  = band_energy(250,  500)
        mid      = band_energy(500,  2000)
        high_mid = band_energy(2000, 6000)
        high     = band_energy(6000, 20000)
        total    = sub_bass+bass+low_mid+mid+high_mid+high+1e-8

        rolloff  = float(librosa.feature.spectral_rolloff(
            y=mono_audio, sr=sample_rate, roll_percent=0.85).mean())
        centroid = float(librosa.feature.spectral_centroid(
            y=mono_audio, sr=sample_rate).mean())

        issues = []
        if rolloff < 8000:               issues.append("High-frequency rolloff detected (dull/muffled sound)")
        if (sub_bass+bass)/total > 0.5:  issues.append("Excessive low-end energy (boomy bass)")
        if mid/total > 0.55:             issues.append("Midrange congestion (muddy mix)")
        if high/total < 0.05:            issues.append("Lacking high-frequency air")
        if not issues:                   issues.append("Frequency balance looks healthy")

        return {
            "dominant_frequency_hz": round(float(dominant_freq), 1),
            "spectral_rolloff_hz":   round(rolloff, 1),
            "spectral_centroid_hz":  round(centroid, 1),
            "band_energy": {
                "sub_bass_20_80hz":     round(sub_bass/total*100, 1),
                "bass_80_250hz":        round(bass/total*100, 1),
                "low_mid_250_500hz":    round(low_mid/total*100, 1),
                "mid_500_2000hz":       round(mid/total*100, 1),
                "high_mid_2000_6000hz": round(high_mid/total*100, 1),
                "high_6000_20000hz":    round(high/total*100, 1),
            },
            "issues": issues,
        }

    # ------------------------------------------------------------------
    # detectVocals (unchanged)
    # ------------------------------------------------------------------
    def detectVocals(self, mono_audio, sample_rate):
        print("[AIEQEngine] Detecting vocals...")
        harmonic, _  = librosa.effects.hpss(mono_audio)
        stft_harm    = np.abs(librosa.stft(harmonic, n_fft=2048, hop_length=512))
        freqs        = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
        vocal_mask           = (freqs >= 300) & (freqs <= 3500)
        vocal_energy         = float(np.mean(stft_harm[vocal_mask]))
        total_energy         = float(np.mean(stft_harm)) + 1e-8
        vocal_presence_ratio = vocal_energy / total_energy

        if vocal_presence_ratio > 0.6:    vocal_presence = "Strong vocal presence"
        elif vocal_presence_ratio > 0.35: vocal_presence = "Moderate vocal presence"
        else:                             vocal_presence = "Weak or no vocal presence (instrumental)"

        zcr       = float(np.mean(librosa.feature.zero_crossing_rate(harmonic)))
        stft_full = np.abs(librosa.stft(mono_audio, n_fft=2048, hop_length=512))
        avg_mag   = np.mean(stft_full, axis=1)
        issues          = []
        presence_mask   = (freqs >= 2000) & (freqs <= 5000)
        presence_energy = float(np.mean(avg_mag[presence_mask]))
        if vocal_presence_ratio > 0.35 and presence_energy < 0.01:
            issues.append("Vocal presence dip (2-5 kHz): may sound distant")
        if vocal_presence_ratio > 0.35:
            issues.append("Recommend gentle boost at 2-4 kHz to improve vocal clarity")
        return {
            "vocal_presence":     vocal_presence,
            "vocal_energy_ratio": round(vocal_presence_ratio, 3),
            "zero_crossing_rate": round(zcr, 4),
            "issues":             issues if issues else ["Vocals sound balanced"],
        }

    # ------------------------------------------------------------------
    # detectInstruments (unchanged)
    # ------------------------------------------------------------------
    def detectInstruments(self, mono_audio, sample_rate):
        print("[AIEQEngine] Detecting instruments...")
        harmonic, percussive = librosa.effects.hpss(mono_audio)
        stft_full = np.abs(librosa.stft(mono_audio,   n_fft=2048, hop_length=512))
        stft_harm = np.abs(librosa.stft(harmonic,     n_fft=2048, hop_length=512))
        stft_perc = np.abs(librosa.stft(percussive,   n_fft=2048, hop_length=512))
        freqs     = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
        avg_full  = np.mean(stft_full, axis=1)
        avg_harm  = np.mean(stft_harm, axis=1)
        avg_perc  = np.mean(stft_perc, axis=1)
        total     = float(np.mean(avg_full)) + 1e-8

        def energy_in(lo, hi, spectrum):
            mask = (freqs >= lo) & (freqs < hi)
            return float(np.mean(spectrum[mask])) if mask.any() else 0.0

        perc_ratio    = float(np.mean(avg_perc)) / total
        bass_harm     = energy_in(60,    250,  avg_harm) / total
        guitar_energy = energy_in(250,   5000, avg_harm) / total
        piano_energy  = energy_in(80,    8000, avg_harm) / total
        brass_energy  = energy_in(200,   2500, avg_harm) / total
        hihat_energy  = energy_in(8000, 20000, avg_perc) / total

        instruments_detected = {}
        if perc_ratio > 0.3:
            instruments_detected["drums_percussion"] = {
                "confidence": "High" if perc_ratio > 0.5 else "Medium",
                "note": "Strong transient/percussive content detected"}
        if bass_harm > 0.15:
            instruments_detected["bass"] = {
                "confidence": "High" if bass_harm > 0.3 else "Medium",
                "note": "Significant low-harmonic content (bass guitar or upright bass)"}
        if guitar_energy > 0.25:
            instruments_detected["guitar_strings"] = {
                "confidence": "High" if guitar_energy > 0.45 else "Medium",
                "note": "Mid-range harmonic energy consistent with guitar or strings"}
        if piano_energy > 0.5 and bass_harm < 0.25:
            instruments_detected["piano_keys"] = {
                "confidence": "Medium",
                "note": "Broad harmonic content consistent with piano or keyboards"}
        if brass_energy > 0.35 and perc_ratio < 0.3:
            instruments_detected["brass_winds"] = {
                "confidence": "Medium",
                "note": "Mid-range tonal content consistent with brass or wind instruments"}
        if hihat_energy > 0.05:
            instruments_detected["cymbals_hihats"] = {
                "confidence": "Medium",
                "note": "High-frequency transient content (cymbals or hi-hats)"}
        if not instruments_detected:
            instruments_detected["unknown"] = {
                "confidence": "Low", "note": "Could not identify specific instruments"}

        return {
            "instruments_detected": instruments_detected,
            "harmonic_ratio":       round(float(np.mean(avg_harm)) / total, 3),
            "percussive_ratio":     round(perc_ratio, 3),
        }
        
    def applyEQAndEnhancement(self, audio_data, sample_rate, raw_gains_db):
        """Applies gain adjustments while clamping extreme boosts/cuts."""
        # Fix #2: Clamp gains to preserve natural mix balance (max +3dB boost, -4dB cut)
        clamped_gains_db = np.clip(raw_gains_db, -4.0, 3.0)
        
        # Convert dB gain back to linear scale
        linear_gains = 10.0 ** (clamped_gains_db / 20.0)

        # Apply STFT processing
        stft = librosa.stft(audio_data, n_fft=2048, hop_length=512)
        stft_mag, stft_phase = librosa.magphase(stft)

        # Fix #3: Add spectral floor protection so room acoustics & tails aren't stripped
        alpha = 0.6  # Gentler noise subtraction multiplier
        stft_processed = stft_mag * linear_gains[:, np.newaxis]
        stft_processed = np.maximum(stft_processed, 0.10 * stft_mag)

        # Reconstruct signal
        enhanced_stft = stft_processed * stft_phase
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=512, length=len(audio_data))
        
        return enhanced_audio

    def enhanceAudio(self, audio_file, output_path="enhanced_output.wav"):
        print(f"[AIEQEngine] Processing file: {audio_file}")
        
        # Load original audio
        original_audio, sr = librosa.load(audio_file, sr=None, mono=True)

        # 1. Analyze instruments (Without Tempo)
        analysis_res = self.detectInstruments(original_audio, sr)
        self._printSummary(analysis_res)

        # 2. Example spectral gain array (adjust to your model/bands output)
        num_freq_bins = 1 + 2048 // 2
        raw_gains_db = np.zeros(num_freq_bins)  # Placeholder for calculated gains

        # 3. Apply EQ and spectral enhancement
        enhanced_processed = self.applyEQAndEnhancement(original_audio, sr, raw_gains_db)

        # Fix #1: Dry/Wet Blend (Parallel Processing to maintain original character)
        dry_wet_ratio = 0.65  # 65% enhanced, 35% original raw track
        min_len = min(len(original_audio), len(enhanced_processed))
        
        final_audio = (dry_wet_ratio * enhanced_processed[:min_len]) + \
                      ((1.0 - dry_wet_ratio) * original_audio[:min_len])

        # Export result
        sf.write(output_path, final_audio, sr)
        print(f"[AIEQEngine] Saved enhanced audio to: {output_path}")
        return final_audio    
        
    def detectTimelineChanges(self, mono_audio, sample_rate):
        """
        Detects significant structural changes in the song over time.
        Returns a dict with:
          - 'all': every detected change point (for the report)
          - 'top': the 10 biggest changes (for the graph)
        """
        print("[AIEQEngine] Scanning timeline for change points...")
        
        # 1. Compute features per 0.5 second hop
        hop_length = int(sample_rate * 0.5)  # 500ms windows
        frame_length = hop_length * 2
        
        # RMS energy (loudness)
        rms = librosa.feature.rms(y=mono_audio, frame_length=frame_length, hop_length=hop_length)
        rms = rms[0]  # flatten
        
        # Spectral Centroid (brightness)
        cent = librosa.feature.spectral_centroid(y=mono_audio, sr=sample_rate, 
                                                hop_length=hop_length, n_fft=frame_length)
        cent = cent[0]
        
        # 2. Normalize features to 0-1 range for combined scoring
        rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-8)
        cent_norm = (cent - np.min(cent)) / (np.max(cent) - np.min(cent) + 1e-8)
        
        # 3. Combined "change score"
        combined = rms_norm + cent_norm
        diff = np.abs(np.diff(combined))
        
        # 4. Find peaks exceeding threshold
        threshold = np.mean(diff) + 1.2 * np.std(diff)
        change_indices = np.where(diff > threshold)[0]
        
        # 5. Collect all candidate changes with their magnitudes
        candidates = []
        for idx in change_indices:
            timestamp = round((idx * hop_length) / sample_rate, 1)
            magnitude = float(diff[idx])
            
            # Avoid duplicates (if two changes happen very close)
            if not candidates or (timestamp - candidates[-1][0]) > 2.0:
                candidates.append((timestamp, magnitude))
        
        # 6. --- LIST A: ALL timestamps (chronological) ---
        all_timestamps = [t for t, m in sorted(candidates, key=lambda x: x[0])]
        
        # 7. --- LIST B: TOP 10 biggest changes (sorted by magnitude) ---
        candidates_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)
        top_changes = candidates_sorted[:10]  # Keep top 10
        top_changes.sort(key=lambda x: x[0])  # Sort chronologically
        top_timestamps = [t for t, m in top_changes]
        
        # 8. Always include the start (0:00) in both lists for context
        if all_timestamps and all_timestamps[0] > 1.0:
            all_timestamps.insert(0, 0.0)
        if top_timestamps and top_timestamps[0] > 1.0:
            top_timestamps.insert(0, 0.0)
        elif not top_timestamps:
            top_timestamps.append(0.0)
        
        print(f"  Found {len(change_indices)} total shifts, keeping {len(top_timestamps)} for graph, {len(all_timestamps)} for report.")
        return {"all": all_timestamps, "top": top_timestamps}

    # ------------------------------------------------------------------
    # _printSummary (unchanged)
    # ------------------------------------------------------------------
    def _printSummary(self):
        r = self.analysis_results
        print("\n" + "="*55)
        print("       JAVEANEQ — AUDIO ANALYSIS REPORT")
        print("="*55)
        era = r["era"]
        print(f"\n🎙  ERA DETECTION")
        print(f"  Detected Era      : {era['era_label']}")
        print(f"  Description       : {era['description']}")
        print(f"  Spectral Rolloff  : {era['spectral_rolloff']} Hz")
        print(f"  Dynamic Range     : {era['dynamic_range']}")
        print(f"  Confidence Scores : {era['confidence_scores']}")
        print("  EQ Recommendations:")
        for hint in era["eq_hints"]:
            print(f"    💡 {hint}")
        freq = r["frequencies"]
        print(f"\n📊 FREQUENCY ANALYSIS")
        print(f"  Dominant Frequency : {freq['dominant_frequency_hz']} Hz")
        print(f"  Spectral Rolloff   : {freq['spectral_rolloff_hz']} Hz")
        print(f"  Spectral Centroid  : {freq['spectral_centroid_hz']} Hz")
        print("  Band Energy Distribution:")
        for band, pct in freq["band_energy"].items():
            bar = "█" * int(pct / 5)
            print(f"    {band:<28} {pct:>5.1f}% {bar}")
        print("  Issues:")
        for issue in freq["issues"]:
            print(f"    ⚠ {issue}")
        voc = r["vocals"]
        print(f"\n🎤 VOCAL DETECTION")
        print(f"  Status       : {voc['vocal_presence']}")
        print(f"  Energy Ratio : {voc['vocal_energy_ratio']}")
        for note in voc["issues"]:
            print(f"    → {note}")
        ins = r["instruments"]
        print(f"\n🎸 INSTRUMENT DETECTION")
        print(f"  Harmonic Ratio   : {ins['harmonic_ratio']}")
        print(f"  Percussive Ratio : {ins['percussive_ratio']}")
        for name, info in ins["instruments_detected"].items():
            print(f"    [{info['confidence']}] {name}: {info['note']}")
        print("\n" + "="*55)