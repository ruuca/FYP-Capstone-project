import numpy as np
from scipy import signal
from audio_file import AudioFile

class AudioProcessor:
    """
    Handles all audio processing tasks.
    Era-aware — adjusts processing strength based on detected recording era.
    Matches the AudioProcessor class in the class diagram.
    """

    def __init__(self, audio_file: AudioFile):
        self.audio_file  = audio_file
        self.era_profile = None

    def setEraProfile(self, era_profile: dict):
        self.era_profile = era_profile
        print(f"[AudioProcessor] Era profile set: {era_profile['era_label']}")
        print(f"[AudioProcessor] Noise threshold : {era_profile['noise_threshold']}")
        print(f"[AudioProcessor] Normalize target: {era_profile['normalize_target']}")

    def preprocess(self):
        """
        Full preprocessing pipeline:
        noise reduction → normalize → brightness boost
        """
        print("\n[AudioProcessor] Starting preprocessing pipeline...")
        self.reduceNoise()
        self.normalize()
        self.brightnessBoost()
        print("[AudioProcessor] Preprocessing complete.\n")
        self._hardLimit(ceiling=0.95)

    # ------------------------------------------------------------------
    # reduceNoise
    # ------------------------------------------------------------------
    def reduceNoise(self):
        """
        Era-aware noise reduction.
        Step 1: High-pass filter removes low-frequency rumble
        Step 2: Spectral gating suppresses background hiss
        """
        if self.era_profile:
            noise_threshold = self.era_profile["noise_threshold"]
            era_label       = self.era_profile["era_label"]
        else:
            noise_threshold = 0.03
            era_label       = "Unknown (using default)"

        print(f"[AudioProcessor] Reducing noise (era: {era_label}, threshold: {noise_threshold})...")

        audio_data  = self.audio_file.get_data().copy()
        sample_rate = self.audio_file.sampleRate
        cleaned     = []

        for ch in range(audio_data.shape[0]):
            channel = audio_data[ch]

            # High-pass filter — era dependent
            # NOTE: 4th order filter at 30Hz causes -24dB at 60Hz (kills kick drum)
            # So for 1970s-1980s we skip it entirely — studio recordings have no rumble
            nyq = sample_rate / 2
            if self.era_profile and "1930s" in self.era_profile.get("era_label", ""):
                # Order 4 at 100Hz — aggressive, real tape rumble needs removing
                b, a    = signal.butter(4, 100 / nyq, btype='high')
                channel = signal.filtfilt(b, a, channel)
                print(f"      [HP] Applied 100Hz order-4 high-pass (1930s rumble removal)")
            elif self.era_profile and "1960s" in self.era_profile.get("era_label", ""):
                # Order 2 at 40Hz — gentle, preserves kick drum at 60-80Hz
                b, a    = signal.butter(2, 40 / nyq, btype='high')
                channel = signal.filtfilt(b, a, channel)
                print(f"      [HP] Applied 40Hz order-2 high-pass (gentle 1960s rumble removal)")
            else:
                # 1970s-1980s: SKIP high-pass filter entirely
                # Studio recordings from this era have no rumble
                # Kick drum (60-80Hz) must be fully preserved
                print(f"      [HP] Skipped for 1970s-1980s — preserving kick drum and bass")

            # --- KICK DRUM & BASS PRESERVATION ---
            # Split signal into LOW and HIGH bands BEFORE gating.
            # Only gate the HIGH band (200Hz+).
            # LOW band (kick + bass) passes through completely UNTOUCHED.
            # This avoids phase cancellation from the old filter-restore approach.
            nyq_gate     = sample_rate / 2
            crossover_hz = 200  # kick drum (60-80Hz) stays in the low band

            # Order 2 for minimal phase distortion at crossover point
            b_lo, a_lo = signal.butter(2, crossover_hz / nyq_gate, btype='low')
            b_hi, a_hi = signal.butter(2, crossover_hz / nyq_gate, btype='high')

            low_band  = signal.filtfilt(b_lo, a_lo, channel)  # kick + bass — NOT gated
            high_band = signal.filtfilt(b_hi, a_hi, channel)  # mids + highs — gated

            # Spectral gating runs ONLY on the high band (200Hz+)
            frame_size = 2048
            hop_size   = 512
            f, t, Zxx  = signal.stft(high_band, fs=sample_rate,
                                     nperseg=frame_size,
                                     noverlap=frame_size - hop_size)
            magnitude    = np.abs(Zxx)
            phase        = np.angle(Zxx)
            frame_energy = np.mean(magnitude, axis=0)
            noise_floor  = np.percentile(frame_energy, 20) * noise_threshold * 8
            mask         = np.where(magnitude > noise_floor, 1.0,
                                    magnitude / (noise_floor + 1e-8))
            _, high_gated = signal.istft(mask * magnitude * np.exp(1j * phase),
                                          fs=sample_rate, nperseg=frame_size,
                                          noverlap=frame_size - hop_size)

            # Trim/pad high_gated to match original length
            orig_len = audio_data.shape[1]
            if len(high_gated) > orig_len:   high_gated = high_gated[:orig_len]
            elif len(high_gated) < orig_len: high_gated = np.pad(high_gated, (0, orig_len - len(high_gated)))

            # Recombine: untouched low band + gated high band
            ch_clean = low_band + high_gated
            print(f"      [Gate] Low band below {crossover_hz}Hz untouched, high band gated")

            cleaned.append(ch_clean)

        self.audio_file._audio_data = np.array(cleaned)
        print("[AudioProcessor] Noise reduction done.")

    # ------------------------------------------------------------------
    # normalize
    # ------------------------------------------------------------------
    def normalize(self):
        """
        RMS-based normalization to a consistent loudness level.
        Ensures EQ and compression receive a predictable input signal.
        """
        target_rms = self.era_profile["normalize_target"] if self.era_profile else 0.10
        print(f"[AudioProcessor] Normalizing (target RMS: {target_rms})...")

        audio_data = self.audio_file._audio_data.copy()
        rms        = np.sqrt(np.mean(audio_data ** 2))
        if rms == 0:
            print("[AudioProcessor] Warning: Silent audio, skipping normalize.")
            return
        gain = np.clip(target_rms / rms, 0.1, 10.0)
        self.audio_file._audio_data = np.clip(audio_data * gain, -1.0, 1.0)
        print(f"[AudioProcessor] Normalization done. Gain applied: {gain:.3f}x")

    # ------------------------------------------------------------------
    # brightnessBoost
    # ------------------------------------------------------------------
    def brightnessBoost(self):
        """
        Boosts high frequencies to restore air and clarity
        lost in vintage recordings.
        """
        print("[AudioProcessor] Applying brightness boost...")
        audio_data  = self.audio_file._audio_data.copy()
        sample_rate = self.audio_file.sampleRate

        if self.era_profile and "1930s" in self.era_profile.get("era_label", ""):
            boost_gain = 2.0    # Strong boost — very dull recordings
            cutoff_hz  = 6000
        elif self.era_profile and "1960s" in self.era_profile.get("era_label", ""):
            boost_gain = 1.6    # Medium boost
            cutoff_hz  = 8000
        else:
            boost_gain = 1.08   # Minimal for 1970s-1980s — just a hint of air
            cutoff_hz  = 14000  # Only the very top frequencies

        nyq    = sample_rate / 2
        b, a   = signal.butter(2, cutoff_hz / nyq, btype='high')
        boosted = np.zeros_like(audio_data)
        for ch in range(audio_data.shape[0]):
            high_shelf  = signal.filtfilt(b, a, audio_data[ch])
            boosted[ch] = audio_data[ch] + (high_shelf * (boost_gain - 1))

        self.audio_file._audio_data = np.clip(boosted, -1.0, 1.0)
        print(f"[AudioProcessor] Brightness boost done (cutoff: {cutoff_hz}Hz, gain: +{round((boost_gain-1)*100)}%)")

    # ------------------------------------------------------------------
    # compress  (PROPERLY IMPLEMENTED)
    # ------------------------------------------------------------------
    def compress(self):
        """
        Professional-grade compressor with:
        - Attack & release envelope follower
        - Soft-knee gain computation
        - Makeup gain to restore loudness after compression
        - Hard limiter as final safety ceiling

        Controls the dynamic range so the final audio is:
        loud, punchy, and free of clipping.
        """
        print("\n[AudioProcessor] Applying compression...")

        # --- Era-aware settings ---
        if self.era_profile and "1970s" in self.era_profile.get("era_label", ""):
            # 1970s-1980s: already somewhat compressed, use lighter settings
            threshold_db = -18.0
            ratio        = 3.0
            attack_ms    = 15.0   # Slower attack = more punch/transient preserved
            release_ms   = 200.0
            knee_db      = 6.0
            print("  Settings: 1970s-1980s (light ratio, slower attack for punch)")

        elif self.era_profile and "1960s" in self.era_profile.get("era_label", ""):
            threshold_db = -20.0
            ratio        = 4.0
            attack_ms    = 10.0
            release_ms   = 150.0
            knee_db      = 4.0
            print("  Settings: 1960s-1970s (medium ratio)")

        else:
            # 1930s-1950s: wide dynamic range, needs more compression
            threshold_db = -24.0
            ratio        = 5.0
            attack_ms    = 8.0
            release_ms   = 120.0
            knee_db      = 3.0
            print("  Settings: 1930s-1950s (higher ratio, tighter)")

        sample_rate = self.audio_file.sampleRate
        audio_data  = self.audio_file._audio_data.copy()

        # Convert attack/release times to per-sample coefficients
        alpha_attack  = np.exp(-1.0 / (attack_ms  * sample_rate / 1000.0))
        alpha_release = np.exp(-1.0 / (release_ms * sample_rate / 1000.0))

        print(f"  Threshold: {threshold_db} dB | Ratio: {ratio}:1 | "
              f"Attack: {attack_ms}ms | Release: {release_ms}ms | Knee: {knee_db}dB")

        # Measure RMS before compression (for makeup gain calculation)
        rms_before = np.sqrt(np.mean(audio_data ** 2)) + 1e-8

        compressed = np.zeros_like(audio_data)

        for ch in range(audio_data.shape[0]):
            channel  = audio_data[ch]
            envelope = 0.0
            output   = np.zeros_like(channel)

            for i, sample in enumerate(channel):
                # --- Envelope follower ---
                abs_sample = abs(sample)
                if abs_sample > envelope:
                    envelope = alpha_attack  * envelope + (1 - alpha_attack)  * abs_sample
                else:
                    envelope = alpha_release * envelope + (1 - alpha_release) * abs_sample

                # Convert envelope to dB
                envelope_db = 20.0 * np.log10(envelope + 1e-8)

                # --- Soft-knee gain computation ---
                half_knee = knee_db / 2.0

                if envelope_db < (threshold_db - half_knee):
                    # Below knee: no compression
                    gain_db = 0.0

                elif envelope_db > (threshold_db + half_knee):
                    # Above knee: full compression
                    gain_db = (threshold_db + (envelope_db - threshold_db) / ratio) - envelope_db

                else:
                    # Inside knee: gradual compression
                    knee_factor = (envelope_db - threshold_db + half_knee) / knee_db
                    gain_db     = (1.0/ratio - 1.0) * (knee_factor ** 2) * half_knee

                # Apply gain reduction
                gain      = 10.0 ** (gain_db / 20.0)
                output[i] = sample * gain

            compressed[ch] = output

        self.audio_file._audio_data = compressed

        # --- Makeup Gain ---
        # Boost the output back up to match the pre-compression loudness
        # so the result sounds loud and full, not squashed and quiet
        rms_after  = np.sqrt(np.mean(compressed ** 2)) + 1e-8
        makeup_gain = np.clip(rms_before / rms_after, 1.0, 4.0)
        self.audio_file._audio_data = compressed * makeup_gain
        print(f"  Makeup gain applied: {makeup_gain:.3f}x ({20*np.log10(makeup_gain):.1f} dB)")

        # --- Hard Limiter ---
        # Final safety ceiling — absolutely prevents any sample
        # from exceeding ±0.95 (leaves tiny headroom, no clipping)
        self._hardLimit(ceiling=0.95)

        print("[AudioProcessor] Compression complete.")
        print(f"  RMS before: {rms_before:.4f} | RMS after: {np.sqrt(np.mean(self.audio_file._audio_data**2)):.4f}")

    # ------------------------------------------------------------------
    # _hardLimit (internal)
    # ------------------------------------------------------------------
    def _hardLimit(self, ceiling: float = 0.95):
        """
        Hard brick-wall limiter.
        Ensures the final audio never exceeds the ceiling level.
        Prevents clipping while keeping loudness high.
        """
        print(f"  [Limiter] Applying hard limit at ±{ceiling}...")
        audio_data = self.audio_file._audio_data
        peak       = np.max(np.abs(audio_data))

        if peak > ceiling:
            # Scale down just enough to bring peak to ceiling
            scale = ceiling / peak
            self.audio_file._audio_data = audio_data * scale
            print(f"  [Limiter] Peak was {peak:.4f}, scaled by {scale:.4f}")
        else:
            print(f"  [Limiter] Peak {peak:.4f} already within ceiling, no limiting needed")

    # ------------------------------------------------------------------
    # applyFilters
    # ------------------------------------------------------------------
    def applyFilters(self, filter_chain):
        print("[AudioProcessor] Applying EQ filters...")
        filter_chain.apply(self.audio_file)
        print("[AudioProcessor] Filters applied.")

    def get_processed_audio(self):
        return self.audio_file.get_data()