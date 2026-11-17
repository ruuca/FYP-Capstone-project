import numpy as np
from scipy import signal
from band import Band

class FilterChain:
    """
    Holds a chain of EQ bands and applies them to audio.
    Implements biquad IIR filters using the Audio EQ Cookbook formulas.
    Matches the FilterChain class in the class diagram.
    """

    def __init__(self):
        self.filters = []   # List of Band objects
        self._bypassed = False

    def addFilter(self, type: str, freq: float, gain: float, Q: float):
        """Add a new EQ band to the chain."""
        band = Band(freq=freq, gain=gain, Q=Q, type=type)
        self.filters.append(band)
        return band

    def bypass(self, state: bool = True):
        """Bypass the filter chain (pass audio through unchanged)."""
        self._bypassed = state
        print(f"[FilterChain] Bypass: {'ON' if state else 'OFF'}")

    def apply(self, audio_file):
        """
        Apply all bands in the chain to the audio file.
        Each band is applied as a biquad IIR filter.
        """
        if self._bypassed:
            print("[FilterChain] Bypassed — no filters applied.")
            return

        if not self.filters:
            print("[FilterChain] No filters in chain.")
            return

        print(f"[FilterChain] Applying {len(self.filters)} EQ band(s)...")

        audio_data  = self.audio_file_ref._audio_data if hasattr(self, 'audio_file_ref') else audio_file._audio_data
        audio_data  = audio_file._audio_data.copy()
        sample_rate = audio_file.sampleRate

        for band in self.filters:
            print(f"  Applying: {band}")
            b, a = self._design_filter(band, sample_rate)
            if b is None:
                continue
            for ch in range(audio_data.shape[0]):
                audio_data[ch] = signal.filtfilt(b, a, audio_data[ch])

        # Safety clip after all filters applied
        audio_file._audio_data = np.clip(audio_data, -1.0, 1.0)
        print("[FilterChain] All bands applied successfully.")

    def _design_filter(self, band: Band, sample_rate: int):
        """
        Design biquad filter coefficients using Audio EQ Cookbook formulas.
        Returns (b, a) coefficients for scipy.signal.filtfilt
        """
        f0 = band.freq
        dB = band.gain
        Q  = band.Q
        fs = sample_rate

        # Safety check
        if f0 <= 0 or f0 >= fs / 2:
            print(f"  [Warning] Frequency {f0}Hz out of range, skipping.")
            return None, None

        A  = 10 ** (dB / 40.0)          # sqrt(10^(dBgain/20))
        w0 = 2 * np.pi * f0 / fs
        cw = np.cos(w0)
        sw = np.sin(w0)

        if band.type == "peak":
            # Peaking EQ filter
            alpha = sw / (2 * Q)
            b0 =  1 + alpha * A
            b1 = -2 * cw
            b2 =  1 - alpha * A
            a0 =  1 + alpha / A
            a1 = -2 * cw
            a2 =  1 - alpha / A

        elif band.type == "lowshelf":
            # Low shelving filter
            alpha = sw / 2 * np.sqrt((A + 1/A) * (1/Q - 1) + 2)
            b0 =      A * ((A+1) - (A-1)*cw + 2*np.sqrt(A)*alpha)
            b1 =  2 * A * ((A-1) - (A+1)*cw)
            b2 =      A * ((A+1) - (A-1)*cw - 2*np.sqrt(A)*alpha)
            a0 =           (A+1) + (A-1)*cw + 2*np.sqrt(A)*alpha
            a1 =     -2 * ((A-1) + (A+1)*cw)
            a2 =           (A+1) + (A-1)*cw - 2*np.sqrt(A)*alpha

        elif band.type == "highshelf":
            # High shelving filter
            alpha = sw / 2 * np.sqrt((A + 1/A) * (1/Q - 1) + 2)
            b0 =       A * ((A+1) + (A-1)*cw + 2*np.sqrt(A)*alpha)
            b1 = -2 * A  * ((A-1) + (A+1)*cw)
            b2 =       A * ((A+1) + (A-1)*cw - 2*np.sqrt(A)*alpha)
            a0 =            (A+1) - (A-1)*cw + 2*np.sqrt(A)*alpha
            a1 =      2 * ((A-1) - (A+1)*cw)
            a2 =            (A+1) - (A-1)*cw - 2*np.sqrt(A)*alpha

        elif band.type == "lowpass":
            alpha = sw / (2 * Q)
            b0 = (1 - cw) / 2
            b1 =  1 - cw
            b2 = (1 - cw) / 2
            a0 =  1 + alpha
            a1 = -2 * cw
            a2 =  1 - alpha

        elif band.type == "highpass":
            alpha = sw / (2 * Q)
            b0 =  (1 + cw) / 2
            b1 = -(1 + cw)
            b2 =  (1 + cw) / 2
            a0 =   1 + alpha
            a1 =  -2 * cw
            a2 =   1 - alpha

        else:
            print(f"  [Warning] Unknown filter type: {band.type}, skipping.")
            return None, None

        # Normalize by a0
        b = np.array([b0/a0, b1/a0, b2/a0])
        a = np.array([1.0,   a1/a0, a2/a0])

        return b, a

    def printChain(self):
        """Print all bands in the chain."""
        if not self.filters:
            print("[FilterChain] Empty — no bands added yet.")
            return
        print(f"\n[FilterChain] {len(self.filters)} band(s):")
        for i, band in enumerate(self.filters):
            print(f"  Band {i+1}: {band}")