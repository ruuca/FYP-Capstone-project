import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import signal
from scipy.io import wavfile


class Visualizer:
    """Generates comparison graphs between original and enhanced audio.
    Shows frequency response, band energy, and instrument enhancement."""

    def __init__(self):
        self.style()

    def style(self):
        """Set up consistent dark theme for all plots."""
        plt.rcParams.update({
            "figure.facecolor":  "#1a1a2e",
            "axes.facecolor":    "#16213e",
            "axes.edgecolor":    "#0f3460",
            "axes.labelcolor":   "#e0e0e0",
            "axes.titlecolor":   "#e0e0e0",
            "xtick.color":       "#e0e0e0",
            "ytick.color":       "#e0e0e0",
            "text.color":        "#e0e0e0",
            "grid.color":        "#0f3460",
            "grid.linewidth":    0.8,
            "font.family":       "sans-serif",
            "font.size":         10,
        })

    def generateComparison(self, original_path: str, enhanced_path: str,
                            analysis_results: dict, output_path: str = "comparison.png"):
        print("\n[Visualizer] Generating comparison graph...")

        # Load both files
        orig_mono, sr = self._load_wav(original_path)
        enh_mono, _   = self._load_wav(enhanced_path)

        # Compute power spectral density
        freqs_o, psd_o = signal.welch(orig_mono, sr, nperseg=8192)
        freqs_e, psd_e = signal.welch(enh_mono,  sr, nperseg=8192)

        # Convert to dB
        psd_o_db = 10 * np.log10(psd_o + 1e-12)
        psd_e_db = 10 * np.log10(psd_e + 1e-12)

        # Smooth for cleaner display
        from scipy.ndimage import uniform_filter1d
        psd_o_db = uniform_filter1d(psd_o_db, size=30)
        psd_e_db = uniform_filter1d(psd_e_db, size=30)

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("JavenEQ — Audio Enhancement Comparison",
                     fontsize=16, fontweight="bold", color="#e94560", y=0.98)

        # -------------------------------------------------------
        # Panel 1: Frequency Spectrum
        # -------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.set_xscale("log")
        ax1.plot(freqs_o, psd_o_db, color="#4a9eff", linewidth=1.2,
                 label="Original", alpha=0.85)
        ax1.plot(freqs_e, psd_e_db, color="#e94560", linewidth=1.2,
                 label="Enhanced", alpha=0.85)
        ax1.fill_between(freqs_e, psd_o_db, psd_e_db,
                         where=(psd_e_db > psd_o_db),
                         alpha=0.15, color="#e94560", label="Boosted")
        ax1.fill_between(freqs_e, psd_o_db, psd_e_db,
                         where=(psd_e_db < psd_o_db),
                         alpha=0.15, color="#4a9eff", label="Cut")
        ax1.set_xlim(20, 20000)
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Power (dB)")
        ax1.set_title("Frequency Spectrum")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{int(x)}Hz" if x < 1000 else f"{int(x/1000)}kHz"))
        # Mark key frequency points
        for freq, label in [(80,"Bass"), (250,"Low-mid"),
                             (2000,"Presence"), (8000,"Air")]:
            ax1.axvline(x=freq, color="#ffdd57", linewidth=0.6, linestyle="--", alpha=0.5)
            ax1.text(freq, ax1.get_ylim()[0]+2, label, fontsize=7,
                     color="#ffdd57", rotation=90, alpha=0.7)

        # -------------------------------------------------------
        # Panel 2: Band Energy Comparison
        # -------------------------------------------------------
        ax2 = axes[0, 1]
        band_labels = ["Sub\n20-80Hz", "Bass\n80-250Hz", "Low-mid\n250-500Hz",
                       "Mid\n500-2kHz", "High-mid\n2-6kHz", "High\n6-20kHz"]
        band_ranges = [(20,80),(80,250),(250,500),(500,2000),(2000,6000),(6000,20000)]
        total_o = np.sum(psd_o) + 1e-12
        total_e = np.sum(psd_e) + 1e-12

        orig_pcts = []
        enh_pcts  = []
        for lo, hi in band_ranges:
            mask = (freqs_o >= lo) & (freqs_o < hi)
            orig_pcts.append(float(np.sum(psd_o[mask]))/total_o*100)
            enh_pcts.append(float(np.sum(psd_e[mask]))/total_e*100)

        x    = np.arange(len(band_labels))
        w    = 0.35
        bars1 = ax2.bar(x - w/2, orig_pcts, w, label="Original",
                        color="#4a9eff", alpha=0.85)
        bars2 = ax2.bar(x + w/2, enh_pcts,  w, label="Enhanced",
                        color="#e94560", alpha=0.85)
        ax2.set_xticks(x)
        ax2.set_xticklabels(band_labels, fontsize=8)
        ax2.set_ylabel("Energy (%)")
        ax2.set_title("Band Energy Distribution")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis="y")
        for bar in bars1:
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                     f"{bar.get_height():.1f}%", ha="center", va="bottom",
                     fontsize=7, color="#4a9eff")
        for bar in bars2:
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                     f"{bar.get_height():.1f}%", ha="center", va="bottom",
                     fontsize=7, color="#e94560")

        # -------------------------------------------------------
        # Panel 3: Vocal Range Comparison
        # -------------------------------------------------------
        ax3 = axes[1, 0]
        vocal_labels = ["Warmth\n200-300Hz", "Body\n300-800Hz",
                        "Clarity\n800Hz-2kHz", "Presence\n2-4kHz",
                        "Sibilance\n6-8kHz", "Air\n10-14kHz"]
        vocal_ranges = [(200,300),(300,800),(800,2000),(2000,4000),(6000,8000),(10000,14000)]

        orig_v, enh_v = [], []
        for lo, hi in vocal_ranges:
            mask = (freqs_o >= lo) & (freqs_o < hi)
            orig_v.append(float(np.sum(psd_o[mask]))/total_o*100)
            enh_v.append(float(np.sum(psd_e[mask]))/total_e*100)

        x = np.arange(len(vocal_labels))
        ax3.bar(x - w/2, orig_v, w, label="Original", color="#4a9eff", alpha=0.85)
        ax3.bar(x + w/2, enh_v,  w, label="Enhanced", color="#e94560", alpha=0.85)
        ax3.set_xticks(x)
        ax3.set_xticklabels(vocal_labels, fontsize=8)
        ax3.set_ylabel("Energy (%)")
        ax3.set_title("🎤 Vocal Range Comparison")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis="y")

        # -------------------------------------------------------
        # Panel 4: Instrument Range Comparison
        # -------------------------------------------------------
        ax4 = axes[1, 1]
        instr_labels = ["Kick\n60-80Hz", "Bass\n80-200Hz", "Guitar\nbody\n500Hz-1k",
                        "Guitar\nbite\n3-5kHz", "Snare\n2-3kHz",
                        "Cymbal\n8-12kHz"]
        instr_ranges = [(60,80),(80,200),(500,1000),(3000,5000),(2000,3000),(8000,12000)]

        orig_i, enh_i = [], []
        for lo, hi in instr_ranges:
            mask = (freqs_o >= lo) & (freqs_o < hi)
            orig_i.append(float(np.sum(psd_o[mask]))/total_o*100)
            enh_i.append(float(np.sum(psd_e[mask]))/total_e*100)

        x = np.arange(len(instr_labels))
        ax4.bar(x - w/2, orig_i, w, label="Original", color="#4a9eff", alpha=0.85)
        ax4.bar(x + w/2, enh_i,  w, label="Enhanced", color="#e94560", alpha=0.85)
        ax4.set_xticks(x)
        ax4.set_xticklabels(instr_labels, fontsize=8)
        ax4.set_ylabel("Energy (%)")
        ax4.set_title("🎸 Instrument Range Comparison")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis="y")

        # -------------------------------------------------------
        # Era label + detected instruments annotation
        # -------------------------------------------------------
        era_label  = analysis_results.get("era", {}).get("era_label", "")
        detected   = analysis_results.get("instruments", {}).get("instruments_detected", {})
        instr_str  = ", ".join(detected.keys()) if detected else "None"
        vocal_str  = analysis_results.get("vocals", {}).get("vocal_presence", "")

        fig.text(0.01, 0.01,
                 f"Era: {era_label}   |   Instruments: {instr_str}   |   {vocal_str}",
                 fontsize=8, color="#888888")

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

        # Save PNG copy
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[Visualizer] Graph saved to: {output_path}")

        # Show as interactive popup window
        print("[Visualizer] Opening comparison graph (close window to continue)...")
        plt.show(block=True)

        return output_path

    def _load_wav(self, path):
        """Load a wav file and return mono float array."""
        sr, data = wavfile.read(path)
        data = data.astype(np.float32)
        if data.max() > 1.0:
            data = data / 32768.0
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        return data, sr