"""
JavenEQ — Main Application UI
PyQt6 desktop app with seekable playback and embedded comparison graph.
Matches the MainApp class in the class diagram.
"""

import sys
import os
import time
import numpy as np
from scipy.io import wavfile
from scipy import signal as scipy_signal

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QTextEdit,
    QFrame, QSizePolicy, QSlider, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
import matplotlib as mpl

import sounddevice as sd


# ================================================================
# Processing Thread
# ================================================================
class ProcessingThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, audio_path: str):
        super().__init__()
        self.audio_path = audio_path

    def run(self):
        try:
            from audio_file      import AudioFile
            from audio_processor import AudioProcessor
            from ai_eq_engine    import AIEQEngine

            self.progress.emit("Loading audio file...")
            audio_file = AudioFile()
            audio_file.load(self.audio_path)

            # Quick normalize before era detection
            self.progress.emit("Running era detection...")
            rms = np.sqrt(np.mean(audio_file.get_data() ** 2))
            if rms > 0:
                gain = np.clip(0.10 / rms, 0.1, 10.0)
                audio_file._audio_data = np.clip(audio_file.get_data() * gain, -1.0, 1.0)

            engine   = AIEQEngine()
            mono     = audio_file.get_mono()
            sr       = audio_file.sampleRate
            era_info = engine.detectEra(mono, sr)
            profile  = engine.getEraProfile()
            self.progress.emit(f"Detected: {era_info['era_label']}")

            # Reload and full preprocess
            self.progress.emit("Reloading original audio...")
            audio_file = AudioFile()
            audio_file.load(self.audio_path)
            processor = AudioProcessor(audio_file)
            processor.setEraProfile(profile)

            self.progress.emit("Reducing noise...")
            processor.reduceNoise()
            self.progress.emit("Normalizing loudness...")
            processor.normalize()
            self.progress.emit("Applying brightness boost...")
            processor.brightnessBoost()

            # --- Generate output filenames based on input filename ---
            # e.g. "Queen - Don't Stop Me Now.mp3" →
            #   preprocessed/Queen - Dont Stop Me Now_preprocessed.wav
            #   enhanced output/Queen - Dont Stop Me Now_enhanced.wav
            import re
            raw_stem  = os.path.splitext(os.path.basename(self.audio_path))[0]
            # Sanitize: remove characters that cause issues in filenames
            safe_stem = re.sub(r'[<>:"/\\|?*]', '', raw_stem).strip()
            # Limit length to avoid overly long filenames
            safe_stem = safe_stem[:60] if len(safe_stem) > 60 else safe_stem

            os.makedirs("preprocessed", exist_ok=True)
            os.makedirs("enhanced output", exist_ok=True)
            preprocessed_path = f"preprocessed/{safe_stem}_preprocessed.wav"
            enhanced_path     = f"enhanced output/{safe_stem}_enhanced.wav"

            audio_file.save(preprocessed_path)

            self.progress.emit("Analysing audio...")
            results   = engine.analyzeAudio(audio_file)
            final_era = results["era"]

            self.progress.emit("Generating adaptive EQ curve...")
            mono_proc  = audio_file.get_mono()
            eq_profile = engine.generateEQCurve(results,
                                                  mono_audio=mono_proc,
                                                  sample_rate=sr)

            self.progress.emit(f"Applying {len(eq_profile.bands)} EQ bands...")
            eq_profile.apply(audio_file)

            self.progress.emit("Compressing...")
            processor.compress()

            audio_file.save(enhanced_path)

            self.progress.emit("Done!")
            self.finished.emit({
                "era":               final_era,
                "results":           results,
                "preprocessed_path": preprocessed_path,
                "enhanced_path":     enhanced_path,
                "eq_bands":          len(eq_profile.bands),
            })

        except Exception as ex:
            import traceback
            self.error.emit(f"{str(ex)}\n\n{traceback.format_exc()}")


# ================================================================
# Comparison Graph Widget (UPDATED - 2x3 Layout)
# ================================================================
class ComparisonGraph(FigureCanvas):
    def __init__(self, parent=None):
        # Make the figure wider to fit 3 columns comfortably
        self.fig = Figure(figsize=(12, 6), facecolor="#1a1a2e")
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._placeholder()

    def _placeholder(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#16213e")
        ax.text(0.5, 0.5, "Load and process a song\nto see the comparison graph",
                ha="center", va="center", fontsize=13,
                color="#555577", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor("#0f3460")
        self.draw()

    def update_graph(self, orig_path, enh_path, analysis_results, save_path=None):
        self.fig.clear()
        for k, v in {
            "axes.facecolor": "#16213e", "axes.edgecolor": "#0f3460",
            "axes.labelcolor": "#e0e0e0", "axes.titlecolor": "#e0e0e0",
            "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
            "text.color": "#e0e0e0", "grid.color": "#0f3460",
        }.items():
            mpl.rcParams[k] = v

        def load_mono(path):
            sr, d = wavfile.read(path)
            d = d.astype(np.float32)
            if d.max() > 1.0: d /= 32768.0
            return (np.mean(d, axis=1) if d.ndim > 1 else d), sr

        from scipy.ndimage import uniform_filter1d
        import librosa
        
        om, sr = load_mono(orig_path)
        em, _  = load_mono(enh_path)
        fo, po = scipy_signal.welch(om, sr, nperseg=8192)
        fe, pe = scipy_signal.welch(em, sr, nperseg=8192)
        po_db  = uniform_filter1d(10*np.log10(po+1e-12), size=30)
        pe_db  = uniform_filter1d(10*np.log10(pe+1e-12), size=30)
        to = np.sum(po)+1e-12
        te = np.sum(pe)+1e-12

        timeline = analysis_results.get("timeline_top", [])
        duration = len(om) / sr
        valid_t = [t for t in timeline if 0 <= t <= duration]

        from matplotlib.gridspec import GridSpec
        self.fig.set_size_inches(14, 7)
        gs = GridSpec(2, 3, figure=self.fig, height_ratios=[1, 1], width_ratios=[1, 1, 1])
        
        ax_spectrum = self.fig.add_subplot(gs[0, 0])
        ax_band     = self.fig.add_subplot(gs[0, 1])
        ax_vocal    = self.fig.add_subplot(gs[0, 2])
        ax_instrument = self.fig.add_subplot(gs[1, 0])
        ax_timeline   = self.fig.add_subplot(gs[1, 1:])
        
        self.fig.suptitle("JavenEQ — Audio Enhancement Comparison",
                          fontsize=14, fontweight="bold", color="#e94560", y=0.99)

        w = 0.35

        # ==============================================================
        # Panel 1: Frequency Spectrum
        # ==============================================================
        ax = ax_spectrum
        ax.set_xscale("log")
        ax.plot(fo, po_db, "#FFB74D", lw=1.2, label="Original", alpha=0.85)
        ax.plot(fe, pe_db, "#00E5FF", lw=1.2, label="Enhanced", alpha=0.85)
        ax.fill_between(fe, po_db, pe_db, where=(pe_db>po_db), alpha=0.15, color="#00E5FF")
        ax.fill_between(fe, po_db, pe_db, where=(pe_db<po_db), alpha=0.15, color="#FFB74D")
        ax.set_xlim(20, 20000)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Frequency (Hz)", fontsize=8)
        ax.set_ylabel("Power (dB)", fontsize=8)
        ax.set_title("Frequency Spectrum", fontsize=9)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x,_: f"{int(x)}Hz" if x<1000 else f"{int(x/1000)}kHz"))
        for f,l in [(80,"Bass"),(250,"Low-mid"),(2000,"Presence"),(8000,"Air")]:
            ax.axvline(x=f, color="#ffdd57", lw=0.6, ls="--", alpha=0.5)
            ax.text(f, ax.get_ylim()[0]+1, l, fontsize=6, color="#ffdd57",
                    rotation=90, alpha=0.7)

        # ==============================================================
        # Panel 2: Band Energy Distribution
        # ==============================================================
        ax = ax_band
        bl = ["Sub\n20-80Hz","Bass\n80-250Hz","Low-mid\n250-500Hz",
              "Mid\n500-2kHz","High-mid\n2-6kHz","High\n6-20kHz"]
        br = [(20,80),(80,250),(250,500),(500,2000),(2000,6000),(6000,20000)]
        op = [np.sum(po[(fo>=lo)&(fo<hi)])/to*100 for lo,hi in br]
        ep = [np.sum(pe[(fe>=lo)&(fe<hi)])/te*100 for lo,hi in br]
        x = np.arange(len(bl))
        b1 = ax.bar(x-w/2, op, w, label="Original", color="#FFB74D", alpha=0.85)
        b2 = ax.bar(x+w/2, ep, w, label="Enhanced", color="#00E5FF", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(bl, fontsize=6)
        ax.set_ylabel("Energy (%)", fontsize=8)
        ax.set_title("Band Energy Distribution", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")
        for bar in list(b1)+list(b2):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom",
                    fontsize=6, color=bar.get_facecolor())

        # ==============================================================
        # Panel 3: Vocal Range Comparison
        # ==============================================================
        ax = ax_vocal
        vl = ["Warmth\n200-300Hz","Body\n300-800Hz","Clarity\n800Hz-2kHz",
              "Presence\n2-4kHz","Sibilance\n6-8kHz","Air\n10-14kHz"]
        vr = [(200,300),(300,800),(800,2000),(2000,4000),(6000,8000),(10000,14000)]
        ov = [np.sum(po[(fo>=lo)&(fo<hi)])/to*100 for lo,hi in vr]
        ev = [np.sum(pe[(fe>=lo)&(fe<hi)])/te*100 for lo,hi in vr]
        xv = np.arange(len(vl))
        ax.bar(xv-w/2, ov, w, label="Original", color="#FFB74D", alpha=0.85)
        ax.bar(xv+w/2, ev, w, label="Enhanced", color="#00E5FF", alpha=0.85)
        ax.set_xticks(xv)
        ax.set_xticklabels(vl, fontsize=6)
        ax.set_ylabel("Energy (%)", fontsize=8)
        ax.set_title("🎤 Vocal Range", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

        # ==============================================================
        # Panel 4: Instrument Range Comparison
        # ==============================================================
        ax = ax_instrument
        il = ["Kick\n60-80Hz", "Bass\n80-200Hz", "Guitar\nbody\n500-1k",
              "Guitar\nbite\n3-5kHz", "Snare\n2-3kHz", "Cymbal\n8-12kHz"]
        ir = [(60,80), (80,200), (500,1000), (3000,5000), (2000,3000), (8000,12000)]
        oi = [np.sum(po[(fo>=lo)&(fo<hi)])/to*100 for lo,hi in ir]
        ei = [np.sum(pe[(fe>=lo)&(fe<hi)])/te*100 for lo,hi in ir]
        xi = np.arange(len(il))
        ax.bar(xi-w/2, oi, w, label="Original", color="#FFB74D", alpha=0.85)
        ax.bar(xi+w/2, ei, w, label="Enhanced", color="#00E5FF", alpha=0.85)
        ax.set_xticks(xi)
        ax.set_xticklabels(il, fontsize=6)
        ax.set_ylabel("Energy (%)", fontsize=8)
        ax.set_title("🎸 Instrument Range", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

                # ==============================================================
        # Panel 5: Song Timeline - TWO WAVES (No Fill, All Shifts Visible)
        # ==============================================================
        ax = ax_timeline
        
        # Compute RMS for BOTH original and enhanced audio
        frame_length = 2048
        hop_length = 512
        
        # Original audio RMS (Light Cyan)
        rms_orig = librosa.feature.rms(y=om, frame_length=frame_length, hop_length=hop_length)[0]
        rms_times = np.arange(len(rms_orig)) * hop_length / sr
        
        # Enhanced audio RMS (Warm Amber)
        rms_enh = librosa.feature.rms(y=em, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Plot Original volume (Light Cyan - NO FILL to avoid grey overlap)
        ax.plot(rms_times, rms_orig, color="#4a9eff", linewidth=1.8, 
                label="Original Signal level", alpha=0.85)
        
        # Plot Enhanced volume (Warm Amber - NO FILL)
        ax.plot(rms_times, rms_enh, color="#FFB74D", linewidth=2.0, 
                label="Enhanced Signal level", alpha=0.9)
        
        # Add timeline change points
        if timeline:
            valid_t = [t for t in timeline if 0 <= t <= duration]
            for t in valid_t:
                ax.axvline(x=t, color="#e94560", linestyle=":", linewidth=1.5, alpha=0.8)
        
        # Clean formatting
        ax.set_xlabel("Time (MM:SS)", fontsize=8)
        ax.set_ylabel("Volume", fontsize=8)
        ax.set_title("📊 Song Timeline – Structural Shifts", fontsize=9)
        ax.grid(True, alpha=0.2, axis="x")
        
        # =============================================================
        # FIX: Show ALL timestamps on x-axis (no cutoff)
        # =============================================================
        if timeline and len(valid_t) > 0:
            # Set x-axis ticks to ALL valid timestamps
            ax.set_xticks(valid_t)
            time_labels = [f"{int(t // 60)}:{int(t % 60):02d}" for t in valid_t]
            ax.set_xticklabels(time_labels, fontsize=6, rotation=45, ha="right")
        else:
            # Fallback: automatic ticks
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda x, _: f"{int(x // 60)}:{int(x % 60):02d}"))
        
        # Legend
        if timeline and len(valid_t) > 0:
            ax.plot([], [], color="#e94560", linestyle=":", linewidth=1.5,
                    label=f"Structural shifts ({len(valid_t)})")
            ax.legend(fontsize=7, loc="upper right")
        else:
            ax.legend(fontsize=7, loc="upper right")
        
        # Set x-axis limits to show the full song
        ax.set_xlim(0, duration)

        # ==============================================================
        # Bottom Footer
        # ==============================================================
        era_label = analysis_results.get("era",{}).get("era_label","")
        instr = ", ".join(analysis_results.get("instruments",{})
                         .get("instruments_detected",{}).keys())
        vocal = analysis_results.get("vocals",{}).get("vocal_presence","")
        self.fig.text(0.01, 0.01,
                      f"Era: {era_label}  |  Instruments: {instr}  |  {vocal}  |  {len(valid_t)} major shifts",
                      fontsize=8, color="#666688")

        self.fig.tight_layout(rect=[0, 0.03, 1, 0.97])

        if save_path:
            self.fig.savefig(save_path, dpi=150, bbox_inches="tight",
                             facecolor=self.fig.get_facecolor())
        self.draw()

        # ============================================================
        # Bottom Footer
        # ============================================================
        era_label = analysis_results.get("era",{}).get("era_label","")
        instr = ", ".join(analysis_results.get("instruments",{})
                         .get("instruments_detected",{}).keys())
        vocal = analysis_results.get("vocals",{}).get("vocal_presence","")
        self.fig.text(0.01, 0.01,
                      f"Era: {era_label}  |  Instruments: {instr}  |  {vocal}  |  {len(valid_t)} major shifts",
                      fontsize=8, color="#666688")

        self.fig.tight_layout(rect=[0, 0.03, 1, 0.97])

        if save_path:
            self.fig.savefig(save_path, dpi=150, bbox_inches="tight",
                             facecolor=self.fig.get_facecolor())
        self.draw()


# ================================================================
# Main Application
# ================================================================
class MainApp(QMainWindow):
    """
    JavenEQ Main Application Window.
    Matches the MainApp class in the class diagram.
    """

    def __init__(self):
        super().__init__()

        # State
        self.audio_path        = None
        self.preprocessed_path = None
        self.enhanced_path     = None
        self.analysis_results  = None
        self.processing_thread = None

        # Playback state
        self._current_audio    = None   # numpy array of current audio
        self._current_sr       = None   # sample rate
        self._current_version  = None   # "original" or "enhanced"
        self._total_samples    = 0
        self._play_start_time  = None   # wall clock when play started
        self._play_start_samp  = 0      # sample index where play started
        self._is_playing       = False
        self._user_dragging    = False  # True while user drags slider

        # Timer for seek slider updates
        self._seek_timer = QTimer()
        self._seek_timer.setInterval(200)
        self._seek_timer.timeout.connect(self._tick_seek)

        self.showMainScreen()

    # ──────────────────────────────────────────────────────────────
    # showMainScreen
    # ──────────────────────────────────────────────────────────────
    def showMainScreen(self):
        self.setWindowTitle("JavenEQ — AI Audio Equalizer & Enhancer")
        self.setMinimumSize(1200, 760)
        self.minimumWidth = 1100
        self._apply_theme()

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0,0,0,0)
        root.setSpacing(0)
        root.addWidget(self._build_left_panel(), stretch=0)
        root.addWidget(self._build_right_panel(), stretch=1)
        self.show()

    def _build_left_panel(self):
        panel = QFrame()
        panel.setFixedWidth(330)
        panel.setStyleSheet("background:#16213e; border-right:1px solid #0f3460;")

        L = QVBoxLayout(panel)
        L.setContentsMargins(14, 14, 14, 14)
        L.setSpacing(8)

        # Title
        t = QLabel("JavenEQ")
        t.setFont(QFont("Arial", 22, QFont.Weight.Bold))
        t.setStyleSheet("color:#e94560;")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        L.addWidget(t)

        sub = QLabel("AI Audio Equalizer & Enhancer")
        sub.setFont(QFont("Arial", 9))
        sub.setStyleSheet("color:#888899;")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        L.addWidget(sub)

        L.addWidget(self._hr())

        # File label
        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setStyleSheet("color:#aaaacc; font-size:10px;")
        self.lbl_file.setAlignment(Qt.AlignmentFlag.AlignCenter)
        L.addWidget(self.lbl_file)

        # Buttons
        self.btn_load = self._btn("📂  Load Audio File", "#0f3460", "#1a4a80")
        self.btn_load.clicked.connect(self.loadAudioFile)
        L.addWidget(self.btn_load)

        self.btn_enhance = self._btn("⚡  Enhance Audio", "#e94560", "#c73250")
        self.btn_enhance.setEnabled(False)
        self.btn_enhance.clicked.connect(self._run_pipeline)
        L.addWidget(self.btn_enhance)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(
            "QProgressBar{background:#0f3460;border-radius:4px;height:6px;}"
            "QProgressBar::chunk{background:#e94560;border-radius:4px;}")
        L.addWidget(self.progress_bar)

        # Status
        self.lbl_status = QLabel("Ready — load a song to begin")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color:#666688; font-size:9px;")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        L.addWidget(self.lbl_status)

        L.addWidget(self._hr())

        # Era badge
        era_hdr = QLabel("DETECTED ERA")
        era_hdr.setStyleSheet("color:#555577; font-size:9px; font-weight:bold;")
        L.addWidget(era_hdr)

        self.lbl_era = QLabel("—")
        self.lbl_era.setWordWrap(True)
        self.lbl_era.setStyleSheet(
            "color:#e94560; font-size:11px; font-weight:bold;"
            "background:#0f3460; border-radius:6px; padding:6px;")
        L.addWidget(self.lbl_era)

        L.addWidget(self._hr())

        # =============================================================
        # ANALYSIS REPORT - BIGGER (stretch=2)
        # =============================================================
        rpt_hdr = QLabel("ANALYSIS REPORT")
        rpt_hdr.setStyleSheet("color:#555577; font-size:9px; font-weight:bold;")
        L.addWidget(rpt_hdr)

        self.txt_report = QTextEdit()
        self.txt_report.setReadOnly(True)
        self.txt_report.setStyleSheet(
            "QTextEdit{background:#0d1b2a; color:#aaaacc;"
            "border:1px solid #0f3460; border-radius:6px;"
            "font-family:Consolas,monospace; font-size:9px; padding:6px;}")
        self.txt_report.setPlaceholderText("Analysis results will appear here...")
        L.addWidget(self.txt_report, stretch=2)  # BIGGER - was stretch=1

        L.addWidget(self._hr())

        # =============================================================
        # MEDIA PLAYER - MINIMIZED (smaller, more compact)
        # =============================================================
        player_card = QFrame()
        player_card.setMinimumHeight(170)  # Reduced from 220
        player_card.setMaximumHeight(185)  # Reduced from 240
        player_card.setStyleSheet(
            "QFrame{background:#0d1b2a; border-radius:8px;"
            "border:1px solid #1a3a5c; padding:4px;}")
        player_layout = QVBoxLayout(player_card)
        player_layout.setContentsMargins(8, 6, 8, 6)  # Tighter margins
        player_layout.setSpacing(3)  # Reduced spacing

        # Now Playing
        now_playing_hdr = QLabel("♫  NOW PLAYING")
        now_playing_hdr.setStyleSheet("color:#555577; font-size:7px; font-weight:bold; border:none;")
        player_layout.addWidget(now_playing_hdr)

        # Song title
        self.lbl_song_title = QLabel("No song loaded")
        self.lbl_song_title.setWordWrap(True)
        self.lbl_song_title.setFixedHeight(16)  # Smaller
        self.lbl_song_title.setStyleSheet("color:#e0e0e0; font-size:9px; font-weight:bold; border:none;")
        player_layout.addWidget(self.lbl_song_title)

        # Version toggle - slightly smaller
        version_row = QHBoxLayout()
        version_row.setSpacing(6)
        version_row.addStretch()

        self.btn_ver_orig = QPushButton("Original")
        self.btn_ver_orig.setFixedSize(85, 28)  # Slightly smaller
        self.btn_ver_orig.setCheckable(True)
        self.btn_ver_orig.setChecked(True)
        self.btn_ver_orig.clicked.connect(lambda: self._set_version("original"))
        self.btn_ver_orig.setStyleSheet(self._tab_style())

        self.btn_ver_enh = QPushButton("Enhanced")
        self.btn_ver_enh.setFixedSize(85, 28)  # Slightly smaller
        self.btn_ver_enh.setCheckable(True)
        self.btn_ver_enh.clicked.connect(lambda: self._set_version("enhanced"))
        self.btn_ver_enh.setStyleSheet(self._tab_style())

        version_row.addWidget(self.btn_ver_orig)
        version_row.addWidget(self.btn_ver_enh)
        version_row.addStretch()
        player_layout.addLayout(version_row)

        # Seek bar - compact
        time_row = QHBoxLayout()
        time_row.setSpacing(3)

        self.lbl_cur_time = QLabel("0:00")
        self.lbl_cur_time.setFixedWidth(26)
        self.lbl_cur_time.setStyleSheet("color:#888899; font-size:8px; border:none;")
        time_row.addWidget(self.lbl_cur_time)

        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setMinimumHeight(14)
        self.seek_slider.setRange(0, 1000)
        self.seek_slider.setValue(0)
        self.seek_slider.setEnabled(False)
        self.seek_slider.setStyleSheet(
            "QSlider::groove:horizontal{background:#1a3a5c;height:3px;border-radius:2px;}"
            "QSlider::handle:horizontal{background:#e94560;width:10px;height:10px;"
            "border-radius:5px;margin:-3px 0;}"
            "QSlider::sub-page:horizontal{background:#e94560;border-radius:2px;}")
        self.seek_slider.sliderPressed.connect(self._seek_pressed)
        self.seek_slider.sliderReleased.connect(self._seek_released)
        self.seek_slider.sliderMoved.connect(self._seek_moved)
        time_row.addWidget(self.seek_slider)

        self.lbl_tot_time = QLabel("0:00")
        self.lbl_tot_time.setFixedWidth(26)
        self.lbl_tot_time.setStyleSheet("color:#888899; font-size:8px; border:none;")
        self.lbl_tot_time.setAlignment(Qt.AlignmentFlag.AlignRight)
        time_row.addWidget(self.lbl_tot_time)

        player_layout.addLayout(time_row)

        # Transport controls - MINIMIZED (smaller buttons)
        transport_row = QHBoxLayout()
        transport_row.setSpacing(6)
        transport_row.addStretch()

        self.btn_rewind = QPushButton("⏮")
        self.btn_rewind.setFixedSize(36, 36)  # Smaller
        self.btn_rewind.setStyleSheet(self._transport_style())
        self.btn_rewind.setEnabled(False)
        self.btn_rewind.clicked.connect(self._rewind)
        transport_row.addWidget(self.btn_rewind)

        self.btn_play_pause = QPushButton("▶")
        self.btn_play_pause.setFixedSize(44, 44)  # Smaller (was 50)
        self.btn_play_pause.setStyleSheet(self._transport_style(primary=True))
        self.btn_play_pause.setEnabled(False)
        self.btn_play_pause.clicked.connect(self._toggle_play_pause)
        transport_row.addWidget(self.btn_play_pause)

        self.btn_forward = QPushButton("⏭")
        self.btn_forward.setFixedSize(36, 36)  # Smaller
        self.btn_forward.setStyleSheet(self._transport_style())
        self.btn_forward.setEnabled(False)
        self.btn_forward.clicked.connect(self._fast_forward)
        transport_row.addWidget(self.btn_forward)

        transport_row.addStretch()
        player_layout.addLayout(transport_row)

        # Volume control - compact
        vol_row = QHBoxLayout()
        vol_row.addStretch()

        lbl_vol = QLabel("🔊")
        lbl_vol.setFixedWidth(18)
        lbl_vol.setStyleSheet("color:#aaaacc; font-size:9px; border:none;")
        vol_row.addWidget(lbl_vol)

        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setFixedWidth(70)  # Smaller
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(80)
        self.vol_slider.setStyleSheet(
            "QSlider::groove:horizontal{background:#1a3a5c;height:3px;border-radius:2px;}"
            "QSlider::handle:horizontal{background:#aaaacc;width:8px;height:8px;"
            "border-radius:4px;margin:-2px 0;}"
            "QSlider::sub-page:horizontal{background:#4a9eff;border-radius:2px;}")
        self.vol_slider.valueChanged.connect(self._on_volume_changed)
        vol_row.addWidget(self.vol_slider)

        self.lbl_vol_pct = QLabel("80%")
        self.lbl_vol_pct.setFixedWidth(26)
        self.lbl_vol_pct.setStyleSheet("color:#888899; font-size:8px; border:none;")
        vol_row.addWidget(self.lbl_vol_pct)

        vol_row.addStretch()
        player_layout.addLayout(vol_row)

        L.addWidget(player_card)

        # Aliases
        self.btn_orig = self.btn_ver_orig
        self.btn_enh = self.btn_ver_enh
        self.btn_stop = self.btn_play_pause

        L.addWidget(self._hr())

        # Exit
        btn_exit = self._btn("✕  Exit", "#2a2a3a", "#333344", h=32)
        btn_exit.clicked.connect(self.exit)
        L.addWidget(btn_exit)

        # ── Footer ──
        footer = QFrame()
        footer.setStyleSheet(
            "QFrame{background:#0d1b2a; border-radius:6px;"
            "border:1px solid #0f3460; padding:6px; margin-top:6px;}")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(10, 6, 10, 6)
        footer_layout.setSpacing(12)

        self.lbl_buffer = QLabel("🔬 FFT Size: 8192")
        self.lbl_buffer.setStyleSheet(
            "color:#4a9eff; font-size:8px; font-family:monospace; border:none;")
        footer_layout.addWidget(self.lbl_buffer)

        badge = QLabel("🔒 100% LOCAL  ◆ NO CLOUD")
        badge.setStyleSheet(
            "background:#1a4a3a; color:#00e676; font-size:7px; font-weight:bold;"
            "border-radius:4px; padding:3px 10px; border:1px solid #00e676;")
        footer_layout.addWidget(badge)

        footer_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        L.addWidget(footer)

        return panel

    def _build_right_panel(self):
        panel = QFrame()
        panel.setStyleSheet("background:#1a1a2e;")
        L = QVBoxLayout(panel)
        L.setContentsMargins(8,8,8,8)
        self.graph = ComparisonGraph(panel)
        L.addWidget(self.graph)
        return panel

    # ──────────────────────────────────────────────────────────────
    # loadAudioFile
    # ──────────────────────────────────────────────────────────────
    def loadAudioFile(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Audio File", "",
            "Audio Files (*.mp3 *.wav *.flac *.ogg *.m4a);;All Files (*)")
        if not path:
            return
        self.audio_path = path
        self.lbl_file.setText(f"📄 {os.path.basename(path)}")
        self.lbl_status.setText("File loaded — click Enhance Audio")
        self.btn_enhance.setEnabled(True)
        self.lbl_era.setText("—")
        self.txt_report.clear()
        self.graph._placeholder()
        # Reset media player
        sd.stop()
        self._is_playing = False
        self._seek_timer.stop()
        self.seek_slider.setValue(0)
        self.seek_slider.setEnabled(False)
        self.lbl_cur_time.setText("0:00")
        self.lbl_tot_time.setText("0:00")
        self.lbl_song_title.setText(os.path.splitext(os.path.basename(path))[0][:40])
        self.btn_play_pause.setText("▶")
        self.btn_play_pause.setEnabled(False)
        self.btn_rewind.setEnabled(False)
        self.btn_forward.setEnabled(False)
        self.btn_ver_orig.setEnabled(False)
        self.btn_ver_enh.setEnabled(False)

    # ──────────────────────────────────────────────────────────────
    # _run_pipeline
    # ──────────────────────────────────────────────────────────────
    def _run_pipeline(self):
        if not self.audio_path:
            return
        self.btn_enhance.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.btn_orig.setEnabled(False)
        self.btn_enh.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.lbl_status.setText("Processing...")

        self.processing_thread = ProcessingThread(self.audio_path)
        self.processing_thread.progress.connect(
            lambda m: self.lbl_status.setText(f"⚙ {m}"))
        self.processing_thread.finished.connect(self._on_done)
        self.processing_thread.error.connect(self._on_error)
        self.processing_thread.start()
        self._current_safe_stem = None  # will be set when processing finishes

    def _on_done(self, data: dict):
        self.preprocessed_path = data["preprocessed_path"]
        self.enhanced_path     = data["enhanced_path"]
        self.analysis_results  = data["results"]
        self.btn_load.setEnabled(True)
        self.btn_enhance.setEnabled(True)
        self.progress_bar.setVisible(False)

        # Enable media player controls
        self.btn_ver_orig.setEnabled(True)
        self.btn_ver_enh.setEnabled(True)
        self.btn_play_pause.setEnabled(True)
        self.btn_rewind.setEnabled(True)
        self.btn_forward.setEnabled(True)
        self.seek_slider.setEnabled(True)

        # Update song title in media player
        import os
        song_name = os.path.splitext(os.path.basename(self.audio_path))[0]
        # Trim long names for display
        display_name = song_name[:40] + "..." if len(song_name) > 40 else song_name
        self.lbl_song_title.setText(display_name)

        # Set Enhanced as default selected version
        self._set_version("enhanced")

        self.showEnhancedResult(data)

    def _on_error(self, msg: str):
        self.btn_load.setEnabled(True)
        self.btn_enhance.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("❌ Error occurred")
        QMessageBox.critical(self, "Error", msg)

    # ──────────────────────────────────────────────────────────────
    # showEnhancedResult
    # ──────────────────────────────────────────────────────────────
    def showEnhancedResult(self, data: dict):
        r     = data["results"]
        era   = r.get("era",{})
        freq  = r.get("frequencies",{})
        voc   = r.get("vocals",{})
        instr = r.get("instruments",{})

        self.lbl_era.setText(era.get("era_label","Unknown"))
        self.lbl_status.setText(f"✅ Done! {data['eq_bands']} EQ bands applied")

        lines = []
        lines += ["━━ ERA ━━",
                  f"  {era.get('era_label','')}",
                  f"  Rolloff : {era.get('spectral_rolloff',0):.0f} Hz",
                  f"  Dyn Range: {era.get('dynamic_range',0):.1f}",
                  ""]
        lines += ["━━ FREQUENCY BANDS ━━"]
        for band, pct in freq.get("band_energy",{}).items():
            bar = "█" * int(pct/6)
            lines.append(f"  {band:<26} {pct:>4.1f}% {bar}")
        lines += ["","━━ ISSUES ━━"]
        for issue in freq.get("issues",[]):
            lines.append(f"  ⚠ {issue}")
        lines += ["","━━ VOCALS ━━",
                  f"  {voc.get('vocal_presence','')}",
                  f"  Energy: {voc.get('vocal_energy_ratio',0):.3f}",
                  "","━━ INSTRUMENTS ━━",
                  f"  Tempo: {instr.get('tempo','')}"]
        for name, info in instr.get("instruments_detected",{}).items():
            lines.append(f"  [{info['confidence']}] {name}")
        
        # ✅ FIXED: Moved outside the loop
        lines += ["━━ EQ HINTS ━━"]
        for hint in era.get("eq_hints", []):
            lines.append(f"  💡 {hint}")
        
        # <--- TIMELINE SECTION (Shows ALL change points) --->
        lines += ["", "━━ ⏱️ TIMELINE CHANGE POINTS ━━"]
        timeline = r.get("timeline_all", [])  # <-- ALL timestamps for the report
        if timeline:
            # Format all timestamps nicely (e.g., 0:00, 0:45, 1:23)
            formatted = []
            for t in timeline:
                mins = int(t // 60)
                secs = int(t % 60)
                formatted.append(f"{mins}:{secs:02d}")
            
            # Split into lines of ~10 timestamps for better readability
            chunk_size = 10
            chunks = [formatted[i:i+chunk_size] for i in range(0, len(formatted), chunk_size)]
            
            for chunk in chunks:
                lines.append("  " + "  |  ".join(chunk))
            
            lines.append("  (Drag slider to these points to hear the shifts)")
        else:
            lines.append("  No major structural changes detected.")

        self.txt_report.setText("\n".join(lines))

        # Render graph
        self.lbl_status.setText("Rendering graph...")
        QApplication.processEvents()
        try:
            import re
            raw_stem  = os.path.splitext(os.path.basename(self.audio_path))[0]
            safe_stem = re.sub(r'[<>:"/\\|?*]', '', raw_stem).strip()[:60]
            graph_path = f"{safe_stem}_comparison.png"
            self.graph.update_graph(
                self.preprocessed_path,
                self.enhanced_path,
                r,
                save_path=graph_path)
        except Exception as e:
            self.lbl_status.setText(f"Graph error: {e}")
            return

        self.lbl_status.setText(
            f"✅ Done! {data['eq_bands']} EQ bands applied")

    # ──────────────────────────────────────────────────────────────
    # playSong
    # ──────────────────────────────────────────────────────────────
    def playSong(self, version: str = "enhanced"):
        """Play original or enhanced audio from current slider position."""
        path = (self.preprocessed_path if version == "original"
                else self.enhanced_path)
        if not path or not os.path.exists(path):
            self.lbl_status.setText("⚠ File not found")
            return

        try:
            # Load audio if switching version or not loaded yet
            if self._current_version != version or self._current_audio is None:
                sr, data = wavfile.read(path)
                data = data.astype(np.float32)
                if data.max() > 1.0:
                    data /= 32768.0
                self._current_audio   = data
                self._current_sr      = sr
                self._current_version = version
                self._total_samples   = len(data)
                # Update total time label
                total_sec = self._total_samples / sr
                self.lbl_tot_time.setText(self._fmt(total_sec))
                self.seek_slider.setEnabled(True)

            # Calculate start sample from slider position
            frac         = self.seek_slider.value() / 1000.0
            start_sample = int(frac * self._total_samples)
            start_sample = max(0, min(start_sample, self._total_samples - 1))

            sd.stop()
            self._seek_timer.stop()

            self._play_start_samp = start_sample
            self._play_start_time = time.time()
            self._is_playing      = True

            # Apply volume
            volume = getattr(self, '_volume', 0.8)
            audio_slice = self._current_audio[start_sample:] * volume
            sd.play(audio_slice, self._current_sr)

            # Update play/pause button to show pause icon
            self.btn_play_pause.setText("⏸")

            label = "Original" if version == "original" else "Enhanced"
            self.lbl_status.setText(
                f"▶ Playing {label} from {self.lbl_cur_time.text()}")
            self._seek_timer.start()

        except Exception as e:
            self.lbl_status.setText(f"Playback error: {e}")

    # ──────────────────────────────────────────────────────────────
    # Seek slider methods
    # ──────────────────────────────────────────────────────────────
    # ── Media Player Methods ──────────────────────────────────────
    def _set_version(self, version: str):
        """Switch between Original and Enhanced version."""
        self._current_version = None  # force reload
        was_playing = self._is_playing
        if was_playing:
            sd.stop()
            self._seek_timer.stop()
            self._is_playing = False

        # Update tab button styles
        active   = "background:#e94560; color:#fff; border-radius:5px; font-size:10px; font-weight:bold; border:none;"
        inactive = "background:#1a3a5c; color:#aaaacc; border-radius:5px; font-size:10px; border:none;"
        if version == "original":
            self.btn_ver_orig.setStyleSheet(active)
            self.btn_ver_enh.setStyleSheet(inactive)
            self.btn_ver_orig.setChecked(True)
            self.btn_ver_enh.setChecked(False)
        else:
            self.btn_ver_enh.setStyleSheet(active)
            self.btn_ver_orig.setStyleSheet(inactive)
            self.btn_ver_enh.setChecked(True)
            self.btn_ver_orig.setChecked(False)

        # Resume playback in new version if it was playing
        if was_playing:
            self.playSong(version)
        else:
            self.lbl_status.setText(
                f"Version: {'Original' if version == 'original' else 'Enhanced'} — press ▶ to play")

    def _toggle_play_pause(self):
        """Toggle between play and pause."""
        if self._is_playing:
            sd.stop()
            self._seek_timer.stop()
            self._is_playing = False
            self.btn_play_pause.setText("▶")
            self.lbl_status.setText("⏸ Paused")
        else:
            version = "original" if (self.btn_ver_orig.isChecked()) else "enhanced"
            self.playSong(version)

    def _rewind(self):
        """Jump back 10 seconds."""
        if self._current_audio is None:
            return
        current_sample = self._play_start_samp
        if self._is_playing:
            elapsed = time.time() - self._play_start_time
            current_sample = self._play_start_samp + int(elapsed * self._current_sr)
        new_sample = max(0, current_sample - int(10 * self._current_sr))
        new_frac   = new_sample / self._total_samples
        self.seek_slider.setValue(int(new_frac * 1000))
        if self._is_playing:
            sd.stop()
            self._play_start_samp = new_sample
            self._play_start_time = time.time()
            sd.play(self._current_audio[new_sample:], self._current_sr)
        else:
            self._play_start_samp = new_sample
        self.lbl_cur_time.setText(self._fmt(new_sample / self._current_sr))
        self.lbl_status.setText(f"⏮ Rewound 10s")

    def _fast_forward(self):
        """Jump forward 10 seconds."""
        if self._current_audio is None:
            return
        current_sample = self._play_start_samp
        if self._is_playing:
            elapsed = time.time() - self._play_start_time
            current_sample = self._play_start_samp + int(elapsed * self._current_sr)
        new_sample = min(self._total_samples - 1,
                         current_sample + int(10 * self._current_sr))
        new_frac   = new_sample / self._total_samples
        self.seek_slider.setValue(int(new_frac * 1000))
        if self._is_playing:
            sd.stop()
            self._play_start_samp = new_sample
            self._play_start_time = time.time()
            sd.play(self._current_audio[new_sample:], self._current_sr)
        else:
            self._play_start_samp = new_sample
        self.lbl_cur_time.setText(self._fmt(new_sample / self._current_sr))
        self.lbl_status.setText(f"⏭ Skipped 10s")

    def _on_volume_changed(self, value: int):
        """Update volume display."""
        self.lbl_vol_pct.setText(f"{value}%")
        # Note: sounddevice doesn't support runtime volume change,
        # volume will apply on next play
        self._volume = value / 100.0

    def _tab_style(self):
        return ("QPushButton{background:#1a3a5c; color:#aaaacc; border-radius:6px;"
                "font-size:11px; font-weight:bold; border:none; padding:6px;}"
                "QPushButton:checked{background:#e94560; color:#ffffff; font-weight:bold;}"
                "QPushButton:hover{background:#1e4a72;}")

    def _transport_style(self, primary=False):
        if primary:
            # CIRCLE play button (perfect circle)
            return ("QPushButton{background:#e94560; color:#fff; border-radius:25px;"
                    "font-size:18px; border:none; font-weight:bold;}"
                    "QPushButton:hover{background:#ff6080;}"
                    "QPushButton:disabled{background:#3a2233; color:#554455;}")
        else:
            # Round transport buttons (rewind/forward)
            return ("QPushButton{background:#1a3a5c; color:#e0e0e0; border-radius:22px;"
                    "font-size:16px; border:none;}"
                    "QPushButton:hover{background:#1e4a72;}"
                    "QPushButton:disabled{background:#151f2a; color:#333344;}")

    def _seek_pressed(self):
        """User started dragging — pause timer updates."""
        self._user_dragging = True
        self._seek_timer.stop()

    def _seek_moved(self, value: int):
        """Update time label as user drags."""
        if self._current_audio is None or self._current_sr is None:
            return
        frac = value / 1000.0
        sec  = frac * (self._total_samples / self._current_sr)
        self.lbl_cur_time.setText(self._fmt(sec))

    def _seek_released(self):
        """User released slider — jump to new position."""
        self._user_dragging = False
        if self._current_audio is None:
            return

        frac         = self.seek_slider.value() / 1000.0
        start_sample = int(frac * self._total_samples)
        start_sample = max(0, min(start_sample, self._total_samples - 1))
        sec          = start_sample / self._current_sr
        self.lbl_cur_time.setText(self._fmt(sec))

        # If was playing, restart from new position
        if self._is_playing:
            sd.stop()
            self._play_start_samp = start_sample
            self._play_start_time = time.time()
            sd.play(self._current_audio[start_sample:], self._current_sr)
            label = "Original" if self._current_version == "original" else "Enhanced"
            self.lbl_status.setText(
                f"▶ Jumped to {self._fmt(sec)} — {label}")
            self._seek_timer.start()
        else:
            # Not playing — just park the position
            self._play_start_samp = start_sample
            self.lbl_status.setText(f"📍 Position set to {self._fmt(sec)}")

    def _tick_seek(self):
        """Called every 200ms to move slider during playback."""
        if not self._is_playing or self._user_dragging:
            return
        if self._current_audio is None:
            return

        elapsed    = time.time() - self._play_start_time
        cur_sample = self._play_start_samp + int(elapsed * self._current_sr)

        if cur_sample >= self._total_samples:
            self._is_playing = False
            self._seek_timer.stop()
            self.seek_slider.setValue(0)
            self.lbl_cur_time.setText("0:00")
            self.btn_play_pause.setText("▶")
            self.lbl_status.setText("⏹ Playback finished")
            return

        # Update slider without triggering sliderMoved signal
        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(int(cur_sample / self._total_samples * 1000))
        self.seek_slider.blockSignals(False)

        self.lbl_cur_time.setText(self._fmt(cur_sample / self._current_sr))

    def _stop_playback(self):
        sd.stop()
        self._is_playing = False
        self._seek_timer.stop()
        self.btn_play_pause.setText("▶")
        self.lbl_status.setText("⏹ Stopped")

    # ──────────────────────────────────────────────────────────────
    # exit
    # ──────────────────────────────────────────────────────────────
    def exit(self):
        sd.stop()
        self._seek_timer.stop()
        QApplication.quit()

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────
    
    def _fmt(self, seconds: float) -> str:
        s = max(0, int(seconds))
        return f"{s//60}:{s%60:02d}"

    def _btn(self, text, bg, hover, h=42):
        b = QPushButton(text)
        b.setFixedHeight(h)
        b.setStyleSheet(
            f"QPushButton{{background:{bg};color:#e0e0e0;border:none;"
            f"border-radius:6px;font-size:11px;font-weight:bold;padding:6px;}}"
            f"QPushButton:hover{{background:{hover};}}"
            f"QPushButton:disabled{{background:#2a2a3a;color:#555566;}}")
        return b

    def _hr(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color:#0f3460;")
        return line

    def _apply_theme(self):
        self.setStyleSheet(
            "QMainWindow{background:#1a1a2e;}"
            "QWidget{background:#1a1a2e;color:#e0e0e0;}"
            "QScrollBar:vertical{background:#16213e;width:8px;border-radius:4px;}"
            "QScrollBar::handle:vertical{background:#0f3460;border-radius:4px;}")


# ================================================================
# Entry Point
# ================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainApp()
    sys.exit(app.exec())