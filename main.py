"""
JavenEQ - Main entry point
Pipeline: Load → Quick Normalize → Era Detection → Full Preprocess
        → Detect → Adaptive EQ → Compression → Save → Visualize
"""

import sys
import numpy as np
from audio_file import AudioFile
from audio_processor import AudioProcessor
from ai_eq_engine import AIEQEngine
from graph import Visualizer


def quick_normalize(audio_file):
    """Quick normalize before era detection to avoid sub-bass bias."""
    audio_data = audio_file.get_data().copy()
    rms = np.sqrt(np.mean(audio_data ** 2))
    if rms == 0:
        return
    gain = np.clip(0.10 / rms, 0.1, 10.0)
    audio_file._audio_data = np.clip(audio_data * gain, -1.0, 1.0)


def main():
    # -------------------------------------------------------
    # Step 1: Load the audio file
    # -------------------------------------------------------
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_audio_file>")
        print("Example: python main.py data/my_song.mp3")
        sys.exit(1)

    audio_path = sys.argv[1]
    audio_file = AudioFile()
    try:
        audio_file.load(audio_path)
    except Exception as e:
        print(f"[Error] Could not load audio file: {e}")
        sys.exit(1)

    # -------------------------------------------------------
    # Step 2: Quick normalize → Era detection
    # -------------------------------------------------------
    print("\n[JavenEQ] Quick normalize before era detection...")
    quick_normalize(audio_file)

    print("[JavenEQ] Running era detection...")
    engine   = AIEQEngine()
    mono     = audio_file.get_mono()
    sr       = audio_file.sampleRate
    era_info = engine.detectEra(mono, sr)
    profile  = engine.getEraProfile()

    print(f"\n[JavenEQ] Detected: {era_info['era_label']}")
    print(f"[JavenEQ] Confidence: {era_info['confidence_scores']}")

    # -------------------------------------------------------
    # Step 3: Reload + full preprocessing
    # -------------------------------------------------------
    print("\n[JavenEQ] Reloading original audio for full preprocessing...")
    audio_file = AudioFile()
    audio_file.load(audio_path)

    processor = AudioProcessor(audio_file)
    processor.setEraProfile(profile)
    processor.preprocess()

    # Save preprocessed version
    preprocessed_path = "preprocessed/output_preprocessed.wav"
    audio_file.save(preprocessed_path)
    print(f"\n[JavenEQ] Preprocessed saved to: {preprocessed_path}")

    # -------------------------------------------------------
    # Step 4: Full analysis
    # -------------------------------------------------------
    results = engine.analyzeAudio(audio_file)
    final_era = results["era"]

    # -------------------------------------------------------
    # Step 5: Generate ADAPTIVE EQ curve
    # Pass mono audio so instrument levels can be measured
    # -------------------------------------------------------
    print("\n[JavenEQ] Generating adaptive EQ curve...")
    mono_processed = audio_file.get_mono()
    eq_profile = engine.generateEQCurve(results,
                                         mono_audio=mono_processed,
                                         sample_rate=sr)

    # -------------------------------------------------------
    # Step 6: Apply EQ enhancement
    # -------------------------------------------------------
    print("\n[JavenEQ] Applying EQ enhancement...")
    eq_profile.apply(audio_file)

    # -------------------------------------------------------
    # Step 7: Compression
    # -------------------------------------------------------
    processor.compress()

    # -------------------------------------------------------
    # Step 8: Save enhanced output
    # -------------------------------------------------------
    enhanced_path = "enhanced output/output_enhanced.wav"
    audio_file.save(enhanced_path)

    # -------------------------------------------------------
    # Step 9: Generate comparison graph
    # -------------------------------------------------------
    print("\n[JavenEQ] Generating comparison graph...")
    try:
        viz = Visualizer()
        graph_path = viz.generateComparison(
            original_path  = preprocessed_path,
            enhanced_path  = enhanced_path,
            analysis_results = results,
            output_path    = "comparison_graph.png"
        )
        print(f"[JavenEQ] Graph saved to: {graph_path}")
    except Exception as e:
        print(f"[JavenEQ] Graph generation failed: {e}")
        graph_path = None

    print(f"\n{'='*55}")
    print(f"✅ JavenEQ Enhancement Complete!")
    print(f"   Era detected    : {final_era['era_label']}")
    print(f"   EQ bands applied: {len(eq_profile.bands)}")
    print(f"   Preprocessed    : {preprocessed_path}")
    print(f"   Enhanced output : {enhanced_path}")
    if graph_path:
        print(f"   Comparison graph: {graph_path}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()