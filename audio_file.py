import librosa
import soundfile as sf
import numpy as np

class AudioFile:
    """
    Represents a loaded audio file.
    Matches the AudioFile class in the class diagram.
    """

    def __init__(self):
        self.path = None
        self.bitDepth = None
        self.channels = None
        self.sampleRate = None

        # Audio data (not in class diagram but needed internally)
        self._audio_data = None

    def load(self, path: str):
        """Load an audio file from the given path."""
        print(f"[AudioFile] Loading file: {path}")

        self.path = path

        # Load audio using librosa (converts to mono float32 by default)
        audio_data, sample_rate = librosa.load(path, sr=None, mono=False)

        self.sampleRate = sample_rate

        # Handle mono vs stereo
        if audio_data.ndim == 1:
            self.channels = 1
        else:
            self.channels = audio_data.shape[0]

        # Bit depth (librosa loads as float32)
        self.bitDepth = 32

        # Store as 2D array (channels x samples) for consistency
        if audio_data.ndim == 1:
            self._audio_data = audio_data[np.newaxis, :]  # shape: (1, samples)
        else:
            self._audio_data = audio_data

        print(f"[AudioFile] Loaded successfully!")
        print(f"  Sample Rate : {self.sampleRate} Hz")
        print(f"  Channels    : {self.channels}")
        print(f"  Duration    : {self._audio_data.shape[-1] / self.sampleRate:.2f} seconds")
        print(f"  Bit Depth   : {self.bitDepth}-bit (float)")

    def save(self, output_path: str):
        """Save the audio data to a file."""
        if self._audio_data is None:
            print("[AudioFile] No audio data to save.")
            return

        # Convert back to (samples, channels) format for soundfile
        if self.channels == 1:
            data_to_save = self._audio_data[0]  # flatten to 1D
        else:
            data_to_save = self._audio_data.T   # transpose to (samples, channels)

        sf.write(output_path, data_to_save, self.sampleRate)
        print(f"[AudioFile] Saved to: {output_path}")

    def get_mono(self):
        """Return a mono version of the audio (averaged across channels)."""
        if self._audio_data is None:
            return None
        return np.mean(self._audio_data, axis=0)

    def get_data(self):
        """Return the raw audio data array."""
        return self._audio_data