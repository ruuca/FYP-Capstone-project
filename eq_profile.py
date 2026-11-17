import json
from band import Band
from filter_chain import FilterChain

class EQProfile:
    """
    Stores an era-specific EQ curve as a collection of bands.
    Matches the EQProfile class in the class diagram.
    """

    def __init__(self, era: str = "", genre: str = ""):
        self.bands = []     # List of Band objects
        self.era   = era
        self.genre = genre

    def addBand(self, freq: float, gain: float, Q: float, type: str):
        """Add a band to this EQ profile."""
        band = Band(freq=freq, gain=gain, Q=Q, type=type)
        self.bands.append(band)
        return band

    def apply(self, audio_file):
        """
        Apply this EQ profile to an audio file
        by building a FilterChain and running it.
        """
        print(f"\n[EQProfile] Applying EQ profile: era={self.era}, genre={self.genre}")
        print(f"[EQProfile] {len(self.bands)} band(s) to apply:")
        for band in self.bands:
            print(f"  {band}")

        chain = FilterChain()
        chain.filters = self.bands
        chain.apply(audio_file)
        print("[EQProfile] EQ profile applied.\n")

    def savePreset(self, path: str):
        """Save this EQ profile to a JSON file for reuse."""
        data = {
            "era":   self.era,
            "genre": self.genre,
            "bands": [
                {
                    "freq": b.freq,
                    "gain": b.gain,
                    "Q":    b.Q,
                    "type": b.type
                }
                for b in self.bands
            ]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[EQProfile] Preset saved to: {path}")

    @staticmethod
    def loadPreset(path: str):
        """Load a saved EQ profile from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        profile = EQProfile(era=data["era"], genre=data.get("genre", ""))
        for b in data["bands"]:
            profile.addBand(b["freq"], b["gain"], b["Q"], b["type"])
        print(f"[EQProfile] Preset loaded from: {path}")
        return profile

    def __repr__(self):
        return f"EQProfile(era={self.era}, genre={self.genre}, bands={len(self.bands)})"