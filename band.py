class Band:
    """
    Represents a single EQ band.
    Matches the Band class in the class diagram.

    Attributes:
        freq  : Center frequency in Hz (e.g. 1000.0)
        gain  : Gain in dB — positive = boost, negative = cut (e.g. +3.0, -2.0)
        Q     : Quality factor — controls bandwidth. Higher Q = narrower band.
                Typical values: 0.5 (wide) to 4.0 (narrow)
        type  : Filter type — "peak", "lowshelf", "highshelf", "lowpass", "highpass"
    """

    def __init__(self, freq: float, gain: float, Q: float, type: str):
        self.freq = freq
        self.gain = gain
        self.Q    = Q
        self.type = type

    def __repr__(self):
        return (f"Band(type={self.type}, freq={self.freq}Hz, "
                f"gain={self.gain:+.1f}dB, Q={self.Q})")