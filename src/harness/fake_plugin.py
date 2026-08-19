"""A minimal stand-in processor, used to exercise PluginRenderer without a real audio plugin."""

import numpy as np


class FakeGainPlugin:
    """Multiplies the input signal by a fixed gain; tracks how many times it was reset."""

    def __init__(self, gain: float = 1.0):
        self.gain = gain
        self.reset_count = 0  # lets tests assert PluginRenderer resets state when expected

    def reset(self) -> None:
        self.reset_count += 1

    def process(self, audio: np.ndarray, *, sample_rate: int, buffer_size: int) -> np.ndarray:
        return self.gain * audio