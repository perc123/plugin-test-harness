import numpy as np
from pedalboard import Pedalboard

class PedalboardProcessor:
    """Wraps a Pedalboard and exposes a process() method compatible with PluginRenderer."""

    def __init__(self, board: Pedalboard):
        self.board = board

    def reset(self) -> None:
        """Pedalboard has no internal state to reset, but this method is required by PluginRenderer."""
        pass

    def process(self, audio: np.ndarray, *, sample_rate: int, buffer_size: int) -> np.ndarray:
        """Process the input audio through the pedalboard.

        Args:
            audio: A 2D numpy array of shape (channels, samples).
            sample_rate: The sample rate of the audio.
            buffer_size: The size of the processing buffer (not used in this implementation).

        Returns:
            A 2D numpy array of processed audio with the same shape as the input.
        """
        return self.board(audio, sample_rate=sample_rate, buffer_size=buffer_size, reset=True)  # Pedalboard processes the entire audio at once