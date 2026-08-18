import numpy as np

class FakeGainPlugin:
    def __init__(self, gain: float = 1.0):
        self.gain = gain
        self.reset_count = 0
        self.process_count = 0

    def reset(self) -> None:
        self.reset_count += 1
        self.process_count = 0

    def process(self, audio: np.ndarray, *, sample_rate: int, buffer_size: int) -> np.ndarray:
        self.process_count += 1
        
        return self.gain * audio