from dataclasses import dataclass

import numpy as np

@dataclass
class RenderResult:
    audio: np.ndarray
    sample_rate: int
    buffer_size: int

class PluginRenderer:
    def __init__(self, processor):
         self.processor = processor

    def render(self, 
               audio: np.ndarray,
               *,
               sample_rate: int,
               buffer_size: int,
               reset: bool = True,
            ) -> RenderResult:
            if audio.ndim != 2:
                raise ValueError("Audio must be a 2D array with shape (channels, samples).")

            if not np.isfinite(audio).all():
                raise ValueError("Audio contains non-finite values.")

            if reset:
                 self.processor.reset()

            output = self.processor.process(
                 audio,
                 sample_rate=sample_rate,
                 buffer_size=buffer_size,
            )

            return RenderResult(audio=output, sample_rate=sample_rate, buffer_size=buffer_size)