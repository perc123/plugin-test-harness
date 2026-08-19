"""Drives an audio plugin ("processor") over a signal and captures its output.

A processor is any object exposing:
    reset() -> None
    process(audio, *, sample_rate, buffer_size) -> np.ndarray
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class RenderResult:
    """Output of a single PluginRenderer.render() call, plus the settings used to produce it."""
    audio: np.ndarray
    sample_rate: int
    buffer_size: int


class PluginRenderer:
    """Wraps a processor and validates/renders audio through it."""

    def __init__(self, processor):
         self.processor = processor

    def render(self,
               audio: np.ndarray,
               *,
               sample_rate: int,
               buffer_size: int,
               reset: bool = True,
            ) -> RenderResult:
            # Audio must be (channels, samples); reject anything else (e.g. bare 1D mono arrays)
            # before it reaches the processor.
            if audio.ndim != 2:
                raise ValueError("Audio must be a 2D array with shape (channels, samples).")

            if not np.isfinite(audio).all():
                raise ValueError("Audio contains non-finite values.")

            # Most plugins carry internal state (filters, envelopes); reset it so renders are
            # reproducible unless the caller explicitly wants state to carry over (reset=False).
            if reset:
                 self.processor.reset()

            output = self.processor.process(
                 audio,
                 sample_rate=sample_rate,
                 buffer_size=buffer_size,
            )

            return RenderResult(audio=output, sample_rate=sample_rate, buffer_size=buffer_size)