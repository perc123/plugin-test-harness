from __future__ import annotations

import numpy as np

def sine(
        frequency: float,
        duration: float,
        sample_rate: int,
        amplitude: float = 1.0,
        channels: int =1,
) -> np.ndarray:
    """
    Generate a degterministic sinusoidal test signal.

    Audio is represented throughout the harness as:
        (channels, samples)

        Keeping this convention in one place prevents individual tests from accidentally using incompatible channel/signal formats.
    """
    if frequency <= 0:
        raise ValueError("Frequency must be greater than 0")

    if duration <= 0:
        raise ValueError("Duration must be greater than 0")

    if sample_rate <= 0:
        raise ValueError("Sample rate must be greater than 0")

    if channels <= 0:
        raise ValueError("Channels must be greater than 0")

    if amplitude < 0:
        raise ValueError("Amplitude must not be negative")

    sample_count = int(round(duration * sample_rate))

    # arange is used rather than linspace so that the number of generated samples is determined explicitly by the same rate and duration.
    t = np.arange(sample_count, dtype=np.float64) / sample_rate

    signal = amplitude * np.sin(2 * np.pi * frequency * t)

    # Add the channel dimention even for mono. This gives every signal generator in the harness the same (channels, samples) interface.
    return np.tile(signal, (channels, 1))