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
        raise ValueError("frequency must be greater than 0")

    if duration <= 0:
        raise ValueError("duration must be greater than 0")

    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than 0")

    if channels <= 0:
        raise ValueError("channels must be greater than 0")

    if amplitude < 0:
        raise ValueError("amplitude must not be negative")

    sample_count = int(round(duration * sample_rate))

    # arange is used rather than linspace so that the number of generated samples is determined explicitly by the same rate and duration.
    t = np.arange(sample_count, dtype=np.float64) / sample_rate

    signal = amplitude * np.sin(2 * np.pi * frequency * t)

    # Add the channel dimention even for mono. This gives every signal generator in the harness the same (channels, samples) interface.
    return np.tile(signal, (channels, 1))


def impulse(
        length: int,
        amplitude: float = 1.0,
        channels: int = 1,
        position: int = 0,
) -> np.ndarray:
    """Generate a discrete impulse.

    An impulse is particularly useful for measuring systems that introduce delay or otherwise modify the impulse response.
    """
    if length <= 0:
        raise ValueError("length must be greater than zero")

    if channels <= 0:
        raise ValueError("channels must be greater than zero")

    if amplitude < 0:
        raise ValueError("amplitude must not be negative")

    if not 0 <= position < length:
        raise ValueError("position must be withing the signal")

    signal = np.zeros((channels, length), dtype=np.float64)

    # The impulse is places at an explicitly controlled sample index.
    # This makes latency measurements reproducible later.
    signal[:, position] = amplitude

    return signal

def silence(
        duration: float,
        sample_rate: int,
        channels: int = 1,
) -> np.ndarray: 
    """Generate digital silence."""
    if duration <= 0:
        raise ValueError("duration must be greater than 0")

    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than 0")

    if channels <= 0:
        raise ValueError("channels must be greater than 0")

    sample_count = int(round(duration * sample_rate))

    return np.zeros(
        (channels, sample_count),
        dtype=np.float64,
    )

def white_noise(
    duration: float,
    sample_rate: int,
    amplitude: float = 1.0,
    channels : int = 1,
    seed: int = 0,
) -> np.ndarray:
     """Generate reproducible white noise.
     
     The seed is explicit so that a failing test can be reproduced exacctly."""
     if duration <= 0:
          raise ValueError("duration must be greater than 0")

     if sample_rate <= 0:
          raise ValueError("sample_rate must be greater than 0")

     if amplitude < 0:
          raise ValueError("amplitude must not be negative")

     if channels <= 0:
          raise ValueError("channels must be greater than 0")

     sample_count = int(round(duration * sample_rate))

     rng = np.random.default_rng(seed)

     # Generate each channel independently. Using an explicit Generator keeps this function isolated from global NumPy RNG state.
     noise = rng.uniform(
          low=-amplitude,
          high=amplitude,
          size=(channels, sample_count),
     )

     return noise