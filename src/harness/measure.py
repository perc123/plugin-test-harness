from __future__ import annotations

import numpy as np

def peak_amplitude(audio: np.ndarray) -> float:
    """Return the largest absolute sample value in the audio signal.
    
    The measurment is taken across signal with peak of 1.0 corresponds to 0 dBFS.
    """

    if audio.size == 0:
        raise ValueError("audio contains non-finite values")
    
    return float(np.max(np.abs(audio)))

def rms(audio: np.ndarray) -> float:
    """Return the RMS level across all samples/channels.
    
    RMS describes the effective magnitude of a signal and is therefore different from peak amplitude. 
    For a full-scale sine wave, RMS is approximately 1/sqrt(2)."""

    if audio.size == 0:
        raise ValueError("audio must not be empty")

    if not np.isfinite(audio).all():
        raise ValueError("audio contains non-finite values")

    return float(np.sqrt(np.mean(np.square(audio))))

def amplitude_to_dbfs(amplitude: float) -> float:
    """Convert a linear amplitude to dBFS.

    0 dBFS corresponds to a normalized amplitude of 1.0.

    The small floor prevents log10(0) from producing -inf. This is useful because measurement functions should finite values whenever possible.
    """

    if amplitude < 0:
        raise ValueError("amplitude must not be negative")

    floor = 1e-30

    return float(20.0 * np.log10(max(amplitude, floor)))

def peak_dbfs(audio: np.ndarray) -> float:
    """Return the peak level of an audio signal in dBFS."""

    return amplitude_to_dbfs(peak_amplitude(audio))

def rms_dbfs(audio: np.ndarray) -> float:
    """Return the RMS level of an audio signal in dBFS."""

    return amplitude_to_dbfs(rms(audio))