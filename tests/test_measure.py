import numpy as np
import pytest

from harness.signals import sine

from harness.measure import (
    amplitude_to_dbfs,
    peak_amplitude,
    peak_dbfs,
    rms,
    rms_dbfs,
)

def test_rms_of_silence_is_zero():
    audio = np.zeros((1, 1024))

    assert peak_amplitude(audio) == 0.0

def test_rms_of_silence_is_zero():
    audio = np.zeros((1, 1024))

    assert rms(audio) == 0.0

def test_silence_is_at_floor_in_dbfs():
    audio = np.zeros((1, 1024))

    assert peak_dbfs(audio) == pytest.approx(-600.0)

def test_peak_amplitude():
    audio = np.array( [
        [0.1, -0.25, 0.5, -0.3],
    ])

    assert peak_amplitude(audio) == pytest.approx(0.5)

@pytest.mark.parametrize(
    ("amplitude", "expected_dbfs"),
    [
        (1.0, 0.0),
        (0.5, -6.020599913),
        (0.25, -12.041199826),
        (0.1, -20.0),
    ],
)

def test_amplitude_todbfs(amplitude, expected_dbfs):
    assert amplitude_to_dbfs(amplitude) == pytest.approx(
        expected_dbfs,
        abs=1e-6,
    )
def test_rms_of_full_scale_signal():
    audio = np.array([
        [1.0, -1.0, 1.0, -1.0],
    ])

    assert rms(audio) == pytest.approx(1.0)

def test_rms_of_constant_half_scale_signal():
    audio = np.full((1, 1024), 0.5)

    assert rms(audio) == pytest.approx(0.5)

def test_rms_of_sine(test_signal):
    expected = 0.25 / np.sqrt(2)

    measured = rms(test_signal)

    assert measured == pytest.approx(
        expected,
        rel=1e-4,
    )

def test_rms_cannot_exceed_peak():
    audio = np.array([
        [0.1, -0.7, 0.3, -0.4, 0.2],
    ])

    assert rms(audio) <= peak_amplitude(audio)

def test_peak_is_measured_across_channels():
    audio = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.9],
    ])

    assert peak_amplitude(audio) == pytest.approx(0.9)

def test_rms_is_measured_across_channels():
    audio = np.array([
        [1.0, 1.0],
        [0.0, 0.0]
    ])

    assert rms(audio) == pytest.approx(
        1.0 / np.sqrt(2),
    )

def test_peak_rejects_empty_audio():
    with pytest.raises(ValueError, match="empty"):
        peak_amplitude(np.empty((1, 0)))


def test_rms_rejects_empty_audio():
    with pytest.raises(ValueError, match="empty"):
        rms(np.empty((1, 0)))


def test_peak_rejects_non_finite_audio():
    audio = np.array([[0.0, np.nan, 0.5]])

    with pytest.raises(ValueError, match="non-finite"):
        peak_amplitude(audio)


def test_rms_rejects_non_finite_audio():
    audio = np.array([[0.0, np.inf, 0.5]])

    with pytest.raises(ValueError, match="non-finite"):
        rms(audio)

@pytest.mark.parametrize(
    "amplitude",
    [-1.0 -0.5],
)

def test_dbfs_rejects_negative_amplitude(amplitude):
    with pytest.raises(ValueError, match="negative"):
        amplitude_to_dbfs(amplitude)

def test_sine_peak_matches_requested_amplitude():
    signal = sine(
        frequency=1_000,
        duration=1.0,
        sample_rate=48_000,
        amplitude=0.25,
    )

    assert peak_amplitude(signal) == pytest.approx(
        0.25,
        abs=1e-5,
    )

def test_sine_rms_matches_theoretical_value():
    amplitude = 0.25

    signal = sine(
        frequency=1_000,
        duration=1.0,
        sample_rate=48_000,
        amplitude=amplitude,
    )

    expected_rms = amplitude / np.sqrt(2)

    assert rms(signal) == pytest.approx(
        expected_rms,
        rel=1e-4,
    )

def test_gain_measurement_pipeline(
    renderer,
    test_signal,
    sample_rate,
):
    result = renderer.render(
        test_signal,
        sample_rate=sample_rate,
        buffer_size=512,
        reset=True,
    )

    measured_peak = peak_amplitude(result.audio)

    expected_peak = 0.25 * 2.0

    assert measured_peak == pytest.approx(
        expected_peak,
        abs=1e-12,
    )
