import numpy as np
import pytest

from harness.signals import sine

def test_sine_has_expected_shape():
    signal = sine(
        frequency=1_000,
        duration=1.0,
        sample_rate=48_000,
    )

    assert signal.shape == (1, 48_000)

def test_sine_has_expected_amplitude():
    signal = sine(
        frequency=1_000,
        duration=1.0,
        sample_rate=48_000,
        amplitude=0.25,
    )

    # The sampled sine should reach the requested amplitude closely enough for this frequency/sample-rate combination.
    assert np.max(np.abs(signal)) == pytest.approx(
        0.25,
        abs=1e-5,
    )

    def test_sine_is_deterministix():
        first = sine(
            frequency=1_000,
            duration=1.0,
            sample_rate=48_000,
            amplitude=0.25,
        )

        second = sine(
            frequency=1_000,
            duration=1.0,
            sample_rate=48_000,
            amplitude=0.25,
        )

        np.testing.assert_array_equal(first, second)

        def test_sine_support_multiple_chanels():
            signal = sine(
                frequency=1_000,
                duration=1.0,
                sample_rate=48_000,
                channels=2,
            )

            assert signal.shape == (2, 48_000)
            np.testing.assert_array_equal(signal[0], signal[1])