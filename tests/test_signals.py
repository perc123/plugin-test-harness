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

def test_sine_is_deterministic():
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

@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"frequency": 0}, "frequency"),
        ({"frequency": -100}, "frequency"),
        ({"duration": 0}, "duration"),
        ({"duration": -1}, "duration"),
        ({"sample_rate": 0}, "sample_rate"),
        ({"sample_rate": -48_000}, "sample_rate"),
        ({"channels": 0}, "channels"),
        ({"channels": -1}, "channels"),
        ({"amplitude": -1}, "amplitude"),
    ],
)

def test_sine_rejects_invalid_parameters(kwargs, error):
     parameters = {
          "frequency": 1_000,
          "duration": 1.0,
          "sample_rate": 48_000,
          "amplitude": 0.25,
          "channels": 1,
     }

     parameters.update(kwargs)

     with pytest.raises(ValueError, match=error):
          sine(**parameters)


def test_impulse_has_single_nonzero_sample():
     signal = impulse(
          length=1_024,
          amplitude=0.5,
     )

     assert signal.shape == (1, 1_024)
     assert np.count_nonzero(signal) == 1
     assert signal[0, 0] == pytest.approx(0.5)


def test_silence_is_zero():
     signal = silence(
          duration=1.0,
          sample_rate=48_000.,
          channels=2,
     )

     assert signal.shape == (2, 48_000)
     assert np.count_nonzero(signal) == 0