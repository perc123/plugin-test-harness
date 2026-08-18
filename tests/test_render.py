import numpy as np
import pytest


def test_signal_is_deterministic(test_signal):
    assert test_signal.shape == (1, 48_000)
    assert np.isfinite(test_signal).all()
    assert np.max(np.abs(test_signal)) == pytest.approx(0.25)
