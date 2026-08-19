import numpy as np
import pytest

from harness.fake_plugin import FakeGainPlugin
from harness.render import PluginRenderer

@pytest.fixture
def sample_rate():
    return 48_000

@pytest.fixture
def test_signal(sample_rate):
    duration = 1.0  # seconds
    frequency = 1_000.0

    t = np.arange(
        int(duration * sample_rate),
        dtype=np.float64,
    ) / sample_rate
    signal = 0.25 * np.sin(2 *np.pi * frequency * t)

    return signal[np.newaxis, :]

@pytest.fixture
def plugin():
    return FakeGainPlugin(gain=2.0)

@pytest.fixture
def renderer(plugin):
    return PluginRenderer(plugin)
