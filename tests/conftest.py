import numpy as np
import pytest

from harness.fake_plugin import FakeGainPlugin
from harness.render import PluginRenderer
from pedalboard import Pedalboard, Gain
from harness.pedalboard_processor import PedalboardProcessor

# Shared fixtures for all tests under tests/.


@pytest.fixture
def sample_rate():
    return 48_000


@pytest.fixture
def test_signal(sample_rate):
    """1 second, -12 dBFS (0.25 peak) sine wave at 1 kHz, shaped (1, samples) for PluginRenderer."""
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
    """Doubles amplitude (+6 dB); used to verify PluginRenderer applies processing correctly."""
    return FakeGainPlugin(gain=2.0)


@pytest.fixture
def renderer(plugin):
    return PluginRenderer(plugin)

@pytest.fixture
def pedalboard_renderer():
    """Wraps a Pedalboard with a Gain plugin, used to verify PluginRenderer works with real audio plugins."""
    board = Pedalboard([Gain(gain_db=6.0)])  # +6 dB gain
    processor = PedalboardProcessor(board)
    return PluginRenderer(processor)