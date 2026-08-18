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

def test_render_resets_plugin(
        renderer,
        plugin,
        test_signal,
        sample_rate,
):
    renderer.render(
        test_signal,
        sample_rate=sample_rate,
        buffer_size=512,
        reset=True,
    )

    assert plugin.reset_count == 1

    def test_render_can_skip_reset(
        renderer,
        plugin,
        test_signal,
        sample_rate,
    ):
        renderer.render(
            test_signal,
            sample_rate=sample_rate,
            buffer_size=512,
            reset=True,
        )

        renderer.render(
            test_signal,
            sample_rate=sample_rate,
            buffer_size=512,
            reset=False,
        )

        assert plugin.reset_count == 1