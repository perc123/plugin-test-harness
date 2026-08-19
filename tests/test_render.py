"""Behavioral tests for PluginRenderer, driven against FakeGainPlugin as a stand-in processor."""

import numpy as np
import pytest


def test_plugin_renders_finite_audio(
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

    assert result.audio.ndim == 2
    assert result.audio.shape == test_signal.shape
    assert result.sample_rate == sample_rate
    assert result.buffer_size == 512
    assert np.isfinite(result.audio).all()


def test_gain_is_applied(
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

    expected = test_signal * 2.0

    np.testing.assert_allclose(result.audio, expected, rtol=0, atol=1e-12,)


def test_render_is_deterministic(
        renderer,
        test_signal,
        sample_rate,
):
    first = renderer.render(
        test_signal,
        sample_rate=sample_rate,
        buffer_size=512,
        reset=True,
    )

    second = renderer.render(
        test_signal,
        sample_rate=sample_rate,
        buffer_size=512,
        reset=True,
    )

    np.testing.assert_array_equal(
        first.audio,
        second.audio,
    )


# reset=True/False controls whether processor state carries over between renders.
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

# Input validation: render() should reject malformed audio before it reaches the processor.
def test_renderer_rejects_mono_1d_array(
        renderer, sample_rate,
):
    with pytest.raises(ValueError, match="shape"):
        renderer.render(
            np.zeros(1000),
            sample_rate=sample_rate,
            buffer_size=512,
        )

def test_renderer_rejects_non_finite_audio(
        renderer, sample_rate,
):
    audio = np.zeros((1, 1000))
    audio[0, 100] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        renderer.render(
            audio,
            sample_rate=sample_rate,
            buffer_size=512,
        )
