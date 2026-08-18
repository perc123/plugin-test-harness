import numpy as np


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