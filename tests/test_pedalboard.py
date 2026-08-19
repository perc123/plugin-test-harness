import numpy as np

def test_pedalboard_renderer_produces_finite_audio(
        pedalboard_renderer,
        test_signal,
        sample_rate,
):
    result = pedalboard_renderer.render(
        test_signal,
        sample_rate=sample_rate,
        buffer_size=512,
        reset=True,
    )

    # PedalboardRenderer should produce finite audio with the same shape as the input.
    assert result.audio.shape == test_signal.shape 
    assert np.isfinite(result.audio).all() 

def test_pedalboard_gain(
        pedalboard_renderer,
        test_signal,
        sample_rate,
):
    result = pedalboard_renderer.render(
        test_signal,
        sample_rate=sample_rate,
        buffer_size=512,
        reset=True,
    )

    expected = test_signal * (10 ** (6.0 / 20))  # +6 dB gain

    np.testing.assert_allclose(result.audio, expected, rtol=1e-5, atol=1e-7
    )