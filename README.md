# plugin-test-harness

A small test harness for driving audio plugins/processors over synthetic
signals and asserting on their output — useful for regression-testing DSP
code (gain stages, filters, effects, etc.) without needing a real DAW.

## How it fits together

- **`harness.render.PluginRenderer`** — wraps a processor object and
  handles validation (shape, finiteness), optional state reset between
  renders, and packaging the result (`RenderResult`: audio + the
  sample rate / buffer size it was rendered with).
- **A processor** — any object exposing:
  ```python
  reset() -> None
  process(audio, *, sample_rate, buffer_size) -> np.ndarray
  ```
  `harness.fake_plugin.FakeGainPlugin` is a minimal example (applies a
  fixed gain) used by the test suite; a real plugin wrapper (e.g. around
  a `pedalboard` VST/AU plugin) would implement the same interface.
- **`tests/`** — pytest suite exercising `PluginRenderer` against
  `FakeGainPlugin`: correct output shape/dtype, gain is applied, renders
  are deterministic, reset behavior, and input validation.

## Audio representation

The harness represents audio as a NumPy array with shape:

    (channels, samples)

Sample values are represented as float32/float64 values in the
normal audio range, typically [-1.0, 1.0].

## Running the tests

```bash
pip install -e .
pytest -v
```
