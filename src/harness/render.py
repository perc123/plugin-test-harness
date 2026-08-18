from pathlib import Path

import numpy as np
from pedalboard import load_plugin

class PluginRenderer:
    def __init__(self, plugin_path: str | Path):
        self.plugin_path = Path(plugin_path)
        self.plugin = load_plugin(str(self.plugin_path))

    def render(self, 
               audio: np.ndarray,
               *,
               sample_rate: int,
               buffer_size: int,
               reset: bool = True,
            ) -> np.ndarray:
            return self.plugin(audio, sample_rate=sample_rate, buffer_size=buffer_size, reset=reset)
    