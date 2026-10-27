"""ALS со знаковым таргетом и временным затуханием."""

import numpy as np
from .ials import IALS


class CustomTargetALS(IALS):
    def __init__(self, half_life_days=180.0, **kwargs):
        super().__init__(**kwargs)
        self.half_life_days = half_life_days

    def target_values(self, frame, cutoff):
        age_days = np.maximum(0, cutoff - frame.timestamp.to_numpy()) / 86400
        decay = 0.5 + 0.5 * np.exp(-np.log(2) * age_days / self.half_life_days)
        return (frame.rating.to_numpy() - 3.0) / 2.0 * decay
