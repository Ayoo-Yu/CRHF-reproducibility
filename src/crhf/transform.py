"""Training-only median imputation for probability models."""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class FeatureTransform:
    columns: list
    medians: pd.Series

    @classmethod
    def fit(cls, frame, columns):
        if any(c.startswith("target_") for c in columns):
            raise ValueError("Target column entered feature list")
        values = frame[columns].replace([np.inf, -np.inf], np.nan)
        return cls(columns, values.median().fillna(0.0))

    def apply(self, frame):
        return (
            frame[self.columns]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(self.medians)
            .to_numpy(float)
        )
