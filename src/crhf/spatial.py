"""Disaggregated cross-farm availability lags and one-hour changes at exact clock offsets."""

import numpy as np
import pandas as pd


LAGS = (0, 1, 2, 4, 8, 16, 24, 48, 92)


def spatial_features(source, keys):
    """Read exact prior timestamps; missing timestamps stay missing, never ffilled."""
    raw = source[["farm", "Timestamp", "potential", "capacity_proxy", "input_valid"]].copy()
    if raw[["farm", "Timestamp"]].duplicated().any():
        raise ValueError("Duplicate source clock")
    raw["value"] = (raw.potential / raw.capacity_proxy).where(raw.input_valid.eq(True))
    wide = raw.pivot(index="Timestamp", columns="farm", values="value").sort_index()
    issue = pd.DatetimeIndex(keys.Timestamp)
    out = {}
    farms = sorted(wide.columns)
    for farm in farms:
        series = wide[farm]
        for lag in LAGS:
            end = issue - pd.Timedelta(minutes=15 * lag)
            now = series.reindex(end).to_numpy(float)
            past = series.reindex(end - pd.Timedelta(hours=1)).to_numpy(float)
            # The target station is excluded, with its own history held separately.
            own = keys.farm.eq(farm).to_numpy()
            now = np.where(own, np.nan, now)
            past = np.where(own, np.nan, past)
            out[f"sp_{farm}_level_{lag}"] = now
            out[f"sp_{farm}_change_{lag}"] = now - past
    return pd.DataFrame(out, index=keys.index)
