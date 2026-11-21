"""Extend the observation grid with local available-power changes, dispersion and residual inputs."""

import numpy as np
import pandas as pd
from .risk_features import build_feature_bank


AVAILABLE_LAGS = (1, 2, 4, 8, 16)


AVAILABLE_WINDOWS = (4, 8, 16, 96)


def build_available_features(base, config):
    """Return the unchanged risk bank plus available-history feature lists.

    ``observed`` permits the full observed physical/cluster bank; ``history``
    restricts the comparator to available power, its past and calendar/farm ID.
    No current quality flag, actual power, dispatch or risk enters ``history``.
    Invalid bins remain on the time grid and are never interpolated.
    """
    grid, risk_spec = build_feature_bank(base, config)
    valid = grid.input_valid.fillna(False).astype(bool)
    available = grid.x_potential.where(valid)
    grouped = available.groupby(grid.farm, sort=False)
    generated = {}
    for lag in AVAILABLE_LAGS:
        generated[f"a_delta_{lag}"] = grouped.diff(lag)
    for window in AVAILABLE_WINDOWS:
        generated[f"a_std_{window}"] = grouped.transform(
            lambda values, w=window: values.rolling(w, min_periods=2).std()
        )
        mean = grouped.transform(lambda values, w=window: values.rolling(w, min_periods=1).mean())
        generated[f"a_deviation_{window}"] = available - mean
    available_history = list(generated)
    generated["a_input_inconsistent"] = (
        (grid.actual_operational > grid.potential + 0.03 * grid.capacity_proxy)
        .astype(float)
        .where(valid, np.nan)
    )
    grid = pd.concat([grid, pd.DataFrame(generated, index=grid.index)], axis=1)
    full = risk_spec["variants"]["full"]
    calendar = {"x_hour_sin", "x_hour_cos", "x_year_sin", "x_year_cos", "x_farm_code"}
    history = [
        name
        for name in full
        if name == "x_potential" or name.startswith("m_x_potential_") or name in calendar
    ]
    power_spec = {
        "observed": [*full, *available_history, "a_input_inconsistent"],
        "history": [*history, *available_history],
    }
    return grid, risk_spec, power_spec
