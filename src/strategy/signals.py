import numpy as np
import pandas as pd


def generate_signals(
    zscore_series: pd.Series,
    entry_z: float,
    exit_z: float,
    stop_z: float,
) -> pd.Series:
    """
    Generate position labels from a z-score series using a state machine.

    States: FLAT (0), LONG (+1), SHORT (-1)

    Entry:
        FLAT -> LONG  when z < -entry_z
        FLAT -> SHORT when z > +entry_z

    Exit (mean reversion):
        LONG  -> FLAT when z > -exit_z  (crossed back toward zero)
        SHORT -> FLAT when z < +exit_z

    Stop loss (divergence):
        LONG  -> FLAT when z < -stop_z
        SHORT -> FLAT when z > +stop_z

    Parameters
    ----------
    zscore_series : pd.Series
        Time series of z-scores.
    entry_z : float
        Absolute z-score threshold to open a position.
    exit_z : float
        Absolute z-score threshold (toward zero) to close a position.
    stop_z : float
        Absolute z-score threshold (away from zero) to stop out.

    Returns
    -------
    pd.Series
        Integer series of {-1, 0, +1} aligned to zscore_series.index.
    """
    FLAT, LONG, SHORT = 0, 1, -1

    positions = np.zeros(len(zscore_series), dtype=int)
    state = FLAT

    for i, z in enumerate(zscore_series):
        if state == FLAT:
            if z < -entry_z:
                state = LONG
            elif z > entry_z:
                state = SHORT
        elif state == LONG:
            if z > -exit_z:   # crossed back toward zero
                state = FLAT
            elif z < -stop_z: # diverged further -- stop loss
                state = FLAT
        elif state == SHORT:
            if z < exit_z:    # crossed back toward zero
                state = FLAT
            elif z > stop_z:  # diverged further -- stop loss
                state = FLAT

        positions[i] = state

    return pd.Series(positions, index=zscore_series.index, dtype=int)
