"""
Tests 1 & 2: generate_signals state machine correctness.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategy.signals import generate_signals


def _series(values):
    return pd.Series(values, dtype=float)


# ---------------------------------------------------------------------------
# Test 1: basic entry / exit via mean reversion
# ---------------------------------------------------------------------------

def test_basic_entry_exit():
    """
    z = [0, 0, -2.5, -2.0, -1.0, 0.5, 0, 0, 2.5, 2.0, 1.0, -0.5, 0]
    entry_z=2, exit_z=0, stop_z=4

    Expected positions:
        index 0-1:  FLAT  (z never crosses ±2)
        index 2-4:  LONG  (entered at -2.5, z still < 0 so exit_z=0 not crossed)
        index 5:    FLAT  (z=0.5 > -exit_z=0 → exit)
        index 6-7:  FLAT
        index 8-10: SHORT (entered at 2.5, z still > 0)
        index 11:   FLAT  (z=-0.5 < exit_z=0 → exit)
        index 12:   FLAT
    """
    z = _series([0, 0, -2.5, -2.0, -1.0, 0.5, 0, 0, 2.5, 2.0, 1.0, -0.5, 0])
    expected = [0, 0, +1, +1, +1, 0, 0, 0, -1, -1, -1, 0, 0]

    pos = generate_signals(z, entry_z=2, exit_z=0, stop_z=4)

    assert list(pos) == expected, f"Got {list(pos)}"


def test_output_index_preserved():
    z = _series([0, -2.5, -1.0, 0.5])
    pos = generate_signals(z, entry_z=2, exit_z=0, stop_z=4)
    assert list(pos.index) == list(z.index)


def test_output_dtype_integer():
    z = _series([0.0, -2.5, 0.5])
    pos = generate_signals(z, entry_z=2, exit_z=0, stop_z=4)
    assert pos.dtype in (int, np.int64, np.int32)


# ---------------------------------------------------------------------------
# Test 2: stop-loss fires before mean reversion
# ---------------------------------------------------------------------------

def test_stop_loss():
    """
    z = [0, -2.5, -3.0, -4.5, -2.0, 0]
    entry_z=2, exit_z=0, stop_z=4

    index 0:   FLAT
    index 1:   LONG  (z=-2.5 < -2)
    index 2:   LONG  (z=-3.0, stop is z < -4)
    index 3:   FLAT  (z=-4.5 < -stop_z=-4 → stop loss)
    index 4-5: FLAT  (no re-entry: z=-2.0 is not < -2 strictly)
    """
    z = _series([0, -2.5, -3.0, -4.5, -2.0, 0])
    expected = [0, +1, +1, 0, 0, 0]

    pos = generate_signals(z, entry_z=2, exit_z=0, stop_z=4)

    assert list(pos) == expected, f"Got {list(pos)}"


def test_stop_loss_short_side():
    """
    Symmetric: SHORT position stopped out when z blows through +stop_z.
    """
    z = _series([0, 2.5, 3.0, 4.5, 2.0, 0])
    expected = [0, -1, -1, 0, 0, 0]

    pos = generate_signals(z, entry_z=2, exit_z=0, stop_z=4)

    assert list(pos) == expected, f"Got {list(pos)}"


def test_flat_series_stays_flat():
    z = _series([0.0] * 10)
    pos = generate_signals(z, entry_z=2, exit_z=0, stop_z=4)
    assert (pos == 0).all()


def test_no_reentry_same_bar_as_stop():
    """
    After a stop the position must be FLAT on that bar, not immediately re-entered.
    """
    z = _series([0, -2.5, -4.5, -2.5])
    pos = generate_signals(z, entry_z=2, exit_z=0, stop_z=4)
    # bar 2: stop fires → FLAT; bar 3: z=-2.5 < -2 → re-entry allowed
    assert pos.iloc[2] == 0
    assert pos.iloc[3] == +1
