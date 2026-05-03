"""Validation helpers shared across NMPC components."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


def as_vector(
    name: str,
    values: np.ndarray | Sequence[float] | Sequence[int],
    expected_size: int,
    *,
    default: float | None = None,
    dtype: type = float,
) -> np.ndarray:
    """Convert a configured vector to a validated numpy array."""
    array = np.asarray(values, dtype=dtype)
    if array.size == 0 and default is not None:
        return np.full(expected_size, default, dtype=dtype)

    if array.shape != (expected_size,):
        msg = f"Expected {name} shape ({expected_size},), got {array.shape}."
        raise ValueError(msg)

    return array
