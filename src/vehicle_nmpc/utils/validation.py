"""Validation helpers shared across NMPC components."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


def as_array(
    name: str,
    values: np.ndarray | Sequence,
    expected_shape: tuple[int, ...],
    *,
    default: float | None = None,
    dtype: type = float,
) -> np.ndarray:
    """Convert values to a validated numpy array."""
    array = np.asarray(values, dtype=dtype)
    if array.size == 0 and default is not None:
        return np.full(expected_shape, default, dtype=dtype)

    if array.shape != expected_shape:
        msg = f"Expected {name} shape {expected_shape}, got {array.shape}."
        raise ValueError(msg)

    return array


def as_vector(
    name: str,
    values: np.ndarray | Sequence[float] | Sequence[int],
    expected_size: int,
    *,
    default: float | None = None,
    dtype: type = float,
) -> np.ndarray:
    """Convert a configured vector to a validated numpy array."""
    return as_array(name, values, (expected_size,), default=default, dtype=dtype)


def as_matrix(
    name: str,
    values: np.ndarray | Sequence[Sequence[float]] | Sequence[Sequence[int]],
    expected_shape: tuple[int, int],
    *,
    default: float | None = None,
    dtype: type = float,
) -> np.ndarray:
    """Convert a configured matrix to a validated numpy array."""
    return as_array(name, values, expected_shape, default=default, dtype=dtype)


def require_positive(name: str, value: float) -> None:
    """Raise if a physical parameter is not strictly positive."""
    if value <= 0.0:
        msg = f"{name} must be positive, got {value}."
        raise ValueError(msg)


def require_slip(name: str, value: float) -> None:
    """Raise if a slip coefficient would make track speed conversion singular."""
    if not 0.0 <= value < 1.0:
        msg = f"{name} must be in [0.0, 1.0), got {value}."
        raise ValueError(msg)
