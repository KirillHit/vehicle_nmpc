"""Closed-loop controller evaluation workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vehicle_nmpc.controller import BaseController
    from vehicle_nmpc.models import ModelBundle
    from vehicle_nmpc.sim import BaseSimulator


@dataclass(kw_only=True, slots=True)
class ClosedLoopResult:
    """Closed-loop rollout data."""

    states: np.ndarray
    """Simulated state trajectory with shape (n_steps + 1, nx)."""

    controls: np.ndarray
    """Applied control trajectory with shape (n_steps, nu)."""

    stats: list[dict]
    """Per-step controller statistics."""


def run_closed_loop(
    controller: BaseController,
    simulator: BaseSimulator,
    model: ModelBundle,
    *,
    n_steps: int,
) -> ClosedLoopResult:
    """Run a controller and simulator in closed loop."""
    if n_steps <= 0:
        msg = f"n_steps must be positive, got {n_steps}."
        raise ValueError(msg)

    states = np.zeros((n_steps + 1, model.nx), dtype=float)
    controls = np.zeros((n_steps, model.nu), dtype=float)
    stats: list[dict] = []

    states[0] = np.asarray(model.x0, dtype=float)
    controller.reset(states[0])
    simulator.reset(states[0])

    for step in range(n_steps):
        controls[step] = controller.solve(states[step])
        stats.append(controller.get_stats())
        states[step + 1] = simulator.step(states[step], controls[step])

    return ClosedLoopResult(states=states, controls=controls, stats=stats)
