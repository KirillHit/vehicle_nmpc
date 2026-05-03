"""Closed-loop controller evaluation workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vehicle_nmpc.controller import BaseController
    from vehicle_nmpc.models import ModelBundle
    from vehicle_nmpc.sim import BaseSimulator
    from vehicle_nmpc.trajectory import BaseTrajectoryProvider

log = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class ClosedLoopResult:
    """Closed-loop rollout data."""

    states: np.ndarray
    """Simulated state trajectory with shape (n_steps + 1, nx)."""

    controls: np.ndarray
    """Applied control trajectory with shape (n_steps, nu)."""

    reference_states: np.ndarray
    """Reference state trajectory with shape (n_steps + 1, nx)."""

    reference_controls: np.ndarray
    """Reference control trajectory with shape (n_steps, nu)."""

    stats: list[dict]
    """Per-step controller statistics."""


def run_evaluation(
    controller: BaseController,
    simulator: BaseSimulator,
    model: ModelBundle,
    trajectories: list[BaseTrajectoryProvider],
    *,
    n_steps: int,
) -> list[ClosedLoopResult]:
    """Run one controller/simulator pair over a suite of trajectory providers."""
    results: list[ClosedLoopResult] = []
    for trajectory in trajectories:
        log.info("Starting trajectory '%s' run.", trajectory.name)
        result = _run_trajectory(controller, simulator, model, trajectory, n_steps=n_steps)
        log.info("Finished trajectory '%s' run.", trajectory.name)
        results.append(result)
    return results


def _run_trajectory(
    controller: BaseController,
    simulator: BaseSimulator,
    model: ModelBundle,
    trajectory: BaseTrajectoryProvider,
    *,
    n_steps: int,
) -> ClosedLoopResult:
    """Run one closed-loop trajectory rollout."""
    if n_steps <= 0:
        msg = f"n_steps must be positive, got {n_steps}."
        raise ValueError(msg)

    states = np.zeros((n_steps + 1, model.nx), dtype=float)
    controls = np.zeros((n_steps, model.nu), dtype=float)
    reference_states = np.zeros((n_steps + 1, model.nx), dtype=float)
    reference_controls = np.zeros((n_steps, model.nu), dtype=float)
    stats: list[dict] = []

    states[0] = np.asarray(model.x0, dtype=float)
    controller.reset(states[0])
    simulator.reset(states[0])

    for step in range(n_steps):
        reference = trajectory.reference_at(step)
        reference_states[step] = reference.x[0]
        if step == n_steps - 1:
            reference_states[step + 1] = reference.x[1]
        reference_controls[step] = np.zeros(model.nu) if reference.u is None else reference.u[0]
        controls[step] = controller.solve(states[step], reference=reference)
        stats.append(controller.get_stats())
        states[step + 1] = simulator.step(states[step], controls[step])

    return ClosedLoopResult(
        states=states,
        controls=controls,
        reference_states=reference_states,
        reference_controls=reference_controls,
        stats=stats,
    )
