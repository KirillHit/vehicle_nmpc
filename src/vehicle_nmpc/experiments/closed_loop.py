"""Closed-loop controller evaluation workflows."""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from vehicle_nmpc.controller import build_controller
from vehicle_nmpc.metrics import (
    EvaluationMetrics,
    evaluate_control,
    evaluate_performance,
    evaluate_tracking,
    save_control_plot,
    save_itae_plot,
    save_speed_plot,
    save_tracking_error_plot,
    save_trajectory_plot,
)
from vehicle_nmpc.models import build_model
from vehicle_nmpc.problem import build_problem
from vehicle_nmpc.sim import build_simulator
from vehicle_nmpc.trajectory import build_trajectory
from vehicle_nmpc.utils.validation import require_positive

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vehicle_nmpc.controller import BaseController
    from vehicle_nmpc.models import ModelBundle
    from vehicle_nmpc.problem import ProblemBundle
    from vehicle_nmpc.sim import BaseSimulator
    from vehicle_nmpc.trajectory import BaseTrajectoryProvider
    from vehicle_nmpc.utils.config import BaseConfig

log = logging.getLogger(__name__)

_PROGRESS_LOG_INTERVAL_S = 2.0


@dataclass(kw_only=True, slots=True)
class ClosedLoopResult:
    """Closed-loop rollout data."""

    trajectory_name: str
    """Name of the evaluated trajectory provider."""

    states: np.ndarray
    """Simulated state trajectory with shape (n_steps + 1, nx)."""

    controls: np.ndarray
    """Applied control trajectory with shape (n_steps, nu)."""

    reference_states: np.ndarray
    """Reference state trajectory with shape (n_steps + 1, nx)."""

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


def run_configured_evaluation(
    cfg: BaseConfig,
) -> tuple[list[ClosedLoopResult], list[EvaluationMetrics], float]:
    """Build configured closed-loop components, run trajectories, and evaluate metrics."""
    model, problem, controller, simulator, trajectories = prepare_closed_loop(cfg)
    if not trajectories:
        msg = "Closed-loop evaluation requires at least one configured trajectory."
        raise ValueError(msg)

    n_steps = _simulation_steps(cfg.runner.simulation_time, dt=problem.dt)
    results = run_evaluation(controller, simulator, model, trajectories, n_steps=n_steps)
    return results, evaluate_results(results, dt=problem.dt), problem.dt


def save_evaluation_artifacts(
    results: list[ClosedLoopResult],
    metrics: list[EvaluationMetrics],
    output_dir: str | Path,
    *,
    dt: float,
    extra_metrics: Mapping[str, object] | None = None,
) -> None:
    """Save trajectory plots and metrics for closed-loop evaluation results."""
    output_path = Path(output_dir)
    metrics_dir = output_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for result, item in zip(results, metrics, strict=True):
        safe_name = _safe_artifact_name(result.trajectory_name)
        save_trajectory_plot(
            result.states,
            result.reference_states,
            output_path / "plots" / f"{safe_name}.png",
            title=result.trajectory_name,
        )
        save_tracking_error_plot(
            result.states,
            result.reference_states,
            output_path / "plots" / f"{safe_name}_errors.png",
            title=result.trajectory_name,
            dt=dt,
        )
        save_itae_plot(
            result.states,
            result.reference_states,
            output_path / "plots" / f"{safe_name}_itae.png",
            title=result.trajectory_name,
            dt=dt,
        )
        save_control_plot(
            result.controls,
            output_path / "plots" / f"{safe_name}_control.png",
            title=result.trajectory_name,
            dt=dt,
        )
        save_speed_plot(
            result.states,
            result.reference_states,
            output_path / "plots" / f"{safe_name}_speed.png",
            title=result.trajectory_name,
            dt=dt,
        )
        item_data = asdict(item)
        summary.append(item_data)
        _write_json(metrics_dir / f"{safe_name}.json", item_data)
    for name, data in (extra_metrics or {}).items():
        _write_json(metrics_dir / f"{_safe_artifact_name(name)}.json", data)
    _write_json(metrics_dir / "summary.json", summary)


def prepare_closed_loop(
    cfg: BaseConfig,
) -> tuple[
    ModelBundle,
    ProblemBundle,
    BaseController,
    BaseSimulator,
    list[BaseTrajectoryProvider],
]:
    """Build and reset configured closed-loop components."""
    model = build_model(cfg.model)
    problem = build_problem(cfg.problem, model)
    controller = build_controller(cfg.controller, problem, model)
    simulator = build_simulator(cfg.sim, problem, model)
    trajectories = [
        build_trajectory(trajectory_cfg, model, problem) for trajectory_cfg in cfg.trajectories
    ]

    controller.reset(model.x0)
    simulator.reset(model.x0)
    return model, problem, controller, simulator, trajectories


def evaluate_result(result: ClosedLoopResult, *, dt: float) -> EvaluationMetrics:
    """Evaluate all metrics for one closed-loop rollout."""
    return EvaluationMetrics(
        trajectory_name=result.trajectory_name,
        tracking=evaluate_tracking(result, dt=dt),
        performance=evaluate_performance(result, dt=dt),
        control=evaluate_control(result),
    )


def evaluate_results(results: list[ClosedLoopResult], *, dt: float) -> list[EvaluationMetrics]:
    """Evaluate all metrics for a list of closed-loop rollouts."""
    return [evaluate_result(result, dt=dt) for result in results]


def _simulation_steps(simulation_time: float, *, dt: float) -> int:
    """Return the rollout step count required to cover simulation_time seconds."""
    require_positive("simulation_time", simulation_time)
    require_positive("dt", dt)
    return math.ceil(simulation_time / dt)


def _safe_artifact_name(name: str) -> str:
    """Return a filesystem-safe artifact name."""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_") or "artifact"


def _write_json(path: Path, data: object) -> None:
    """Write JSON data to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _log_progress(
    trajectory_name: str,
    step: int,
    n_steps: int,
    state: np.ndarray,
    control: np.ndarray,
) -> None:
    """Log one closed-loop progress sample."""
    log.debug(
        "%s step %d/%d (%.1f%%) state=%s control=%s",
        trajectory_name,
        step,
        n_steps,
        100.0 * step / n_steps,
        np.array2string(state, precision=4, suppress_small=True),
        np.array2string(control, precision=4, suppress_small=True),
    )


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
    stats: list[dict] = []

    trajectory.reset()
    states[0] = trajectory.initial_state()
    controller.reset(states[0])
    simulator.reset(states[0])

    last_progress_log = time.monotonic()
    for step in range(n_steps):
        reference = trajectory.reference_at(states[step])
        reference_states[step] = reference.x[0]
        if step == n_steps - 1:
            reference_states[step + 1] = reference.x[1]
        controls[step] = controller.solve(states[step], reference=reference)
        stats.append(controller.get_stats())
        states[step + 1] = simulator.step(states[step], controls[step])

        now = time.monotonic()
        if now - last_progress_log >= _PROGRESS_LOG_INTERVAL_S or step == n_steps - 1:
            _log_progress(trajectory.name, step + 1, n_steps, states[step + 1], controls[step])
            last_progress_log = now

    return ClosedLoopResult(
        trajectory_name=trajectory.name,
        states=states,
        controls=controls,
        reference_states=reference_states,
        stats=stats,
    )
