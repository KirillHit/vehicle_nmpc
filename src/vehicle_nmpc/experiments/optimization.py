"""Controller parameter optimization workflows."""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import optuna

from vehicle_nmpc.experiments.closed_loop import (
    run_configured_evaluation,
    save_evaluation_artifacts,
)

if TYPE_CHECKING:
    from vehicle_nmpc.metrics import EvaluationMetrics
    from vehicle_nmpc.utils.config import BaseConfig, ObjectiveComponent, OptunaParameter

log = logging.getLogger(__name__)

Aggregation = Callable[[Sequence[float]], float]


def _mean(values: Sequence[float]) -> float:
    """Return mean value."""
    return float(sum(values) / len(values))


def _sum(values: Sequence[float]) -> float:
    """Return sum value."""
    return float(sum(values))


def _max(values: Sequence[float]) -> float:
    """Return maximum value."""
    return float(max(values))


def _min(values: Sequence[float]) -> float:
    """Return minimum value."""
    return float(min(values))


_AGGREGATIONS: Mapping[str, Aggregation] = {
    "mean": _mean,
    "sum": _sum,
    "max": _max,
    "min": _min,
}


def run_optimization(cfg: BaseConfig) -> optuna.Study:
    """Run Optuna optimization over configured closed-loop evaluations."""
    if not cfg.optuna.parameters:
        msg = "Optimize mode requires at least one optuna parameter."
        raise ValueError(msg)
    if not cfg.optuna.objective.components:
        msg = "Optimize mode requires at least one objective component."
        raise ValueError(msg)

    study = optuna.create_study(direction=cfg.optuna.objective.direction)
    study.optimize(
        lambda trial: _objective(trial, cfg),
        n_trials=cfg.optuna.n_trials,
    )

    log.info("Best trial: %d", study.best_trial.number)
    log.info("Best value: %.6g", study.best_value)
    log.info("Best params: %s", study.best_params)
    _run_best_estimate(cfg, study.best_trial)
    return study


def _objective(trial: optuna.Trial, cfg: BaseConfig) -> float:
    """Evaluate one Optuna trial."""
    trial_cfg = _configured_copy(
        cfg,
        {},
        Path(cfg.runner.output_dir) / "trials" / f"trial_{trial.number:04d}",
    )
    for parameter in trial_cfg.optuna.parameters:
        _set_config_value(trial_cfg, parameter.parameter, _suggest_value(trial, parameter))

    try:
        results, metrics = run_configured_evaluation(trial_cfg)
    except Exception as exc:
        trial.set_user_attr("error", repr(exc))
        log.exception("Trial %d failed.", trial.number)
        return math.inf if trial_cfg.optuna.objective.direction == "minimize" else -math.inf

    value = _objective_value(metrics, trial_cfg.optuna.objective.components)
    save_evaluation_artifacts(
        results,
        metrics,
        trial_cfg.runner.output_dir,
        extra_metrics={
            "trial": {
                "number": trial.number,
                "value": value,
                "params": trial.params,
            },
        },
    )
    for item in metrics:
        log.info("Trial %d: %s", trial.number, item.print())
    log.info("Trial %d objective: %.6g", trial.number, value)
    return value


def _run_best_estimate(cfg: BaseConfig, best_trial: optuna.trial.FrozenTrial) -> None:
    """Run and save a standard estimate pass for the best optimized parameters."""
    if not math.isfinite(float(best_trial.value)):
        log.warning("Skipping best estimate because best trial value is not finite.")
        return

    best_cfg = _configured_copy(
        cfg,
        best_trial.params,
        Path(cfg.runner.output_dir) / "best_estimate",
    )
    results, metrics = run_configured_evaluation(best_cfg)
    save_evaluation_artifacts(
        results,
        metrics,
        best_cfg.runner.output_dir,
        extra_metrics={
            "best_trial": {
                "number": best_trial.number,
                "value": best_trial.value,
                "params": best_trial.params,
            },
        },
    )
    for item in metrics:
        log.info("Best estimate: %s", item.print())


def _configured_copy(
    cfg: BaseConfig,
    params: Mapping[str, object],
    output_dir: Path,
) -> BaseConfig:
    """Return a config copy with optimized params and run output directories applied."""
    result = copy.deepcopy(cfg)
    result.runner.output_dir = str(output_dir)
    if "artifacts_dir" in result.problem.params:
        result.problem.params["artifacts_dir"] = str(output_dir / "acados")
    for parameter, value in params.items():
        _set_config_value(result, parameter, value)
    return result


def _suggest_value(trial: optuna.Trial, parameter: OptunaParameter) -> object:
    """Sample one configured Optuna parameter."""
    match parameter.type:
        case "float":
            if parameter.low is None or parameter.high is None:
                msg = f"Float parameter '{parameter.parameter}' requires low and high."
                raise ValueError(msg)
            return trial.suggest_float(
                parameter.parameter,
                parameter.low,
                parameter.high,
                log=parameter.log,
            )
        case "int":
            if parameter.low is None or parameter.high is None:
                msg = f"Int parameter '{parameter.parameter}' requires low and high."
                raise ValueError(msg)
            return trial.suggest_int(parameter.parameter, int(parameter.low), int(parameter.high))
        case "categorical":
            if not parameter.choices:
                msg = f"Categorical parameter '{parameter.parameter}' requires choices."
                raise ValueError(msg)
            return trial.suggest_categorical(parameter.parameter, parameter.choices)
        case _:
            msg = (
                f"Unsupported optuna parameter type '{parameter.type}' for "
                f"'{parameter.parameter}'. Expected float, int, or categorical."
            )
            raise ValueError(msg)


def _set_config_value(cfg: BaseConfig, parameter: str, value: object) -> None:
    """Set a nested config value by dotted path."""
    path = parameter.split(".")
    if not path:
        msg = "Optuna parameter path cannot be empty."
        raise ValueError(msg)

    target: object = cfg
    for item in path[:-1]:
        target = _get_child(target, item, parameter)
    _set_child(target, path[-1], value, parameter)


def _get_child(target: object, key: str, parameter: str) -> object:
    """Read one nested config item."""
    if isinstance(target, dict):
        return target[key]
    if isinstance(target, list):
        return target[int(key)]
    if is_dataclass(target):
        return getattr(target, key)

    msg = f"Cannot navigate through '{key}' in optuna parameter '{parameter}'."
    raise TypeError(msg)


def _set_child(target: object, key: str, value: object, parameter: str) -> None:
    """Set one nested config item."""
    if isinstance(target, dict):
        target[key] = value
        return
    if isinstance(target, list):
        target[int(key)] = value
        return
    if is_dataclass(target):
        setattr(target, key, value)
        return

    msg = f"Cannot set '{key}' in optuna parameter '{parameter}'."
    raise TypeError(msg)


def _objective_value(
    metrics: Sequence[EvaluationMetrics],
    components: Sequence[ObjectiveComponent],
) -> float:
    """Aggregate configured objective components over trajectories."""
    if not metrics:
        msg = "Cannot compute objective without evaluation metrics."
        raise ValueError(msg)

    value = 0.0
    for component in components:
        aggregation = _AGGREGATIONS.get(component.aggregation)
        if aggregation is None:
            available = ", ".join(sorted(_AGGREGATIONS))
            msg = (
                f"Unsupported objective aggregation '{component.aggregation}' for "
                f"metric '{component.metric}'. Available aggregations: {available}."
            )
            raise ValueError(msg)
        metric_values = [_metric_value(item, component.metric) for item in metrics]
        value += component.weight * aggregation(metric_values)
    return float(value)


def _metric_value(metrics: EvaluationMetrics, metric: str) -> float:
    """Read a registered metric value by dotted path."""
    target: object = metrics
    for item in metric.split("."):
        if not is_dataclass(target):
            msg = f"Metric path '{metric}' cannot navigate through non-metric value."
            raise ValueError(msg)
        if item not in {field.name for field in fields(target)}:
            msg = (
                f"Unknown metric '{metric}'. Available metrics: "
                f"{', '.join(_registered_metric_paths(metrics))}."
            )
            raise ValueError(msg)
        target = getattr(target, item)
    if isinstance(target, bool | int | float):
        return float(target)

    msg = f"Metric '{metric}' is not a scalar numeric metric."
    raise TypeError(msg)


def _registered_metric_paths(metrics: EvaluationMetrics) -> list[str]:
    """Return dotted paths to scalar numeric fields in EvaluationMetrics."""
    paths: list[str] = []

    def collect(prefix: str, value: object) -> None:
        if not is_dataclass(value):
            return
        for field in fields(value):
            child = getattr(value, field.name)
            child_path = f"{prefix}.{field.name}" if prefix else field.name
            if is_dataclass(child):
                collect(child_path, child)
            elif isinstance(child, bool | int | float):
                paths.append(child_path)

    collect("", metrics)
    return paths
