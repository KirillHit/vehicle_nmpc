"""Trajectory provider definitions and configuration types."""

from .base import BaseTrajectoryConfig, BaseTrajectoryProvider
from .builder import build_trajectory, register_trajectory
from .providers import (
    SCurveTrajectoryProvider,
    StraightTrajectoryProvider,
    TurnTrajectoryProvider,
    SawTrajectoryProvider,
    FigureEightTrajectoryProvider,
)

__all__ = [
    "BaseTrajectoryConfig",
    "BaseTrajectoryProvider",
    "FigureEightTrajectoryProvider",
    "SawTrajectoryProvider",
    "SCurveTrajectoryProvider",
    "StraightTrajectoryProvider",
    "TurnTrajectoryProvider",
    "build_trajectory",
    "register_trajectory",
]
