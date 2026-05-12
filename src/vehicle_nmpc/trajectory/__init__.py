"""Trajectory provider definitions and configuration types."""

from .base import BaseTrajectoryConfig, BaseTrajectoryProvider
from .builder import build_trajectory, register_trajectory
from .providers import (
    FigureEightTrajectoryProvider,
    SawTrajectoryProvider,
    SCurveTrajectoryProvider,
    StraightTrajectoryProvider,
    TurnTrajectoryProvider,
)

__all__ = [
    "BaseTrajectoryConfig",
    "BaseTrajectoryProvider",
    "FigureEightTrajectoryProvider",
    "SCurveTrajectoryProvider",
    "SawTrajectoryProvider",
    "StraightTrajectoryProvider",
    "TurnTrajectoryProvider",
    "build_trajectory",
    "register_trajectory"
]
