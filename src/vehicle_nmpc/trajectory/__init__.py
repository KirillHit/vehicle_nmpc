"""Trajectory provider definitions and configuration types."""

from .base import BaseTrajectoryConfig, BaseTrajectoryProvider
from .builder import build_trajectory, register_trajectory
from .providers import (
    FigureEightTrajectoryProvider,
    SawTrajectoryProvider,
    SCurveTrajectoryProvider,
    StraightDynTrajectoryProvider,
    StraightTrajectoryProvider,
    TurnTrajectoryProvider,
)

__all__ = [
    "BaseTrajectoryConfig",
    "BaseTrajectoryProvider",
    "FigureEightTrajectoryProvider",
    "SCurveTrajectoryProvider",
    "SawTrajectoryProvider",
    "StraightDynTrajectoryProvider",
    "StraightTrajectoryProvider",
    "TurnTrajectoryProvider",
    "build_trajectory",
    "register_trajectory",
]
