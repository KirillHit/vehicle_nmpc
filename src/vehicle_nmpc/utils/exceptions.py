"""Custom exceptions."""


class VehicleNMPCError(Exception):
    """Base exception."""


class ModelCreationError(VehicleNMPCError):
    """Raised when a model cannot be created."""


class ControllerCreationError(VehicleNMPCError):
    """Raised when a controller cannot be created."""


class ProblemCreationError(VehicleNMPCError):
    """Raised when a problem cannot be created."""


class SimulatorCreationError(VehicleNMPCError):
    """Raised when a simulator cannot be created."""


class TrajectoryCreationError(VehicleNMPCError):
    """Raised when a trajectory provider cannot be created."""
