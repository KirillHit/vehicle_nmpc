"""Custom exceptions."""


class VehicleNMPCError(Exception):
    """Base exception."""


class ModelCreationError(VehicleNMPCError):
    """Raised when a model cannot be created."""
