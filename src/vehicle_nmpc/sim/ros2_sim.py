"""ROS 2 simulator implementation for external plants."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING

from omegaconf import MISSING

from vehicle_nmpc.sim.base import BaseSimulator, BaseSimulatorConfig
from vehicle_nmpc.sim.builder import register_simulator
from vehicle_nmpc.utils.exceptions import SimulatorCreationError
from vehicle_nmpc.utils.validation import as_vector, require_positive

if TYPE_CHECKING:
    import numpy as np

    from vehicle_nmpc.models import ModelBundle
    from vehicle_nmpc.problem import ProblemBundle

_COMMAND_MODE_CONSTANTS = {
    "track_speed": "COMMAND_MODE_TRACK_SPEED",
    "track_torque": "COMMAND_MODE_TRACK_TORQUE",
}


@register_simulator("ros2")
class Ros2Simulator(BaseSimulator):
    """Synchronous ROS 2 client for simulator or robot plant integrations."""

    @dataclass(frozen=True, kw_only=True, slots=True)
    class Config(BaseSimulatorConfig):
        """ROS 2 simulator configuration."""

        node_name: str = "vehicle_nmpc_ros2_sim"
        """ROS 2 node name used by the NMPC client."""

        namespace: str = "/vehicle"
        """ROS namespace where plant services are exposed."""

        step_service: str = "step"
        """Relative or absolute step service name."""

        reset_service: str = "reset"
        """Relative or absolute reset service name."""

        command_mode: str = MISSING
        """Command interpretation mode: track_speed or track_torque."""

        wait_timeout_sec: float = 10.0
        """Timeout for initial service discovery."""

        service_timeout_sec: float = 5.0
        """Timeout for individual service calls."""

        spin_period_sec: float = 0.01
        """Polling period while waiting for service futures."""

        def __post_init__(self) -> None:
            """Validate ROS 2 simulator configuration."""
            require_positive("wait_timeout_sec", self.wait_timeout_sec)
            require_positive("service_timeout_sec", self.service_timeout_sec)
            require_positive("spin_period_sec", self.spin_period_sec)
            if not self.node_name:
                msg = "node_name must be non-empty."
                raise ValueError(msg)
            if not self.step_service:
                msg = "step_service must be non-empty."
                raise ValueError(msg)
            if not self.reset_service:
                msg = "reset_service must be non-empty."
                raise ValueError(msg)
            if self.command_mode not in _COMMAND_MODE_CONSTANTS:
                modes = ", ".join(sorted(_COMMAND_MODE_CONSTANTS))
                msg = f"command_mode must be one of: {modes}."
                raise ValueError(msg)

    def __init__(self, cfg: Config, problem: ProblemBundle, model: ModelBundle) -> None:
        """Initialize the ROS 2 plant client."""
        super().__init__(cfg, problem, model)
        self._rclpy, self._step_type, self._reset_type = self._import_ros_dependencies()
        self._owns_context = not self._rclpy.ok()
        if self._owns_context:
            self._rclpy.init(args=None)

        self._node = self._rclpy.create_node(self._cfg.node_name)
        self._step_client = self._node.create_client(
            self._step_type,
            self._service_name(self._cfg.step_service),
        )
        self._reset_client = self._node.create_client(
            self._reset_type,
            self._service_name(self._cfg.reset_service),
        )
        self._command_mode = self._resolve_command_mode()

        self._wait_for_service(self._step_client, self._cfg.step_service)
        self._wait_for_service(self._reset_client, self._cfg.reset_service)

    def reset(self, x0: np.ndarray) -> None:
        """Reset or synchronize the external plant for a new episode."""
        state = self._validate_state(x0)

        request = self._reset_type.Request()
        request.state = state.tolist()
        response = self._call_service(self._reset_client, request, "reset")
        self._require_success(response, "reset")
        self._validate_state(response.state)

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Apply one control command and return the resulting plant state."""
        self._validate_state(x)
        command = self._validate_control(u)

        request = self._step_type.Request()
        request.dt = self._problem.dt
        request.command_mode = self._command_mode
        request.command = command.tolist()

        response = self._call_service(self._step_client, request, "step")
        self._require_success(response, "step")

        return self._validate_state(response.state)

    def close(self) -> None:
        """Destroy ROS resources owned by this simulator."""
        if hasattr(self, "_node"):
            self._node.destroy_node()
        if getattr(self, "_owns_context", False) and self._rclpy.ok():
            self._rclpy.shutdown()

    def _validate_state(self, x: np.ndarray | list[float]) -> np.ndarray:
        """Convert and validate a state vector."""
        return as_vector("state", x, self._model.nx)

    def _validate_control(self, u: np.ndarray) -> np.ndarray:
        """Convert and validate a control vector."""
        return as_vector("control", u, self._model.nu)

    def _service_name(self, service_name: str) -> str:
        """Return an absolute service name under the configured namespace."""
        if service_name.startswith("/"):
            return service_name
        namespace = self._cfg.namespace.strip("/")
        if not namespace:
            return f"/{service_name}"
        return f"/{namespace}/{service_name}"

    def _resolve_command_mode(self) -> int:
        """Resolve configured command mode to the generated service constant."""
        constant_name = _COMMAND_MODE_CONSTANTS[self._cfg.command_mode]
        return int(getattr(self._step_type.Request, constant_name))

    def _wait_for_service(self, client: object, service_name: str) -> None:
        """Wait until a ROS service becomes available."""
        if client.wait_for_service(timeout_sec=self._cfg.wait_timeout_sec):
            return
        msg = (
            f"ROS 2 service '{self._service_name(service_name)}' was not available "
            f"within {self._cfg.wait_timeout_sec} seconds."
        )
        raise TimeoutError(msg)

    def _call_service(self, client: object, request: object, operation: str) -> object:
        """Call a ROS service and block until a response or timeout is received."""
        future = client.call_async(request)
        deadline = monotonic() + self._cfg.service_timeout_sec
        while self._rclpy.ok():
            self._rclpy.spin_once(self._node, timeout_sec=self._cfg.spin_period_sec)
            if future.done():
                response = future.result()
                if response is None:
                    msg = f"ROS 2 {operation} service returned no response."
                    raise RuntimeError(msg)
                return response
            if monotonic() >= deadline:
                msg = (
                    f"ROS 2 {operation} service call timed out after "
                    f"{self._cfg.service_timeout_sec} seconds."
                )
                raise TimeoutError(msg)

        msg = f"ROS 2 context shut down while waiting for {operation} service response."
        raise RuntimeError(msg)

    @staticmethod
    def _require_success(response: object, operation: str) -> None:
        """Raise if a service response reports failure."""
        if response.success:
            return
        detail = f": {response.message}" if response.message else "."
        msg = f"ROS 2 {operation} service failed{detail}"
        raise RuntimeError(msg)

    @staticmethod
    def _import_ros_dependencies() -> tuple[object, object, object]:
        """Import ROS dependencies lazily so non-ROS simulator paths keep working."""
        try:
            import rclpy
            from vehicle_nmpc_interfaces.srv import ResetSimulation, StepSimulation
        except ImportError as exc:
            msg = (
                "Ros2Simulator requires ROS 2 Python packages and "
                "vehicle_nmpc_interfaces to be available."
            )
            raise SimulatorCreationError(msg) from exc
        return rclpy, StepSimulation, ResetSimulation

    def __del__(self) -> None:
        """Best-effort cleanup for interactive sessions."""
        with suppress(Exception):
            self.close()
