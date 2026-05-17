"""Tracked vehicle kinematic model implementation."""

from dataclasses import dataclass, field

from acados_template import AcadosModel
from casadi import SX, cos, sin, sqrt, vertcat
from omegaconf import MISSING

from vehicle_nmpc.models import BaseModel, BaseModelConfig, ModelBundle, register_model
from vehicle_nmpc.utils.validation import as_vector, require_positive, require_slip


@register_model("tracked_veh_kinematic")
class TrackedVehKinematicModel(BaseModel):
    """Kinematic skid-steering tracked vehicle model."""

    @dataclass(frozen=True, kw_only=True, slots=True)
    class Config(BaseModelConfig):
        """Tracked vehicle kinematic model configuration."""

        sprocket_radius: float = MISSING
        """Drive sprocket radius."""

        track_width: float = MISSING
        """Distance between left and right track center lines."""

        track_contact_length: float = MISSING
        """Track-ground contact length."""

        left_slip: float = 0.0
        """Left track slip coefficient."""

        right_slip: float = 0.0
        """Right track slip coefficient."""

        lateral_resistance: float = MISSING
        """Lateral resistance coefficient."""

        gravity: float = 9.81
        """Gravity acceleration."""

        yaw_rate_sign_epsilon: float = 0.05
        """Smoothing scale for yaw-rate sign."""

        x0: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
        """Default initial state."""

        def __post_init__(self) -> None:
            """Validate physical model parameters."""
            require_positive("sprocket_radius", self.sprocket_radius)
            require_positive("track_width", self.track_width)
            require_positive("track_contact_length", self.track_contact_length)
            require_positive("lateral_resistance", self.lateral_resistance)
            require_positive("gravity", self.gravity)
            require_positive("yaw_rate_sign_epsilon", self.yaw_rate_sign_epsilon)
            require_slip("left_slip", self.left_slip)
            require_slip("right_slip", self.right_slip)

    def __init__(self, cfg: Config) -> None:
        """Initialize the model with the provided configuration."""
        super().__init__(cfg)

    def build(self) -> ModelBundle:
        """Build and return the Acados model bundle."""
        model_name = "tracked_veh_kinematic"

        # states & controls
        pos_x = SX.sym("X")
        pos_y = SX.sym("Y")
        yaw = SX.sym("theta")

        x = vertcat(pos_x, pos_y, yaw)

        omega_l = SX.sym("omega_l")
        omega_r = SX.sym("omega_r")
        u = vertcat(omega_l, omega_r)

        # xdot
        pos_x_dot = SX.sym("X_dot")
        pos_y_dot = SX.sym("Y_dot")
        yaw_dot = SX.sym("theta_dot")

        xdot = vertcat(pos_x_dot, pos_y_dot, yaw_dot)

        # parameters
        sprocket_radius = SX.sym("r")
        track_width = SX.sym("B")
        left_slip = SX.sym("i_l")
        right_slip = SX.sym("i_r")
        lateral_resistance = SX.sym("mu_t")
        track_contact_length = SX.sym("l")
        gravity = SX.sym("g")
        yaw_rate_sign_epsilon = SX.sym("eps_omega")

        p = vertcat(
            sprocket_radius,
            track_width,
            left_slip,
            right_slip,
            lateral_resistance,
            track_contact_length,
            gravity,
            yaw_rate_sign_epsilon,
        )

        left_track_speed = sprocket_radius * omega_l * (1.0 - left_slip)
        right_track_speed = sprocket_radius * omega_r * (1.0 - right_slip)

        yaw_rate = (right_track_speed - left_track_speed) / track_width
        longitudinal_speed = (right_track_speed + left_track_speed) / 2.0
        yaw_rate_sign = self._smooth_sign(yaw_rate, yaw_rate_sign_epsilon)
        tan_beta = (
            track_contact_length
            * yaw_rate
            * yaw_rate
            * yaw_rate_sign
            / (2.0 * lateral_resistance * gravity)
        )
        lateral_speed = -longitudinal_speed * tan_beta

        f_expl = vertcat(
            cos(yaw) * longitudinal_speed - sin(yaw) * lateral_speed,
            sin(yaw) * longitudinal_speed + cos(yaw) * lateral_speed,
            yaw_rate,
        )

        f_impl = xdot - f_expl

        model = AcadosModel()
        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.p = p
        model.name = model_name

        model.x_labels = ["$X$ [m]", "$Y$ [m]", r"$\theta$ [rad]"]
        model.u_labels = [r"$\omega_l$ [rad/s]", r"$\omega_r$ [rad/s]"]
        model.t_label = "$t$ [s]"

        return ModelBundle(
            model=model,
            nx=3,
            nu=2,
            np=8,
            p0=as_vector(
                "p0",
                [
                    self._cfg.sprocket_radius,
                    self._cfg.track_width,
                    self._cfg.left_slip,
                    self._cfg.right_slip,
                    self._cfg.lateral_resistance,
                    self._cfg.track_contact_length,
                    self._cfg.gravity,
                    self._cfg.yaw_rate_sign_epsilon,
                ],
                8,
            ),
            x0=as_vector("x0", self._cfg.x0, 3),
        )

    def _smooth_sign(self, value: SX, eps: SX) -> SX:
        """Return a differentiable sign approximation."""
        return value / sqrt(value * value + eps * eps)
