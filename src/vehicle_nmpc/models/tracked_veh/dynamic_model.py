"""Tracked vehicle dynamic model implementation."""

from dataclasses import dataclass, field

import numpy as np
from acados_template import AcadosModel
from casadi import SX, cos, sign, sin, sqrt, vertcat
from omegaconf import MISSING

from vehicle_nmpc.models import BaseModel, BaseModelConfig, ModelBundle, register_model
from vehicle_nmpc.utils.validation import as_vector


@dataclass(frozen=True, kw_only=True, slots=True)
class TrackedVehicleReferenceModel:
    """Model-specific data needed to generate tracked vehicle references."""

    sprocket_radius: float
    """Drive sprocket radius."""

    track_width: float
    """Distance between left and right track center lines."""

    left_slip: float
    """Left track slip coefficient."""

    right_slip: float
    """Right track slip coefficient."""

    def control_reference(self, speed: np.ndarray, yaw_rate: np.ndarray) -> np.ndarray:
        """Convert body speed and yaw rate references to track angular speeds."""
        left_speed = speed - 0.5 * self.track_width * yaw_rate
        right_speed = speed + 0.5 * self.track_width * yaw_rate
        omega_l = left_speed / (self.sprocket_radius * (1.0 - self.left_slip))
        omega_r = right_speed / (self.sprocket_radius * (1.0 - self.right_slip))
        return np.column_stack((omega_l, omega_r))


@register_model("tracked_veh_dynamic")
class TrackedVehDynamicModel(BaseModel):
    """Dynamic skid-steering tracked vehicle model based on the report formulas."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModelConfig):
        """Tracked vehicle dynamic model configuration."""

        sprocket_radius: float = MISSING
        """Drive sprocket radius."""

        track_width: float = MISSING
        """Distance between left and right track center lines."""

        track_contact_length: float = MISSING
        """Length of the track contact patch."""

        left_slip: float = 0.0
        """Left track slip coefficient."""

        right_slip: float = 0.0
        """Right track slip coefficient."""

        longitudinal_resistance: float = MISSING
        """Coefficient of longitudinal resistance μ_l."""

        lateral_resistance: float = MISSING
        """Coefficient of transverse resistance μ_t."""

        gravity: float = 9.81
        """Acceleration due to gravity."""

        mass: float = MISSING
        """Vehicle mass m."""

        inertia: float = MISSING
        """Moment of inertia I_z."""

        drive_force_coefficient: float = 1.0
        """Mapping from track speed to drive force."""

        x0: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        """Default initial state [X, Y, theta, Vx, Vy, omega]."""

    def __init__(self, cfg: Config) -> None:
        """Initialize the model with the provided configuration."""
        super().__init__(cfg)

    def build(self) -> ModelBundle:  # noqa: PLR0915
        """Build and return the Acados model bundle."""
        model_name = "tracked_veh_dynamic"

        # states & controls
        pos_x = SX.sym("X")
        pos_y = SX.sym("Y")
        yaw = SX.sym("theta")
        vel_x = SX.sym("Vx")
        vel_y = SX.sym("Vy")
        yaw_rate = SX.sym("omega")

        x = vertcat(pos_x, pos_y, yaw, vel_x, vel_y, yaw_rate)

        omega_l = SX.sym("omega_l")
        omega_r = SX.sym("omega_r")
        u = vertcat(omega_l, omega_r)

        # xdot
        pos_x_dot = SX.sym("X_dot")
        pos_y_dot = SX.sym("Y_dot")
        yaw_dot = SX.sym("theta_dot")
        vel_x_dot = SX.sym("Vx_dot")
        vel_y_dot = SX.sym("Vy_dot")
        yaw_rate_dot = SX.sym("omega_dot")

        xdot = vertcat(pos_x_dot, pos_y_dot, yaw_dot, vel_x_dot, vel_y_dot, yaw_rate_dot)

        # parameters
        sprocket_radius = SX.sym("r")
        track_width = SX.sym("B")
        left_slip = SX.sym("i_l")
        right_slip = SX.sym("i_r")
        longitudinal_resistance = SX.sym("mu_l")
        lateral_resistance = SX.sym("mu_t")
        track_contact_length = SX.sym("l")
        gravity = SX.sym("g")
        mass = SX.sym("m")
        inertia = SX.sym("I")
        drive_force_coefficient = SX.sym("k_f")

        p = vertcat(
            sprocket_radius,
            track_width,
            left_slip,
            right_slip,
            longitudinal_resistance,
            lateral_resistance,
            track_contact_length,
            gravity,
            mass,
            inertia,
            drive_force_coefficient,
        )

        left_track_speed = sprocket_radius * omega_l * (1.0 - left_slip)
        right_track_speed = sprocket_radius * omega_r * (1.0 - right_slip)

        r_l = longitudinal_resistance * mass * gravity / 2.0
        r_r = r_l

        delta_speed = right_track_speed - left_track_speed
        sum_speed = right_track_speed + left_track_speed
        eps = 1e-6
        radius_prime = track_width * sum_speed / (2.0 * (delta_speed + eps))
        yaw_rate_cmd = delta_speed / (track_width + eps)

        tan_beta = (
            track_contact_length
            * yaw_rate_cmd
            * yaw_rate_cmd
            / (2.0 * lateral_resistance * gravity)
        )
        cos_beta = 1.0 / sqrt(1.0 + tan_beta * tan_beta)
        sin_beta = tan_beta * cos_beta
        s0 = radius_prime * tan_beta

        speed = sqrt(vel_x * vel_x + vel_y * vel_y)
        a_c = speed * speed / (radius_prime + eps)

        f_y = sign(yaw_rate) * 2.0 * lateral_resistance * mass * gravity * s0 / track_contact_length

        f_l = drive_force_coefficient * left_track_speed
        f_r = drive_force_coefficient * right_track_speed
        m = (f_r - f_l) * track_width / 2.0

        m_r = (
            sign(yaw_rate)
            * lateral_resistance
            * mass
            * gravity
            / track_contact_length
            * (s0 * s0 - (track_contact_length * track_contact_length) / 4.0)
        )

        f_expl = vertcat(
            cos(yaw) * vel_x - sin(yaw) * vel_y,
            sin(yaw) * vel_x + cos(yaw) * vel_y,
            yaw_rate,
            (f_l + f_r - mass * a_c * sin_beta - r_l - r_r) / mass,
            (f_y - mass * a_c * cos_beta) / mass,
            (m - m_r + mass * a_c * s0 * cos_beta) / inertia,
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

        model.x_labels = [
            "$X$ [m]",
            "$Y$ [m]",
            r"$\theta$ [rad]",
            "$V_x$ [m/s]",
            "$V_y$ [m/s]",
            r"$\omega$ [rad/s]",
        ]
        model.u_labels = [r"$\omega_l$ [rad/s]", r"$\omega_r$ [rad/s]"]
        model.t_label = "$t$ [s]"

        return ModelBundle(
            model=model,
            nx=6,
            nu=2,
            np=11,
            p0=as_vector(
                "p0",
                [
                    self._cfg.sprocket_radius,
                    self._cfg.track_width,
                    self._cfg.left_slip,
                    self._cfg.right_slip,
                    self._cfg.longitudinal_resistance,
                    self._cfg.lateral_resistance,
                    self._cfg.track_contact_length,
                    self._cfg.gravity,
                    self._cfg.mass,
                    self._cfg.inertia,
                    self._cfg.drive_force_coefficient,
                ],
                11,
            ),
            x0=as_vector("x0", self._cfg.x0, 6),
            trajectory_reference_model=TrackedVehicleReferenceModel(
                sprocket_radius=self._cfg.sprocket_radius,
                track_width=self._cfg.track_width,
                left_slip=self._cfg.left_slip,
                right_slip=self._cfg.right_slip,
            ),
        )
