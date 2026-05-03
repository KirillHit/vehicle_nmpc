"""Tracked vehicle kinematic model implementation."""

from dataclasses import dataclass, field

from acados_template import AcadosModel
from casadi import SX, cos, sin, vertcat
from omegaconf import MISSING

from vehicle_nmpc.models import BaseModel, BaseModelConfig, ModelBundle, register_model
from vehicle_nmpc.utils.validation import as_vector


@register_model("tracked_veh_kinematic")
class TrackedVehKinematicModel(BaseModel):
    """Kinematic model wrapper that builds an Acados model."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModelConfig):
        """Tracked vehicle kinematic model configuration."""

        width: float = MISSING
        x0: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
        """Default initial state."""

    def __init__(self, cfg: Config) -> None:
        """Initialize the model with the provided configuration."""
        super().__init__(cfg)

    def build(self) -> ModelBundle:
        """Build and return the Acados model bundle."""
        model_name = "pendulum_ode"

        # constants
        cart_mass = 1.0  # [kg]
        bob_mass = 0.1  # [kg]
        gravity = 9.81  # [m/s^2]
        rod_len = 0.8  # [m]

        # states & controls
        pos = SX.sym("x")
        angle = SX.sym("theta")
        vel = SX.sym("v")
        ang_vel = SX.sym("theta_dot")

        x = vertcat(pos, angle, vel, ang_vel)

        force = SX.sym("F")
        u = vertcat(force)

        # xdot
        pos_dot = SX.sym("x_dot")
        angle_dot = SX.sym("theta_dot")
        vel_dot = SX.sym("v_dot")
        ang_acc = SX.sym("theta_ddot")

        xdot = vertcat(pos_dot, angle_dot, vel_dot, ang_acc)

        # dynamics
        c = cos(angle)
        s = sin(angle)
        denom = cart_mass + bob_mass - bob_mass * c * c

        f_expl = vertcat(
            vel,
            ang_vel,
            (-bob_mass * rod_len * s * ang_vel * ang_vel + bob_mass * gravity * c * s + force)
            / denom,
            (
                -bob_mass * rod_len * c * s * ang_vel * ang_vel
                + force * c
                + (cart_mass + bob_mass) * gravity * s
            )
            / (rod_len * denom),
        )

        f_impl = xdot - f_expl

        model = AcadosModel()
        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.name = model_name

        model.x_labels = ["$x$ [m]", r"$\theta$ [rad]", "$v$ [m/s]", r"$\dot{\theta}$ [rad/s]"]
        model.u_labels = ["$F$"]
        model.t_label = "$t$ [s]"

        return ModelBundle(
            model=model,
            nx=4,
            nu=1,
            np=0,
            x0=as_vector("x0", self._cfg.x0, 4),
        )
