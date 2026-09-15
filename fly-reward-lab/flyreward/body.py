"""Torque-driven 12-hinge reduced body, not calibrated fly biomechanics.

Six fixed-azimuth two-link legs attach to a translating body whose orientation
is constrained. Coordinates use mm, mg, seconds; forces use mg*mm/s**2. Geometry,
mass, torque, compliance, and friction are declared engineering assumptions.
The 24 input channels preserve the existing LF/LM/LH/RF/RM/RH, trochanter/tibia,
flexor/extensor order. There are no oscillators or imposed movement trajectories.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numpy as np

LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")
JOINTS = ("trochanter", "tibia")


@dataclass(frozen=True)
class BodyConfig:
    body_mass: float = 1.0
    femur_mass: float = 0.01
    distal_mass: float = 0.006
    femur_length: float = 0.70
    distal_length: float = 0.85
    # Positive relative bend closes the femur-tibia interior angle.
    neutral_hip: float = math.pi / 6
    neutral_knee_bend: float = math.pi * 2 / 9
    gravity: float = 9810.0
    max_torque: float = 150.0
    passive_stiffness: float = 120.0
    coactivation_stiffness: float = 100.0
    joint_damping: float = 1.5
    additional_joint_inertia: float = 0.001
    stop_stiffness: float = 4000.0
    stop_damping: float = 3.0
    ground_stiffness: float = 40000.0
    ground_damping: float = 25.0
    friction: float = 0.7
    tangential_damping: float = 20.0
    body_radius: float = 0.20
    max_substep: float = 0.00025


class ReducedBody:
    """A reduced mechanical plant; feedback channels have no assigned neuron IDs.

    Lumped leg masses give a coupled positive-definite mass matrix. Contact forces
    enter both body and joint equations through foot Jacobians. Link inertial
    bias forces are included; orientation is deliberately locked, so this is not
    a free six-degree-of-freedom rigid-body model or a full fly musculoskeletal model.
    """

    def __init__(self, config: BodyConfig | None = None):
        self.config = config or BodyConfig()
        c = self.config
        if any(not np.isfinite(v) or v < 0 for v in asdict(c).values()):
            raise ValueError("Body parameters must be finite and nonnegative")
        if min(c.body_mass, c.additional_joint_inertia, c.max_substep) <= 0:
            raise ValueError("Mass, inertia, and substep must be positive")
        if c.max_substep > 0.00025:
            raise ValueError("Body substep must not exceed 0.00025 seconds")
        self.schema_version = 1
        self.q = np.zeros(15, dtype=float)
        self.q[2] = c.femur_length * math.sin(c.neutral_hip) + c.distal_length * math.sin(c.neutral_hip + c.neutral_knee_bend) + 0.03
        self.dq = np.zeros(15, dtype=float)
        self.activation = np.zeros((6, 2, 2), dtype=float)
        self.t = 0.0
        self.lower = np.tile([-0.45, -0.80], 6)
        self.upper = np.tile([0.75, 0.80], 6)
        self.hips = np.array([[-.55,.24,0],[0,.24,0],[.55,.24,0],[-.55,-.24,0],[0,-.24,0],[.55,-.24,0]],dtype=float)
        self.radial = np.array([[-.6,.8,0],[0,1,0],[.6,.8,0],[-.6,-.8,0],[0,-1,0],[.6,-.8,0]],dtype=float)
        self.last_contact = np.zeros((6, 3), dtype=float)
        self.last_torques = np.zeros(12, dtype=float)
        self.last_body_contact = np.zeros(3, dtype=float)

    def _point(self, leg, first_length, second_length):
        """Point position, d(position)/dq, and Jdot*dq for a leg point."""
        c = self.config
        i = 3 + 2 * leg
        a = c.neutral_hip + self.q[i]
        b = a + c.neutral_knee_bend + self.q[i + 1]
        w1, w2 = self.dq[i], self.dq[i] + self.dq[i + 1]
        r = self.radial[leg]
        down = np.array([0.,0.,-1.])
        v1 = first_length * (math.cos(a) * r + math.sin(a) * down)
        v2 = second_length * (math.cos(b) * r + math.sin(b) * down)
        d1 = first_length * (-math.sin(a) * r + math.cos(a) * down)
        d2 = second_length * (-math.sin(b) * r + math.cos(b) * down)
        jac = np.zeros((3,15),dtype=float)
        jac[:,:3] = np.eye(3)
        jac[:,i] = d1 + d2
        jac[:,i + 1] = d2
        return self.q[:3] + self.hips[leg] + v1 + v2, jac, -v1 * w1 * w1 - v2 * w2 * w2

    def _contact(self, position, velocity, scale=1.0):
        c = self.config
        if position[2] >= 0:
            return np.zeros(3), np.zeros((3,3))
        normal_damping = c.ground_damping * math.sqrt(scale)
        normal = max(0., -c.ground_stiffness * scale * position[2] - normal_damping * velocity[2])
        speed = np.linalg.norm(velocity[:2])
        tangent = np.zeros(2)
        tangent_coefficient = c.tangential_damping
        if speed > 0:
            tangent_coefficient = min(c.tangential_damping, c.friction * normal / speed)
            tangent = -velocity[:2] * tangent_coefficient
        damping = np.diag([tangent_coefficient,tangent_coefficient,normal_damping if normal>0 else 0])
        return np.array([*tangent, normal]), damping

    def _forces(self):
        c = self.config
        mass = np.zeros((15,15),dtype=float)
        mass[:3,:3] = np.eye(3) * c.body_mass
        mass[3:,3:] = np.eye(12) * c.additional_joint_inertia
        damping = np.zeros((15,15),dtype=float)
        damping[3:,3:] = np.eye(12) * c.joint_damping
        force = np.zeros(15)
        force[2] = -c.body_mass * c.gravity
        for leg in range(6):
            for lm, l1, l2 in [(c.femur_mass,c.femur_length*.5,0.),(c.distal_mass,c.femur_length,c.distal_length*.5)]:
                _, jac, bias = self._point(leg,l1,l2)
                mass += lm * (jac.T @ jac)
                force -= lm * (jac[2] * c.gravity + jac.T @ bias)
            foot, jac, _ = self._point(leg,c.femur_length,c.distal_length)
            contact, contact_damping = self._contact(foot,jac @ self.dq)
            self.last_contact[leg] = contact
            force += jac.T @ contact
            damping += jac.T @ contact_damping @ jac
        angles, speed = self.q[3:], self.dq[3:]
        activation = self.activation.reshape(12,2)
        torque = c.max_torque * (activation[:,0] - activation[:,1])
        torque -= (c.passive_stiffness + c.coactivation_stiffness * activation.sum(axis=1)) * angles
        torque -= c.joint_damping * speed
        below, above = np.minimum(angles-self.lower,0), np.maximum(angles-self.upper,0)
        torque -= c.stop_stiffness * (below+above)
        torque -= c.stop_damping * np.where((below < 0) | (above > 0),speed,0)
        damping[3:,3:] += np.diag(c.stop_damping * ((below < 0) | (above > 0)))
        self.last_torques[:] = torque
        force[3:] += torque
        bottom = self.q[:3].copy()
        bottom[2] -= c.body_radius
        self.last_body_contact, body_damping = self._contact(bottom,self.dq[:3],scale=10.)
        force[:3] += self.last_body_contact
        damping[:3,:3] += body_damping
        return mass, force, damping

    @property
    def velocity(self):
        return self.dq

    def advance(self, activations, dt):
        """Integrate already-filtered muscle activations held over one neural step.

        Inputs use leg-major, joint-major, flexor/extensor order. The caller owns
        muscle filtering; this layer does not add another activation time constant.
        """
        channels = np.asarray(activations,dtype=float)
        if channels.shape not in ((24,),(6,2,2)) or not np.isfinite(channels).all() or np.any(channels<0) or np.any(channels>1):
            raise ValueError("Expected 24 finite motor activations in [0,1], flexor/extensor pairs")
        if not np.isfinite(dt) or not 0 < dt <= 0.1:
            raise ValueError("dt must be finite and in (0,0.1] seconds")
        target = channels.reshape(6,2,2)
        n = math.ceil(dt/self.config.max_substep)
        h = dt/n
        self.activation[:] = target
        for _ in range(n):
            mass, force, damping = self._forces()
            # Linearly implicit dissipation prevents stiff contact/joint damping
            # from injecting numerical energy into light segments.
            self.dq += h * np.linalg.solve(mass+h*damping,force)
            self.q += h * self.dq
        self.t += dt
        if not np.isfinite(self.q).all() or not np.isfinite(self.dq).all():
            raise FloatingPointError("Nonfinite reduced body state")
        # Contacts/sensors correspond to the returned state, not the preceding substep.
        self._forces()
        return self.pose()

    def metadata(self):
        return {
            "schema_version": self.schema_version,
            "model": "reduced_torque_contact_body",
            "parameters": asdict(self.config),
            "units": {"length": "mm", "mass": "mg", "time": "s", "angle": "rad"},
            "channel_order": [f"{leg}_{joint}_{action}" for leg in LEGS for joint in JOINTS for action in ("flexor", "extensor")],
            "joint_limits": {"lower_radians": self.lower.tolist(), "upper_radians": self.upper.tolist(), "type": "compliant penalty stops"},
            "geometry": {"hip_positions_mm": self.hips.tolist(), "fixed_radial_planes": self.radial.tolist()},
            "validation": "Unvalidated masses, geometry, muscle force, damping, joint limits and friction are engineering assumptions, not measured fly physiology.",
            "constraints": "Body orientation and leg azimuths are fixed; no head, abdomen, wing or coxa motor assignment. Two hinges per leg, lumped masses, compliant foot and body contact.",
            "sensory_mapping": "Position, velocity and contact-load channels are diagnostics only; no biological sensory-neuron assignment is made here.",
        }

    def pose(self):
        c = self.config
        points = {leg: {"hip": (self.q[:3] + self.hips[i]).tolist(),
                        "knee": self._point(i, c.femur_length, 0.)[0].tolist(),
                        "foot": self._point(i, c.femur_length, c.distal_length)[0].tolist()}
                  for i, leg in enumerate(LEGS)}
        return {
            "legs": {leg: {"coxa": 0., "trochanter": float(self.q[3 + 2*i]),
                            "tibia": float(self.q[4 + 2*i])} for i, leg in enumerate(LEGS)},
            "bodyPositionMm": self.q[:3].tolist(),
            "legPointsMm": points,
            "bodyRoll": 0., "abdomenAngle": 0., "headAngle": 0.,
            "wingAngles": {"L": 0., "R": 0.},
            "geometrySource": "reduced_torque_contact_body",
        }

    def state_dict(self):
        """JSON-safe checkpoint state; returned arrays do not alias the model."""
        return {"schema_version": self.schema_version, "parameters": asdict(self.config),
                "t": self.t, "q": self.q.tolist(), "velocity": self.dq.tolist(),
                "activation": self.activation.tolist()}

    def restore(self, state):
        """Validate a checkpoint entirely before mutating the running body."""
        if not isinstance(state, dict) or set(state) != {"schema_version", "parameters", "t", "q", "velocity", "activation"}:
            raise ValueError("Unexpected reduced-body checkpoint fields")
        if state["schema_version"] != self.schema_version or state["parameters"] != asdict(self.config):
            raise ValueError("Reduced-body checkpoint schema or parameters do not match")
        if isinstance(state["t"], bool):
            raise ValueError("Body checkpoint time must be a nonnegative finite number")
        try:
            t = float(state["t"])
            q = np.asarray(state["q"], dtype=float)
            velocity = np.asarray(state["velocity"], dtype=float)
            activation = np.asarray(state["activation"], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Body checkpoint arrays must be numeric") from exc
        if not np.isfinite(t) or t < 0 or q.shape != (15,) or velocity.shape != (15,) or activation.shape != (6,2,2):
            raise ValueError("Invalid body checkpoint time or array shapes")
        if not all(np.isfinite(x).all() for x in (q, velocity, activation)):
            raise ValueError("Body checkpoint contains nonfinite values")
        if np.any((activation < 0) | (activation > 1)):
            raise ValueError("Body activations must lie in [0,1]")
        # Generous numerical-health bounds, not a claim of measured anatomical
        # limits. Compliant stops can be penetrated slightly under sustained load.
        if np.any(np.abs(q[:3]) > 1e6) or np.any(np.abs(q[3:]) > math.pi) or np.any(np.abs(velocity) > 1e5):
            raise ValueError("Body checkpoint exceeds numerical-health bounds")
        self.q[:] = q
        self.dq[:] = velocity
        self.activation[:] = activation
        self.t = t
        self._forces()

    def diagnostics(self):
        c = self.config
        feet = [self._point(i,c.femur_length,c.distal_length)[0].tolist() for i in range(6)]
        leg_points = {leg:{"hip":(self.q[:3]+self.hips[i]).tolist(),
                          "knee":self._point(i,c.femur_length,0.)[0].tolist(),
                          "foot":feet[i]} for i,leg in enumerate(LEGS)}
        weight_per_leg = c.gravity * (c.body_mass + 6*(c.femur_mass+c.distal_mass)) / 6
        load = self.last_contact[:,2] / max(weight_per_leg,1e-12)
        return {
            "t":self.t,
            "joint_order":[f"{leg}_{joint}" for leg in LEGS for joint in JOINTS],
            "joint_offsets_radians":self.q[3:].tolist(),
            "joint_velocities_radians_per_second":self.dq[3:].tolist(),
            "body_position_mm":self.q[:3].tolist(),
            "body_velocity_mm_per_second":self.dq[:3].tolist(),
            "foot_positions_mm":feet,
            "leg_points_mm":leg_points,
            "contact_forces_mg_mm_per_second_squared":self.last_contact.tolist(),
            "body_contact_force_mg_mm_per_second_squared":self.last_body_contact.tolist(),
            "net_joint_torques_mg_mm_squared_per_second_squared":self.last_torques.tolist(),
            "proprioception":{
                "position":((self.q[3:]-self.lower)/(self.upper-self.lower)).clip(0,1).reshape(6,2).tolist(),
                "velocity":np.tanh(self.dq[3:]/10.).reshape(6,2).tolist(),
                "load":load.tolist(),
                "contact":(self.last_contact[:,2]>0).tolist(),
                "mapping":"Signals have no neuron assignment; positions, velocities and load are modeled quantities, not measured receptor firing rates."
            },
            "limitations":"Reduced assumed mechanics: fixed body orientation, fixed leg azimuths, two hinges per leg, lumped segment masses, compliant contact, no abdomen/head/wing actuation or calibrated sensory encoding."
        }
