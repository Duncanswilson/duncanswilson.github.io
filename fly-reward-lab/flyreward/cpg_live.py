"""Hybrid whole-CNS rate model with an explicitly isolated front VNC circuit.

The front circuit owns its selected neuron rates. Its real neuron entries feed
outward through the existing whole-CNS graph on the next coarse neural step.
Incoming activity, DA/NPF modulation and body sensors do not feed back into that
circuit: preserving its declared experimental boundary is intentional. This is
not a validated whole-animal or pleasure model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import numpy as np

from .body import ReducedBody
from .cpg import FrontCPGCircuit
from .live_model import LiveSimulation, _digest_json
from .motor import LEGS, JOINTS
from .types import SimulationConfig


@dataclass
class CPGConfig(SimulationConfig):
    """Persist the intervention without altering the original rate-mode config."""
    cpg_stimulus: float = 400.0
    cpg_motor_gain: float = 1.0
    cpg_motor_bias: float = 0.0
    cpg_motor_threshold_scale: float = 1.0
    cpg_motor_target: str = "all"


class CPGSimulation(LiveSimulation):
    MOTOR_MODE = "cpg"
    CHECKPOINT_FORMAT = "flyreward.cpg-body-state"
    CHECKPOINT_VERSION = 1
    STATE_KEYS = LiveSimulation.STATE_KEYS | {"circuit_rates", "circuit_ticks", "body_q", "body_velocity"}

    def __init__(self, graph, config, mapping):
        if config.max_rate != 200:
            raise ValueError("The front circuit uses a 200 Hz ceiling")
        config = CPGConfig(**asdict(config))
        super().__init__(graph, config, mapping)
        self.circuit = FrontCPGCircuit(graph, stimulus=config.cpg_stimulus,
            motor_gain=config.cpg_motor_gain, motor_bias=config.cpg_motor_bias,
            motor_threshold_scale=config.cpg_motor_threshold_scale,
            motor_target=config.cpg_motor_target)
        self.body = ReducedBody()
        if self.dopamine_mask[self.circuit.graph_indices].any():
            raise ValueError("Front circuit ownership conflicts with the dopamine clamp")
        self._body_channel_order = np.asarray([self.channel_index[leg,joint,action]
            for leg in LEGS for joint in JOINTS for action in ("flexor","extensor")])
        self.rates[self.circuit.graph_indices] = self.circuit.rates
        self.channel_rates = self._channel_means()
        self.muscle_activations = self.channel_rates / self.config.max_rate
        self.body.activation[:] = self.muscle_activations[self._body_channel_order].reshape(6,2,2)
        self.joint_offsets[:] = self.body.q[3:]
        self.mechanics.update({
            "gain_radians": float(np.pi),
            "joint_response": "Reduced coupled-mass body; motor torques, passive springs, gravity, friction and compliant floor contact.",
            "joint_target": "No angle target: antagonist activation differences produce torque.",
            "body": "Six two-hinge legs and body translation; orientation, coxa, head, abdomen and wings constrained.",
            "feedback": "Computed proprioceptive diagnostics only; no sensory-neuron feedback assignment.",
            "validation": "Experimental hybrid front circuit and uncalibrated reduced mechanics; not measured pleasure or a validated gait.",
        })
        self._mechanics_config_fingerprint = _digest_json(self.mechanics)
        self._extension_metadata = {"circuit":self.circuit.metadata(), "body":self.body.metadata()}
        self._extension_fingerprint = _digest_json(self._extension_metadata)
        self._mechanism_fingerprint = _digest_json({
            "base": self._mechanism_fingerprint, "extension": self._extension_metadata,
            "mechanics": self.mechanics,
            "source_sha256":{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                             for name in ("cpg.py","cpg_live.py","body.py")},
        })

    def _check_config(self):
        super()._check_config()
        if _digest_json({"circuit":self.circuit.metadata(),"body":self.body.metadata()}) != self._extension_fingerprint:
            raise ValueError("Circuit/body configuration must not change during a run")

    def _advance_circuit(self):
        self.circuit.advance(self.config.dt)
        # Replace the owned neural entries BEFORE motor averaging. The published
        # motor arrays and recurrent whole-CNS input therefore share these rates.
        self.rates[self.circuit.graph_indices] = self.circuit.rates

    def _advance_motor(self, dt):
        self.muscle_activations += (self.channel_rates/self.config.max_rate-self.muscle_activations) * -np.expm1(-dt/self.mechanics["activation_tau_seconds"])
        self.body.advance(self.muscle_activations[self._body_channel_order], dt)
        self.body.t = (self.nstep+1)*self.config.dt
        self.joint_offsets[:] = self.body.q[3:]

    def snapshot(self, include_motor_rates=False):
        result = super().snapshot(include_motor_rates)
        result["motorPose"] = self.body.pose()
        result["body"] = self.body.diagnostics()
        result["motor_mode"] = "cpg"
        result["source"] = "Persistent hybrid MaleCNS simulation: front VNC circuit rates drive assumed muscle torques and reduced body contact mechanics; no scripted movement or pleasure measurement."
        result["circuit"] = {"ticks":int(self.circuit.ticks), "controlled_neuron_count":len(self.circuit.rates),
                             "sensory_feedback":False, "reward_modulation_inside_circuit":False}
        return result

    def _state_arrays(self):
        return {**super()._state_arrays(), "circuit_rates":self.circuit.rates,
                "circuit_ticks":np.asarray(self.circuit.ticks,dtype=np.int64),
                "body_q":self.body.q,"body_velocity":self.body.velocity}

    def _validate_state_arrays(self, arrays):
        super()._validate_state_arrays(arrays)
        for key,shape,dtype in [("circuit_rates",self.circuit.rates.shape,np.float64),
                                ("circuit_ticks",(),np.int64),("body_q",(15,),np.float64),
                                ("body_velocity",(15,),np.float64)]:
            value=arrays[key]
            if value.shape!=shape or value.dtype!=np.dtype(dtype) or not np.isfinite(value).all():
                raise ValueError(f"Checkpoint {key} shape, dtype or finiteness is invalid")
        if int(arrays["circuit_ticks"])<0 or (arrays["circuit_rates"]<0).any() or (arrays["circuit_rates"]>200).any():
            raise ValueError("Checkpoint circuit state out of bounds")
        if not np.array_equal(arrays["circuit_rates"].astype(np.float32),arrays["rates"][self.circuit.graph_indices]):
            raise ValueError("Checkpoint circuit rates disagree with primary neuron rates")
        if not np.array_equal(arrays["body_q"][3:],arrays["joint_offsets"]):
            raise ValueError("Checkpoint body joints disagree with motor offsets")
        if (np.abs(arrays["body_q"][:3])>1e6).any() or (np.abs(arrays["body_velocity"])>1e5).any():
            raise ValueError("Checkpoint body state out of numerical bounds")

    def _restore(self, metadata, arrays):
        self._validate_state_arrays(arrays)
        if int(arrays["circuit_ticks"]) != round(metadata["nstep"]*self.config.dt/self.circuit.dt):
            raise ValueError("Checkpoint circuit clock disagrees with neural time")
        # Prepare and validate a replacement body before any primary state change.
        body = ReducedBody()
        state=body.state_dict()
        state.update(t=metadata["nstep"]*self.config.dt,q=arrays["body_q"].tolist(),
                     velocity=arrays["body_velocity"].tolist(),
                     activation=arrays["muscle_activations"][self._body_channel_order].reshape(6,2,2).tolist())
        body.restore(state)
        super()._restore(metadata,arrays)
        self.circuit.rates[:] = arrays["circuit_rates"]
        self.circuit.ticks = int(arrays["circuit_ticks"])
        self.body = body

    @classmethod
    def from_checkpoint(cls, path, graph, mapping):
        """Restore the exact saved intervention as well as its neural/body state."""
        metadata, arrays = cls._read_checkpoint(path)
        simulation = cls(graph, CPGConfig(**metadata["config"]), mapping)
        simulation._restore(metadata, arrays)
        return simulation
