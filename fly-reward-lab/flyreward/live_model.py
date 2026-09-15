"""Continuous state for one connectome rate model and its explicit motor bridge.

No replay, time wrapping, background scheduler, wall-clock promise, or trajectory
history is provided. Neural equations are the primary equations in model.py.
The anatomical and physiological limitations of that hypothesis model still
apply. Motor offsets use the uncalibrated mechanics in motor.py.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from numbers import Integral
import os
from pathlib import Path
import tempfile
import uuid

import numpy as np

from .model import (
    ADAPTATION_DESENSITIZATION_RATE, ADAPTATION_RECOVERY_RATE,
    DA_CLEARANCE_RATE, DA_OCCUPANCY_KD, DA_RELEASE_RATE, NPF_TAU, RATE_TAU,
    _advance_dopamine, _clamp_dopamine, _dopamine_projection,
    _normalized_fast_coupling, _validate_config,
)
from .motor import JOINTS, LEGS, MECHANICS
from .types import Graph, SimulationConfig


def _json_text(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False,
                      default=lambda item: item.item() if isinstance(item, np.generic) else item.tolist())


def _digest_json(value) -> str:
    return hashlib.sha256(_json_text(value).encode("utf-8")).hexdigest()


def _digest_arrays(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        array = np.ascontiguousarray(arrays[key])
        digest.update(_json_text([key, array.dtype.str, list(array.shape)]).encode("utf-8"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _integer_ids(values, name: str) -> list[int]:
    try:
        values = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain integer body IDs") from exc
    if not values or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
                         for value in values):
        raise ValueError(f"{name} must contain integer body IDs")
    result = [int(value) for value in values]
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique body IDs")
    return result


class LiveSimulation:
    """A persistent single-fly state, advanced only when step() is called.

    ``config.duration`` is retained for provenance and finite-batch comparison;
    it never limits or resets this engine. Recording cadence is also irrelevant:
    every neural step advances muscle and joint state. Configuration is copied
    on construction and must remain fixed for this engine's lifetime.
    """

    CHECKPOINT_FORMAT = "flyreward.live-state"
    CHECKPOINT_VERSION = 1

    def __init__(self, graph: Graph, config: SimulationConfig, mapping: dict):
        _validate_config(config)
        self.config = replace(config)
        self._config_fingerprint = _digest_json(asdict(self.config))
        self.coupling, self.neurotransmitters = _normalized_fast_coupling(graph)
        self.dopamine_mask = self.neurotransmitters == "dopamine"
        if self.config.dopamine_drive > 0 and not self.dopamine_mask.any():
            raise ValueError("dopamine stimulation requires identified dopamine neurons")
        self.dopamine_projection, self.dopamine_recipients = _dopamine_projection(graph, self.dopamine_mask)
        self._neuron_count = len(graph.ids)
        graph_ids = _integer_ids(graph.ids, "Graph neuron IDs")
        lookup = {body_id: i for i, body_id in enumerate(graph_ids)}
        self.motor_neuron_ids = _integer_ids(mapping.get("motor_neuron_ids", ()), "Motor neuron IDs")
        missing = set(self.motor_neuron_ids) - set(lookup)
        if missing:
            raise ValueError(f"Motor IDs are absent from graph: {sorted(missing)[:8]}")
        self.motor_indices = np.asarray([lookup[body] for body in self.motor_neuron_ids], dtype=np.intp)
        motor_ids = set(self.motor_neuron_ids)
        self.channels = []
        self.channel_indices = []
        self.channel_index = {}
        for channel in mapping.get("channels", ()):
            try:
                key = (channel["leg"], channel["joint"], channel["action"])
                ids = _integer_ids(channel["body_ids"], "Motor channel IDs")
            except KeyError as exc:
                raise ValueError("Each motor channel requires leg, joint, action and body_ids") from exc
            if key in self.channel_index:
                raise ValueError(f"Duplicate motor channel: {key}")
            if not set(ids) <= motor_ids:
                raise ValueError(f"Motor channel contains IDs outside the recorded motor set: {key}")
            self.channel_index[key] = len(self.channels)
            self.channels.append({"leg": key[0], "joint": key[1], "action": key[2], "body_ids": ids})
            self.channel_indices.append(np.asarray([lookup[body] for body in ids], dtype=np.intp))
        expected = {(leg, joint, action) for leg in LEGS for joint in JOINTS for action in ("flexor", "extensor")}
        if set(self.channel_index) != expected:
            raise ValueError("Live motor mapping must supply all 24 explicit antagonist channels")
        self._mapping_fingerprint = _digest_json({"motor_neuron_ids": self.motor_neuron_ids, "channels": self.channels})
        # Dtype, shape, IDs, transmitter labels and cell annotations accompany
        # exact CSR contents, so checkpoints cannot silently change specimens.
        counts = graph.connectivity.tocsr()
        self._graph_fingerprint = _digest_arrays({
            "ids": np.asarray(graph_ids, dtype=np.int64),
            "connectivity_data": counts.data,
            "connectivity_indices": counts.indices,
            "connectivity_indptr": counts.indptr,
            "connectivity_shape": np.asarray(counts.shape, dtype=np.int64),
            "neurotransmitters": np.asarray(graph.neurotransmitters, dtype=str),
            "cell_types": np.asarray(graph.cell_types, dtype=str),
        })
        self.mechanics = dict(MECHANICS)
        self._mechanics_config_fingerprint = _digest_json(self.mechanics)
        self._mechanism_fingerprint = _digest_json({
            "rate_tau": RATE_TAU, "release": DA_RELEASE_RATE,
            "clearance": DA_CLEARANCE_RATE, "occupancy_kd": DA_OCCUPANCY_KD,
            "recovery": ADAPTATION_RECOVERY_RATE, "desensitization": ADAPTATION_DESENSITIZATION_RATE,
            "npf_tau": NPF_TAU, "mechanics": self.mechanics,
            "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                              for name in ("live_model.py", "model.py", "motor.py")},
        })
        rng = np.random.default_rng(self.config.seed)
        self.basal = rng.uniform(0.03, 0.07, self._neuron_count).astype(np.float32)
        self.rates = self.basal * self.config.max_rate
        _clamp_dopamine(self.rates, self.dopamine_mask, self.config)
        self.dopamine = np.zeros(self._neuron_count, dtype=np.float32)
        self.sensitivity = np.ones(self._neuron_count, dtype=np.float32)
        self.npf = 0.0
        self.nstep = 0  # Python integer: no fixed replay horizon or int32 rollover.
        self.run_id = str(uuid.uuid4())
        self.channel_rates = self._channel_means()
        self.muscle_activations = self.channel_rates / self.config.max_rate
        self.joint_offsets = np.zeros(len(LEGS) * len(JOINTS), dtype=np.float64)
        self._alpha_rate = -np.expm1(-self.config.dt / RATE_TAU)
        self._alpha_npf = -np.expm1(-self.config.dt / NPF_TAU)
        self._failure_reason = None

    @property
    def t(self) -> float:
        return self.nstep * self.config.dt

    @property
    def fingerprints(self) -> dict[str, str]:
        return {"graph": self._graph_fingerprint, "mapping": self._mapping_fingerprint,
                "config": self._config_fingerprint, "mechanism": self._mechanism_fingerprint}

    def _check_config(self) -> None:
        if _digest_json(asdict(self.config)) != self._config_fingerprint:
            raise ValueError("LiveSimulation configuration must not be mutated after construction")
        if _digest_json(self.mechanics) != self._mechanics_config_fingerprint:
            raise ValueError("LiveSimulation mechanics must not be mutated after construction")

    def _check_healthy(self) -> None:
        if self._failure_reason is not None:
            raise RuntimeError("LiveSimulation advancement failed; restore a valid checkpoint before continuing: " + self._failure_reason)

    def _channel_means(self) -> np.ndarray:
        # motor_playback converts recorded float32 rates to float64 before its
        # group means; retain that arithmetic and exact within-channel ID order.
        return np.asarray([self.rates[indices].astype(np.float64).mean()
                           for indices in self.channel_indices], dtype=np.float64)

    def step(self, steps: int = 1) -> None:
        """Advance in place, retaining only the current state; zero is a no-op."""
        if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, Integral) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        self._check_config()
        self._check_healthy()
        try:
            for _ in range(int(steps)):
                self._step_once()
        except BaseException as exc:
            # Advancement is in-place to avoid copying a whole brain each step.
            # An interrupted step may therefore be partial. Never publish or
            # checkpoint that state, including on a shutdown finalizer.
            self._failure_reason = (type(exc).__name__ + ": " + str(exc))[:512]
            raise

    def _step_once(self) -> None:
        release = (DA_RELEASE_RATE / self.config.max_rate) * (self.dopamine_projection @ self.rates)
        occupancy = _advance_dopamine(self.dopamine, self.sensitivity, release, self.config)
        self.npf += (self.config.npf_drive - self.npf) * self._alpha_npf
        self.npf = float(np.clip(self.npf, 0, 1))
        modulation = 0.10 * occupancy * self.sensitivity + 0.15 * self.npf
        drive = self.basal + (self.config.network_gain / self.config.max_rate) * (self.coupling @ self.rates)
        drive += modulation
        target = self.config.max_rate * np.clip(drive, 0, 1)
        self.rates += self._alpha_rate * (target - self.rates)
        np.clip(self.rates, 0, self.config.max_rate, out=self.rates)
        _clamp_dopamine(self.rates, self.dopamine_mask, self.config)
        self.channel_rates = self._channel_means()
        # Use the same difference of recorded frame times as motor_playback.
        # Neural dynamics above always use the original configured dt.
        dt = (self.nstep + 1) * self.config.dt - self.nstep * self.config.dt
        self.muscle_activations += (self.channel_rates / self.config.max_rate - self.muscle_activations) * -np.expm1(-dt / self.mechanics["activation_tau_seconds"])
        for leg_idx, leg in enumerate(LEGS):
            for joint_idx, joint in enumerate(JOINTS):
                flex = self.muscle_activations[self.channel_index[leg, joint, "flexor"]]
                extend = self.muscle_activations[self.channel_index[leg, joint, "extensor"]]
                target_angle = self.mechanics["gain_radians"] * (flex - extend)
                column = 2 * leg_idx + joint_idx
                self.joint_offsets[column] += (target_angle - self.joint_offsets[column]) * -np.expm1(-dt / self.mechanics["joint_tau_seconds"])
        self.nstep += 1
        if not all(np.isfinite(value).all() for value in (
            self.rates, self.dopamine, self.sensitivity, self.muscle_activations, self.joint_offsets
        )) or not np.isfinite(self.npf):
            raise FloatingPointError("Nonfinite live simulation state; no snapshot should be served")

    def snapshot(self, include_motor_rates: bool = False) -> dict:
        """Return JSON-safe current values, without advancing or retaining history."""
        self._check_config()
        self._check_healthy()
        occupancy = self.dopamine / (self.dopamine + DA_OCCUPANCY_KD)
        recipients = self.dopamine_recipients
        has_recipients = bool(recipients.any())
        legs = {}
        for index, leg in enumerate(LEGS):
            legs[leg] = {"coxa": 0.0, "trochanter": float(self.joint_offsets[index * 2]),
                         "tibia": float(self.joint_offsets[index * 2 + 1])}
        result = {
            "t": float(self.t), "step_count": self.nstep,
            "dopamine": float(np.mean(self.dopamine[recipients])) if has_recipients else 0.0,
            "npf": float(self.npf),
            "rate": float(np.mean(self.rates[self.dopamine_mask])) if self.dopamine_mask.any() else 0.0,
            "mean_rate_hz": float(np.mean(self.rates)),
            "dopamine_sensitivity": float(np.mean(self.sensitivity[recipients])) if has_recipients else 1.0,
            "dopamine_reward_proxy": float(np.mean((occupancy * self.sensitivity)[recipients])) if has_recipients else 0.0,
            "saturation_fraction": float(np.mean(self.rates >= self.config.max_rate * 0.95)),
            "motorMean": float(np.mean(self.rates[self.motor_indices].astype(np.float64))),
            "channelRates": self.channel_rates.tolist(), "angles": self.joint_offsets.tolist(),
            "motorPose": {"legs": legs, "bodyRoll": 0.0, "abdomenAngle": 0.0,
                          "headAngle": 0.0, "wingAngles": {"L": 0.0, "R": 0.0}},
            "rate_ceiling_hz": float(self.config.max_rate),
            "neuron_count": self._neuron_count, "motor_neuron_count": len(self.motor_neuron_ids),
            "source": "Current persistent primary simulation state, not a replay or biological recording; motor mechanics remain assumptions.",
        }
        if include_motor_rates:
            result["motor_neuron_ids"] = list(self.motor_neuron_ids)
            result["motor_rates_hz"] = self.rates[self.motor_indices].tolist()
        result.update({
            "run_id": self.run_id, "step": self.nstep, "sim_time": result["t"],
            "channel_rates_hz": result["channelRates"],
            "dopamine_rate_hz": result["rate"], "motor_mean_hz": result["motorMean"],
        })
        return result

    def _state_arrays(self) -> dict[str, np.ndarray]:
        return {"basal": self.basal, "rates": self.rates,
                "dopamine": self.dopamine, "sensitivity": self.sensitivity,
                "npf": np.asarray(self.npf, dtype=np.float64),
                "muscle_activations": self.muscle_activations,
                "joint_offsets": self.joint_offsets}

    def save_checkpoint(self, path) -> None:
        """Atomically replace one NPZ checkpoint, with numeric arrays and JSON only."""
        self._check_config()
        self._check_healthy()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays = self._state_arrays()
        self._validate_state_arrays(arrays)
        metadata = {"format": self.CHECKPOINT_FORMAT, "version": self.CHECKPOINT_VERSION,
                    "config": asdict(self.config), "nstep": self.nstep, "run_id": self.run_id,
                    "fingerprints": self.fingerprints,
                    "state_sha256": _digest_arrays(arrays)}
        metadata["checkpoint_sha256"] = _digest_json(metadata)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=destination.parent,
                                             prefix=destination.name + ".", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(handle, metadata=np.asarray(_json_text(metadata)), **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            # Persist the directory entry as well as file contents. Oracle's
            # Linux filesystem supports this; it also works on local macOS.
            directory_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @classmethod
    def _read_checkpoint(cls, path) -> tuple[dict, dict[str, np.ndarray]]:
        expected = {"metadata", "basal", "rates", "dopamine", "sensitivity", "npf",
                    "muscle_activations", "joint_offsets"}
        with np.load(Path(path), allow_pickle=False) as archive:
            if set(archive.files) != expected:
                raise ValueError("Checkpoint has missing or unexpected state arrays")
            raw = archive["metadata"]
            if raw.shape != () or raw.dtype.kind not in "US":
                raise ValueError("Checkpoint metadata must be a scalar JSON string")
            metadata = json.loads(str(raw.item()))
            arrays = {key: archive[key].copy() for key in expected - {"metadata"}}
        if metadata.get("format") != cls.CHECKPOINT_FORMAT or metadata.get("version") != cls.CHECKPOINT_VERSION:
            raise ValueError("Unsupported live checkpoint format/version")
        if metadata.get("checkpoint_sha256") != _digest_json({key: value for key, value in metadata.items() if key != "checkpoint_sha256"}):
            raise ValueError("Checkpoint metadata checksum mismatch")
        if metadata.get("state_sha256") != _digest_arrays(arrays):
            raise ValueError("Checkpoint state checksum mismatch")
        if not isinstance(metadata.get("nstep"), int) or isinstance(metadata["nstep"], bool) or metadata["nstep"] < 0:
            raise ValueError("Checkpoint step count must be a nonnegative integer")
        try:
            if str(uuid.UUID(metadata["run_id"])) != metadata["run_id"]:
                raise ValueError("noncanonical UUID")
        except (KeyError, ValueError, AttributeError, TypeError) as exc:
            raise ValueError("Checkpoint run_id must be a canonical UUID") from exc
        return metadata, arrays

    def _validate_state_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        shapes = {"basal": (self._neuron_count,), "rates": (self._neuron_count,),
                  "dopamine": (self._neuron_count,), "sensitivity": (self._neuron_count,),
                  "npf": (), "muscle_activations": (24,), "joint_offsets": (12,)}
        for key, shape in shapes.items():
            expected_dtype = np.dtype(np.float32 if key in {"basal", "rates", "dopamine", "sensitivity"} else np.float64)
            if arrays[key].shape != shape or arrays[key].dtype != expected_dtype or not np.isfinite(arrays[key]).all():
                raise ValueError(f"Checkpoint {key} shape, dtype or finiteness is invalid")
        for key in ("dopamine", "sensitivity", "npf", "muscle_activations"):
            if ((arrays[key] < 0) | (arrays[key] > 1)).any():
                raise ValueError(f"Checkpoint {key} is out of bounds")
        if ((arrays["rates"] < 0) | (arrays["rates"] > self.config.max_rate + 1e-4)).any():
            raise ValueError("Checkpoint rates exceed the configured bounds")
        if ((arrays["basal"] < 0.029999) | (arrays["basal"] > 0.070001)).any():
            raise ValueError("Checkpoint basal drive is outside its modeled bounds")
        if (np.abs(arrays["joint_offsets"]) > self.mechanics["gain_radians"] + 1e-8).any():
            raise ValueError("Checkpoint joint offsets exceed mechanical bounds")

    def _restore(self, metadata: dict, arrays: dict[str, np.ndarray]) -> None:
        if metadata.get("fingerprints") != self.fingerprints or _digest_json(metadata.get("config")) != self._config_fingerprint:
            raise ValueError("Checkpoint graph, mapping, config or mechanism fingerprint mismatch")
        self._validate_state_arrays(arrays)
        channel_rates = np.asarray([arrays["rates"][indices].astype(np.float64).mean()
                                    for indices in self.channel_indices], dtype=np.float64)
        # Validate everything before mutating the existing live state.
        self.basal = arrays["basal"]
        self.rates = arrays["rates"]
        self.dopamine = arrays["dopamine"]
        self.sensitivity = arrays["sensitivity"]
        self.npf = float(arrays["npf"])
        self.muscle_activations = arrays["muscle_activations"]
        self.joint_offsets = arrays["joint_offsets"]
        self.nstep = metadata["nstep"]
        self.run_id = metadata["run_id"]
        self.channel_rates = channel_rates
        self._failure_reason = None

    def restore_checkpoint(self, path) -> None:
        """Restore into a matching engine, rejecting any incompatible checkpoint."""
        self._check_config()
        self._restore(*self._read_checkpoint(path))

    @classmethod
    def from_checkpoint(cls, path, graph: Graph, mapping: dict) -> "LiveSimulation":
        """Recreate the saved continuous state without executing a neural step."""
        metadata, arrays = cls._read_checkpoint(path)
        simulation = cls(graph, SimulationConfig(**metadata["config"]), mapping)
        simulation._restore(metadata, arrays)
        return simulation
