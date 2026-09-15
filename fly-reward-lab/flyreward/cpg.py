"""Connectome-derived front-leg rhythm circuit, with an explicit tonic input.

Independent NumPy/SciPy implementation of Pugliese et al.'s rectified-tanh rate
ODE. The bundled asset records the published cell/edge selection and exact
current MaleCNS counts. This circuit is a model of motor rhythms, not pleasure.
No sine wave, random motion, phase clock, or recorded trajectory drives its rates.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from .types import Graph

ASSET_PATH = Path(__file__).with_name("assets") / "cpg_front_v1.json"


def load_cpg_asset(path: Path | str = ASSET_PATH) -> dict:
    """Read the versioned, attributed factual network asset."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("format") != "flyreward.cpg-front" or value.get("version") != 1:
        raise ValueError("Unsupported CPG asset format")
    return value


class FrontCPGCircuit:
    """A persistent front-leg circuit with exclusive ownership of its rates.

    Graph indices identify where the caller publishes these rates. The caller
    must not add a second baseline/neuromodulator update to these same cells.
    Other full-connectome cells may receive these rates as inputs; their activity
    is not fed back into this experimentally isolated circuit.
    """

    def __init__(self, graph: Graph, *, stimulus: float = 400.0,
                 silenced_cell_types=(), internal_dt: float = 0.001,
                 asset_path: Path | str = ASSET_PATH):
        if not np.isfinite(stimulus) or not 0 <= stimulus <= 2000:
            raise ValueError("CPG tonic stimulus must be finite and between 0 and 2000")
        if not np.isfinite(internal_dt) or internal_dt <= 0 or internal_dt > 0.001:
            raise ValueError("CPG internal timestep must be positive and no larger than 1 ms")
        onset_steps = 0.02 / internal_dt
        if not np.isclose(onset_steps, round(onset_steps), rtol=0, atol=1e-9):
            raise ValueError("CPG timestep must divide the 20 ms stimulus onset")
        asset_bytes = Path(asset_path).read_bytes()
        asset = load_cpg_asset(asset_path)
        neurons, params = asset["neurons"], asset["parameters"]
        self.ids = np.asarray([n["id"] for n in neurons], dtype=np.int64)
        self.cell_types = np.asarray([n["type"] for n in neurons])
        self.silenced_cell_types = tuple(sorted(set(silenced_cell_types)))
        if set(self.silenced_cell_types) - set(self.cell_types):
            raise ValueError("Unknown CPG cell type requested for silencing")
        lookup = {int(body): i for i, body in enumerate(graph.ids)}
        if len(lookup) != len(graph.ids):
            raise ValueError("Graph body IDs must be unique")
        missing = [int(body) for body in self.ids if int(body) not in lookup]
        if missing:
            raise ValueError(f"CPG body IDs absent from graph: {missing[:8]}")
        self.graph_indices = np.asarray([lookup[int(body)] for body in self.ids], dtype=np.intp)
        self.publish_indices = np.arange(len(self.ids), dtype=np.intp)
        self.motor_indices = np.flatnonzero([n["motor"] for n in neurons])
        self.motor_neuron_ids = self.ids[self.motor_indices].copy()
        edges = np.asarray(asset["edges"], dtype=np.int64)
        source, target, counts = edges.T
        actual = np.asarray(graph.connectivity[self.graph_indices[target],
                                                self.graph_indices[source]]).ravel()
        if not np.array_equal(actual, counts):
            raise ValueError("CPG synapse counts do not match the current graph")
        signs = np.asarray([n["sign"] for n in neurons], dtype=float)
        expected_sign = {"acetylcholine": 1, "gaba": -1, "glutamate": -1}
        for index in np.unique(source):
            graph_nt = str(graph.neurotransmitters[self.graph_indices[index]]).lower()
            if expected_sign.get(graph_nt, 0) != signs[index]:
                raise ValueError(f"CPG transmitter sign conflicts with graph for body {self.ids[index]}")
        self.coupling = csr_matrix((counts * signs[source] * params["synapse_multiplier"],
                                   (target, source)), shape=(len(self.ids), len(self.ids)))
        size = np.asarray([n["size"] for n in neurons], dtype=float) / params["size_normalization_median"]
        if np.any(~np.isfinite(size)) or np.any(size <= 0):
            raise ValueError("CPG cell sizes must be positive and finite")
        self._gain_over_cap = (params["gain"] / size) / params["rate_cap_hz"]
        self._threshold = params["threshold"] * size
        self._tau = float(params["tau_seconds"])
        self.rate_cap_hz = float(params["rate_cap_hz"])
        self.dt = self.internal_dt = float(internal_dt)
        self.stimulus = float(stimulus)
        self._stimulus_onset_tick = int(round(onset_steps))
        self._input = np.where(np.isin(self.ids, params["stimulated_body_ids"]), self.stimulus, 0.)
        self._enabled = ~np.isin(self.cell_types, self.silenced_cell_types)
        self.rates = np.zeros(len(self.ids), dtype=np.float64)
        self.ticks = 0
        self._metadata = {
            "model": "male-cns-front-cpg-v1", "asset_sha256": hashlib.sha256(asset_bytes).hexdigest(),
            "neurons": len(self.ids), "edges": len(edges), "motor_neurons": len(self.motor_indices),
            "internal_dt_seconds": self.dt, "rate_cap_hz": self.rate_cap_hz,
            "stimulus": self.stimulus, "stimulated_body_ids": params["stimulated_body_ids"],
            "stimulus_onset_seconds": params["stimulus_onset_seconds"],
            "silenced_cell_types": list(self.silenced_cell_types),
            "solver": params["solver"], "parameter_choice": params["parameter_choice"],
            "source_commit": asset["reference"]["commit"], "source_paper": asset["reference"]["paper"],
            "adaptation": asset["adaptation"]["description"], "scope": asset["adaptation"]["scope"],
            "coupling_policy": "Selected circuit rates replace global-model rates; no external feedback into circuit.",
            "neuromodulation_policy": "No dopamine/NPF modulation inside the CPG; tonic DNg100 input is independent.",
            "sign_policy": "ACh excitatory; GABA/glutamate inhibitory. Published silent transmitter classes retain zero outgoing weights.",
        }

    def metadata(self) -> dict:
        """Immutable configuration/provenance only; deliberately excludes state."""
        return deepcopy(self._metadata)

    def _derivative(self, rates: np.ndarray, stimulus: np.ndarray) -> np.ndarray:
        drive = self.coupling @ rates + stimulus - self._threshold
        target = np.maximum(self.rate_cap_hz * np.tanh(self._gain_over_cap * drive), 0.)
        target *= self._enabled
        return (target - rates) / self._tau

    def advance(self, dt: float) -> np.ndarray:
        """Advance persistent neural state by an integer number of internal steps."""
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("CPG duration must be nonnegative and finite")
        steps = dt / self.dt
        count = int(round(steps))
        if not np.isclose(steps, count, rtol=0, atol=1e-8):
            raise ValueError("CPG duration must be a multiple of the internal timestep")
        h = self.dt
        for _ in range(count):
            stimulus = self._input if self.ticks >= self._stimulus_onset_tick else 0.
            rates = self.rates
            k1 = self._derivative(rates, stimulus)
            k2 = self._derivative(rates + h * k1 / 2, stimulus)
            k3 = self._derivative(rates + h * k2 / 2, stimulus)
            k4 = self._derivative(rates + h * k3, stimulus)
            self.rates += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            self.ticks += 1
        return self.rates
