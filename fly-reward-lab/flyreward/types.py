from dataclasses import dataclass, field
from typing import Any
import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class Graph:
    """Connectivity is nonnegative synapse counts: row=target, column=source."""
    ids: np.ndarray
    connectivity: csr_matrix
    neurotransmitters: np.ndarray
    cell_types: np.ndarray
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationConfig:
    duration: float = 2.0
    dt: float = 0.01
    seed: int = 7
    preset: str = "baseline"
    dopamine_drive: float = 0.0
    reuptake_factor: float = 1.0
    npf_drive: float = 0.0
    tolerance: bool = True
    max_rate: float = 100.0
    network_gain: float = 0.6
    record_every: int = 1


@dataclass
class SimulationResult:
    config: dict[str, Any]
    time: np.ndarray
    traces: dict[str, np.ndarray]
    metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
    neuron_recording: dict[str, Any] = field(default_factory=dict)
