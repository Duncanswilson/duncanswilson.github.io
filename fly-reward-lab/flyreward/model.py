"""Bounded, connectome-constrained rate *hypothesis* model.

This is not a reproduction of Shiu et al., a biophysical fly emulation, or a
validated model of subjective experience. Time constants, receptor occupancy,
global modulatory effects, and the default 100 Hz ceiling are modeling choices.
"""

from dataclasses import asdict
from numbers import Integral, Real

import numpy as np
from scipy.sparse import csr_matrix

from .types import Graph, SimulationConfig, SimulationResult


RATE_TAU = 0.05
DA_RELEASE_RATE = 2.0
DA_CLEARANCE_RATE = 2.0
DA_OCCUPANCY_KD = 0.25
ADAPTATION_RECOVERY_RATE = 0.2
ADAPTATION_DESENSITIZATION_RATE = 0.8
NPF_TAU = 0.3


def presets() -> dict[str, SimulationConfig]:
    """Return independent configurations, with no claim of biological optimality."""
    settings = {
        "baseline": {},
        "dopamine": {"dopamine_drive": 0.5, "reuptake_factor": 0.25},
        "npf": {"npf_drive": 0.6},
        "combined": {
            "dopamine_drive": 0.5, "reuptake_factor": 0.25, "npf_drive": 0.6
        },
        "combined_no_tolerance": {
            "dopamine_drive": 0.5, "reuptake_factor": 0.25,
            "npf_drive": 0.6, "tolerance": False
        },
        "maximal_dopamine": {
            "dopamine_drive": 1.0, "reuptake_factor": 0.05,
            "npf_drive": 0.0, "tolerance": False
        },
    }
    return {name: SimulationConfig(preset=name, **values)
            for name, values in settings.items()}


def _canonical_nt(values: np.ndarray) -> np.ndarray:
    labels = np.asarray([str(value).strip().lower() for value in values])
    aliases = {
        "ach": "acetylcholine", "acetylcholine": "acetylcholine",
        "gaba": "gaba", "glut": "glutamate", "glutamate": "glutamate",
        "da": "dopamine", "dopamine": "dopamine",
        "ser": "serotonin", "5ht": "serotonin", "5-ht": "serotonin",
        "serotonin": "serotonin", "oct": "octopamine",
        "octopamine": "octopamine",
        "his": "histamine", "ha": "histamine", "histamine": "histamine",
        "": "unknown", "none": "unknown", "nan": "unknown",
        "na": "unknown", "unassigned": "unknown", "unknown": "unknown",
    }
    # Preserve unexpected labels for diagnostics, rather than concealing an
    # unsupported transmitter as though the source annotation were missing.
    return np.asarray([aliases.get(value, value) for value in labels])


def _validate_config(config: SimulationConfig) -> int:
    for name in ("duration", "dt", "dopamine_drive", "reuptake_factor",
                 "npf_drive", "max_rate", "network_gain"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
    if not 0 < config.dt <= 0.02:
        raise ValueError("dt must be in (0, 0.02] seconds for this rate model")
    if config.duration < 5 * config.dt:
        raise ValueError("duration must contain at least five integration steps")
    quotient = config.duration / config.dt
    if not np.isfinite(quotient) or abs(quotient - round(quotient)) > 1e-7:
        raise ValueError("duration must be an integer multiple of dt")
    for name in ("dopamine_drive", "npf_drive"):
        if not 0 <= getattr(config, name) <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    if not 0 <= config.reuptake_factor <= 5:
        raise ValueError("reuptake_factor must be between 0 and 5")
    if not 0 < config.max_rate <= 1000:
        raise ValueError("max_rate must be in (0, 1000] Hz")
    if not 0 <= config.network_gain <= 0.95:
        raise ValueError("network_gain must be between 0 and 0.95")
    if not isinstance(config.tolerance, (bool, np.bool_)):
        raise ValueError("tolerance must be boolean")
    for name in ("record_every", "seed"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer")
    if config.record_every < 1 or config.seed < 0:
        raise ValueError("record_every must be positive and seed nonnegative")
    return int(round(quotient))


def _normalized_fast_coupling(graph: Graph) -> tuple[csr_matrix, np.ndarray]:
    """Rows are postsynaptic targets; columns are presynaptic sources.

    Normalize by total incoming synapse count, then assign source-transmitter
    signs. DA/5HT/OCT/unknown have zero *fast* coupling, not a claimed biological
    sign. Glutamate is assumed inhibitory because receptor-specific signs are
    absent. Histamine is assumed inhibitory from fly Ort/HisCl1 chloride-channel
    physiology; a complete per-target receptor map is absent. Synapse counts
    are structural priors, not measured conductances.
    """
    count = len(graph.ids)
    if count == 0:
        raise ValueError("graph must contain neurons")
    if graph.connectivity.shape != (count, count):
        raise ValueError("connectivity shape must match neuron IDs")
    if len(graph.neurotransmitters) != count or len(graph.cell_types) != count:
        raise ValueError("neurotransmitter and cell-type arrays must match neuron IDs")
    if len(np.unique(graph.ids)) != count:
        raise ValueError("neuron IDs must be unique")
    matrix = csr_matrix(graph.connectivity, dtype=np.float32, copy=True)
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError("connectivity must contain finite nonnegative synapse counts")
    matrix.sum_duplicates()
    incoming = np.asarray(matrix.sum(axis=1)).ravel()
    if not np.isfinite(incoming).all():
        raise ValueError("incoming synapse sums must be finite")
    inv = np.zeros(count, dtype=np.float32)
    np.divide(1.0, incoming, out=inv, where=incoming > 0)
    nt = _canonical_nt(graph.neurotransmitters)
    signs = np.zeros(count, dtype=np.float32)
    signs[nt == "acetylcholine"] = 1
    signs[np.isin(nt, ("gaba", "glutamate", "histamine"))] = -1
    matrix.data *= signs[matrix.indices]
    matrix.data *= np.repeat(inv, np.diff(matrix.indptr))
    matrix.eliminate_zeros()
    return matrix, nt


def _clamp_dopamine(rates: np.ndarray, mask: np.ndarray,
                    config: SimulationConfig) -> None:
    if config.dopamine_drive > 0:
        rates[mask] = config.dopamine_drive * config.max_rate


def _dopamine_projection(graph: Graph, dopamine_mask: np.ndarray) -> tuple[csr_matrix, np.ndarray]:
    """Anatomical DA source-to-target projection, normalized within each target.

    This uses chemical synapse counts as a locality/relative-weight prior. It
    omits extrasynaptic diffusion and does not identify postsynaptic receptors.
    """
    projection = csr_matrix(graph.connectivity, dtype=np.float32, copy=True)
    projection.data *= dopamine_mask[projection.indices]
    projection.eliminate_zeros()
    incoming = np.asarray(projection.sum(axis=1)).ravel()
    recipients = incoming > 0
    inv = np.zeros(len(graph.ids), dtype=np.float32)
    np.divide(1.0, incoming, out=inv, where=recipients)
    projection.data *= np.repeat(inv, np.diff(projection.indptr))
    return projection, recipients


def _advance_dopamine(concentration: np.ndarray, sensitivity: np.ndarray,
                      release: np.ndarray, config: SimulationConfig) -> np.ndarray:
    """Advance bounded local concentration and sensitivity; return occupancy."""
    total = release + DA_CLEARANCE_RATE * config.reuptake_factor
    equilibrium = np.zeros_like(concentration)
    np.divide(release, total, out=equilibrium, where=total > 0)
    concentration += (equilibrium - concentration) * -np.expm1(-total * config.dt)
    np.clip(concentration, 0, 1, out=concentration)
    occupancy = concentration / (concentration + DA_OCCUPANCY_KD)
    if config.tolerance:
        adapt_total = ADAPTATION_RECOVERY_RATE + ADAPTATION_DESENSITIZATION_RATE * occupancy
        adapt_equilibrium = ADAPTATION_RECOVERY_RATE / adapt_total
        sensitivity += (adapt_equilibrium - sensitivity) * -np.expm1(-adapt_total * config.dt)
        np.clip(sensitivity, 0, 1, out=sensitivity)
    else:
        sensitivity.fill(1)
    return occupancy


def _recording_indices(graph: Graph, record_neuron_ids) -> np.ndarray:
    """Resolve actual graph IDs, retaining caller order without synthesizing IDs."""
    if record_neuron_ids is None:
        return np.empty(0, dtype=np.intp)
    if isinstance(record_neuron_ids, (str, bytes)):
        raise ValueError("record_neuron_ids must be a sequence of graph neuron IDs")
    try:
        requested = list(record_neuron_ids)
    except TypeError as exc:
        raise ValueError("record_neuron_ids must be a sequence of graph neuron IDs") from exc
    if not requested:
        return np.empty(0, dtype=np.intp)
    try:
        unique = set(requested)
    except TypeError as exc:
        raise ValueError("record_neuron_ids must contain individual graph neuron IDs") from exc
    if len(unique) != len(requested):
        raise ValueError("record_neuron_ids must not contain duplicate IDs")
    if any(isinstance(value, (bool, np.bool_)) for value in requested):
        raise ValueError("record_neuron_ids must contain graph neuron IDs, not booleans")
    lookup = {neuron_id: index for index, neuron_id in enumerate(graph.ids)}
    missing = [value for value in requested if value not in lookup]
    if missing:
        raise ValueError(f"record_neuron_ids contains IDs absent from graph: {missing[:8]}")
    return np.asarray([lookup[value] for value in requested], dtype=np.intp)


def run_simulation(graph: Graph, config: SimulationConfig,
                   record_neuron_ids=None) -> SimulationResult:
    """Simulate all graph neurons and record aggregate, explicitly labeled proxies.

    dopamine_drive > 0 clamps identified DA neurons to that fraction of max_rate.
    reuptake_factor multiplies clearance; zero disables this abstract clearance.
    npf_drive drives an abstract global NPF-associated modulatory variable. It
    does not select purported NPF neurons, introduce human receptors, or imply
    that NPF identity can be inferred from the supplied small-molecule NT classes.

    Dopamine states are per-target, driven by actual DA source-to-target synapse
    counts normalized over each target's DA inputs. Unconnected targets receive
    no DA modulation; extrasynaptic signaling and receptor maps are omitted.
    A paired shadow trajectory receives a small non-DA probe at 60--80% of the
    duration, using the same modulatory states as the primary trajectory. Its
    deviation measures network responsiveness without changing the main run.

    Optional record_neuron_ids selects existing graph body IDs, in caller order.
    Their actual primary-trajectory rates are copied at exactly the aggregate
    recording times, including the initial and final state. No derived proxy or
    population average substitutes for an individual neuron's state. None or an
    empty selection leaves the result's neuron_recording dictionary empty.
    """
    steps = _validate_config(config)
    coupling, nt = _normalized_fast_coupling(graph)
    recording_indices = _recording_indices(graph, record_neuron_ids)
    recorded_rates = []
    dopamine_mask = nt == "dopamine"
    n_dopamine = int(dopamine_mask.sum())
    if config.dopamine_drive > 0 and n_dopamine == 0:
        raise ValueError("dopamine stimulation requires identified dopamine neurons")
    n = len(graph.ids)
    dopamine_projection, dopamine_recipients = _dopamine_projection(graph, dopamine_mask)
    n_recipients = int(dopamine_recipients.sum())
    rng = np.random.default_rng(config.seed)
    basal = rng.uniform(0.03, 0.07, n).astype(np.float32)
    rates = basal * config.max_rate
    _clamp_dopamine(rates, dopamine_mask, config)
    probe = np.zeros(n, dtype=np.float32)
    eligible = np.flatnonzero(~dopamine_mask)
    if eligible.size:
        selected = rng.choice(eligible, max(1, int(np.ceil(0.05 * len(eligible)))), replace=False)
        probe[selected] = 0.02
    else:
        selected = np.empty(0, dtype=int)
    probe_start = max(1, int(np.ceil(0.6 * steps)))
    probe_stop = max(probe_start + 1, int(np.ceil(0.8 * steps)))
    alpha_rate = -np.expm1(-config.dt / RATE_TAU)
    alpha_npf = -np.expm1(-config.dt / NPF_TAU)
    dopamine = np.zeros(n, dtype=np.float32)
    sensitivity = np.ones(n, dtype=np.float32)
    npf = 0.0
    shadow = None
    trace_names = (
        "mean_rate_hz", "saturation_fraction", "dopamine_population_rate_hz",
        "dopamine_concentration", "dopamine_sensitivity", "dopamine_reward_proxy",
        "npf_exposure_proxy", "responsiveness_hz", "probe_on",
    )
    traces: dict[str, list[float]] = {name: [] for name in trace_names}
    time = []
    auc = {name: 0.0 for name in (
        "dopamine_concentration", "dopamine_reward_proxy", "npf_exposure_proxy",
        "mean_rate_hz", "saturation_fraction", "dopamine_population_rate_hz"
    )}
    peak_saturation = 0.0
    response_total = 0.0
    response_steps = 0
    peak_response = 0.0

    def snapshot(response: float, active: bool) -> dict[str, float]:
        occupancy = dopamine / (dopamine + DA_OCCUPANCY_KD)
        return {
            "mean_rate_hz": float(np.mean(rates)),
            "saturation_fraction": float(np.mean(rates >= 0.95 * config.max_rate)),
            "dopamine_population_rate_hz": float(np.mean(rates[dopamine_mask])) if n_dopamine else 0.0,
            "dopamine_concentration": float(np.mean(dopamine[dopamine_recipients])) if n_recipients else 0.0,
            "dopamine_sensitivity": float(np.mean(sensitivity[dopamine_recipients])) if n_recipients else 1.0,
            "dopamine_reward_proxy": float(np.mean((occupancy * sensitivity)[dopamine_recipients])) if n_recipients else 0.0,
            "npf_exposure_proxy": npf,
            "responsiveness_hz": response,
            "probe_on": float(active),
        }

    previous = snapshot(0.0, False)

    def record(step: int, values: dict[str, float]) -> None:
        time.append(step * config.dt)
        for name in trace_names:
            traces[name].append(values[name])
        if recording_indices.size:
            recorded_rates.append(rates[recording_indices].copy())

    record(0, previous)
    for step in range(1, steps + 1):
        active = probe_start <= step < probe_stop
        if step == probe_start:
            shadow = rates.copy()
        release = (DA_RELEASE_RATE / config.max_rate) * (dopamine_projection @ rates)
        occupancy = _advance_dopamine(dopamine, sensitivity, release, config)
        npf += (config.npf_drive - npf) * alpha_npf
        npf = float(np.clip(npf, 0, 1))
        # DA follows anatomical targets; receptor dynamics remain hypothetical.
        # NPF remains a uniform abstract drive because no NPF map was supplied.
        modulation = 0.10 * occupancy * sensitivity + 0.15 * npf

        def update(state: np.ndarray, additional: np.ndarray | float = 0.0) -> None:
            drive = basal + (config.network_gain / config.max_rate) * (coupling @ state)
            drive += modulation
            drive += additional
            target = config.max_rate * np.clip(drive, 0, 1)
            state += alpha_rate * (target - state)
            np.clip(state, 0, config.max_rate, out=state)
            _clamp_dopamine(state, dopamine_mask, config)

        update(rates)
        response = 0.0
        if shadow is not None:
            update(shadow, probe if active else 0.0)
            response = float(np.mean(np.abs(shadow[eligible] - rates[eligible]))) if eligible.size else 0.0
        values = snapshot(response, active)
        if not all(np.isfinite(value) for value in values.values()):
            raise FloatingPointError("nonfinite simulation state; results discarded")
        for name in auc:
            auc[name] += 0.5 * config.dt * (previous[name] + values[name])
        peak_saturation = max(peak_saturation, values["saturation_fraction"])
        peak_response = max(peak_response, response)
        if active:
            response_total += response
            response_steps += 1
        if step % config.record_every == 0 or step == steps:
            record(step, values)
        previous = values

    metrics = {
        "dopamine_exposure_auc": auc["dopamine_concentration"],
        "dopamine_concentration_mean": auc["dopamine_concentration"] / config.duration,
        "dopamine_reward_proxy_mean": auc["dopamine_reward_proxy"] / config.duration,
        "npf_exposure_proxy_mean": auc["npf_exposure_proxy"] / config.duration,
        "mean_rate_hz": auc["mean_rate_hz"] / config.duration,
        "dopamine_population_rate_hz_mean": auc["dopamine_population_rate_hz"] / config.duration,
        "mean_saturation_fraction": auc["saturation_fraction"] / config.duration,
        "peak_saturation_fraction": peak_saturation,
        "mean_responsiveness_hz": response_total / response_steps if response_steps else 0.0,
        "peak_responsiveness_hz": peak_response,
        "final_dopamine_sensitivity": previous["dopamine_sensitivity"],
        "final_dopamine_concentration": previous["dopamine_concentration"],
    }
    metadata = {
        "model_kind": "hypothesis_connectome_rate_model",
        "biological_validation": "not_validated; not a Shiu et al. reproduction",
        "subjective_experience": "not measured or established; all reward readouts are proxies",
        "neuron_count": n,
        "fast_coupling_edges": int(coupling.nnz),
        "dopamine_neuron_count": n_dopamine,
        "dopamine_recipient_neuron_count": n_recipients,
        "dopamine_projection_edges": int(dopamine_projection.nnz),
        "dopamine_mapping": "canonical neurotransmitter identity; all identified dopamine cells",
        "dopamine_clamped_neuron_count": n_dopamine if config.dopamine_drive > 0 else 0,
        "dopamine_clamp_hz": config.dopamine_drive * config.max_rate,
        "dopamine_state_units": "dimensionless bounded local concentration; traces average over anatomical DA recipients, not measured molarity",
        "dopamine_modulation": "per-target states driven by DA outgoing synapse counts; row normalization uses only DA inputs; abstract identical receptor dynamics",
        "dopamine_locality_caveat": "chemical synapses define targets; extrasynaptic signaling and receptor maps are absent; no DA incoming edges means zero DA modulation",
        "npf_mapping": "abstract_global_modulatory_drive_no_cell_or_receptor_mapping",
        "npf_caveat": "NPF-associated scalar is hypothetical; no NPF-cell identification or human mu-opioid receptor is modeled",
        "glutamate_sign_assumption": "inhibitory; receptor-specific exceptions are not represented",
        "histamine_sign_assumption": "inhibitory, based on fly Ort/HisCl1 histamine-gated chloride channels; no complete receptor-specific map",
        "histamine_sign_sources": [
            "https://pubmed.ncbi.nlm.nih.gov/12196539/",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC6335465/",
        ],
        "zero_fast_coupling_transmitters": sorted(set(nt) - {"acetylcholine", "gaba", "glutamate", "histamine"}),
        "neurotransmitter_counts": {str(label): int(np.count_nonzero(nt == label)) for label in np.unique(nt)},
        "unmapped_transmitter_counts": {
            str(label): int(np.count_nonzero(nt == label))
            for label in sorted(set(nt) - {
                "acetylcholine", "gaba", "glutamate", "histamine",
                "dopamine", "serotonin", "octopamine", "unknown"
            })
        },
        "unknown_transmitter_neuron_count": int(np.count_nonzero(nt == "unknown")),
        "normalization": "target-row incoming synapse count; source-column transmitter sign",
        "rate_ceiling": "config.max_rate is an arbitrary refractory-inspired numerical bound, not a measured universal fly limit",
        "probe": {
            "kind": "paired shadow trajectory; same local DA and abstract global NPF states",
            "start_seconds": (probe_start - 1) * config.dt,
            "stop_seconds": (probe_stop - 1) * config.dt,
            "target_count": int(len(selected)),
            "input_fraction_of_max_rate": 0.02,
            "readout": "mean absolute rate deviation over non-dopamine neurons during probe",
            "limitation": "tests responses to a small numerical input, not naturalistic behavior or consciousness",
        },
        "parameters": {
            "rate_tau_seconds": RATE_TAU,
            "dopamine_release_rate_per_second": DA_RELEASE_RATE,
            "base_clearance_rate_per_second": DA_CLEARANCE_RATE,
            "occupancy_kd": DA_OCCUPANCY_KD,
            "adaptation_recovery_rate_per_second": ADAPTATION_RECOVERY_RATE,
            "adaptation_desensitization_rate_per_second": ADAPTATION_DESENSITIZATION_RATE,
            "npf_tau_seconds": NPF_TAU,
        },
        "limits": [
            "Synapse counts do not determine physiological synaptic strengths.",
            "Rate units omit spike timing, electrical synapses, detailed plasticity and morphology.",
            "Dopamine cell classes can convey different or opposite reinforcement signals.",
            "A higher numerical proxy does not establish greater biological reward or pleasure.",
        ],
    }
    neuron_recording = {}
    if recording_indices.size:
        neuron_recording = {
            "ids": np.asarray(graph.ids)[recording_indices].tolist(),
            "rates_hz": np.stack(recorded_rates),
            "neurotransmitters": nt[recording_indices].tolist(),
            "cell_types": np.asarray(graph.cell_types)[recording_indices].tolist(),
            "source": "Actual per-neuron primary simulation rate state in Hz, indexed by graph body ID; sampled at result.time, not synthesized from population means or reward proxies.",
        }
    return SimulationResult(
        config=asdict(config), time=np.asarray(time),
        traces={name: np.asarray(values) for name, values in traces.items()},
        metrics=metrics, metadata=metadata, neuron_recording=neuron_recording,
    )
