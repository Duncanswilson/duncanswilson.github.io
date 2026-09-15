from dataclasses import replace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from flyreward.model import (
    _advance_dopamine, _canonical_nt, _clamp_dopamine, _dopamine_projection,
    _normalized_fast_coupling, presets, run_simulation,
)
from flyreward.types import Graph, SimulationConfig


def graph_fixture():
    # 0 (ACh) -> 1; 1 (GABA) -> 2; 2 (DA) -> 0,3; 3 unknown -> 1.
    return Graph(
        ids=np.array([10, 20, 30, 40]),
        connectivity=csr_matrix((
            [2.0, 3.0, 4.0, 1.0, 2.0],
            ([1, 2, 0, 3, 1], [0, 1, 2, 2, 3])
        ), shape=(4, 4)),
        neurotransmitters=np.array(["ACH", "GABA", "DA", "unknown"]),
        cell_types=np.array(["excitatory", "inhibitory", "PAM", "unclassified"]),
    )


def test_orientation_and_transmitter_signs():
    coupling, _ = _normalized_fast_coupling(graph_fixture())
    # An isolated presynaptic signal reaches the specified target only.
    np.testing.assert_allclose(coupling @ [1, 0, 0, 0], [0, 0.5, 0, 0])
    np.testing.assert_allclose(coupling @ [0, 1, 0, 0], [0, 0, -1, 0])
    np.testing.assert_allclose(coupling @ [0, 0, 1, 1], np.zeros(4))


def test_histamine_aliases_are_preserved_and_inhibit_anatomical_target():
    np.testing.assert_array_equal(_canonical_nt(np.array(["HIS", "HA", "histamine"])),
                                  ["histamine", "histamine", "histamine"])
    graph = graph_fixture()
    graph.neurotransmitters = np.array(["histamine", "gaba", "dopamine", "unknown"])
    coupling, nt = _normalized_fast_coupling(graph)
    assert nt[0] == "histamine"
    np.testing.assert_allclose(coupling @ [1, 0, 0, 0], [0, -0.5, 0, 0])


def test_unmapped_transmitter_label_retained_with_explicit_diagnostics():
    graph = graph_fixture()
    graph.neurotransmitters = np.array(["acetylcholine", "gaba", "dopamine", "tyramine"])
    coupling, nt = _normalized_fast_coupling(graph)
    assert nt[3] == "tyramine"
    np.testing.assert_array_equal(coupling @ [0, 0, 0, 1], 0)
    result = run_simulation(graph, SimulationConfig(duration=0.1))
    assert result.metadata["unmapped_transmitter_counts"] == {"tyramine": 1}
    assert result.metadata["unknown_transmitter_neuron_count"] == 0
    assert "tyramine" in result.metadata["zero_fast_coupling_transmitters"]


def test_clamp_selects_only_dopamine_by_transmitter_not_cell_name():
    graph = graph_fixture()
    graph.cell_types[0] = "PAM"
    _, nt = _normalized_fast_coupling(graph)
    rates = np.array([3.0, 5.0, 7.0, 9.0])
    _clamp_dopamine(rates, nt == "dopamine", SimulationConfig(dopamine_drive=1))
    np.testing.assert_array_equal(rates, [3, 5, 100, 9])


def test_dopamine_release_follows_actual_edge_orientation_and_locality():
    graph = graph_fixture()
    projection, recipients = _dopamine_projection(graph, np.array([False, False, True, False]))
    np.testing.assert_array_equal(recipients, [True, False, False, True])
    # DA neuron 2 projects to targets 0 and 3 only, despite their distinct counts.
    release = 0.02 * (projection @ [0, 0, 100, 0])
    np.testing.assert_allclose(release, [2, 0, 0, 2])
    concentration = np.zeros(4)
    sensitivity = np.ones(4)
    occupancy = _advance_dopamine(concentration, sensitivity, release, SimulationConfig())
    assert concentration[0] > 0 and concentration[3] > 0
    np.testing.assert_array_equal(concentration[[1, 2]], 0)
    np.testing.assert_array_equal(occupancy[[1, 2]], 0)
    np.testing.assert_array_equal(sensitivity[[1, 2]], 1)


def test_adaptation_recovers_when_dopamine_is_absent():
    concentration = np.zeros(3)
    sensitivity = np.full(3, 0.2)
    _advance_dopamine(concentration, sensitivity, np.zeros(3), SimulationConfig())
    assert np.all(sensitivity > 0.2)
    assert np.all(sensitivity < 1)


def test_da_neurons_without_outgoing_edges_create_no_local_modulation():
    graph = graph_fixture()
    graph.connectivity = csr_matrix((4, 4))
    result = run_simulation(graph, SimulationConfig(duration=0.1, dopamine_drive=1))
    assert result.metadata["dopamine_recipient_neuron_count"] == 0
    np.testing.assert_array_equal(result.traces["dopamine_concentration"], 0)
    np.testing.assert_array_equal(result.traces["dopamine_reward_proxy"], 0)


def test_reduced_reuptake_monotonic_for_identical_clamped_drive():
    graph = graph_fixture()
    normal = SimulationConfig(duration=0.5, dopamine_drive=0.5, reuptake_factor=1)
    normal_result = run_simulation(graph, normal)
    slow_result = run_simulation(graph, replace(normal, reuptake_factor=0.1))
    assert np.all(slow_result.traces["dopamine_concentration"] >= normal_result.traces["dopamine_concentration"])
    assert slow_result.metrics["dopamine_exposure_auc"] > normal_result.metrics["dopamine_exposure_auc"]
    np.testing.assert_array_equal(normal_result.traces["dopamine_population_rate_hz"], 50)


def test_no_tolerance_preserves_sensitivity_and_adaptation_remains_bounded():
    config = SimulationConfig(duration=0.5, dopamine_drive=1)
    adapted = run_simulation(graph_fixture(), config)
    unadapted = run_simulation(graph_fixture(), replace(config, tolerance=False))
    np.testing.assert_array_equal(unadapted.traces["dopamine_sensitivity"], 1)
    assert 0 < adapted.metrics["final_dopamine_sensitivity"] < 1
    assert unadapted.metrics["dopamine_reward_proxy_mean"] > adapted.metrics["dopamine_reward_proxy_mean"]


def test_deterministic_and_probe_does_not_modify_primary_trajectory():
    config = SimulationConfig(duration=0.25, dopamine_drive=0.5, npf_drive=0.2)
    first = run_simulation(graph_fixture(), config)
    second = run_simulation(graph_fixture(), config)
    for name in first.traces:
        np.testing.assert_array_equal(first.traces[name], second.traces[name])
    assert first.metrics == second.metrics
    # Probe has zero response before its scheduled onset and a nonzero response
    # during the intervention, measured against a distinct primary trajectory.
    active = first.traces["probe_on"] > 0
    first_active = np.flatnonzero(active)[0]
    np.testing.assert_array_equal(first.traces["responsiveness_hz"][:first_active], 0)
    assert first.metrics["mean_responsiveness_hz"] > 0


def test_all_presets_bounded_finite_and_maximum_has_measurable_effect():
    results = {name: run_simulation(graph_fixture(), replace(config, duration=0.25))
               for name, config in presets().items()}
    assert set(results) == {"baseline", "dopamine", "npf", "combined",
                            "combined_no_tolerance", "maximal_dopamine"}
    for result in results.values():
        assert all(np.isfinite(value).all() for value in result.traces.values())
        assert all(np.isfinite(value) for value in result.metrics.values())
        for key in ("dopamine_concentration", "dopamine_sensitivity",
                    "dopamine_reward_proxy", "npf_exposure_proxy", "saturation_fraction"):
            assert np.all((result.traces[key] >= 0) & (result.traces[key] <= 1))
        assert np.all((result.traces["mean_rate_hz"] >= 0) & (result.traces["mean_rate_hz"] <= 100))
        assert result.metadata["npf_mapping"].startswith("abstract_global")
    assert results["maximal_dopamine"].metrics["dopamine_exposure_auc"] > results["baseline"].metrics["dopamine_exposure_auc"]
    np.testing.assert_array_equal(results["maximal_dopamine"].traces["dopamine_population_rate_hz"], 100)


def test_metrics_independent_of_record_downsampling():
    config = SimulationConfig(duration=0.25, dopamine_drive=0.5)
    full = run_simulation(graph_fixture(), config)
    sparse = run_simulation(graph_fixture(), replace(config, record_every=7))
    assert full.metrics == sparse.metrics
    assert sparse.time[-1] == config.duration


@pytest.mark.parametrize("changes", [
    {"dt": 0}, {"dt": 0.05}, {"duration": 0.001}, {"duration": 0.255},
    {"dopamine_drive": 1.1}, {"npf_drive": -1}, {"reuptake_factor": -1},
    {"network_gain": 1.5}, {"max_rate": 0}, {"seed": -1}, {"seed": 2.5},
    {"record_every": 0}, {"record_every": 1.5}, {"tolerance": "no"},
    {"npf_drive": float("nan")}, {"duration": float("inf")},
])
def test_invalid_configuration_rejected(changes):
    with pytest.raises(ValueError):
        run_simulation(graph_fixture(), replace(SimulationConfig(), **changes))


def test_missing_dopamine_mapping_cannot_silently_run_da_stimulation():
    graph = graph_fixture()
    graph.neurotransmitters[:] = "unknown"
    with pytest.raises(ValueError, match="identified dopamine"):
        run_simulation(graph, SimulationConfig(dopamine_drive=1))


def test_negative_connectivity_rejected():
    graph = graph_fixture()
    graph.connectivity.data[0] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        run_simulation(graph, SimulationConfig())


def test_recorded_neuron_ids_align_with_actual_state_and_requested_order():
    graph = graph_fixture()
    config = SimulationConfig(duration=0.25, dopamine_drive=0.5, npf_drive=0.2)
    complete = run_simulation(graph, config, record_neuron_ids=[10, 20, 30, 40])
    selected = run_simulation(graph, config, record_neuron_ids=[40, 10, 30])
    recording = selected.neuron_recording
    assert recording["ids"] == [40, 10, 30]
    assert recording["neurotransmitters"] == ["unknown", "acetylcholine", "dopamine"]
    assert recording["cell_types"] == ["unclassified", "excitatory", "PAM"]
    assert "Actual per-neuron" in recording["source"]
    assert recording["rates_hz"].shape == (len(selected.time), 3)
    np.testing.assert_array_equal(recording["rates_hz"], complete.neuron_recording["rates_hz"][:, [3, 0, 2]])
    # The clamped DA source is recorded at its own actual 50 Hz, not a proxy.
    np.testing.assert_array_equal(recording["rates_hz"][:, 2], 50)
    np.testing.assert_array_equal(complete.neuron_recording["rates_hz"].mean(axis=1), complete.traces["mean_rate_hz"])
    expected_initial = np.random.default_rng(config.seed).uniform(0.03, 0.07, 4).astype(np.float32) * config.max_rate
    expected_initial[2] = 50
    np.testing.assert_array_equal(complete.neuron_recording["rates_hz"][0], expected_initial)


def test_neuron_recording_has_exact_aggregate_times_with_downsampling_and_final():
    config = SimulationConfig(duration=0.25, dopamine_drive=0.5)
    full = run_simulation(graph_fixture(), config, record_neuron_ids=[20, 30])
    sparse = run_simulation(graph_fixture(), replace(config, record_every=7), record_neuron_ids=[20, 30])
    steps = [0, 7, 14, 21, 25]
    np.testing.assert_array_equal(sparse.time, full.time[steps])
    np.testing.assert_array_equal(sparse.neuron_recording["rates_hz"], full.neuron_recording["rates_hz"][steps])
    assert sparse.neuron_recording["rates_hz"].shape == (5, 2)
    assert sparse.time[0] == 0
    assert sparse.time[-1] == config.duration


@pytest.mark.parametrize("selection", [None, [], np.array([], dtype=np.int64)])
def test_no_selected_neurons_keeps_recording_empty(selection):
    result = run_simulation(graph_fixture(), SimulationConfig(duration=0.1), record_neuron_ids=selection)
    assert result.neuron_recording == {}


def test_selected_recording_leaves_base_trajectory_and_metadata_unchanged():
    config = SimulationConfig(duration=0.25, dopamine_drive=0.5, npf_drive=0.2)
    base = run_simulation(graph_fixture(), config)
    recorded = run_simulation(graph_fixture(), config, record_neuron_ids=[20, 40])
    np.testing.assert_array_equal(recorded.time, base.time)
    for name in base.traces:
        np.testing.assert_array_equal(recorded.traces[name], base.traces[name])
    assert recorded.config == base.config
    assert recorded.metrics == base.metrics
    assert recorded.metadata == base.metadata


@pytest.mark.parametrize("selection, message", [
    ([20, 20], "duplicate"), ([20, np.int64(20)], "duplicate"),
    ([999], "absent"), ([20, 999], "absent"), ([True], "booleans"),
    (20, "sequence"), ("20", "sequence"), ([[20]], "individual"),
])
def test_invalid_recorded_neuron_ids_rejected(selection, message):
    with pytest.raises(ValueError, match=message):
        run_simulation(graph_fixture(), SimulationConfig(duration=0.1), record_neuron_ids=selection)
