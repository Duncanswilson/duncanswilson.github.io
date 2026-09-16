"""Causal and numerical checks for a connectome-generated motor rhythm."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.signal import find_peaks
from scipy.sparse import csr_matrix

from flyreward.cpg import FrontCPGCircuit, load_cpg_asset
from flyreward.types import Graph


def cpg_graph():
    asset = load_cpg_asset()
    cells, edges = asset['neurons'], np.asarray(asset['edges'], dtype=np.int64)
    source, target, count = edges.T
    return Graph(ids=np.asarray([c['id'] for c in cells], dtype=np.int64),
                 connectivity=csr_matrix((count, (target, source)), shape=(len(cells), len(cells))),
                 neurotransmitters=np.asarray([c['nt'] for c in cells]),
                 cell_types=np.asarray([c['type'] for c in cells]))


def record(circuit, seconds=3., sample_dt=.01):
    frames = []
    for _ in range(round(seconds / sample_dt)):
        circuit.advance(sample_dt)
        frames.append(circuit.rates[circuit.motor_indices].copy())
    return np.asarray(frames)


def test_tonic_input_generates_persistent_motor_rhythms_in_both_front_legs():
    circuit = FrontCPGCircuit(cpg_graph())
    frames = record(circuit)[100:]
    oscillating = np.ptp(frames, axis=0) > 1.
    assert oscillating.sum() >= 8
    asset = load_cpg_asset()
    cells = {c['id']: c for c in asset['neurons']}
    sides = {cells[int(body)]['side'] for body in circuit.motor_neuron_ids[oscillating]}
    assert sides == {'L', 'R'}
    for trace in frames[:, oscillating].T:
        assert len(find_peaks(trace, prominence=.25)[0]) >= 10
    assert circuit.ticks == 3000
    assert frames.min() >= 0
    assert frames.max() <= 200


@pytest.mark.parametrize('cell_type', ['IN17A001', 'INXXX466'])
def test_core_ablation_abolishes_sustained_motor_rhythm(cell_type):
    circuit = FrontCPGCircuit(cpg_graph(), silenced_cell_types=[cell_type])
    frames = record(circuit)[100:]
    assert np.ptp(frames, axis=0).max() < 1e-5
    assert np.all(circuit.rates[circuit.cell_types == cell_type] == 0)


def test_absent_descending_input_is_silent_and_onset_is_exact():
    circuit = FrontCPGCircuit(cpg_graph(), stimulus=0)
    circuit.advance(2.)
    assert not np.any(circuit.rates)
    circuit = FrontCPGCircuit(cpg_graph())
    circuit.advance(.020)
    assert not np.any(circuit.rates)
    circuit.advance(.001)
    assert np.any(circuit.rates)


def test_chunking_preserves_state_and_metadata_excludes_mutable_state():
    graph = cpg_graph()
    first, second = FrontCPGCircuit(graph), FrontCPGCircuit(graph)
    metadata = first.metadata()
    first.advance(.137)
    for dt in [.013, .020, .001, .003, .100]:
        second.advance(dt)
    np.testing.assert_array_equal(first.rates, second.rates)
    assert first.ticks == second.ticks == 137
    assert first.metadata() == metadata
    metadata['stimulus'] = 0
    assert first.metadata()['stimulus'] == 400


def test_one_ms_solver_matches_half_ms_reference():
    graph = cpg_graph()
    coarse = record(FrontCPGCircuit(graph), seconds=2.)
    fine = record(FrontCPGCircuit(graph, internal_dt=.0005), seconds=2.)
    np.testing.assert_allclose(coarse, fine, atol=.12, rtol=.02)


def test_changed_anatomical_weight_and_transmitter_fail_closed():
    graph = cpg_graph()
    graph.connectivity.data[0] += 1
    with pytest.raises(ValueError, match='synapse counts'):
        FrontCPGCircuit(graph)
    graph = cpg_graph()
    graph.neurotransmitters[graph.ids == 10056] = 'gaba'
    with pytest.raises(ValueError, match='transmitter sign'):
        FrontCPGCircuit(graph)


def test_invalid_integration_requests_are_rejected():
    graph = cpg_graph()
    with pytest.raises(ValueError, match='timestep'):
        FrontCPGCircuit(graph, internal_dt=.01)
    circuit = FrontCPGCircuit(graph)
    for dt in [-1, np.inf, np.nan, .0001]:
        with pytest.raises(ValueError):
            circuit.advance(dt)
    assert circuit.ticks == 0


@pytest.mark.parametrize('kwargs',[
    {'motor_gain':0}, {'motor_gain':np.inf}, {'motor_gain':501},
    {'motor_bias':-1}, {'motor_bias':np.nan},
    {'motor_threshold_scale':-1}, {'motor_threshold_scale':1.1},
])
def test_invalid_motor_interventions_are_rejected(kwargs):
    with pytest.raises(ValueError,match='motor_'):
        FrontCPGCircuit(cpg_graph(),**kwargs)


def recruited_circuit(**kwargs):
    from flyreward.server import continuous_config
    config = continuous_config('cpg','recruited')
    params = dict(stimulus=config.cpg_stimulus, motor_gain=config.cpg_motor_gain,
                  motor_bias=config.cpg_motor_bias,
                  motor_threshold_scale=config.cpg_motor_threshold_scale,
                  motor_target=config.cpg_motor_target)
    return FrontCPGCircuit(cpg_graph(),**(params | kwargs))


def test_recruitment_amplifies_real_motor_cells_in_both_front_legs():
    circuit = recruited_circuit()
    frames = record(circuit)[100:]
    cells = {c['id']:c for c in load_cpg_asset()['neurons']}
    selected = circuit.metadata()['motor_stimulation']['target_body_ids']
    assert {cells[body]['type'] for body in selected} == {
        'Tr flexor MN','Acc. tr flexor MN','Ti extensor MN'}
    expected = {body for body,c in cells.items() if c['motor'] and c['type'] in {
        'Tr flexor MN','Acc. tr flexor MN','Ti extensor MN'}}
    assert set(selected) == expected and len(selected) == 24
    unselected = ~np.isin(circuit.ids,selected)
    baseline = FrontCPGCircuit(cpg_graph())
    np.testing.assert_array_equal(circuit._gain_over_cap[unselected],baseline._gain_over_cap[unselected])
    np.testing.assert_array_equal(circuit._threshold[unselected],baseline._threshold[unselected])
    for side in ('L','R'):
        indices = [i for i,body in enumerate(circuit.motor_neuron_ids)
                   if cells[int(body)]['side']==side and cells[int(body)]['type'] in ('Tr flexor MN','Acc. tr flexor MN')]
        assert np.ptp(frames[:,indices].mean(axis=1)) > 30.
    assert frames.min() >= 0 and frames.max() <= 200.


@pytest.mark.parametrize('intervention',[
    {'stimulus':0}, {'silenced_cell_types':['IN17A001']},
    {'silenced_cell_types':['INXXX466']},
])
def test_recruited_motor_rhythm_still_requires_the_neural_circuit(intervention):
    frames = record(recruited_circuit(**intervention))[100:]
    assert np.ptp(frames,axis=0).max() < 1e-5


def test_recruited_circuit_numerical_refinement():
    coarse = record(recruited_circuit())
    fine = record(recruited_circuit(internal_dt=.0005))
    assert np.max(np.abs(coarse-fine)) < 2.
    assert np.sqrt(np.mean((coarse-fine)**2)) < .15
    np.testing.assert_allclose(np.ptp(coarse[100:],axis=0),np.ptp(fine[100:],axis=0),atol=.5,rtol=.01)
