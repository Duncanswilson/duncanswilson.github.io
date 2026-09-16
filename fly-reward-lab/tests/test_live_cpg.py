"""Causal integration and complete persistence of the hybrid motor/body state."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from flyreward.cpg import load_cpg_asset
from flyreward.cpg_live import CPGConfig, CPGSimulation
from flyreward.live_model import LiveSimulation, _digest_arrays, _digest_json, _json_text
from flyreward.server import continuous_config
from flyreward.types import Graph


def specimen():
    asset=load_cpg_asset()
    cells=asset['neurons']
    mapping=json.loads((Path(__file__).parents[1]/'results/motor/motor_mapping.json').read_text())
    ids=[c['id'] for c in cells]
    extra=sorted(set(mapping['motor_neuron_ids'])-set(ids))+[1999999,2000000]
    lookup={body:i for i,body in enumerate(ids+extra)}
    edges=np.asarray(asset['edges'])
    source,target,counts=edges.T
    # One measured-circuit cell projects into a clearly artificial observer.
    rows=np.r_[target,lookup[2000000]]
    cols=np.r_[source,lookup[800173]]
    values=np.r_[counts,50]
    graph=Graph(ids=np.asarray(ids+extra),connectivity=sparse.csr_matrix((values,(rows,cols)),shape=(len(lookup),len(lookup))),
                neurotransmitters=np.asarray([c['nt'] for c in cells]+['acetylcholine']*(len(extra)-2)+['dopamine','acetylcholine']),
                cell_types=np.asarray([c['type'] for c in cells]+['test_cell']*len(extra)))
    return graph,mapping


@pytest.fixture(scope='module')
def graph_and_mapping():
    return specimen()


def engine(spec):
    graph,mapping=spec
    return CPGSimulation(graph,continuous_config('cpg'),mapping)


def assert_same(first,second):
    assert first.nstep==second.nstep
    for key,value in first._state_arrays().items():
        np.testing.assert_array_equal(value,second._state_arrays()[key],err_msg=key)
    np.testing.assert_array_equal(first.channel_rates,second.channel_rates)
    assert first.snapshot(True)==second.snapshot(True)


def test_raw_rates_channels_and_body_share_the_same_current_state(graph_and_mapping):
    live=engine(graph_and_mapping)
    live.step(35)
    frame=live.snapshot(True)
    np.testing.assert_array_equal(live.rates[live.circuit.graph_indices],live.circuit.rates.astype(np.float32))
    raw=dict(zip(frame['motor_neuron_ids'],frame['motor_rates_hz']))
    means=[np.mean([raw[body] for body in ch['body_ids']]) for ch in live.channels]
    np.testing.assert_array_equal(means,frame['channel_rates_hz'])
    np.testing.assert_array_equal(frame['angles'],live.body.q[3:])
    assert frame['motorPose']==live.body.pose()
    assert live.circuit.ticks==350
    assert live.body.t==live.t


def test_circuit_output_reaches_an_external_primary_neuron(graph_and_mapping):
    first,second=engine(graph_and_mapping),engine(graph_and_mapping)
    body_index=np.flatnonzero(graph_and_mapping[0].ids==800173)[0]
    observer=np.flatnonzero(graph_and_mapping[0].ids==2000000)[0]
    second.rates[body_index]=80
    second.circuit.rates[second.circuit.ids==800173]=80
    first.step();second.step()
    assert second.rates[observer]>first.rates[observer]+1


def test_restart_and_chunking_preserve_neurons_body_and_identity(graph_and_mapping,tmp_path):
    original=engine(graph_and_mapping)
    original.step(27)
    checkpoint=tmp_path/'hybrid.npz'
    original.save_checkpoint(checkpoint)
    resumed=CPGSimulation.from_checkpoint(checkpoint,*graph_and_mapping)
    assert_same(original,resumed)
    original.step(17)
    resumed.step(5);resumed.snapshot();resumed.step(12)
    assert_same(original,resumed)
    with pytest.raises(ValueError,match='unexpected|format'):
        LiveSimulation.from_checkpoint(checkpoint,*graph_and_mapping)


@pytest.mark.parametrize('corruption',['body_nan','body_shape','body_angle','rates','clock','mode','fingerprint'])
def test_corruption_fails_without_mutating_live_state(graph_and_mapping,tmp_path,corruption):
    live=engine(graph_and_mapping);live.step(4)
    path=tmp_path/'state.npz';live.save_checkpoint(path)
    metadata,arrays=live._read_checkpoint(path)
    before=live.snapshot(True)
    if corruption=='body_nan':arrays['body_q'][0]=np.nan
    elif corruption=='body_shape':arrays['body_q']=arrays['body_q'][:-1]
    elif corruption=='body_angle':arrays['body_q'][4]+=0.1
    elif corruption=='rates':arrays['circuit_rates'][0]+=1
    elif corruption=='clock':arrays['circuit_ticks']+=1
    elif corruption=='mode':metadata['motor_mode']='rate'
    elif corruption=='fingerprint':metadata['fingerprints']['mechanism']='different'
    metadata['state_sha256']=_digest_arrays(arrays)
    metadata['checkpoint_sha256']=_digest_json({k:v for k,v in metadata.items() if k!='checkpoint_sha256'})
    with path.open('wb') as f:
        np.savez_compressed(f,metadata=np.asarray(_json_text(metadata)),**arrays)
    with pytest.raises(ValueError):live.restore_checkpoint(path)
    assert live.snapshot(True)==before


def test_observation_does_not_advance_any_state(graph_and_mapping):
    live=engine(graph_and_mapping);live.step(5)
    before={key:value.copy() for key,value in live._state_arrays().items()}
    for _ in range(10):live.snapshot(True)
    for key,value in before.items():np.testing.assert_array_equal(value,live._state_arrays()[key])


def test_nondefault_motor_stimulation_restarts_exactly_and_rejects_mismatch(graph_and_mapping,tmp_path):
    config = CPGConfig(**asdict(continuous_config('cpg')))
    config.cpg_motor_gain = 20.
    config.cpg_motor_bias = 50.
    config.cpg_motor_target = 'trochanter_flexors_tibia_extensors'
    graph, mapping = graph_and_mapping
    stimulated = CPGSimulation(graph, config, mapping)
    stimulated.step(25)
    path = tmp_path/'stimulated.npz'
    stimulated.save_checkpoint(path)
    resumed = CPGSimulation.from_checkpoint(path,graph,mapping)
    assert resumed.config.cpg_motor_gain == 20.
    assert resumed.config.cpg_motor_bias == 50.
    assert resumed.config.cpg_motor_target == 'trochanter_flexors_tibia_extensors'
    assert_same(stimulated,resumed)
    stimulated.step(11)
    resumed.step(4);resumed.step(7)
    assert_same(stimulated,resumed)
    frame = resumed.snapshot(True)
    raw = dict(zip(frame['motor_neuron_ids'],frame['motor_rates_hz']))
    np.testing.assert_array_equal(frame['channel_rates_hz'],
        [np.mean([raw[body] for body in ch['body_ids']]) for ch in resumed.channels])
    mismatched = engine(graph_and_mapping)
    before = mismatched.snapshot(True)
    with pytest.raises(ValueError,match='fingerprint'):
        mismatched.restore_checkpoint(path)
    assert mismatched.snapshot(True) == before


def test_recruited_mode_moves_both_front_legs_after_settling(graph_and_mapping):
    graph,mapping = graph_and_mapping
    live = CPGSimulation(graph,continuous_config('cpg','recruited'),mapping)
    frames = []
    for step in range(400):
        live.step()
        if step >= 200:
            frames.append(live.body.q[3:].copy())
    excursion = np.ptp(frames,axis=0)*180/np.pi
    assert excursion[:2].max() > 2.
    assert excursion[6:8].max() > 2.
    # Keeping the last real activation cannot sustain an invented body rhythm.
    frozen = live.body.activation.copy()
    settled = []
    for step in range(300):
        live.body.advance(frozen,.01)
        if step >= 200:
            settled.append(live.body.q.copy())
    assert np.ptp(settled,axis=0).max() < 1e-5
