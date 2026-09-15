from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from flyreward.live_model import LiveSimulation
import flyreward.live_model as live_module
from flyreward.model import run_simulation
from flyreward.motor import JOINTS, LEGS, motor_playback
from flyreward.types import Graph, SimulationConfig


def specimen():
    ids = np.concatenate([np.arange(1000, 1048), [2000, 3000]])
    n = len(ids)
    rows = list(range(n)) + list(range(0, 48, 3)) + list(range(1, 48, 5))
    cols = [(i + 1) % n for i in range(n)] + [48] * 16 + [49] * 10
    counts = [1 + i % 4 for i in range(n)] + [7] * 16 + [2] * 10
    nt = np.asarray(["acetylcholine" if i % 3 else "gaba" for i in range(48)] + ["dopamine", "histamine"])
    graph = Graph(ids=ids, connectivity=csr_matrix((counts, (rows, cols)), shape=(n, n)),
                  neurotransmitters=nt, cell_types=np.asarray([f"cell_{body}" for body in ids]))
    channels = []
    for leg in LEGS:
        for joint in JOINTS:
            for action in ("flexor", "extensor"):
                index = len(channels)
                channels.append({"leg": leg, "joint": joint, "action": action,
                                 "body_ids": [1000 + 2 * index, 1001 + 2 * index]})
    mapping = {"motor_neuron_ids": list(range(1000, 1048)), "channels": channels}
    config = SimulationConfig(duration=0.2, dt=0.01, dopamine_drive=0.7,
                              reuptake_factor=0.25, npf_drive=0.6)
    return graph, config, mapping


@pytest.mark.parametrize("tolerance", [True, False])
def test_live_initial_and_twenty_steps_equal_finite_primary_rates_and_motor_pose(tolerance):
    graph, config, mapping = specimen()
    config = replace(config, tolerance=tolerance)
    finite = run_simulation(graph, config, record_neuron_ids=graph.ids)
    playback = motor_playback(finite, mapping, "finite", "Finite comparison", "test")
    live = LiveSimulation(graph, config, mapping)
    for step in range(21):
        if step:
            live.step()
        frame = live.snapshot(include_motor_rates=True)
        assert live.nstep == step
        assert frame["step"] == step
        assert frame["sim_time"] == finite.time[step]
        np.testing.assert_array_equal(live.rates, finite.neuron_recording["rates_hz"][step])
        np.testing.assert_array_equal(frame["motor_rates_hz"], finite.neuron_recording["rates_hz"][step, :48])
        np.testing.assert_array_equal(frame["channel_rates_hz"], playback["channels_hz"][step])
        np.testing.assert_array_equal(frame["angles"], playback["joint_offsets_radians"][step])
        assert frame["dopamine"] == finite.traces["dopamine_concentration"][step]
        assert frame["npf"] == finite.traces["npf_exposure_proxy"][step]
        assert frame["mean_rate_hz"] == finite.traces["mean_rate_hz"][step]
        assert frame["dopamine_sensitivity"] == finite.traces["dopamine_sensitivity"][step]
        assert frame["dopamine_reward_proxy"] == finite.traces["dopamine_reward_proxy"][step]
        json.dumps(frame, allow_nan=False)


def assert_same_state(first, second):
    assert first.nstep == second.nstep
    assert first.npf == second.npf
    for key in first._state_arrays():
        np.testing.assert_array_equal(first._state_arrays()[key], second._state_arrays()[key])
    np.testing.assert_array_equal(first.channel_rates, second.channel_rates)


def test_chunk_boundaries_do_not_reset_or_change_state():
    graph, config, mapping = specimen()
    together = LiveSimulation(graph, config, mapping)
    chunked = LiveSimulation(graph, config, mapping)
    together.step(113)
    for steps in (1, 17, 0, 20, 75):
        chunked.step(steps)
        chunked.snapshot()
    assert_same_state(together, chunked)
    assert chunked.t > config.duration
    assert chunked.nstep == 113


def test_checkpoint_restart_preserves_all_state_identity_and_continuation(tmp_path):
    graph, config, mapping = specimen()
    original = LiveSimulation(graph, config, mapping)
    original.step(37)
    path = tmp_path / "fly.npz"
    original.save_checkpoint(path)
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[key].dtype.kind != "O" for key in archive.files)
    resumed = LiveSimulation.from_checkpoint(path, graph, mapping)
    assert_same_state(original, resumed)
    assert resumed.run_id == original.run_id
    assert resumed.snapshot(True) == original.snapshot(True)
    original.step(61)
    resumed.step(7)
    resumed.step(54)
    assert_same_state(original, resumed)
    assert resumed.run_id == original.run_id
    assert resumed.nstep == 98


def test_restoring_existing_engine_does_not_reinitialize_muscles_or_joints(tmp_path):
    graph, config, mapping = specimen()
    saved = LiveSimulation(graph, config, mapping)
    saved.step(30)
    path = tmp_path / "fly.npz"
    saved.save_checkpoint(path)
    other = LiveSimulation(graph, config, mapping)
    other.step(2)
    assert not np.array_equal(other.muscle_activations, saved.muscle_activations)
    other.restore_checkpoint(path)
    assert_same_state(saved, other)
    assert saved.run_id == other.run_id


@pytest.mark.parametrize("mismatch", ["graph", "mapping", "config"])
def test_checkpoint_fingerprints_reject_changed_inputs_without_mutating_engine(tmp_path, mismatch):
    graph, config, mapping = specimen()
    source = LiveSimulation(graph, config, mapping)
    source.step(10)
    path = tmp_path / "fly.npz"
    source.save_checkpoint(path)
    if mismatch == "graph":
        graph = deepcopy(graph)
        graph.connectivity.data[0] += 1
    elif mismatch == "mapping":
        mapping = deepcopy(mapping)
        mapping["channels"][0]["body_ids"] = list(mapping["channels"][1]["body_ids"])
    else:
        config = replace(config, seed=8)
    target = LiveSimulation(graph, config, mapping)
    target.step(3)
    before = target.snapshot(True)
    with pytest.raises(ValueError, match="fingerprint"):
        target.restore_checkpoint(path)
    assert target.snapshot(True) == before


@pytest.mark.parametrize("corruption", ["rates", "metadata", "object_array", "missing_array", "truncated"])
def test_corrupt_checkpoint_fails_closed(tmp_path, corruption):
    graph, config, mapping = specimen()
    live = LiveSimulation(graph, config, mapping)
    live.step(10)
    path = tmp_path / "fly.npz"
    live.save_checkpoint(path)
    if corruption == "truncated":
        path.write_bytes(path.read_bytes()[:100])
    else:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        if corruption == "rates":
            arrays["rates"][0] += 1
        elif corruption == "metadata":
            metadata = json.loads(str(arrays["metadata"].item()))
            metadata["nstep"] = 0
            arrays["metadata"] = np.asarray(json.dumps(metadata))
        elif corruption == "object_array":
            arrays["rates"] = np.asarray([{"not": "numeric state"}], dtype=object)
        else:
            del arrays["muscle_activations"]
        np.savez_compressed(path, **arrays)
    before = live.snapshot(True)
    with pytest.raises(Exception):
        live.restore_checkpoint(path)
    assert live.snapshot(True) == before


def test_atomic_checkpoint_failure_preserves_previous_file(tmp_path, monkeypatch):
    graph, config, mapping = specimen()
    live = LiveSimulation(graph, config, mapping)
    live.step(4)
    path = tmp_path / "fly.npz"
    live.save_checkpoint(path)
    previous = path.read_bytes()
    live.step(8)

    def failed_write(*args, **kwargs):
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(np, "savez_compressed", failed_write)
    with pytest.raises(OSError):
        live.save_checkpoint(path)
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]
    restored = LiveSimulation.from_checkpoint(path, graph, mapping)
    assert restored.nstep == 4


def test_live_storage_does_not_grow_with_steps_or_snapshots():
    graph, config, mapping = specimen()
    live = LiveSimulation(graph, config, mapping)
    names = set(vars(live))
    sizes = {key: len(value) for key, value in vars(live).items() if isinstance(value, (list, dict))}
    array_bytes = sum(value.nbytes for value in live._state_arrays().values())
    for _ in range(500):
        live.step(10)
        frame = live.snapshot(include_motor_rates=True)
        assert len(frame["motor_rates_hz"]) == 48
    assert live.nstep == 5000
    assert live.t == 50.0
    assert set(vars(live)) == names
    assert {key: len(value) for key, value in vars(live).items() if isinstance(value, (list, dict))} == sizes
    assert sum(value.nbytes for value in live._state_arrays().values()) == array_bytes
    assert "motor_rates_hz" not in live.snapshot()


@pytest.mark.parametrize("problem", ["missing_motor", "duplicate_motor", "absent_motor", "missing_channel", "duplicate_channel", "unknown_channel_id"])
def test_live_mapping_requires_actual_unique_ids_and_complete_channels(problem):
    graph, config, mapping = specimen()
    if problem == "missing_motor":
        del mapping["motor_neuron_ids"]
    elif problem == "duplicate_motor":
        mapping["motor_neuron_ids"].append(1000)
    elif problem == "absent_motor":
        mapping["motor_neuron_ids"].append(999999)
    elif problem == "missing_channel":
        mapping["channels"].pop()
    elif problem == "duplicate_channel":
        mapping["channels"].append(deepcopy(mapping["channels"][0]))
    else:
        mapping["channels"][0]["body_ids"] = [2000]
    with pytest.raises(ValueError):
        LiveSimulation(graph, config, mapping)


@pytest.mark.parametrize("steps", [-1, 0.5, True])
def test_invalid_step_count_rejected(steps):
    graph, config, mapping = specimen()
    live = LiveSimulation(graph, config, mapping)
    with pytest.raises(ValueError):
        live.step(steps)
    assert live.nstep == 0


def test_input_config_copy_and_detected_engine_config_mutation():
    graph, config, mapping = specimen()
    live = LiveSimulation(graph, config, mapping)
    config.npf_drive = 1
    assert live.config.npf_drive == 0.6
    live.config.npf_drive = 1
    with pytest.raises(ValueError, match="mutated"):
        live.step()


def test_advance_beyond_twenty_seconds_matches_batch_without_wrapping():
    graph, config, mapping = specimen()
    finite = run_simulation(graph, replace(config, duration=21.0), record_neuron_ids=graph.ids)
    expected_motor = motor_playback(finite, mapping, "long", "Long comparison", "test")
    live = LiveSimulation(graph, config, mapping)
    identity = live.run_id
    live.step(1999)
    before_boundary = live.snapshot()
    live.step(101)
    assert live.nstep == 2100
    assert live.t == 21.0
    assert live.run_id == identity == before_boundary["run_id"]
    np.testing.assert_array_equal(live.rates, finite.neuron_recording["rates_hz"][-1])
    np.testing.assert_array_equal(live.joint_offsets, expected_motor["joint_offsets_radians"][-1])
    assert live.snapshot()["step"] > before_boundary["step"]


def test_model_source_fingerprint_change_rejects_checkpoint(tmp_path, monkeypatch):
    graph, config, mapping = specimen()
    source = LiveSimulation(graph, config, mapping)
    source.step(5)
    path = tmp_path / "fly.npz"
    source.save_checkpoint(path)
    original_read_bytes = Path.read_bytes

    def changed_model_source(file):
        content = original_read_bytes(file)
        return content + b"\n# changed model implementation\n" if file.name == "model.py" else content

    monkeypatch.setattr(Path, "read_bytes", changed_model_source)
    target = LiveSimulation(graph, config, mapping)
    assert target.fingerprints["mechanism"] != source.fingerprints["mechanism"]
    before = target.snapshot(True)
    with pytest.raises(ValueError, match="fingerprint"):
        target.restore_checkpoint(path)
    assert target.snapshot(True) == before
    with pytest.raises(ValueError, match="fingerprint"):
        LiveSimulation.from_checkpoint(path, graph, mapping)


def test_partial_step_failure_cannot_publish_or_overwrite_last_good_checkpoint(tmp_path, monkeypatch):
    graph, config, mapping = specimen()
    live = LiveSimulation(graph, config, mapping)
    live.step(5)
    path = tmp_path / "fly.npz"
    live.save_checkpoint(path)
    before = live.snapshot(True)
    good_bytes = path.read_bytes()

    def interrupted_update(concentration, sensitivity, release, configuration):
        concentration[:] = 0.91
        raise RuntimeError("interrupted after partially updating dopamine")

    with monkeypatch.context() as patch:
        patch.setattr(live_module, "_advance_dopamine", interrupted_update)
        with pytest.raises(RuntimeError, match="partially updating"):
            live.step()
        for action in (live.snapshot, live.step, lambda: live.save_checkpoint(path)):
            with pytest.raises(RuntimeError, match="restore a valid checkpoint"):
                action()
        assert path.read_bytes() == good_bytes
        live.restore_checkpoint(path)
        assert live.snapshot(True) == before
    live.step()
    assert live.nstep == 6
    assert live.run_id == before["run_id"]


def test_nonfinite_state_cannot_replace_last_good_checkpoint(tmp_path):
    graph, config, mapping = specimen()
    live = LiveSimulation(graph, config, mapping)
    live.step(5)
    path = tmp_path / "fly.npz"
    live.save_checkpoint(path)
    previous = path.read_bytes()
    live.rates[0] = np.nan
    with pytest.raises(ValueError, match="finiteness"):
        live.save_checkpoint(path)
    assert path.read_bytes() == previous


def test_mutated_mechanics_fail_before_advancing_or_checkpointing(tmp_path):
    graph, config, mapping = specimen()
    live = LiveSimulation(graph, config, mapping)
    live.mechanics["gain_radians"] = 9
    for action in (live.step, live.snapshot, lambda: live.save_checkpoint(tmp_path / "fly.npz")):
        with pytest.raises(ValueError, match="mechanics must not be mutated"):
            action()
    assert live.nstep == 0
