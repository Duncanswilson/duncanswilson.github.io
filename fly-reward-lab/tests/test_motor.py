"""Functional checks of actual recorded rates and the explicit motor bridge."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from flyreward.model import run_simulation
from flyreward.motor import JOINTS, LEGS, MECHANICS, motor_playback, write_motor_script
from flyreward.types import Graph, SimulationConfig, SimulationResult


def synthetic_mapping():
    channels = []
    for leg in LEGS:
        for joint in JOINTS:
            for action in ("flexor", "extensor"):
                index = len(channels)
                channels.append({"leg": leg, "joint": joint, "action": action,
                                 "body_ids": [1000 + 2 * index, 1001 + 2 * index]})
    return {"channels": channels}


def synthetic_result(duration=0.1):
    # Reversed recording order is intentionally different from mapping order.
    ids = list(reversed(range(1000, 1048))) + [9999]
    times = np.arange(round(duration / 0.01) + 1, dtype=float) * 0.01
    rates = np.zeros((len(times), len(ids)), dtype=np.float32)
    return SimulationResult(
        config=asdict(SimulationConfig(duration=duration)),
        time=times,
        traces={"dopamine_concentration": np.zeros(len(times)),
                "npf_exposure_proxy": np.zeros(len(times)),
                "dopamine_population_rate_hz": np.zeros(len(times)),
                "dopamine_reward_proxy": np.zeros(len(times))},
        metrics={},
        neuron_recording={
            "ids": ids, "rates_hz": rates,
            "neurotransmitters": ["glutamate"] * len(ids),
            "cell_types": [f"synthetic_motor_{body}" for body in ids],
            "source": "Actual per-neuron primary simulation rate state in Hz, indexed by graph body ID",
        },
    )


def set_rates(result, body_ids, values):
    index = {body: col for col, body in enumerate(result.neuron_recording["ids"])}
    result.neuron_recording["rates_hz"][:, [index[body] for body in body_ids]] = values


def playback(result, mapping=None):
    return motor_playback(result, mapping or synthetic_mapping(), "test", "Test run", "report.html")


def test_channel_means_use_exact_body_ids_with_different_recording_order():
    result = synthetic_result(duration=0.02)
    set_rates(result, [1000, 1001], [[20, 40], [40, 60], [60, 80]])
    set_rates(result, [1002, 1003], [[10, 30], [30, 50], [50, 70]])
    set_rates(result, [9999], 99)
    output = playback(result)
    means = np.asarray(output["channels_hz"])
    np.testing.assert_array_equal(means[:, 0], [30, 50, 70])
    np.testing.assert_array_equal(means[:, 1], [20, 40, 60])
    np.testing.assert_array_equal(means[:, 2:], 0)
    # Unmapped-but-recorded motor cells affect the motor population mean only.
    np.testing.assert_allclose(output["motor_mean_hz"], result.neuron_recording["rates_hz"].mean(axis=1), rtol=1e-7)


@pytest.mark.parametrize("action, sign", [("flexor", 1), ("extensor", -1)])
def test_independent_antagonist_stimulation_has_correct_sign_and_joint(action, sign):
    result = synthetic_result(duration=1.0)
    channel = next(c for c in synthetic_mapping()["channels"]
                   if (c["leg"], c["joint"], c["action"]) == ("RM", "tibia", action))
    set_rates(result, channel["body_ids"], 40)
    angles = np.asarray(playback(result)["joint_offsets_radians"])
    target_col = LEGS.index("RM") * 2 + JOINTS.index("tibia")
    assert sign * angles[-1, target_col] > 0
    assert abs(angles[-1, target_col]) < MECHANICS["gain_radians"] * 0.4
    assert np.all(sign * np.diff(angles[:, target_col]) >= 0)
    np.testing.assert_array_equal(np.delete(angles, target_col, axis=1), 0)


def test_reward_and_dopamine_proxy_changes_cannot_change_motor_pose():
    result = synthetic_result()
    set_rates(result, [1000, 1001], 35)
    original = playback(result)
    altered = deepcopy(result)
    altered.traces["dopamine_concentration"][:] = 0.99
    altered.traces["npf_exposure_proxy"][:] = 0.88
    altered.traces["dopamine_population_rate_hz"][:] = 100
    altered.traces["dopamine_reward_proxy"][:] = 1
    altered.metrics["dopamine_reward_proxy_mean"] = 100000
    modified = playback(altered)
    assert original["dopamine"] != modified["dopamine"]
    assert original["npf"] != modified["npf"]
    assert original["rate"] != modified["rate"]
    for key in original:
        if key not in {"dopamine", "npf", "rate"}:
            assert original[key] == modified[key], key


def test_constant_motor_input_converges_without_an_autonomous_oscillator():
    result = synthetic_result(duration=3.0)
    set_rates(result, [1000, 1001], 50)
    angles = np.asarray(playback(result)["joint_offsets_radians"])
    target = MECHANICS["gain_radians"] * 0.5
    expected = target * -np.expm1(-result.time / MECHANICS["joint_tau_seconds"])
    np.testing.assert_allclose(angles[:, 0], expected, atol=1e-12, rtol=1e-12)
    assert np.all(np.diff(angles[:, 0]) >= 0)
    assert np.ptp(angles[result.time >= 2, 0]) < 1e-7
    np.testing.assert_array_equal(angles[:, 1:], 0)


def test_bridge_consumes_actual_model_recording_api():
    ids = np.arange(1000, 1048, dtype=np.int64)
    graph = Graph(ids=ids, connectivity=csr_matrix((48, 48)),
                  neurotransmitters=np.repeat("glutamate", 48),
                  cell_types=np.asarray([f"motor_{body}" for body in ids]))
    result = run_simulation(graph, SimulationConfig(duration=0.05), record_neuron_ids=ids[::-1])
    output = playback(result)
    columns = {body: i for i, body in enumerate(result.neuron_recording["ids"])}
    expected = np.stack([
        result.neuron_recording["rates_hz"][:, [columns[body] for body in channel["body_ids"]]].astype(float).mean(axis=1)
        for channel in synthetic_mapping()["channels"]
    ], axis=1)
    np.testing.assert_array_equal(output["channels_hz"], expected)
    np.testing.assert_array_equal(output["time"], result.time)
    # The payload contains portable JSON values, not NumPy arrays or scalars.
    assert json.loads(json.dumps(output, allow_nan=False))["recorded_neuron_count"] == 48


@pytest.mark.parametrize("problem", ["absent", "duplicate", "missing_id", "rows", "columns"])
def test_missing_duplicate_or_misaligned_recordings_fail(problem):
    result = synthetic_result()
    rec = result.neuron_recording
    if problem == "absent":
        result.neuron_recording = {}
    elif problem == "duplicate":
        rec["ids"][1] = rec["ids"][0]
    elif problem == "missing_id":
        column = rec["ids"].index(1000)
        for key in ("ids", "neurotransmitters", "cell_types"):
            del rec[key][column]
        rec["rates_hz"] = np.delete(rec["rates_hz"], column, axis=1)
    elif problem == "rows":
        rec["rates_hz"] = rec["rates_hz"][:-1]
    else:
        rec["rates_hz"] = rec["rates_hz"][:, :-1]
    with pytest.raises(ValueError):
        playback(result)


def test_integer_coercion_cannot_conceal_duplicate_recorded_ids():
    result = synthetic_result()
    rec = result.neuron_recording
    rec["ids"].append(str(rec["ids"][0]))
    rec["rates_hz"] = np.column_stack([rec["rates_hz"], rec["rates_hz"][:, 0]])
    rec["neurotransmitters"].append(rec["neurotransmitters"][0])
    rec["cell_types"].append(rec["cell_types"][0])
    with pytest.raises(ValueError):
        playback(result)


@pytest.mark.parametrize("key", ["neurotransmitters", "cell_types"])
def test_recording_annotation_columns_must_align_with_ids(key):
    result = synthetic_result()
    result.neuron_recording[key] = result.neuron_recording[key][:-1]
    with pytest.raises(ValueError):
        playback(result)


@pytest.mark.parametrize("trace", ["dopamine_concentration", "npf_exposure_proxy", "dopamine_population_rate_hz"])
def test_hud_trace_lengths_must_match_motor_frame_times(trace):
    result = synthetic_result()
    result.traces[trace] = result.traces[trace][:-1]
    with pytest.raises(ValueError):
        playback(result)


@pytest.mark.parametrize("change", ["coarsened_times", "record_every", "missing_initial", "missing_final"])
def test_bridge_requires_complete_full_resolution_neural_frames(change):
    result = synthetic_result()
    if change == "coarsened_times":
        result.time *= 2
        result.config["duration"] *= 2
    elif change == "record_every":
        result.config["record_every"] = 2
    else:
        selected = slice(1, None) if change == "missing_initial" else slice(None, -1)
        result.time = result.time[selected]
        result.neuron_recording["rates_hz"] = result.neuron_recording["rates_hz"][selected]
        result.traces = {key: values[selected] for key, values in result.traces.items()}
    with pytest.raises(ValueError):
        playback(result)


@pytest.mark.parametrize("problem", ["negative", "nonfinite", "above_ceiling", "backward_time"])
def test_nonphysical_rates_or_time_order_fail(problem):
    result = synthetic_result()
    if problem == "backward_time":
        result.time[3] = result.time[2]
    else:
        result.neuron_recording["rates_hz"][3, 0] = {
            "negative": -1, "nonfinite": float("nan"), "above_ceiling": 101
        }[problem]
    with pytest.raises(ValueError):
        playback(result)


def test_generated_js_replays_exact_frames_and_interpolates_without_clock_input(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js required for generated playback integration")
    result = synthetic_result()
    set_rates(result, [1000, 1001], np.linspace(0, 50, len(result.time))[:, None])
    run = playback(result)
    path = tmp_path / "motor.js"
    payload = {"legs": list(LEGS), "joints": list(JOINTS), "runs": [run], "note": "</script>"}
    write_motor_script(path, payload)
    assert "</script>" not in path.read_text()
    script = "global.window=global;require(process.argv[1]);process.stdout.write(JSON.stringify([FlyMotor.sample('test',.02),FlyMotor.sample('test',.025),FlyMotor.sample('test',99)]));"
    response = subprocess.run([node, "-e", script, str(path)], check=True, capture_output=True, text=True)
    exact, between, end = json.loads(response.stdout)
    angles = np.asarray(run["joint_offsets_radians"])
    np.testing.assert_allclose(exact["angles"], angles[2], atol=1e-12)
    np.testing.assert_allclose(between["angles"], (angles[2] + angles[3]) / 2, atol=1e-12)
    np.testing.assert_allclose(end["angles"], angles[-1], atol=1e-12)
    assert exact["motorPose"]["bodyRoll"] == 0
    assert exact["motorPose"]["wingAngles"] == {"L": 0, "R": 0}
    assert exact["motorPose"]["legs"]["LF"]["trochanter"] == exact["angles"][0]
