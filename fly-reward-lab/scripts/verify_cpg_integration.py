#!/usr/bin/env python3
"""Quantify the real-graph hybrid model without modifying any live deployment."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import tempfile
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from flyreward.server import graph_factory


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path,
                        help="Directory containing the verified MaleCNS files/cache; no download is performed")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Explicit directory for CPG_VERIFICATION.json and hybridtrace.npz")
    parser.add_argument("--seconds", type=float, default=5.0,
                        help="Simulated duration, in whole 10 ms steps (default: 5)")
    parser.add_argument("--warmup-seconds", type=float, default=2.0,
                        help="Exclude earlier samples from variability measurements (default: 2)")
    parser.add_argument("--motor-stimulation", choices=("baseline", "recruited"), default="baseline",
                        help="Explicit motor-neuron intervention to verify")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive and finite")
    if not math.isfinite(args.warmup_seconds) or not 0 <= args.warmup_seconds < args.seconds:
        parser.error("--warmup-seconds must be finite, nonnegative and less than --seconds")
    if not args.data_dir.is_dir():
        parser.error("--data-dir must be an existing directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = {name: sha(REPO / "flyreward" / name) for name in
               ("live_model.py", "server.py", "model.py", "cpg.py", "cpg_live.py", "body.py")}
    sources["visualizer/fly.js"] = sha(REPO / "visualizer/fly.js")
    started = time.perf_counter()
    print("Loading actual MaleCNS graph and hybrid model", flush=True)
    factory = graph_factory(args.data_dir, motor_mode="cpg", motor_stimulation=args.motor_stimulation)
    engine, metadata = factory()
    quotient = args.seconds / engine.config.dt
    steps = round(quotient)
    if not math.isclose(quotient, steps, rel_tol=0, abs_tol=1e-7):
        parser.error("--seconds must be a whole number of neural integration steps")
    loaded = time.perf_counter()
    print(f"Loaded {metadata['neuron_count']} neurons; circuit {engine.circuit.metadata()}", flush=True)
    leg_order = metadata["legs"]
    selected_ids = engine.circuit.motor_neuron_ids
    raw_index = {body: i for i, body in enumerate(engine.motor_neuron_ids)}
    selected_raw_indices = [raw_index[int(body)] for body in selected_ids]
    channels = metadata["channels"]
    primary_indices = [np.asarray([raw_index[int(body)] for body in ch["body_ids"]])
                       for ch in channels]
    samples = {key: [] for key in ("time", "selected_rates", "channels", "angles", "body", "geometry")}
    agreement_checks = 0

    def sample():
        nonlocal agreement_checks
        frame = engine.snapshot(include_motor_rates=True)
        raw = np.asarray(frame["motor_rates_hz"], dtype=float)
        means = np.asarray([raw[indices].mean() for indices in primary_indices])
        np.testing.assert_array_equal(means, frame["channel_rates_hz"])
        np.testing.assert_array_equal(engine.rates[engine.circuit.graph_indices],
                                      engine.circuit.rates.astype(np.float32))
        assert np.isfinite(raw).all() and np.all(raw >= 0) and np.all(raw <= frame["rate_ceiling_hz"])
        assert frame["motor_mode"] == "cpg"
        agreement_checks += 1
        pose = frame["motorPose"]
        samples["time"].append(frame["sim_time"])
        samples["selected_rates"].append(raw[selected_raw_indices])
        samples["channels"].append(means)
        samples["angles"].append(frame["angles"])
        samples["body"].append(pose["bodyPositionMm"])
        samples["geometry"].append([[pose["legPointsMm"][leg][joint] for joint in ("hip", "knee", "foot")]
                                    for leg in leg_order])

    sample()
    for i in range(steps):
        engine.step()
        sample()
        if (i + 1) % 100 == 0:
            print(f"Integrated {engine.t:.2f} simulated seconds in {time.perf_counter()-loaded:.2f} wall seconds", flush=True)
    integrated = time.perf_counter()
    arrays = {key: np.asarray(value) for key, value in samples.items()}
    steady = arrays["time"] >= args.warmup_seconds
    if steady.sum() < 2:
        parser.error("The analysis window must include at least two sampled states")
    projection = np.asarray([[85., 22., 0.], [0., 48., -85.]])
    projected_geometry = arrays["geometry"] @ projection.T
    projected_body = arrays["body"] @ projection.T
    trace = args.output_dir / "hybridtrace.npz"
    np.savez_compressed(trace, time_seconds=arrays["time"], selected_motor_ids=selected_ids,
                        selected_motor_rates_hz=arrays["selected_rates"], channel_rates_hz=arrays["channels"],
                        joint_angles_radians=arrays["angles"], body_position_mm=arrays["body"],
                        leg_points_mm=arrays["geometry"], projected_leg_points_actor_pixels=projected_geometry,
                        projected_body_actor_pixels=projected_body)
    print("Testing full-graph checkpoint reconstruction and exact continuation", flush=True)
    with tempfile.TemporaryDirectory(prefix="cpg-checkpoint-", dir=args.output_dir) as checkpoint_dir:
        checkpoint = Path(checkpoint_dir) / "checkpoint.npz"
        engine.save_checkpoint(checkpoint)
        resumed, _ = factory()
        resumed.restore_checkpoint(checkpoint)
    assert engine.snapshot(True) == resumed.snapshot(True)
    assert engine.run_id == resumed.run_id
    for key, value in engine._state_arrays().items():
        np.testing.assert_array_equal(value, resumed._state_arrays()[key])
    engine.step(2)
    resumed.step()
    resumed.step()
    for key, value in engine._state_arrays().items():
        np.testing.assert_array_equal(value, resumed._state_arrays()[key])
    assert engine.snapshot(True) == resumed.snapshot(True)
    for name, digest in sources.items():
        path = REPO / name if name.startswith("visualizer/") else REPO / "flyreward" / name
        assert sha(path) == digest, f"Source changed during verification: {name}"

    motor_ptp = np.ptp(arrays["selected_rates"][steady], axis=0)
    channel_ptp = np.ptp(arrays["channels"][steady], axis=0)
    joint_ptp = np.ptp(arrays["angles"][steady], axis=0)
    geometry_ptp = np.ptp(projected_geometry[steady], axis=0)
    feet_bounds = np.linalg.norm(geometry_ptp[:, 2, :], axis=-1)
    point_bounds = np.linalg.norm(geometry_ptp, axis=-1)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak_rss if sys.platform == "darwin" else peak_rss * 1024
    report = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Local quantitative test using the complete real MaleCNS graph; no live VPS state was changed.",
        "model": metadata["model"], "config": asdict(engine.config),
        "neuron_count": metadata["neuron_count"], "directed_edge_count": metadata["directed_edge_count"],
        "circuit": engine.circuit.metadata(), "body": engine.body.metadata(),
        "sample_period_seconds": engine.config.dt, "trajectory_seconds": args.seconds,
        "samples": len(arrays["time"]),
        "analysis_window_seconds": [args.warmup_seconds, args.seconds],
        "raw_circuit_motor_neurons": len(selected_ids),
        "raw_circuit_motor_ptp_hz": {str(int(body)): float(ptp) for body, ptp in zip(selected_ids, motor_ptp)},
        "raw_circuit_motor_max_ptp_hz": float(motor_ptp.max()),
        "channels": [{"leg": ch["leg"], "joint": ch["joint"], "action": ch["action"],
                      "minimum_hz": float(arrays["channels"][steady, i].min()),
                      "maximum_hz": float(arrays["channels"][steady, i].max()),
                      "peak_to_peak_hz": float(channel_ptp[i])} for i, ch in enumerate(channels)],
        "joint_peak_to_peak_degrees": {f"{leg}_{joint}": float(joint_ptp[2*i+j] * 180 / np.pi)
                                        for i, leg in enumerate(leg_order) for j, joint in enumerate(("trochanter", "tibia"))},
        "body_peak_to_peak_mm": np.ptp(arrays["body"][steady], axis=0).tolist(),
        "body_projected_peak_to_peak_actor_pixels": np.ptp(projected_body[steady], axis=0).tolist(),
        "foot_projected_bbox_diagonal_actor_pixels": dict(zip(leg_order, feet_bounds.tolist())),
        "leg_point_projected_bbox_diagonal_actor_pixels": {
            leg: dict(zip(("hip", "knee", "foot"), point_bounds[i].tolist()))
            for i, leg in enumerate(leg_order)},
        "maximum_leg_point_projected_bbox_diagonal_actor_pixels": float(point_bounds.max()),
        "projection_note": "Uses FlyActor projectBodyPoint: [85*x+22*y,48*y-85*z], before room scale/pixel rasterization. Foot values are bounding-box diagonal upper bounds, not exaggerated display motion.",
        "same_snapshot_channel_agreement": {"snapshots_checked": agreement_checks, "channels_each": 24, "exact": True},
        "checkpoint_roundtrip_and_two_step_continuation": {"exact_all_numeric_arrays_and_snapshot": True,
                                                          "run_identity_preserved": True},
        "performance": {"load_wall_seconds": loaded-started,
                        "trajectory_wall_seconds": integrated-loaded,
                        "simulated_seconds_per_wall_second": args.seconds/(integrated-loaded),
                        "peak_rss_bytes_including_second_engine_for_restart_test": int(peak_bytes)},
        "source_sha256": sources, "trace_sha256": sha(trace),
        "limitations": ["Only the declared front circuit has autonomous rhythmic dynamics; surrounding cells use the prior rate model.",
                        "Tonic CPG input is independent of dopamine/NPF; body diagnostics are not fed into sensory neurons.",
                        "Motor-rate variability does not by itself establish visibly substantial movement, gait or subjective pleasure."]
    }
    destination = args.output_dir / "CPG_VERIFICATION.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(destination), "trace": str(trace),
                      "raw_motor_max_ptp_hz": float(motor_ptp.max()),
                      "channel_max_ptp_hz": float(channel_ptp.max()),
                      "joint_max_ptp_degrees": float(joint_ptp.max()*180/np.pi),
                      "foot_max_projected_bound_actor_pixels": float(feet_bounds.max()),
                      "leg_point_max_projected_bound_actor_pixels": float(point_bounds.max()),
                      "checkpoint_exact": True, "performance": report["performance"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
