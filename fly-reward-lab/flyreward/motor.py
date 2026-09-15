"""Recorded motor-neuron output and an explicit, uncalibrated joint bridge.

Network rates are recorded without changing the neural model. Muscle activation
and joint dynamics below are engineering assumptions, not measured fly mechanics.
There is no oscillator, gait generator, reward-to-motion mapping or animation seed.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from numbers import Integral
from pathlib import Path
import time

import numpy as np

from .types import SimulationResult

LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")
JOINTS = ("trochanter", "tibia")
MECHANICS = {
    "activation_tau_seconds": 0.04,
    "joint_tau_seconds": 0.12,
    "gain_radians": 1.2,
    "activation": "mean motor-neuron rate per labeled muscle group divided by config.max_rate, then first-order filtering",
    "joint_target": "gain_radians * (flexor_activation - extensor_activation)",
    "initial_conditions": "muscle activation starts at normalized initial neural rate; joint offsets start at zero",
    "muscle_weighting": "equal mean weighting within each group; no measured motor-unit force or muscle recruitment calibration",
    "joint_response": "overdamped first-order relaxation toward target; no autonomous oscillation",
    "body": "fixed root, head, abdomen and wings; no assigned motor action where the map is insufficient",
    "feedback": "one-way motor output replay; no physical contact solver, proprioception or body-to-neuron feedback",
    "validation": "uncalibrated kinematic approximation; not a validated neuromechanical fly simulation",
}


def motor_playback(result: SimulationResult, mapping: dict, run_id: str,
                   label: str, source: str) -> dict:
    """Map exact selected-neuron traces to explicit antagonist channel pairs."""
    recording = result.neuron_recording
    if not recording:
        raise ValueError("Motor playback requires individually recorded neuron rates")
    ids = list(recording["ids"])
    if any(isinstance(body_id, (bool, np.bool_)) or not isinstance(body_id, Integral) for body_id in ids):
        raise ValueError("Recorded motor IDs must be integers")
    if len(ids) != len(set(ids)):
        raise ValueError("Recorded motor IDs must be unique")
    for key in ("neurotransmitters", "cell_types"):
        if len(recording.get(key, ())) != len(ids):
            raise ValueError(f"Recorded {key} must align with motor body IDs")
    index = {int(body_id): i for i, body_id in enumerate(ids)}
    rates = np.asarray(recording["rates_hz"], dtype=np.float64)
    times = np.asarray(result.time, dtype=np.float64)
    if times.ndim != 1 or rates.shape != (len(times), len(ids)) or len(times) < 2:
        raise ValueError("Motor rates must align exactly with result times and body IDs")
    if not np.isfinite(rates).all() or (rates < 0).any():
        raise ValueError("Motor rates must be finite and nonnegative")
    if not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("Motor times must be finite and strictly increasing")
    dt = float(result.config["dt"])
    duration = float(result.config["duration"])
    if not np.isfinite(dt) or not np.isfinite(duration) or dt <= 0 or duration <= 0:
        raise ValueError("Motor playback requires finite positive neural duration and dt")
    quotient = duration / dt
    if not np.isfinite(quotient) or abs(quotient-round(quotient)) > 1e-7:
        raise ValueError("Motor duration must be an integer multiple of neural dt")
    if result.config.get("record_every") != 1 or len(times) != round(quotient)+1:
        raise ValueError("Motor mechanics require every neural frame, including initial and final states")
    if not np.allclose(times, np.arange(len(times))*dt, rtol=0, atol=1e-9):
        raise ValueError("Motor recording times must exactly follow the configured neural dt")
    for key in ("dopamine_concentration", "npf_exposure_proxy", "dopamine_population_rate_hz"):
        values = np.asarray(result.traces.get(key, ()), dtype=np.float64)
        if values.shape != times.shape or not np.isfinite(values).all():
            raise ValueError(f"Motor HUD trace {key} must be finite and frame-aligned")
    ceiling = float(result.config["max_rate"])
    if not np.isfinite(ceiling) or ceiling <= 0 or (rates > ceiling + 1e-4).any():
        raise ValueError("Motor rate normalization requires a valid simulation ceiling")
    channels = mapping["channels"]
    channel_index = {}
    means = []
    for channel in channels:
        key = (channel["leg"], channel["joint"], channel["action"])
        if key in channel_index:
            raise ValueError(f"Duplicate motor channel: {key}")
        body_ids = channel["body_ids"]
        if not body_ids or len(body_ids) != len(set(body_ids)):
            raise ValueError(f"Motor channel must contain unique identified neurons: {key}")
        missing = set(body_ids) - set(index)
        if missing:
            raise ValueError(f"Motor channel has unrecorded body IDs: {sorted(missing)}")
        channel_index[key] = len(means)
        means.append(rates[:, [index[body_id] for body_id in body_ids]].mean(axis=1))
    expected = {(leg, joint, action) for leg in LEGS for joint in JOINTS for action in ("flexor", "extensor")}
    if set(channel_index) != expected:
        raise ValueError("Motor mapping must explicitly supply all 24 leg/joint/antagonist channels")
    means = np.stack(means, axis=1)
    activation = np.zeros_like(means)
    activation[0] = means[0] / ceiling
    angles = np.zeros((len(times), len(LEGS) * len(JOINTS)), dtype=np.float64)
    for step, dt in enumerate(np.diff(times), 1):
        activation[step] = activation[step-1] + (means[step] / ceiling - activation[step-1]) * -np.expm1(-dt / MECHANICS["activation_tau_seconds"])
        for leg_idx, leg in enumerate(LEGS):
            for joint_idx, joint in enumerate(JOINTS):
                flex = activation[step, channel_index[leg, joint, "flexor"]]
                extend = activation[step, channel_index[leg, joint, "extensor"]]
                target = MECHANICS["gain_radians"] * (flex - extend)
                column = 2 * leg_idx + joint_idx
                angles[step, column] = angles[step-1, column] + (target - angles[step-1, column]) * -np.expm1(-dt / MECHANICS["joint_tau_seconds"])
    if not np.isfinite(angles).all():
        raise FloatingPointError("Motor bridge produced a nonfinite joint angle")
    max_change = float(np.max(np.abs(np.diff(angles, axis=0))))
    return {
        "id": run_id, "label": label, "duration": float(times[-1]), "source": source,
        "time": times.tolist(), "channels_hz": means.tolist(),
        "joint_offsets_radians": angles.tolist(),
        "dopamine": np.asarray(result.traces["dopamine_concentration"]).tolist(),
        "npf": np.asarray(result.traces["npf_exposure_proxy"]).tolist(),
        "rate": np.asarray(result.traces["dopamine_population_rate_hz"]).tolist(),
        "motor_mean_hz": rates.mean(axis=1).tolist(),
        "rate_ceiling_hz": ceiling,
        "recorded_neuron_count": len(ids),
        "config": result.config,
        "diagnostics": {
            "max_absolute_joint_offset_degrees": float(np.max(np.abs(angles)) * 180 / np.pi),
            "max_frame_change_degrees": max_change * 180 / np.pi,
            "final_joint_offsets_degrees": (angles[-1] * 180 / np.pi).tolist(),
            "final_motor_rate_range_hz": [float(rates[-1].min()), float(rates[-1].max())],
            "max_last_second_joint_change_degrees": float(np.max(np.ptp(angles[times >= max(times[0], times[-1]-1)], axis=0)) * 180 / np.pi),
        },
    }


def write_motor_script(path: Path, payload: dict) -> None:
    """Portable replay with fixed mechanical assumptions, loaded without fetch."""
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    script = '''/* Actual individually recorded model motor outputs; assumed joint mechanics. */
(function(global){"use strict";
const data=__DATA__;
function sample(run,time){
  if(typeof run==='string')run=data.runs.find(r=>r.id===run);
  if(!run||!Number.isFinite(time))throw new TypeError('A recorded motor run and finite time are required');
  const times=run.time;let left=0,right=times.length-1;
  const t=Math.max(times[0],Math.min(time,times[right]));
  while(right-left>1){const mid=(left+right)>>1;if(times[mid]<=t)left=mid;else right=mid;}
  const f=(t-times[left])/(times[right]-times[left]);
  const scalar=values=>values[left]+f*(values[right]-values[left]);
  const columns=values=>values[left].map((v,i)=>v+f*(values[right][i]-v));
  const angles=columns(run.joint_offsets_radians),rates=columns(run.channels_hz);
  const legs={};data.legs.forEach((leg,i)=>{legs[leg]={coxa:0,trochanter:angles[i*2],tibia:angles[i*2+1]};});
  return {dopamine:scalar(run.dopamine),npf:scalar(run.npf),rate:scalar(run.rate),
    motorMean:scalar(run.motor_mean_hz),channelRates:rates,angles,
    motorPose:{legs,bodyRoll:0,abdomenAngle:0,headAngle:0,wingAngles:{L:0,R:0}}};
}
global.FlyMotor=Object.freeze({...data,sample});
})(window);
'''.replace("__DATA__", encoded)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script, encoding="utf-8")


def run_motor_experiment(args) -> None:
    """Re-run the saved configuration and comparisons while recording every MN."""
    from .cli import _load_replay_config, _software_provenance, write_json
    from .data import load_graph
    from .model import presets, run_simulation
    from .motor_mapping import build_motor_mapping
    from .report import write_report

    winner = _load_replay_config(args.config, duration=args.duration)
    if not np.isfinite(args.comparison_duration) or args.comparison_duration <= 0:
        raise ValueError("comparison-duration must be finite and positive")
    print(f"Loading full connectome from {args.data_dir}", flush=True)
    graph = load_graph(args.data_dir)
    mapping = build_motor_mapping(args.data_dir, graph)
    all_ids = mapping["motor_neuron_ids"]
    print(f"Recording {len(all_ids)} identified motor neurons; {len(mapping['channels'])} mapped antagonist channels", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "motor_mapping.json", mapping)
    configs = [("best_tested_combination", "Best tested combination", winner)]
    labels = {"baseline":"Baseline", "dopamine":"Dopamine boost", "npf":"NPF-associated drive", "combined":"Combined", "combined_no_tolerance":"Combined, no tolerance", "maximal_dopamine":"Maximal dopamine"}
    configs += [(name,labels[name],replace(config,duration=args.comparison_duration,dt=winner.dt,seed=winner.seed,record_every=1)) for name,config in presets().items()]
    # Full resolution is required for muscle/joint integration; keep the saved neural parameters otherwise.
    configs[0] = (configs[0][0], configs[0][1], replace(winner, record_every=1))
    software = _software_provenance()
    software["motor_source_sha256"] = {name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ("motor.py","motor_mapping.py","types.py")}
    playback_runs=[]
    results=[]
    recordings=[]
    for i,(run_id,label,config) in enumerate(configs,1):
        tick=time.perf_counter()
        print(f"[{i}/{len(configs)}] {label}: {config.duration:g}s, recording individual motor rates", flush=True)
        result=run_simulation(graph,config,record_neuron_ids=all_ids)
        rec=result.neuron_recording
        destination=args.output/run_id
        destination.mkdir(parents=True,exist_ok=True)
        raw_path=destination/'neuron_rates.npz'
        np.savez_compressed(raw_path,time=result.time,body_ids=np.asarray(rec['ids'],dtype=np.int64),rates_hz=rec['rates_hz'],neurotransmitters=np.asarray(rec['neurotransmitters']),cell_types=np.asarray(rec['cell_types']))
        checksum=hashlib.sha256(raw_path.read_bytes()).hexdigest()
        record_info={"ids":rec["ids"],"shape":list(rec['rates_hz'].shape),"source":rec['source'],"rates_file":"neuron_rates.npz","sha256":checksum}
        source=f"../results/motor/{run_id}/report.html"
        # Relative paths follow the requested output directory, including custom destinations.
        import os
        source=Path(os.path.relpath(destination/'report.html',args.visualizer_data.parent)).as_posix()
        playback=motor_playback(result,mapping,run_id,label,source)
        playback['raw_rates_source']=Path(os.path.relpath(raw_path,args.visualizer_data.parent)).as_posix()
        playback['raw_rates_sha256']=checksum
        playback_runs.append(playback)
        result.neuron_recording=record_info
        result.metadata['motor_recording']=record_info
        result.metadata['motor_mechanics']=MECHANICS
        result.metadata['wall_seconds']=time.perf_counter()-tick
        write_json(destination/'result.json',asdict(result))
        write_report([result],graph.metadata,destination/'report.html')
        results.append(asdict(result))
        recordings.append({"id":run_id,"raw_rates_file":str(raw_path.relative_to(args.output)),"sha256":checksum,"diagnostics":playback['diagnostics']})
        print(f"  done in {result.metadata['wall_seconds']:.2f}s; max joint offset {playback['diagnostics']['max_absolute_joint_offset_degrees']:.3f} degrees", flush=True)
    payload={
        "schema_version":1,"mode":"recorded_simulated_motor_outputs","legs":list(LEGS),"joints":list(JOINTS),
        "mechanics":MECHANICS,"channels":mapping['channels'],"mapping_summary":mapping['summary'],
        "recorded_neuron_count":len(all_ids),"mapped_neuron_count":len(set(body for c in mapping['channels'] for body in c['body_ids'])),
        "runs":playback_runs,"sources":mapping['sources'],
        "mapping_source":Path(os.path.relpath(args.output/'motor_mapping.json',args.visualizer_data.parent)).as_posix(),
        "provenance":"Rates are actual state variables from identified motor neurons in the connectome-constrained hypothesis model. They are not biological recordings. Leg mapping uses annotated muscle groups; activation and joint mechanics remain assumptions. Unmapped body parts are stationary.",
    }
    write_json(args.output/'motor_playback.json',payload)
    write_json(args.output/'results.json',{"schema_version":1,"graph":graph.metadata,"results":results,"optimization":None,"motor_recordings":recordings,"software":software,"created_utc":datetime.now(timezone.utc).isoformat()})
    write_motor_script(args.visualizer_data,payload)
    print(f"Motor playback: {args.visualizer_data.resolve()}",flush=True)
