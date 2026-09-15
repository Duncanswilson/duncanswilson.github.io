from __future__ import annotations
import argparse
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
import hashlib
from importlib import metadata as package_metadata
import json
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
from . import __version__
from .types import SimulationConfig
from .experiments import OBJECTIVES, candidates, rank_results


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, default=_json_default, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def _graph(args):
    from .data import load_graph, make_demo_graph
    print("Loading synthetic demonstration graph" if args.demo else f"Loading MaleCNS from {args.data_dir}", flush=True)
    graph = make_demo_graph(seed=args.seed) if args.demo else load_graph(args.data_dir, max_neurons=args.max_neurons)
    print(f"Loaded {len(graph.ids):,} neurons, {graph.connectivity.nnz:,} directed neuron pairs", flush=True)
    return graph


def _load_replay_config(path: Path, duration: float | None = None) -> SimulationConfig:
    """Load an exact saved configuration, with only an explicit duration override."""
    from .model import _validate_config
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Replay file must contain a JSON object")
    if "winner" in payload:
        winner = payload["winner"]
        if not isinstance(winner, dict) or not isinstance(winner.get("config"), dict):
            raise ValueError("Replay winner must contain a config object")
        payload = winner["config"]
    elif "config" in payload:
        if not isinstance(payload["config"], dict):
            raise ValueError("Replay config must be a JSON object")
        payload = payload["config"]
    expected = {field.name for field in fields(SimulationConfig)}
    unknown = set(payload) - expected
    if unknown:
        raise ValueError(f"Unknown configuration fields: {', '.join(sorted(unknown))}")
    missing = expected - set(payload)
    if missing:
        raise ValueError(f"Replay requires a complete saved configuration; missing fields: {', '.join(sorted(missing))}")
    config = SimulationConfig(**payload)
    if duration is not None:
        config = replace(config, duration=duration)
    if not isinstance(config.preset, str) or not config.preset:
        raise ValueError("preset must be a nonempty string")
    # Validate before loading a large graph or using the stored demo graph seed.
    _validate_config(config)
    return config


def _software_provenance() -> dict:
    package_dir = Path(__file__).resolve().parent
    versions = {}
    for name in ("numpy", "scipy", "pandas", "pyarrow"):
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {
        "package_version": __version__,
        "source_sha256": {name: hashlib.sha256((package_dir / name).read_bytes()).hexdigest()
                          for name in ("model.py", "data.py", "experiments.py")},
        "source_hash_scope": "Package source files on disk at command start",
        "dependency_versions": versions,
    }


def _execute(args):
    from .model import presets, run_simulation
    from .report import write_report
    software = _software_provenance()
    if args.command == "replay":
        config = _load_replay_config(args.config, duration=args.duration)
        configs = [config]
        # Synthetic graph generation and simulation must share the saved seed.
        args.seed = config.seed
    elif args.command == "optimize":
        base = SimulationConfig(duration=args.duration, dt=args.dt, seed=args.seed, record_every=args.record_every)
        configs = candidates(base, args.trials)
    else:
        available = presets()
        names = list(available) if args.presets == "all" else args.presets.split(",")
        unknown = set(names) - set(available)
        if unknown:
            raise ValueError(f"Unknown presets: {', '.join(sorted(unknown))}")
        configs = [replace(available[n], duration=args.duration, dt=args.dt, seed=args.seed,
                           record_every=args.record_every) for n in names]
    graph = _graph(args)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    started = time.perf_counter()
    for i, config in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {config.preset}: running {config.duration:g}s simulated time", flush=True)
        tick = time.perf_counter()
        result = run_simulation(graph, config)
        result.metadata["wall_seconds"] = time.perf_counter() - tick
        results.append(result)
        # A replay label comes from an external JSON file, not the preset list;
        # retain it in the record without interpreting it as a filesystem path.
        filename = "replay" if args.command == "replay" else config.preset
        write_json(args.output / f"{filename}.json", asdict(result))
        print(f"  done in {result.metadata['wall_seconds']:.2f}s", flush=True)
    optimization = rank_results(results, args.target) if args.command == "optimize" else None
    provenance = dict(graph.metadata)
    provenance.update({"retained_neurons": len(graph.ids), "directed_pairs": graph.connectivity.nnz,
                       "python": platform.python_version(), "platform": platform.platform(),
                       "created_utc": datetime.now(timezone.utc).isoformat(),
                       "wall_seconds": time.perf_counter() - started,
                       "software": software,
                       "model": "Connectome-constrained rate model with idealized neuromodulation; not a validated brain emulation"})
    if args.command == "replay":
        provenance["replay"] = {"configuration_file": str(args.config.resolve()),
                                "configuration_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                                "duration_override": args.duration,
                                "graph_loading": "Reloaded from requested data directory or explicit synthetic demo; saved configuration is not a graph snapshot",
                                "requested_data_dir": str(args.data_dir.resolve()) if not args.demo else None,
                                "max_neurons": args.max_neurons}
    payload = {"schema_version": 1, "graph": provenance, "results": [asdict(r) for r in results], "optimization": optimization}
    write_json(args.output / "results.json", payload)
    if optimization:
        write_json(args.output / "optimization.json", optimization)
        print(f"Best tested configuration: {optimization['winner']['preset']} (score {optimization['winner']['score']:.6g})")
    path = write_report(results, provenance, args.output / "report.html", optimization=optimization)
    print(f"Report: {path.resolve()}")


def parser():
    p = argparse.ArgumentParser(description="MaleCNS reward simulation. Outputs are neural/model proxies, not measured pleasure.")
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("download", help="Download and verify the official v1.0 tables")
    d.add_argument("--data-dir", type=Path, default=Path("data"))
    motor = sub.add_parser("motor", help="Record identified motor neurons and export leg-joint playback")
    motor.add_argument("--data-dir", type=Path, default=Path("data"))
    motor.add_argument("--config", type=Path, required=True, help="Saved configuration or optimization winner")
    motor.add_argument("--duration", type=float, default=20.0)
    motor.add_argument("--comparison-duration", type=float, default=2.0)
    motor.add_argument("--output", type=Path, default=Path("results/motor"))
    motor.add_argument("--visualizer-data", type=Path, default=Path("visualizer/motor-signals.js"))
    for name in ("run", "optimize", "inspect", "replay"):
        q = sub.add_parser(name)
        q.add_argument("--data-dir", type=Path, default=Path("data"))
        q.add_argument("--demo", action="store_true", help="Explicitly use synthetic data; never an automatic fallback")
        q.add_argument("--max-neurons", type=int, help="Optional annotated subset; omit for the full retained graph")
        if name != "replay":
            q.add_argument("--seed", type=int, default=7)
        if name == "inspect":
            continue
        q.add_argument("--output", type=Path, default=Path("results") / name)
        if name == "replay":
            q.add_argument("--config", type=Path, required=True,
                           help="Saved raw configuration, result JSON, or optimization JSON")
            q.add_argument("--duration", type=float,
                           help="Optional simulated seconds override; all other saved settings are preserved")
            continue
        q.add_argument("--duration", type=float, default=2.0, help="Simulated seconds")
        q.add_argument("--dt", type=float, default=0.01, help="Integration timestep in seconds")
        q.add_argument("--record-every", type=int, default=1)
        if name == "run":
            q.add_argument("--presets", default="all", help="all or comma-separated preset names")
        else:
            q.add_argument("--trials", type=int, default=12, help="1–54 finite-grid candidates")
            q.add_argument("--target", choices=list(OBJECTIVES), default="reward_proxy")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.command == "download":
            from .data import download_data
            print(json.dumps(download_data(args.data_dir), default=_json_default, indent=2))
        elif args.command == "inspect":
            graph = _graph(args)
            print(json.dumps(graph.metadata, default=_json_default, indent=2))
        elif args.command == "motor":
            from .motor import run_motor_experiment
            run_motor_experiment(args)
        else:
            _execute(args)
    except subprocess.CalledProcessError as error:
        p.exit(2, f"Error: External command failed (exit code {error.returncode}). Partial downloads are retained for retry.\n")
    except (ValueError, FileNotFoundError, OSError) as error:
        p.exit(2, f"Error: {error}\n")
