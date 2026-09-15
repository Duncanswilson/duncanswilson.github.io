#!/usr/bin/env python3
"""Rebuild the CPG asset from a pinned author checkout and verified MaleCNS cache.

Example (paths are inputs, not downloaded or modified by this script):
  python scripts/build_cpg_asset.py --upstream /path/Pugliese_2026 \
    --graph-cache /path/compiled/cache-key --manifest /path/data/manifest.json \
    --output flyreward/assets/cpg_front_v1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

UPSTREAM_COMMIT = "faee4b06869855ae0164cbf217fb6ec28ef3521b"
TABLE = "data/imac t1 connectome data/wTable_20260210_vncRoisOnly.csv"
WEIGHTS = "data/imac t1 connectome data/W_20260210_vncRoisOnly.csv"


def build(upstream: Path, graph_cache: Path, manifest: Path) -> dict:
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"Expected upstream revision {UPSTREAM_COMMIT}, got {commit}")
    table = pd.read_csv(upstream / TABLE).set_index("bodyId")
    author = pd.read_csv(upstream / WEIGHTS, index_col=0)
    author.columns = author.columns.astype(np.int64)
    if not np.array_equal(author.index.to_numpy(), table.index.to_numpy()) or not np.array_equal(author.columns, table.index):
        raise ValueError("Upstream matrix and neuron table ID ordering disagree")
    reference = author.to_numpy()
    with np.load(graph_cache / "neurons.npz", allow_pickle=False) as archive:
        graph_ids = archive["ids"].copy()
        graph_nt = archive["neurotransmitters"].copy()
    graph = load_npz(graph_cache / "connectivity.npz")
    lookup = {int(body): i for i, body in enumerate(graph_ids)}
    keep = np.asarray([int(body) in lookup for body in table.index])
    ids = table.index.to_numpy()[keep]
    indices = np.asarray([lookup[int(body)] for body in ids])
    raw = graph[indices][:, indices].toarray().T
    ref = reference[np.ix_(keep, keep)]
    author_nonzero = ref != 0
    signed = np.where(author_nonzero & (raw >= 5), raw * np.sign(ref), 0)
    source, target = np.nonzero(signed)
    selected = table.loc[ids]
    neurons = []
    nt_sign = {"acetylcholine": 1, "gaba": -1, "glutamate": -1}
    for i, (body, row) in enumerate(selected.iterrows()):
        signs = np.unique(np.sign(signed[i][signed[i] != 0]))
        if len(signs) > 1:
            raise ValueError(f"Inconsistent presynaptic sign for {body}")
        sign = int(signs[0]) if len(signs) else 0
        if sign and nt_sign.get(str(graph_nt[indices[i]]), 0) != sign:
            raise ValueError(f"Presynaptic sign conflicts with current annotation for {body}")
        side = row.rootSide if pd.notna(row.rootSide) else row.somaSide
        neurons.append({"id": int(body), "type": str(row["type"]) if pd.notna(row["type"]) else "",
                        "size": int(row["size"]), "nt": str(row.consensusNt) if pd.notna(row.consensusNt) else "unclear",
                        "sign": sign, "motor": row["class"] == "motor neuron",
                        "side": str(side) if pd.notna(side) else ""})
    inputs = [TABLE, WEIGHTS]
    return {
        "format": "flyreward.cpg-front", "version": 1,
        "reference": {
            "paper": "https://doi.org/10.1101/2025.09.12.675944", "code": "https://github.com/smpuglie/Pugliese_2026",
            "commit": UPSTREAM_COMMIT, "code_license": "MIT as stated in upstream README; equation reimplemented independently",
            "data_license": "CC-BY-4.0",
            "attribution": "Pugliese et al.; MaleCNS collaboration: FlyEM/HHMI Janelia, University of Cambridge, MRC LMB, Google Research",
            "files": {p: hashlib.sha256((upstream / p).read_bytes()).hexdigest() for p in inputs},
        },
        "adaptation": {
            "description": "Published MaleCNS T1 neuron selection, neuron sizes, presynaptic sign and nonzero VNC edge mask; retained edges use exact current MaleCNS v1.0 full-CNS synapse counts >=5. This is an adaptation, not an exact reproduction of the published network or VNC-only counts. No edges absent from the published mask are added.",
            "excluded_body_ids": [int(body) for body in table.index[~keep]],
            "excluded_reason": "Untraced sensory body absent from the imported Traced population; silent in reproduced published baseline.",
            "original_nodes": len(table), "original_edges": int(np.count_nonzero(reference)),
            "current_edges": int(len(source)),
            "mapped_original_edges_exact": int(np.sum(author_nonzero & (raw == abs(ref)))),
            "mapped_original_edges_smaller_than_current": int(np.sum(author_nonzero & (raw > abs(ref)))),
            "mapped_original_edges_larger_than_current": int(np.sum(author_nonzero & (raw < abs(ref)))),
            "current_weights_sha256": json.loads(manifest.read_text())["files"]["connectome-weights-male-cns-v1.0-minconf-0.5.feather"]["sha256"],
            "scope": "Both front-leg neuropils only; no middle/hind CPG and no claim of coordinated gait.",
        },
        "parameters": {
            "size_normalization_median": float(table["size"].median()), "tau_seconds": .02, "gain": 1.,
            "threshold": 7.5, "rate_cap_hz": 200., "synapse_multiplier": .03, "stimulus": 400.,
            "stimulus_onset_seconds": .02, "stimulated_body_ids": [10045, 10056],
            "default_internal_dt_seconds": .001, "solver": "classical fourth-order Runge-Kutta",
            "parameter_choice": "Deterministic published distribution means; no sampled or fitted per-cell physiology.",
        },
        "neuron_columns": None, "neurons": neurons,
        "edge_columns": ["source_index", "target_index", "synapse_count"],
        "edges": [[int(i), int(j), int(abs(signed[i, j]))] for i, j in zip(source, target)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["upstream", "graph-cache", "manifest", "output"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    value = build(args.upstream, args.graph_cache, args.manifest)
    encoded = (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(json.dumps({"sha256": hashlib.sha256(encoded).hexdigest(), "neurons": len(value["neurons"]),
                      "edges": len(value["edges"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
