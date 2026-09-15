"""Pinned MaleCNS v1.0 data, integrity verification, and sparse graph import.

Anatomical weights remain unsigned synapse counts. A neurotransmitter prediction
is not a synaptic sign or receptor identity. Peptide expression is unavailable in
these tables, so the real-data NPF mask is deliberately empty.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
from scipy import sparse

from .types import Graph

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS = "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
# Size and MD5 were verified against Google Storage response headers, 2026-09-14.
SOURCES = {
    ANNOTATIONS: {"bytes": 14483314, "md5": "50a7718770c57220f160ba4f431ab89e", "generation": "1780494878811468"},
    NEUROTRANSMITTERS: {"bytes": 43282834, "md5": "3d842b12fe5c49eefade528d7dd24a1f", "generation": "1780894899156750"},
    WEIGHTS: {"bytes": 1051241946, "md5": "f30e9dcca25cfd021bf1e7b3d975599e", "generation": "1780494887545976"},
}
IMPORT_VERSION = 1
NT_NAMES = {"acetylcholine", "gaba", "glutamate", "dopamine", "serotonin", "octopamine", "histamine", "unknown"}


def _digests(path: Path) -> dict:
    md5 = hashlib.md5(usedforsecurity=False)
    sha = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            size += len(block)
            md5.update(block)
            sha.update(block)
    return {"bytes": size, "md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def _verified(path: Path, expected: dict) -> dict:
    result = _digests(path)
    if result["bytes"] != expected["bytes"] or result["md5"] != expected["md5"]:
        raise ValueError(f"Integrity check failed for {path.name}; preserve/remove the damaged file and download again")
    return result


def _atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def download_data(data_dir: str | Path) -> dict:
    """Resume interrupted transfers and verify publisher MD5 plus local SHA256.

    Uses curl's Range support and five retries; transfers below 1 KiB/s for
    120 seconds time out. Complete verified files require no network request.
    Incomplete transfers remain as .part files for resuming.
    """
    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": "MaleCNS v1.0", "license": "CC-BY-4.0",
        "attribution": "MaleCNS collaboration: FlyEM/HHMI Janelia, University of Cambridge, MRC LMB, Google Research",
        "source_page": "https://male-cns.janelia.org/download/",
        "verified_at": datetime.now(timezone.utc).isoformat(), "files": {},
    }
    for name, expected in SOURCES.items():
        path = directory / name
        url = BASE_URL + name
        if not path.exists():
            part = directory / (name + ".part")
            # A previous process may have finished the transfer before it could
            # atomically rename the file; avoid an unnecessary HTTP 416 retry.
            if not part.exists() or part.stat().st_size != expected["bytes"]:
                subprocess.run([
                    "curl", "--fail", "--location", "--show-error", "--retry", "5",
                    "--retry-all-errors", "--connect-timeout", "30",
                    "--speed-time", "120", "--speed-limit", "1024", "--continue-at", "-",
                    "--output", str(part), url + "?generation=" + expected["generation"],
                ], check=True)
            result = _verified(part, expected)
            part.replace(path)
        else:
            result = _verified(path, expected)
        manifest["files"][name] = {**result, "url": url, "generation": expected["generation"], "publisher_md5_verified": True}
        _atomic_json(directory / "manifest.json", manifest)
    return manifest


def _integer_ids(values, label: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError(f"{label} must contain non-null integer IDs")
    if values.size and (np.any(values <= 0) or np.any(values > np.iinfo(np.uint32).max)):
        raise ValueError(f"{label} has IDs outside the published MaleCNS uint32 range")
    return values.astype(np.int64, copy=False)


class _SeenIds:
    """Exact unique endpoint count with a compact bitmap (not a Python set)."""
    def __init__(self):
        self.bits = np.zeros(0, dtype=np.uint8)

    def add(self, ids):
        if not ids.size:
            return
        needed = int(ids.max() // 8) + 1
        if needed > self.bits.size:
            self.bits.resize(max(needed, self.bits.size * 2), refcheck=False)
        np.bitwise_or.at(self.bits, ids >> 3, (1 << (ids & 7)).astype(np.uint8))

    def count(self):
        lut = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
        return sum(int(lut[self.bits[i:i + 8_000_000]].sum()) for i in range(0, self.bits.size, 8_000_000))

    def contains(self, ids):
        valid = (ids >> 3) < self.bits.size
        result = np.zeros(ids.size, dtype=bool)
        result[valid] = (self.bits[ids[valid] >> 3] & (1 << (ids[valid] & 7))) != 0
        return result


def _prepare_neurons(annotations: pd.DataFrame, nt: pd.DataFrame, max_neurons=None):
    required = {"bodyId", "status", "type", "instance"}
    if not required.issubset(annotations.columns) or not {"body", "consensus_nt"}.issubset(nt.columns):
        raise ValueError("Unexpected MaleCNS annotation/neurotransmitter schema")
    annotation_ids = _integer_ids(annotations["bodyId"].to_numpy(), "annotation bodyId")
    nt_ids = _integer_ids(nt["body"].to_numpy(), "neurotransmitter body")
    if pd.Index(annotation_ids).has_duplicates or pd.Index(nt_ids).has_duplicates:
        raise ValueError("Duplicate neuron IDs in annotations or neurotransmitter table")
    selected = annotations.loc[annotations["status"].eq("Traced")].sort_values("bodyId").copy()
    curated_count = len(selected)
    if max_neurons is not None:
        if isinstance(max_neurons, bool) or not isinstance(max_neurons, (int, np.integer)) or max_neurons < 1:
            raise ValueError("max_neurons must be a positive integer")
        selected = selected.iloc[:int(max_neurons)].copy()
    if selected.empty:
        raise ValueError("No curated neurons (status == Traced) in annotations")
    ids = selected["bodyId"].to_numpy(dtype=np.int64)
    nts = nt.set_index("body").reindex(ids)
    raw_nt = nts["consensus_nt"].fillna("unknown").astype(str).str.strip().str.lower()
    canonical = raw_nt.replace({"unclear": "unknown", "": "unknown"})
    unexpected = set(canonical) - NT_NAMES
    if unexpected:
        raise ValueError(f"Unrecognized consensus neurotransmitters: {sorted(unexpected)}")
    types = selected["type"].fillna("").astype(str).to_numpy(dtype=str)
    instances = selected["instance"].fillna("").astype(str)
    npf_candidates = instances.str.contains(r"(?:^|[^A-Za-z])NPF", regex=True).to_numpy(dtype=bool)
    mask = {
        "dopamine": canonical.to_numpy(dtype=str) == "dopamine",
        "npf": np.zeros(len(ids), dtype=bool),
        "pam": np.array([bool(re.match(r"^PAM\d", value)) for value in types]),
        "npf_name_candidate": npf_candidates,
    }
    metadata = {
        "dataset": "MaleCNS v1.0", "synthetic": False,
        "population_rule": "Retain annotation rows with status exactly Traced; retain missing types and unknown transmitters",
        "annotation_rows": len(annotations), "curated_population": curated_count,
        "retained_neurons": len(ids), "excluded_annotated_segments": len(annotations) - curated_count,
        "excluded_by_max_neurons": curated_count - len(ids),
        "annotation_status_counts": annotations["status"].fillna("missing").value_counts().to_dict(),
        "neurotransmitter_source": "consensus_nt; unclear/null canonicalized to unknown",
        "neurotransmitter_counts": canonical.value_counts().to_dict(),
        "missing_neurotransmitter_rows": int((~pd.Index(ids).isin(nt_ids)).sum()),
        "missing_cell_types": int((types == "").sum()),
        "npf_available": False,
        "npf_reason": "No peptide-expression annotations in these three official files; NPF-like names are candidates only",
        "npf_name_candidate_ids": ids[npf_candidates].tolist(),
        "pam_warning": "PAM label is anatomical; the mask is not a validated appetitive-only population",
        "dopamine_neurons": int(mask["dopamine"].sum()),
    }
    return ids, canonical.to_numpy(dtype=str), types, mask, metadata


def _compile_graph(annotations, nt, weights_path: Path, max_neurons=None) -> Graph:
    ids, neurotransmitters, types, masks, metadata = _prepare_neurons(annotations, nt, max_neurons)
    index = pd.Index(ids)
    rows, cols, counts = [], [], []
    seen = _SeenIds()
    edge_rows = excluded_edges = crossing_edges = retained_weight = total_weight = 0
    with pa.memory_map(str(weights_path), "r") as stream:
        reader = ipc.open_file(stream)
        if set(reader.schema.names) != {"body_pre", "body_post", "weight"}:
            raise ValueError("Unexpected connectivity schema; expected body_pre, body_post, weight")
        for name in reader.schema.names:
            if not pa.types.is_integer(reader.schema.field(name).type):
                raise ValueError(f"Connectivity {name} must have integer Arrow type")
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            if any(batch.column(name).null_count for name in batch.schema.names):
                raise ValueError("Null values in connectivity")
            pre = _integer_ids(batch.column("body_pre").to_numpy(), "body_pre")
            post = _integer_ids(batch.column("body_post").to_numpy(), "body_post")
            weight = batch.column("weight").to_numpy()
            if np.any(weight < 0):
                raise ValueError("Negative synapse counts in connectivity")
            # Uint32 per edge bounds make int64 aggregation safe for this release.
            if np.any(weight > np.iinfo(np.uint32).max):
                raise ValueError("Implausibly large synapse count")
            seen.add(pre)
            seen.add(post)
            source = index.get_indexer(pre)
            target = index.get_indexer(post)
            keep = (source >= 0) & (target >= 0)
            edge_rows += len(weight)
            excluded_edges += int((~keep).sum())
            crossing_edges += int(((source >= 0) ^ (target >= 0)).sum())
            total_weight += int(weight.sum(dtype=np.int64))
            retained_weight += int(weight[keep].sum(dtype=np.int64))
            if keep.any():
                rows.append(target[keep].astype(np.int32))
                cols.append(source[keep].astype(np.int32))
                counts.append(weight[keep].astype(np.int64))
    if not edge_rows:
        raise ValueError("Connectivity table is empty")
    if rows:
        row = np.concatenate(rows)
        col = np.concatenate(cols)
        data = np.concatenate(counts)
        del rows, cols, counts
        matrix = sparse.coo_matrix((data, (row, col)), shape=(len(ids), len(ids))).tocsr()
        before_zero_removal = matrix.nnz
        duplicate_edges = len(data) - before_zero_removal
        matrix.eliminate_zeros()
    else:
        matrix = sparse.csr_matrix((len(ids), len(ids)), dtype=np.int64)
        duplicate_edges = 0
    if matrix.data.size and np.any(matrix.data < 0):
        raise ValueError("Synapse-count overflow after duplicate-edge aggregation")
    observed_segments = seen.count()
    observed_curated = int(seen.contains(ids).sum())
    metadata.update({
        "connectivity_orientation": "row=target, column=source",
        "connectivity_values": "nonnegative anatomical synapse counts; no physiological signs applied",
        "bulk_edge_rows": edge_rows, "excluded_edge_rows": excluded_edges,
        "boundary_crossing_edge_rows": crossing_edges,
        "bulk_synapse_count": total_weight, "retained_synapse_count": retained_weight,
        "bulk_distinct_segments": observed_segments,
        "excluded_bulk_segments": observed_segments - observed_curated,
        "retained_neurons_without_bulk_edges": len(ids) - observed_curated,
        "retained_connection_pairs": matrix.nnz,
        "duplicate_retained_edge_rows_aggregated": duplicate_edges,
        "import_version": IMPORT_VERSION,
    })
    return Graph(ids, matrix, neurotransmitters, types, masks, metadata)


def load_graph(data_dir: str | Path, max_neurons=None) -> Graph:
    """Load the pinned official files, recheck hashes, and cache sparse imports."""
    directory = Path(data_dir)
    provenance = {}
    for name, expected in SOURCES.items():
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}; run the download command first")
        provenance[name] = _verified(path, expected)
    fingerprint = hashlib.sha256(json.dumps({"version": IMPORT_VERSION, "sources": provenance, "max_neurons": max_neurons}, sort_keys=True).encode()).hexdigest()[:20]
    cache = directory / "compiled" / fingerprint
    info_path = cache / "metadata.json"
    if info_path.exists():
        metadata = json.loads(info_path.read_text())
        with np.load(cache / "neurons.npz", allow_pickle=False) as arrays:
            ids = arrays["ids"]
            neurotransmitters = arrays["neurotransmitters"]
            types = arrays["cell_types"]
            masks = {name.removeprefix("mask_"): arrays[name] for name in arrays.files if name.startswith("mask_")}
        matrix = sparse.load_npz(cache / "connectivity.npz")
        if matrix.shape != (len(ids), len(ids)) or np.any(matrix.data < 0) or not np.isfinite(matrix.data).all():
            raise ValueError("Malformed compiled graph cache; remove the compiled cache directory")
        _integer_ids(ids, "cached bodyId")
        if pd.Index(ids).has_duplicates or any(len(v) != len(ids) for v in (neurotransmitters, types, *masks.values())):
            raise ValueError("Malformed cached neuron mapping; remove the compiled cache directory")
        if set(neurotransmitters) - NT_NAMES or not {"dopamine", "npf"}.issubset(masks):
            raise ValueError("Malformed cached neurotransmitters or masks")
        metadata["loaded_from_cache"] = True
        return Graph(ids, matrix, neurotransmitters, types, masks, metadata)
    annotations = feather.read_feather(directory / ANNOTATIONS)
    nt = feather.read_feather(directory / NEUROTRANSMITTERS)
    graph = _compile_graph(annotations, nt, directory / WEIGHTS, max_neurons)
    graph.metadata.update({"source_hashes": provenance, "source_page": "https://male-cns.janelia.org/download/", "cache_key": fingerprint, "loaded_from_cache": False})
    cache.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(cache / "connectivity.npz", graph.connectivity)
    np.savez_compressed(cache / "neurons.npz", ids=graph.ids, neurotransmitters=graph.neurotransmitters, cell_types=graph.cell_types, **{"mask_" + name: value for name, value in graph.masks.items()})
    _atomic_json(info_path, graph.metadata)
    return graph


def make_demo_graph(seed: int = 7) -> Graph:
    """Small synthetic graph for tests and software demonstrations only."""
    rng = np.random.default_rng(seed)
    n = 48
    weights = rng.poisson(2, (n, n)) * (rng.random((n, n)) < 0.14)
    np.fill_diagonal(weights, 0)
    nt = np.array(["acetylcholine"] * n, dtype="U16")
    nt[:5] = "dopamine"
    nt[5:13] = "gaba"
    nt[13:17] = "glutamate"
    masks = {"dopamine": nt == "dopamine", "npf": np.arange(n) >= n - 3, "pam": np.zeros(n, dtype=bool)}
    return Graph(np.arange(1, n + 1, dtype=np.int64), sparse.csr_matrix(weights), nt, np.array([f"synthetic_{i}" for i in range(n)]), masks, {"dataset": "SYNTHETIC TEST GRAPH", "synthetic": True, "seed": seed, "npf_available": True, "warning": "Fabricated cells and weights for software tests only; not MaleCNS or a biological reward model"})
