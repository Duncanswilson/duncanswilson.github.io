"""Auditable motor output identities from the official MaleCNS annotations.

This module assigns no activity, force, gait, phase, or joint-angle trajectories.
Only six explicitly named tibia/trochanter motor types receive action channels.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ANNOTATIONS, SOURCES, _integer_ids, _verified
from .types import Graph

MAPPING_VERSION = 1
LEG_ORDER = ("LF", "LM", "LH", "RF", "RM", "RH")
TYPE_ACTIONS = {
    "Tr flexor MN": ("trochanter", "flexor"),
    "Acc. tr flexor MN": ("trochanter", "flexor"),
    "Tr extensor MN": ("trochanter", "extensor"),
    "Ti flexor MN": ("tibia", "flexor"),
    "Acc. ti flexor MN": ("tibia", "flexor"),
    "Ti extensor MN": ("tibia", "extensor"),
}
LEG_SEGMENTS = {
    "fl": ("F", "T1", {"ProLN", "ProAN", "VProN"}),
    "ml": ("M", "T2", {"MesoLN"}),
    "hl": ("H", "T3", {"MetaLN"}),
}
ANNOTATION_FIELDS = (
    "bodyId", "status", "superclass", "subclass", "type", "instance",
    "somaSide", "somaNeuromere", "exitNerve", "rootSide", "mancType",
    "mancBodyid", "matchingNotes",
)
PRIMARY_SOURCES = [
    {
        "id": "malecns_annotations",
        "url": "https://male-cns.janelia.org/download/",
        "title": "MaleCNS v1.0 official annotations",
        "supports": "Exact current neuron IDs, Traced status, motor superclass, named types, leg subclasses, soma side, neuromere and exit nerve",
    },
    {
        "id": "malecns_typing",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12636603/",
        "title": "Sexual dimorphism in the complete connectome of the Drosophila male central nervous system",
        "supports": "Instance labels combine type and somaSide for these neurons; their suffix is not independent nerve-side evidence",
    },
    {
        "id": "manc_motor_targets",
        "url": "https://elifesciences.org/articles/96084",
        "title": "Cheong et al., Organization of circuits linking descending input to motor output in MANC (2026)",
        "supports": "Figures 5 and 7: motor type and nerve nomenclature, fl/ml/hl categories, muscle-target matching and serial homology; Figure 13: flexor and accessory flexor have similar basic joint action",
        "limits": "MANC is a different specimen; targets were assigned by morphology and homology, not peripheral muscles observed in MaleCNS",
    },
    {
        "id": "ipsilateral_leg_anatomy",
        "url": "https://elifesciences.org/articles/42692",
        "title": "Venkatasubramanian et al., Stereotyped terminal axon branching of leg motor neurons (2019)",
        "supports": "Figure 1 and results describe segment-specific ipsilateral leg-MN projection: T1 foreleg, T2 middle leg, T3 hind leg",
    },
    {
        "id": "tibia_motor_physiology",
        "url": "https://elifesciences.org/articles/56754",
        "title": "Azevedo et al., A size principle for recruitment of Drosophila leg motor neurons (2020)",
        "supports": "Identified tibia flexor motor neurons cause flexion with different force per spike; a uniform rate-to-force conversion is not measured biology",
    },
    {
        "id": "fanc_motor_atlas",
        "url": "https://www.nature.com/articles/s41586-024-07389-x",
        "title": "Azevedo et al., Connectomic reconstruction of a female Drosophila ventral nerve cord (2024)",
        "supports": "Motor target atlas links CNS reconstructions to leg muscles using genetic lines and X-ray nanotomography",
    },
]


def _clean(value):
    if value is None or pd.isna(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def classify_motor_annotation(row: dict) -> tuple[dict | None, str | None]:
    """Strict identity gate; ambiguous anatomy remains unmapped with a reason."""
    if row.get("status") != "Traced":
        return None, "not_curated_traced"
    if row.get("superclass") != "vnc_motor":
        return None, "outside_supported_vnc_leg_motor_class"
    cell_type = row.get("type")
    if cell_type not in TYPE_ACTIONS:
        return None, "type_has_no_supported_tibia_or_trochanter_action"
    subclass = row.get("subclass")
    if subclass not in LEG_SEGMENTS:
        return None, "missing_or_conflicting_leg_subclass"
    segment, neuromere, nerves = LEG_SEGMENTS[subclass]
    side = row.get("somaSide")
    if side not in ("L", "R"):
        return None, "missing_or_ambiguous_soma_side"
    # Instance is derived from soma side: a consistency check, not a second
    # anatomical determination of output side.
    if row.get("instance") != f"{cell_type}_{side}":
        return None, "instance_and_soma_side_conflict"
    if row.get("somaNeuromere") != neuromere:
        return None, "leg_subclass_and_neuromere_conflict"
    if row.get("exitNerve") not in nerves:
        return None, "exit_nerve_not_supported_for_named_leg_output"
    if row.get("mancType") != cell_type:
        return None, "missing_or_conflicting_manc_type_cross_reference"
    joint, action = TYPE_ACTIONS[cell_type]
    leg = side + segment
    return {
        "leg": leg, "joint": joint, "action": action,
        "channel": f"{leg}_{joint}_{action}",
        "joint_anatomy": "coxa-trochanter" if joint == "trochanter" else "femur-tibia",
        "identity_evidence": "explicit published motor type with concordant subclass, neuromere, nerve and MANC type",
        "side_evidence": "somaSide plus published ipsilateral leg-MN projection; no independent nerve-side field in this Feather table",
        "mechanical_calibration": "unavailable",
    }, None


def _build_from_annotations(annotations: pd.DataFrame, graph: Graph, provenance: dict) -> dict:
    missing = set(ANNOTATION_FIELDS) - set(annotations.columns)
    if missing:
        raise ValueError(f"Motor mapping annotation fields missing: {sorted(missing)}")
    source_ids = _integer_ids(annotations["bodyId"].to_numpy(), "annotation bodyId")
    graph_ids = _integer_ids(graph.ids, "graph IDs")
    if pd.Index(source_ids).has_duplicates or pd.Index(graph_ids).has_duplicates:
        raise ValueError("Motor mapping requires unique annotation and graph IDs")
    if graph.connectivity.shape != (len(graph_ids), len(graph_ids)):
        raise ValueError("Graph dimensions do not match neuron IDs")
    candidates = annotations.loc[
        annotations["status"].eq("Traced") & annotations["superclass"].isin(("vnc_motor", "cb_motor"))
    ].sort_values("bodyId")
    index = pd.Index(graph_ids)
    indices = index.get_indexer(candidates["bodyId"])
    excluded_from_graph = candidates.loc[indices < 0, "bodyId"].astype(int).tolist()
    retained = candidates.loc[indices >= 0]
    retained_indices = indices[indices >= 0]
    channels = {}
    for leg in LEG_ORDER:
        for joint in ("trochanter", "tibia"):
            for action in ("flexor", "extensor"):
                name = f"{leg}_{joint}_{action}"
                channels[name] = {
                    "name": name, "leg": leg, "joint": joint, "action": action,
                    "joint_anatomy": "coxa-trochanter" if joint == "trochanter" else "femur-tibia",
                    "body_ids": [], "graph_indices": [], "motor_types": [],
                    "evidence_status": "named_motor_target_with_ipsilateral_side_inference",
                }
    records, unmapped = [], []
    for values, graph_index in zip(retained[list(ANNOTATION_FIELDS)].to_dict("records"), retained_indices):
        row = {key: _clean(value) for key, value in values.items()}
        body_id = int(row["bodyId"])
        if row["mancBodyid"] is not None:
            if not float(row["mancBodyid"]).is_integer():
                raise ValueError(f"Noninteger MANC cross-reference for {body_id}")
            row["mancBodyid"] = int(row["mancBodyid"])
        mapped, reason = classify_motor_annotation(row)
        record = {
            "body_id": body_id, "graph_index": int(graph_index),
            "annotations": row, "mapped": mapped is not None,
            "mapping": mapped, "unmapped_reason": reason,
        }
        records.append(record)
        if mapped is None:
            unmapped.append(record)
        else:
            channel = channels[mapped["channel"]]
            channel["body_ids"].append(body_id)
            channel["graph_indices"].append(int(graph_index))
            if row["type"] not in channel["motor_types"]:
                channel["motor_types"].append(row["type"])
    for channel in channels.values():
        channel["count"] = len(channel["body_ids"])
        channel["motor_types"].sort()
        channel["available"] = bool(channel["body_ids"])
    mapped_ids = sorted(body for channel in channels.values() for body in channel["body_ids"])
    if len(mapped_ids) != len(set(mapped_ids)):
        raise ValueError("A motor neuron was assigned to multiple action channels")
    return {
        "schema_version": 1, "mapping_version": MAPPING_VERSION,
        "dataset": "MaleCNS v1.0", "source_annotation": provenance,
        "graph_cache_key": graph.metadata.get("cache_key"),
        "leg_order": list(LEG_ORDER),
        "motor_neuron_ids": [r["body_id"] for r in records],
        "motor_graph_indices": [r["graph_index"] for r in records],
        "mapped_motor_neuron_ids": mapped_ids,
        "channels": list(channels.values()), "motor_neurons": records,
        "unmapped": unmapped, "sources": PRIMARY_SOURCES,
        "summary": {
            "annotated_traced_motor_neurons": len(candidates),
            "retained_motor_neurons": len(records), "mapped_motor_neurons": len(mapped_ids),
            "unmapped_motor_neurons": len(unmapped), "motor_neurons_excluded_from_graph": len(excluded_from_graph),
            "available_channels": sum(channel["available"] for channel in channels.values()),
            "retained_by_superclass": retained["superclass"].value_counts().to_dict(),
            "retained_by_subclass": retained["subclass"].value_counts().to_dict(),
            "unmapped_reasons": pd.Series([r["unmapped_reason"] for r in unmapped], dtype=str).value_counts().to_dict(),
        },
        "excluded_motor_neuron_ids": excluded_from_graph,
        "identity_rules": [
            "Record only actual graph IDs annotated Traced and vnc_motor or cb_motor.",
            "Map only six exact Ti/Tr flexor/extensor type strings; accessorily named flexors enter the corresponding flexion channel.",
            "Require type/mancType, instance/somaSide, subclass/neuromere and exit nerve consistency.",
            "Leg side is a documented ipsilateral-projection inference from somaSide; instance is derived from the same field.",
            "No activity, arbitrary ID, neurotransmitter sign, or graph position assigns an action.",
        ],
        "limits": [
            "Motor rates are simulated, not measured physiology or observed behavior.",
            "The annotation table does not reconstruct muscle attachments, force, moment arms, joint limits or load feedback.",
            "Pooling accessory and main flexors by mean rate is a downstream model choice, not a calibrated force sum.",
            "Motor target names include cross-specimen morphology and serial-homology assignments; individual confidence values are absent here.",
            "Unmapped means absent from this strict two-joint adapter, not necessarily that the muscle target is unknown to biology.",
            "Wing, neck, head, abdomen, tarsal and coxal-rotation outputs are recorded but receive no inferred animation action.",
        ],
    }


def build_motor_mapping(data_dir: str | Path, graph: Graph) -> dict:
    """Return a JSON-ready mapping and recording inventory for an actual graph.

    `motor_neuron_ids` can be passed directly to run_simulation's
    record_neuron_ids parameter; each channel lists a subset of those exact IDs.
    Every call verifies the official annotation file against its pinned checksum
    and the graph's source provenance. No file is written by this function.
    """
    if graph.metadata.get("synthetic") is not False:
        raise ValueError("Motor mapping requires the actual MaleCNS graph")
    path = Path(data_dir) / ANNOTATIONS
    provenance = _verified(path, SOURCES[ANNOTATIONS])
    expected = graph.metadata.get("source_hashes", {}).get(ANNOTATIONS, {}).get("sha256")
    if provenance["sha256"] != expected:
        raise ValueError("Motor annotations do not match the graph's source provenance")
    provenance.update({"file": ANNOTATIONS, "url": "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/" + ANNOTATIONS})
    return _build_from_annotations(pd.read_feather(path), graph, provenance)


def write_motor_mapping(path: str | Path, mapping: dict) -> None:
    """Write a reviewable JSON manifest atomically."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(mapping, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(output)
