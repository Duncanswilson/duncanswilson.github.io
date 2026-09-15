"""Motor identity gates, using explicitly fabricated fixtures for software tests."""
import copy

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from flyreward.motor_mapping import (
    ANNOTATION_FIELDS, TYPE_ACTIONS, _build_from_annotations,
    build_motor_mapping, classify_motor_annotation,
)
from flyreward.types import Graph


def annotation(**changes):
    row = {name: None for name in ANNOTATION_FIELDS}
    row.update(bodyId=10, status="Traced", superclass="vnc_motor", subclass="fl",
               type="Ti extensor MN", instance="Ti extensor MN_L", somaSide="L",
               somaNeuromere="T1", exitNerve="ProLN", mancType="Ti extensor MN")
    row.update(changes)
    return row


@pytest.mark.parametrize("cell_type", TYPE_ACTIONS)
def test_only_exact_named_action_is_assigned(cell_type):
    row = annotation(type=cell_type, mancType=cell_type, instance=cell_type + "_L")
    result, reason = classify_motor_annotation(row)
    joint, action = TYPE_ACTIONS[cell_type]
    assert reason is None
    assert result["leg"] == "LF"
    assert (result["joint"], result["action"]) == (joint, action)
    assert "ipsilateral" in result["side_evidence"]


@pytest.mark.parametrize("change,reason", [
    ({"type": "MNfl999"}, "type_has_no_supported_tibia_or_trochanter_action"),
    ({"type": "ti extensor MN"}, "type_has_no_supported_tibia_or_trochanter_action"),
    ({"somaSide": None}, "missing_or_ambiguous_soma_side"),
    ({"somaSide": "R"}, "instance_and_soma_side_conflict"),
    ({"somaNeuromere": "T3"}, "leg_subclass_and_neuromere_conflict"),
    ({"exitNerve": "MetaLN"}, "exit_nerve_not_supported_for_named_leg_output"),
    ({"mancType": "Ti flexor MN"}, "missing_or_conflicting_manc_type_cross_reference"),
    ({"mancType": None}, "missing_or_conflicting_manc_type_cross_reference"),
    ({"subclass": "wm"}, "missing_or_conflicting_leg_subclass"),
    ({"superclass": "vnc_intrinsic"}, "outside_supported_vnc_leg_motor_class"),
    ({"status": "Orphan"}, "not_curated_traced"),
])
def test_ambiguous_or_conflicting_anatomy_is_unmapped(change, reason):
    result, actual_reason = classify_motor_annotation(annotation(**change))
    assert result is None
    assert actual_reason == reason


def test_hind_right_side_and_segment_use_annotations_not_neuron_number():
    row = annotation(subclass="hl", somaSide="R", instance="Ti extensor MN_R",
                     somaNeuromere="T3", exitNerve="MetaLN", bodyId=123456)
    first, _ = classify_motor_annotation(row)
    row["bodyId"] = 987654
    second, _ = classify_motor_annotation(row)
    assert first == second
    assert first["channel"] == "RH_tibia_extensor"


def test_prothoracic_accessory_nerve_allowed_for_trochanter():
    row = annotation(type="Tr extensor MN", mancType="Tr extensor MN",
                     instance="Tr extensor MN_L", exitNerve="ProAN")
    assert classify_motor_annotation(row)[0]["channel"] == "LF_trochanter_extensor"


def fixture_graph(ids):
    n = len(ids)
    return Graph(np.array(ids, dtype=np.int64), csr_matrix((n,n)),
                 np.array(["unknown"]*n), np.array(["fixture"]*n),
                 metadata={"synthetic": True})


def test_all_motor_cells_recorded_but_unmapped_cells_never_receive_actions():
    rows = [annotation(), annotation(bodyId=20, superclass="cb_motor", type="MN1"),
            annotation(bodyId=30, type="unidentified"), annotation(bodyId=40)]
    graph = fixture_graph([30,10,20,99])
    manifest = _build_from_annotations(pd.DataFrame(rows), graph, {"fixture": True})
    assert manifest["motor_neuron_ids"] == [10,20,30]
    assert manifest["motor_graph_indices"] == [1,2,0]
    assert manifest["mapped_motor_neuron_ids"] == [10]
    assert manifest["excluded_motor_neuron_ids"] == [40]
    assert {row["body_id"] for row in manifest["unmapped"]} == {20,30}
    channel = next(c for c in manifest["channels"] if c["name"] == "LF_tibia_extensor")
    assert channel["body_ids"] == [10] and channel["graph_indices"] == [1]
    assert manifest["summary"]["available_channels"] == 1


def test_duplicate_source_ids_and_graph_ids_are_rejected():
    rows = pd.DataFrame([annotation(), annotation()])
    with pytest.raises(ValueError, match="unique"):
        _build_from_annotations(rows, fixture_graph([10]), {})
    with pytest.raises(ValueError, match="unique"):
        _build_from_annotations(rows.iloc[:1], fixture_graph([10,10]), {})


def test_missing_schema_is_rejected():
    rows = pd.DataFrame([annotation()]).drop(columns="exitNerve")
    with pytest.raises(ValueError, match="fields missing"):
        _build_from_annotations(rows, fixture_graph([10]), {})


def test_public_builder_rejects_synthetic_graph_before_reading_files(tmp_path):
    with pytest.raises(ValueError, match="actual MaleCNS"):
        build_motor_mapping(tmp_path, fixture_graph([10]))
