import hashlib
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pytest

from flyreward.data import _compile_graph, _prepare_neurons, _verified, download_data, make_demo_graph


def fixture_tables():
    annotations = pd.DataFrame({
        "bodyId": [12, 11, 13, 14], "status": ["Traced", "Traced", "Glia", "Traced"],
        "type": ["PAM04", "", "glia", "NPFL1-I"],
        "instance": ["PAM04_R", "unknown_L", "glia", "NPFL1-I_L"],
    })
    nt = pd.DataFrame({"body": [11, 12, 13], "consensus_nt": ["acetylcholine", "dopamine", "unclear"]})
    return annotations, nt


def write_weights(tmp_path, pre, post, weight):
    path = tmp_path / "edges.feather"
    feather.write_feather(pa.table({"body_pre": pre, "body_post": post, "weight": weight}), path, chunksize=2)
    return path


def test_directed_counts_filter_duplicates_and_missing_nt(tmp_path):
    a, nt = fixture_tables()
    path = write_weights(tmp_path, [11, 11, 12, 13, 99], [12, 12, 11, 11, 100], [3, 4, 2, 8, 1])
    graph = _compile_graph(a, nt, path)
    assert graph.ids.tolist() == [11, 12, 14]
    assert graph.connectivity.toarray().tolist() == [[0, 2, 0], [7, 0, 0], [0, 0, 0]]
    assert graph.neurotransmitters.tolist() == ["acetylcholine", "dopamine", "unknown"]
    assert graph.metadata["missing_neurotransmitter_rows"] == 1
    assert graph.metadata["excluded_annotated_segments"] == 1
    assert graph.metadata["bulk_distinct_segments"] == 5
    assert graph.metadata["excluded_bulk_segments"] == 3
    assert graph.metadata["boundary_crossing_edge_rows"] == 1
    assert graph.metadata["duplicate_retained_edge_rows_aggregated"] == 1
    assert graph.metadata["retained_neurons_without_bulk_edges"] == 1
    assert graph.masks["dopamine"].tolist() == [False, True, False]
    assert graph.masks["pam"].tolist() == [False, True, False]
    assert not graph.masks["npf"].any()
    assert graph.metadata["npf_name_candidate_ids"] == [14]


@pytest.mark.parametrize("bad", [[1, -1], [1.0, 2.5], [1, None]])
def test_bad_weights_fail_closed(tmp_path, bad):
    a, nt = fixture_tables()
    path = write_weights(tmp_path, [11, 12], [12, 11], bad)
    with pytest.raises(ValueError):
        _compile_graph(a, nt, path)


@pytest.mark.parametrize("which", ["annotations", "nt"])
def test_duplicate_neuron_ids_fail_closed(which):
    a, nt = fixture_tables()
    if which == "annotations":
        a = pd.concat([a, a.iloc[:1]], ignore_index=True)
    else:
        nt = pd.concat([nt, nt.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        _prepare_neurons(a, nt)


def test_null_and_fractional_ids_fail_closed():
    a, nt = fixture_tables()
    a["bodyId"] = [11.5, 12, 13, 14]
    with pytest.raises(ValueError, match="integer"):
        _prepare_neurons(a, nt)


def test_unknown_nt_label_fails_closed():
    a, nt = fixture_tables()
    nt.loc[0, "consensus_nt"] = "made-up-pleasure-chemical"
    with pytest.raises(ValueError, match="Unrecognized"):
        _prepare_neurons(a, nt)


def test_max_neurons_has_explicit_order_and_exclusion(tmp_path):
    a, nt = fixture_tables()
    path = write_weights(tmp_path, [11], [12], [3])
    graph = _compile_graph(a, nt, path, max_neurons=1)
    assert graph.ids.tolist() == [11]
    assert graph.metadata["excluded_by_max_neurons"] == 2
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            _prepare_neurons(a, nt, bad)


def test_checksum_rejects_tampering(tmp_path):
    path = tmp_path / "data.bin"
    original = b"official fixture"
    path.write_bytes(original)
    expected = {"bytes": len(original), "md5": hashlib.md5(original).hexdigest()}
    assert _verified(path, expected)["sha256"] == hashlib.sha256(original).hexdigest()
    path.write_bytes(b"untrusted fixture")
    with pytest.raises(ValueError, match="Integrity"):
        _verified(path, expected)


def test_downloader_resumes_part_and_reuses_verified_complete_file(tmp_path):
    payload = b"test fixture payload"
    name = "fixture.feather"
    source = {name: {"bytes": len(payload), "md5": hashlib.md5(payload).hexdigest(), "generation": "123"}}
    (tmp_path / (name + ".part")).write_bytes(payload[:5])

    def fake_download(args, check):
        assert check
        assert args[args.index("--continue-at") + 1] == "-"
        assert args[args.index("--speed-time") + 1] == "120"
        assert args[args.index("--speed-limit") + 1] == "1024"
        Path(args[args.index("--output") + 1]).write_bytes(payload)

    with patch("flyreward.data.SOURCES", source), patch("flyreward.data.subprocess.run", side_effect=fake_download) as call:
        first = download_data(tmp_path)
        second = download_data(tmp_path)
    assert call.call_count == 1
    assert not (tmp_path / (name + ".part")).exists()
    assert first["files"][name]["publisher_md5_verified"]
    assert second["files"][name]["sha256"] == hashlib.sha256(payload).hexdigest()


def test_completed_part_is_verified_and_committed_without_network(tmp_path):
    payload = b"finished but not renamed"
    name = "fixture.feather"
    source = {name: {"bytes": len(payload), "md5": hashlib.md5(payload).hexdigest(), "generation": "123"}}
    (tmp_path / (name + ".part")).write_bytes(payload)
    with patch("flyreward.data.SOURCES", source), patch("flyreward.data.subprocess.run") as call:
        download_data(tmp_path)
    call.assert_not_called()
    assert (tmp_path / name).read_bytes() == payload


def test_exhausted_download_timeout_preserves_partial_file_for_restart(tmp_path):
    payload = b"partial resumable fixture"
    name = "fixture.feather"
    source = {name: {"bytes": len(payload), "md5": hashlib.md5(payload).hexdigest(), "generation": "123"}}
    part = tmp_path / (name + ".part")

    def stalled_download(args, check):
        part.write_bytes(payload[:7])
        # curl returns 28 when its low-speed timeout exhausts the retries.
        raise subprocess.CalledProcessError(28, args)

    def resumed_download(args, check):
        assert part.read_bytes() == payload[:7]
        assert args[args.index("--continue-at") + 1] == "-"
        part.write_bytes(payload)

    with patch("flyreward.data.SOURCES", source):
        with patch("flyreward.data.subprocess.run", side_effect=stalled_download):
            with pytest.raises(subprocess.CalledProcessError):
                download_data(tmp_path)
        assert part.read_bytes() == payload[:7]
        assert not (tmp_path / name).exists()
        with patch("flyreward.data.subprocess.run", side_effect=resumed_download):
            download_data(tmp_path)
    assert not part.exists()
    assert (tmp_path / name).read_bytes() == payload


def test_demo_is_reproducible_and_unmistakably_synthetic():
    one, two = make_demo_graph(), make_demo_graph()
    assert one.metadata["synthetic"] is True
    assert "SYNTHETIC" in one.metadata["dataset"]
    assert (one.connectivity != two.connectivity).nnz == 0
