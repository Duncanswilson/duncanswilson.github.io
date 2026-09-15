import json
from dataclasses import asdict
import numpy as np
import pytest
from flyreward.cli import main, write_json
from flyreward.types import SimulationConfig


def test_demo_run_writes_portable_bundle(tmp_path):
    main(["run", "--demo", "--duration", "0.1", "--presets", "baseline,maximal_dopamine", "--output", str(tmp_path)])
    payload = json.loads((tmp_path / "results.json").read_text())
    assert len(payload["results"]) == 2
    assert (tmp_path / "report.html").exists()
    assert payload["results"][1]["config"]["dopamine_drive"] == 1
    assert "synthetic" in json.dumps(payload["graph"]).lower()
    software = payload["graph"]["software"]
    assert software["package_version"]
    assert set(software["source_sha256"]) == {"model.py", "data.py", "experiments.py"}
    assert all(len(value) == 64 for value in software["source_sha256"].values())
    assert set(software["dependency_versions"]) == {"numpy", "scipy", "pandas", "pyarrow"}


def test_small_search_records_actual_winner(tmp_path):
    main(["optimize", "--demo", "--duration", "0.1", "--trials", "3", "--target", "dopamine", "--output", str(tmp_path)])
    p = json.loads((tmp_path / "optimization.json").read_text())
    assert p["trials"] == 3
    assert p["winner"]["score"] == max(row["score"] for row in p["ranking"])


def test_json_nonfinite_fails_without_replacing_existing_file(tmp_path):
    path = tmp_path / "valid.json"
    write_json(path, {"a": np.int64(5)})
    with pytest.raises(ValueError):
        write_json(path, {"bad": float("inf")})
    assert json.loads(path.read_text()) == {"a": 5}


@pytest.mark.parametrize("shape", ["winner", "result", "raw"])
def test_replay_saved_winner_preserves_settings_and_reproduces_metrics(tmp_path, shape):
    search = tmp_path / "search"
    main(["optimize", "--demo", "--duration", "0.15", "--dt", "0.005", "--seed", "19",
          "--record-every", "3", "--trials", "3", "--output", str(search)])
    optimization = json.loads((search / "optimization.json").read_text())
    winner = optimization["winner"]
    original = json.loads((search / f"{winner['preset']}.json").read_text())
    source = tmp_path / "saved.json"
    source.write_text(json.dumps({"winner": optimization, "result": original,
                                  "raw": winner["config"]}[shape]))
    output = tmp_path / "replay"
    main(["replay", "--demo", "--config", str(source), "--output", str(output)])
    payload = json.loads((output / "results.json").read_text())
    replayed = payload["results"][0]
    assert replayed["config"] == winner["config"]
    assert replayed["metrics"] == original["metrics"]
    assert replayed["traces"] == original["traces"]
    assert payload["optimization"] is None
    assert (output / "replay.json").exists()
    assert (output / "report.html").exists()
    extended_output = tmp_path / "longer"
    main(["replay", "--demo", "--config", str(source), "--duration", "0.2", "--output", str(extended_output)])
    extended = json.loads((extended_output / "replay.json").read_text())
    assert extended["config"] == {**winner["config"], "duration": 0.2}
    assert extended["time"][-1] == pytest.approx(0.2)


@pytest.mark.parametrize("invalid,expected", [
    ({**asdict(SimulationConfig()), "unrecognized_parameter": 1}, "Unknown configuration fields: unrecognized_parameter"),
    ({key: value for key, value in asdict(SimulationConfig()).items() if key != "seed"}, "missing fields: seed"),
    ({**asdict(SimulationConfig()), "dt": "invalid"}, "dt must be a finite number"),
    ({"winner": {"config": []}}, "winner must contain a config object"),
])
def test_replay_rejects_malformed_config_before_graph_loading(tmp_path, capsys, invalid, expected):
    source = tmp_path / "bad.json"
    source.write_text(json.dumps(invalid))
    output = tmp_path / "uncreated"
    with pytest.raises(SystemExit) as caught:
        main(["replay", "--config", str(source), "--output", str(output)])
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert expected in captured.err
    assert "Loading MaleCNS" not in captured.out
    assert not output.exists()
