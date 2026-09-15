"""Exercise CLI failure exit through the real Uvicorn loop and worker thread."""
from pathlib import Path
import subprocess
import sys

import pytest

from flyreward.live_model import LiveSimulation
from test_live_model import specimen


FAILED_WORKER_SCRIPT = r"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / "tests"))
from test_live_model import specimen
from flyreward.live_model import LiveSimulation
import flyreward.server as server

phase, state_dir = sys.argv[1:]

def factory():
    if phase == "factory":
        raise RuntimeError("injected factory failure")
    graph, config, mapping = specimen()
    engine = LiveSimulation(graph, config, mapping)
    def failed_step(steps):
        # Mimic an in-place update failing after it has altered memory.
        engine.rates[0] = float("nan")
        engine.nstep += 1
        raise FloatingPointError("injected step failure")
    engine.step = failed_step
    return engine, {}

server.graph_factory = lambda *args: factory
sys.argv = ["flyreward.server", "--data-dir", str(Path(state_dir) / "unused-data"),
            "--state-dir", state_dir, "--host", "127.0.0.1", "--port", "0"]
server.main()
"""


@pytest.mark.parametrize("phase", ["factory", "step"])
def test_cli_worker_failure_exits_nonzero_and_preserves_checkpoint(tmp_path, phase):
    graph, config, mapping = specimen()
    engine = LiveSimulation(graph, config, mapping)
    engine.step(7)
    checkpoint = tmp_path / "checkpoint.npz"
    engine.save_checkpoint(checkpoint)
    original = checkpoint.read_bytes()

    result = subprocess.run(
        [sys.executable, "-c", FAILED_WORKER_SCRIPT, phase, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=15,
    )

    assert result.returncode == 1, result.stderr
    assert f"injected {phase} failure" in result.stderr
    assert "Application shutdown complete." in result.stderr
    assert checkpoint.read_bytes() == original
    restored = LiveSimulation.from_checkpoint(checkpoint, graph, mapping)
    assert restored.nstep == 7
    assert restored.run_id == engine.run_id
