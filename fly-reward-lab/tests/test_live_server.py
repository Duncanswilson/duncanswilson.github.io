import asyncio
import json
import time

from fastapi.testclient import TestClient
import pytest

from flyreward.live_model import LiveSimulation
from flyreward.server import SimulationService, create_app, stream_states
from test_live_model import specimen


def service_at(tmp_path, **kwargs):
    graph, config, mapping = specimen()
    def factory():
        return LiveSimulation(graph, config, mapping), {"channels": mapping["channels"]}
    return SimulationService(factory, tmp_path / "checkpoint.npz", **kwargs)


def await_status(service, status="running"):
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        if service.health()["status"] == status:
            return
        time.sleep(0.002)
    pytest.fail(f"Worker did not become {status}: {service.health()}")


def test_simulation_advances_without_observers_and_resumes_identity(tmp_path):
    service = service_at(tmp_path)
    service.start()
    try:
        await_status(service)
        first = json.loads(service.latest()[1])
        time.sleep(0.08)
        second = json.loads(service.latest()[1])
        assert second["step"] > first["step"]
    finally:
        service.stop()
    stopped_step = service.last_checkpoint_step
    restarted = service_at(tmp_path)
    restarted.start()
    try:
        await_status(restarted)
        state = json.loads(restarted.latest()[1])
        assert state["step"] >= stopped_step
        assert state["run_id"] == second["run_id"]
        assert state["stream_id"] != second["stream_id"]
    finally:
        restarted.stop()


def test_one_worker_per_state_directory(tmp_path):
    first, second = service_at(tmp_path), service_at(tmp_path)
    first.start()
    try:
        with pytest.raises(RuntimeError, match="Another simulation"):
            second.start()
    finally:
        first.stop()
    second.start()
    second.stop()


def test_read_only_api_snapshot_neurons_cors_and_lifespan(tmp_path):
    service = service_at(tmp_path)
    with TestClient(create_app(service)) as client:
        await_status(service)
        health = client.get("/api/health")
        assert health.status_code == 200 and health.json()["ready"]
        state = client.get("/api/state", headers={"Origin": "https://duncanscottwilson.com"})
        assert state.headers["access-control-allow-origin"] == "https://duncanscottwilson.com"
        assert state.headers["cache-control"] == "no-store"
        assert "motor_rates_hz" not in state.json()
        assert len(state.json()["channel_rates_hz"]) == 24
        neurons = client.get("/api/neurons").json()
        assert len(neurons["motor_rates_hz"]) == len(neurons["motor_neuron_ids"]) == 48
        assert client.get("/api/meta").json()["run_id"] == state.json()["run_id"]
        assert client.post("/api/state", json={"reset": True}).status_code == 405
        assert client.post("/api/reset").status_code == 404
        assert "access-control-allow-origin" not in client.get("/api/state", headers={"Origin": "https://untrusted.invalid"}).headers
    assert service.health()["status"] == "stopped"
    assert service.checkpoint.exists()


def test_corrupt_checkpoint_does_not_reset_and_api_fails_closed(tmp_path):
    checkpoint = tmp_path / "checkpoint.npz"
    checkpoint.write_bytes(b"not a checkpoint")
    service = service_at(tmp_path)
    with TestClient(create_app(service)) as client:
        await_status(service, "error")
        for route in ("health", "state", "neurons", "meta", "stream"):
            assert client.get(f"/api/{route}").status_code == 503
    assert checkpoint.read_bytes() == b"not a checkpoint"


def test_stalled_progress_is_unavailable_even_with_snapshot(tmp_path):
    service = service_at(tmp_path)
    service.status = "running"
    service.last_progress = time.monotonic()-16
    service.state_json = "{}"
    service.step = 8
    assert service.health()["status"] == "stale"
    assert not service.health()["ready"]
    with TestClient(create_app(service), raise_server_exceptions=True) as client:
        await_status(service)
        with service.lock:
            service.status = "error"
        assert client.get("/api/state").status_code == 503


def test_failed_step_preserves_last_good_checkpoint(tmp_path):
    graph, config, mapping = specimen()
    def factory():
        engine = LiveSimulation(graph, config, mapping)
        def broken_step(n):
            raise FloatingPointError("test failure")
        engine.step = broken_step
        return engine, {}
    service = SimulationService(factory, tmp_path / "checkpoint.npz")
    service.start()
    try:
        await_status(service, "error")
        assert not service.health()["ready"]
        restored = LiveSimulation.from_checkpoint(service.checkpoint, graph, mapping)
        assert restored.nstep == 0
    finally:
        service.stop()


def test_stream_publishes_latest_without_advancing_model(tmp_path):
    service = service_at(tmp_path)
    service.status = "running"
    service.last_progress = time.monotonic()
    service.step = 7
    service.state_json = '{"step":7}'
    class Request:
        async def is_disconnected(self):
            return False
    async def inspect():
        stream = stream_states(service, Request(), poll_seconds=0)
        assert await anext(stream) == "retry: 2000\n\n"
        assert await anext(stream) == 'event: state\ndata: {"step":7}\n\n'
        service.step, service.state_json = 9, '{"step":9}'
        assert await anext(stream) == 'event: state\ndata: {"step":9}\n\n'
        await stream.aclose()
    asyncio.run(inspect())
    assert service.thread is None and service.step == 9


@pytest.mark.parametrize("kwargs", [{"batch_steps":0}, {"checkpoint_seconds":float("nan")},
                                    {"target_realtime_factor":0}])
def test_invalid_worker_settings_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        service_at(tmp_path, **kwargs)
