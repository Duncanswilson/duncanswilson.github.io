"""Exercise the complete ASGI response, including pre-generator disconnects."""
import asyncio
import signal
import sys
import time

from fastapi import HTTPException
import pytest

from flyreward.server import SimulationService, create_app, stream_states
import flyreward.server as server_module


class DisconnectedRequest:
    async def is_disconnected(self):
        return True


def ready_stream_endpoint(tmp_path, capacity=2):
    # No neural worker or filesystem operation is needed to test observer slots.
    service = SimulationService(lambda: None, tmp_path / "unused.npz")
    service.status = "running"
    service.last_progress = time.monotonic()
    service.state_json = '{"step":1}'
    service.step = 1
    app = create_app(service, max_clients=capacity)
    endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/api/stream")
    return app, endpoint


async def waiting_receive():
    await asyncio.Event().wait()


async def discard_send(message):
    pass


@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
def test_header_send_failure_releases_slot_before_generator_starts(tmp_path, spec_version):
    app, endpoint = ready_stream_endpoint(tmp_path)
    scope = {"type": "http", "asgi": {"spec_version": spec_version}}

    async def failed_headers(message):
        assert message["type"] == "http.response.start"
        raise OSError("peer disconnected while sending headers")

    async def exercise():
        # More failures than capacity must not permanently exhaust admission.
        for _ in range(4):
            response = await endpoint(DisconnectedRequest())
            assert app.state.stream_clients == 1
            with pytest.raises(Exception) as failure:
                await asyncio.wait_for(response(scope, waiting_receive, failed_headers), timeout=1)
            assert not isinstance(failure.value, TimeoutError)
            assert app.state.stream_clients == 0
            await response.body_iterator.aclose()
        healthy = await endpoint(DisconnectedRequest())
        await asyncio.wait_for(healthy(scope, waiting_receive, discard_send), timeout=1)
        assert app.state.stream_clients == 0

    asyncio.run(exercise())


@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
def test_cancellation_during_header_send_releases_unstarted_slot(tmp_path, spec_version):
    app, endpoint = ready_stream_endpoint(tmp_path)
    scope = {"type": "http", "asgi": {"spec_version": spec_version}}

    async def exercise():
        started = asyncio.Event()

        async def blocked_headers(message):
            assert message["type"] == "http.response.start"
            started.set()
            await asyncio.Event().wait()

        response = await endpoint(DisconnectedRequest())
        task = asyncio.create_task(response(scope, waiting_receive, blocked_headers))
        await asyncio.wait_for(started.wait(), timeout=1)
        assert app.state.stream_clients == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert app.state.stream_clients == 0
        await response.body_iterator.aclose()

    asyncio.run(exercise())


def test_admission_is_reserved_before_response_await_and_released_once(tmp_path):
    app, endpoint = ready_stream_endpoint(tmp_path)
    scope = {"type": "http", "asgi": {"spec_version": "2.4"}}

    async def exercise():
        first = await endpoint(DisconnectedRequest())
        second = await endpoint(DisconnectedRequest())
        assert app.state.stream_clients == 2
        with pytest.raises(HTTPException) as failure:
            await endpoint(DisconnectedRequest())
        assert failure.value.status_code == 503
        await first(scope, waiting_receive, discard_send)
        assert app.state.stream_clients == 1
        # Defensive repeated cleanup must not consume another observer's slot.
        first._release_slot()
        assert app.state.stream_clients == 1
        third = await endpoint(DisconnectedRequest())
        assert app.state.stream_clients == 2
        await second(scope, waiting_receive, discard_send)
        await third(scope, waiting_receive, discard_send)
        assert app.state.stream_clients == 0

    asyncio.run(exercise())


def test_stream_stops_when_service_shutdown_requested_even_if_client_stays_connected(tmp_path):
    service = SimulationService(lambda: None, tmp_path / "unused.npz")
    service.status = "running"
    service.last_progress = time.monotonic()
    service.state_json = '{"step":1}'
    service.step = 1

    class ConnectedRequest:
        async def is_disconnected(self):
            return False

    async def exercise():
        stream = stream_states(service, ConnectedRequest(), poll_seconds=0)
        assert await anext(stream) == "retry: 2000\n\n"
        assert "event: state" in await anext(stream)
        service.stop_event.set()
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(stream), timeout=1)

    asyncio.run(exercise())


def test_main_signals_observers_before_uvicorn_begins_connection_drain(tmp_path, monkeypatch):
    import uvicorn
    services = []
    actual_service_type = server_module.SimulationService
    actual_handle_exit = uvicorn.Server.handle_exit

    def capture_service(*args, **kwargs):
        service = actual_service_type(*args, **kwargs)
        services.append(service)
        return service

    def check_base_exit(self, sig, frame):
        # This is the boundary before Uvicorn can wait on active SSE requests.
        assert services[0].stop_event.is_set()
        actual_handle_exit(self, sig, frame)

    def run_signal_check(self, sockets=None):
        assert not services[0].stop_event.is_set()
        self.handle_exit(signal.SIGTERM, None)
        assert self.should_exit
        assert self.config.timeout_graceful_shutdown == 5

    monkeypatch.setattr(server_module, "SimulationService", capture_service)
    monkeypatch.setattr(uvicorn.Server, "handle_exit", check_base_exit)
    monkeypatch.setattr(uvicorn.Server, "run", run_signal_check)
    monkeypatch.setattr(sys, "argv", ["flyreward.server", "--data-dir", str(tmp_path / "data"),
                                     "--state-dir", str(tmp_path / "state")])
    server_module.main()
    assert len(services) == 1
