"""One persistent model, independent of its read-only HTTP/SSE observers.

Run one worker behind a TLS proxy. The state-directory lock rejects accidental
duplicate workers. All neural operations belong to one thread; HTTP handlers
only read immutable published snapshots. No complete trajectory is retained.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
import fcntl
import json
import logging
import math
from pathlib import Path
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .data import download_data, load_graph
from .live_model import LiveSimulation
from .motor import LEGS, MECHANICS
from .motor_mapping import build_motor_mapping
from .types import SimulationConfig

LOG = logging.getLogger("flyreward.live")
PUBLIC_ORIGINS = ["https://duncanscottwilson.com", "https://www.duncanscottwilson.com",
                  "https://duncanswilson.github.io"]


def continuous_config() -> SimulationConfig:
    """Fixed input configuration used by the previously published best run.

    These are idealized model parameters, not pharmacological doses or a
    validated measure of pleasure. duration is provenance, never a live cutoff.
    """
    return SimulationConfig(duration=20, dt=0.01, seed=7, preset="continuous_best_tested",
                            dopamine_drive=1, reuptake_factor=0.1, npf_drive=1,
                            tolerance=False, max_rate=100, network_gain=0.6, record_every=1)


def encode(value) -> str:
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


class SimulationService:
    def __init__(self, factory, checkpoint: Path, *, batch_steps=5,
                 checkpoint_seconds=60.0, target_realtime_factor=1.0):
        if (not isinstance(batch_steps, int) or isinstance(batch_steps, bool) or batch_steps < 1
                or not math.isfinite(checkpoint_seconds) or checkpoint_seconds <= 0
                or not math.isfinite(target_realtime_factor) or target_realtime_factor <= 0):
            raise ValueError("Positive batch, checkpoint interval and target speed required")
        self.factory = factory
        self.checkpoint = Path(checkpoint)
        self.batch_steps = batch_steps
        self.checkpoint_seconds = checkpoint_seconds
        self.target_realtime_factor = target_realtime_factor
        self.stream_id = str(uuid.uuid4())
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread = None
        self.status = "starting"
        self.started = time.monotonic()
        self.last_progress = None
        self.state_json = None
        self.neurons_json = None
        self.meta_json = None
        self.step = -1
        self.last_checkpoint_step = None
        self._state_lock = None

    def start(self):
        if self.thread is not None:
            raise RuntimeError("Service already started")
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        handle = (self.checkpoint.parent / "worker.lock").open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            raise RuntimeError("Another simulation owns this state directory; use one worker") from None
        self._state_lock = handle
        self.thread = threading.Thread(target=self._run, name="connectome", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            # A graceful shutdown waits for the current neural batch/checkpoint.
            self.thread.join()
        if self._state_lock:
            fcntl.flock(self._state_lock, fcntl.LOCK_UN)
            self._state_lock.close()
            self._state_lock = None

    def _publish(self, engine, start_sim, start_wall):
        now = time.monotonic()
        state = engine.snapshot(include_motor_rates=True)
        state.update(stream_id=self.stream_id, server_time=time.time(),
                     uptime_seconds=now-self.started,
                     realtime_factor=(state["sim_time"]-start_sim)/max(now-start_wall, 0.001),
                     checkpoint_step=self.last_checkpoint_step)
        neurons = encode(state)
        state.pop("motor_neuron_ids")
        state.pop("motor_rates_hz")
        encoded = encode(state)
        with self.lock:
            self.state_json, self.neurons_json = encoded, neurons
            self.last_progress = now
            self.step = state["step"]
            self.status = "running"

    def _run(self):
        engine = None
        try:
            engine, metadata = self.factory()
            if self.checkpoint.exists():
                # Fail closed on incompatible/corrupt state, never silently reset.
                engine.restore_checkpoint(self.checkpoint)
                self.last_checkpoint_step = engine.nstep
            else:
                engine.save_checkpoint(self.checkpoint)
                self.last_checkpoint_step = engine.nstep
            metadata.update(run_id=engine.run_id, stream_id=self.stream_id,
                            config=asdict(engine.config), fingerprints=engine.fingerprints,
                            checkpoint_interval_seconds=self.checkpoint_seconds,
                            target_realtime_factor=self.target_realtime_factor)
            self.meta_json = encode(metadata)
            start_sim, start_wall = engine.t, time.monotonic()
            last_checkpoint = start_wall
            self._publish(engine, start_sim, start_wall)
            while not self.stop_event.is_set():
                batch_start = time.monotonic()
                engine.step(self.batch_steps)
                now = time.monotonic()
                if now-last_checkpoint >= self.checkpoint_seconds:
                    engine.save_checkpoint(self.checkpoint)
                    self.last_checkpoint_step = engine.nstep
                    last_checkpoint = time.monotonic()
                self._publish(engine, start_sim, start_wall)
                # Pace each batch; never skip neural steps or catch up in a burst.
                desired = self.batch_steps*engine.config.dt/self.target_realtime_factor
                self.stop_event.wait(max(0, desired-(time.monotonic()-batch_start)))
            engine.save_checkpoint(self.checkpoint)
            self.last_checkpoint_step = engine.nstep
            with self.lock:
                self.status = "stopped"
        except Exception:
            # Preserve the last known good disk checkpoint on any model failure.
            LOG.exception("Continuous simulation stopped; retaining its last good checkpoint")
            with self.lock:
                self.status = "error"

    def health(self):
        with self.lock:
            age = None if self.last_progress is None else time.monotonic()-self.last_progress
            status = self.status
            if status == "running" and age > 15:
                status = "stale"
            return {"status": status, "ready": status == "running", "step": self.step,
                    "seconds_since_progress": age, "stream_id": self.stream_id,
                    "uptime_seconds": time.monotonic()-self.started}

    def latest(self, neurons=False):
        with self.lock:
            return self.step, self.neurons_json if neurons else self.state_json


def graph_factory(data_dir: Path, download=False):
    def factory():
        if download:
            download_data(data_dir)
        graph = load_graph(data_dir)
        mapping = build_motor_mapping(data_dir, graph)
        engine = LiveSimulation(graph, continuous_config(), mapping)
        metadata = {
            "model": "MaleCNS persistent rate hypothesis model",
            "neuron_count": len(graph.ids), "directed_edge_count": int(graph.connectivity.nnz),
            "recorded_neuron_count": len(mapping["motor_neuron_ids"]),
            "mapped_neuron_count": len({int(body) for ch in mapping["channels"] for body in ch["body_ids"]}),
            "legs": list(LEGS), "channels": mapping["channels"], "mechanics": MECHANICS,
            "assumptions": [
                "Neuromodulation is an idealized hypothesis; dopamine is not a measurement of pleasure.",
                "Motor rates are current simulated neuron outputs. Muscles and joints use uncalibrated mechanics.",
                "No body physics or sensory feedback. Unmapped body parts stay fixed.",
                "Constant inputs may converge to a steady pose even while neural time advances.",
                "A crash can lose progress since the last checkpoint. A restart changes stream_id.",
            ],
            "source": "https://male-cns.janelia.org/download/", "license": "CC-BY-4.0",
            "attribution": "MaleCNS collaboration; see the source dataset and published model documentation.",
        }
        return engine, metadata
    return factory


async def stream_states(service, request, *, poll_seconds=0.1, heartbeat_seconds=10):
    last_step = -1
    last_send = time.monotonic()
    yield "retry: 2000\n\n"
    while not service.stop_event.is_set() and not await request.is_disconnected():
        if not service.health()["ready"]:
            return
        step, payload = service.latest()
        if payload is not None and step > last_step:
            yield f"event: state\ndata: {payload}\n\n"
            last_step, last_send = step, time.monotonic()
        elif time.monotonic()-last_send >= heartbeat_seconds:
            yield ": heartbeat\n\n"
            last_send = time.monotonic()
        await asyncio.sleep(poll_seconds)


class ReservedStreamingResponse(StreamingResponse):
    """Release an observer slot over the whole ASGI response lifecycle.

    A generator's finally cannot release a slot when sending response headers
    fails before its first iteration. This wrapper also covers that boundary.
    """
    def __init__(self, *args, release_slot, **kwargs):
        self._release_slot_callback = release_slot
        self._slot_released = False
        try:
            super().__init__(*args, **kwargs)
        except BaseException:
            self._release_slot()
            raise

    def _release_slot(self):
        if not self._slot_released:
            self._slot_released = True
            self._release_slot_callback()

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._release_slot()


def create_app(service: SimulationService, *, viewer_dir: Path | None = None,
               allowed_origins=None, max_clients=32) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        service.start()
        try:
            yield
        finally:
            await asyncio.to_thread(service.stop)

    app = FastAPI(title="Fly Reward Lab live motor stream", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(CORSMiddleware, allow_origins=allowed_origins or PUBLIC_ORIGINS,
                       allow_methods=["GET"], allow_headers=[], allow_credentials=False)
    app.state.stream_clients = 0

    def ready():
        if not service.health()["ready"]:
            raise HTTPException(503, "Simulation is starting, stalled or unavailable; retry later")

    def json_response(payload):
        return Response(payload, media_type="application/json", headers={"Cache-Control": "no-store"})

    @app.get("/api/health")
    async def health():
        result = service.health()
        return Response(encode(result), status_code=200 if result["ready"] else 503,
                        media_type="application/json", headers={"Cache-Control": "no-store"})

    @app.get("/api/meta")
    async def meta():
        ready()
        return json_response(service.meta_json)

    @app.get("/api/state")
    async def state():
        ready()
        return json_response(service.latest()[1])

    @app.get("/api/neurons")
    async def neurons():
        ready()
        return json_response(service.latest(neurons=True)[1])

    @app.get("/api/stream")
    async def stream(request: Request):
        ready()
        if app.state.stream_clients >= max_clients:
            raise HTTPException(503, "Observer capacity reached; retry later")
        app.state.stream_clients += 1

        def release_slot():
            app.state.stream_clients -= 1

        return ReservedStreamingResponse(stream_states(service, request), release_slot=release_slot,
                                         media_type="text/event-stream", headers={
            "Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    if viewer_dir and viewer_dir.is_dir():
        app.mount("/viewer", StaticFiles(directory=viewer_dir, html=True), name="viewer")
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="Fetch and verify pinned official files if absent")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint-seconds", type=float, default=60)
    parser.add_argument("--viewer-dir", type=Path, default=Path(__file__).resolve().parent.parent / "visualizer")
    args = parser.parse_args()
    service = SimulationService(graph_factory(args.data_dir, args.download),
                                args.state_dir / "checkpoint.npz", checkpoint_seconds=args.checkpoint_seconds)
    import uvicorn

    class ShutdownAwareServer(uvicorn.Server):
        async def on_tick(self, counter):
            with service.lock:
                worker_failed = service.status == "error"
            if worker_failed:
                # End observers before request draining, then let the container
                # restart this CLI from the last good checkpoint.
                service.stop_event.set()
                self.should_exit = True
            return await super().on_tick(counter)

        def handle_exit(self, sig, frame):
            # Signal the model and observers before Uvicorn drains requests.
            # Lifespan cleanup still joins the owner and its final checkpoint.
            service.stop_event.set()
            super().handle_exit(sig, frame)

    config = uvicorn.Config(create_app(service, viewer_dir=args.viewer_dir),
                            host=args.host, port=args.port, workers=1,
                            proxy_headers=False, access_log=False,
                            timeout_graceful_shutdown=5)
    ShutdownAwareServer(config).run()
    with service.lock:
        worker_failed = service.status == "error"
    if worker_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
