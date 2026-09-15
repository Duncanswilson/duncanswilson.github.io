# Fly motor-neuron simulator

[Open the continuous live viewer](https://duncanscottwilson.com/fly-reward-lab/).

## Backend source

The continuous simulator and deployment code are committed directly on the repository's default branch, `master`:

- [Neuron updates and neuromodulatory inputs](flyreward/live_model.py): `LiveSimulation._step_once()` adds dopamine/NPF modulation to each neural step.
- [Dopamine dynamics](flyreward/model.py): anatomical projections, dopamine firing clamps, clearance and sensitivity.
- [Server and fixed input settings](flyreward/server.py): `continuous_config()` configures the running model; the server exposes read-only HTTP/SSE observations.
- [Tests](tests/), [container and Compose deployment](deploy/README.md), and [VM installation scripts](cloud/README.md).

From this directory, install and run the tests:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -c requirements-live-lock.txt '.[live,test]'
.venv/bin/python -m pytest
```

The [cloud backend archive](cloud/fly-backend.tar.gz) is retained unchanged for installers that verify its checksum. Its Python backend matches the unpacked source. The unpacked `visualizer/` contains the current FlyPaint design; the archive retains its original viewer. GitHub Pages serves the frontend only; merging source does not redeploy the running VPS.

The live page observes an independently running server using read-only HTTPS requests and server-sent events. It displays actual model motor states, connection status, and simulated time. A lost or stale connection holds the last received pose and displays its status.

The recorded viewer and search-report page have been removed. Existing [motor mapping](results/motor/motor_mapping.json) and raw per-neuron recordings remain available at their original paths.

This is a connectome-constrained hypothesis model using public MaleCNS v1.0 data, not measured pleasure or a biologically validated simulation. Muscle and joint mechanics are assumptions; unmapped body parts remain still. See [motor mapping and primary sources](MOTOR_MAPPING.md), [model equations](MODEL.md), and [recorded verification](MOTOR_VERIFICATION.json).

The backend URL in `live-config.js` is public. The viewer sends no write requests and uses no API key, tracking, or external runtime libraries. JavaScript and a working backend connection are required for live state.
