# Fly motor-neuron simulator

[Open the continuous live viewer](https://duncanscottwilson.com/fly-reward-lab/).

The live page observes an independently running server using read-only HTTPS requests and server-sent events. It displays actual model motor states, connection status, and simulated time. A lost or stale connection holds the last received pose and displays its status.

The recorded viewer and search-report page have been removed. Existing [motor mapping](results/motor/motor_mapping.json) and raw per-neuron recordings remain available at their original paths.

This is a connectome-constrained hypothesis model using public MaleCNS v1.0 data, not measured pleasure or a biologically validated simulation. Muscle and joint mechanics are assumptions; unmapped body parts remain still. See [motor mapping and primary sources](MOTOR_MAPPING.md), [model equations](MODEL.md), and [recorded verification](MOTOR_VERIFICATION.json).

The backend URL in `live-config.js` is public. The viewer sends no write requests and uses no API key, tracking, or external runtime libraries. JavaScript and a working backend connection are required for live state.
