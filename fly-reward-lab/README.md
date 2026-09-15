# Fly motor-neuron simulator

[Open the continuous live viewer](https://duncanscottwilson.com/fly-reward-lab/) or [open the preserved recorded viewer](recorded.html).

The live page observes an independently running server using read-only HTTPS requests and server-sent events. It displays actual model motor states, connection status, simulated time, and a bounded recent trace. It has no replay timeline. A lost or stale connection holds the last received pose and displays its status; no recording substitutes for live state.

The recorded viewer remains self-contained, including its original playback data and controls. Existing [recorded experiment reports](results/search/report.html), [motor mapping](results/motor/motor_mapping.json), and all raw per-neuron recordings remain available at their original paths.

This is a connectome-constrained hypothesis model using public MaleCNS v1.0 data, not measured pleasure or a biologically validated simulation. Muscle and joint mechanics are assumptions; unmapped body parts remain still. See [motor mapping and primary sources](MOTOR_MAPPING.md), [model equations](MODEL.md), and [recorded verification](MOTOR_VERIFICATION.json).

The backend URL in `live-config.js` is public. The viewer sends no write requests and uses no API key, tracking, or external runtime libraries. JavaScript and a working backend connection are required for live state; the preserved recording works offline.
