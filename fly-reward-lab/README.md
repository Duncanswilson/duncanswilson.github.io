# Fly motor-neuron simulator

[Open the simulator](https://duncanscottwilson.com/fly-reward-lab/).

Static, browser-based replay of a connectome-constrained hypothesis model using the public MaleCNS v1.0 data from HHMI Janelia and collaborators. It records 815 identified motor neurons; 164 are mapped to 24 antagonist channels for two joints on each of six legs. The inspector exposes firing rates, body IDs, raw recordings, and joint offsets.

The neural rates are simulated. Muscle and joint mechanics are uncalibrated assumptions; unmapped body parts remain stationary, and body physics and sensory feedback are absent. The viewer contains no scripted writhing. The best-tested 20-second replay settles into a steady pose after small joint changes.

See [motor mapping and primary sources](MOTOR_MAPPING.md), [model equations](MODEL.md), and [verification results](MOTOR_VERIFICATION.json). Data source: [MaleCNS v1.0](https://male-cns.janelia.org/download/). Background: [Google Research's connectome announcement](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/).

All visualizer code and playback data are embedded in `index.html`. Reports, the motor mapping, and full per-neuron NPZ recordings are served as static files. No server computation, cookies, tracking, API key, or external runtime dependency is required.
