# Hedonome

[Open the live simulator](https://duncanscottwilson.com/hedonome/).

This directory contains the GitHub Pages frontend. Its source templates are in
[`../fly-reward-lab/visualizer/`](../fly-reward-lab/visualizer/).
The continuous simulation, tests, deployment scripts and model documentation
remain in [`../fly-reward-lab/`](../fly-reward-lab/).

The viewer observes the existing read-only HTTPS/SSE backend configured in
`live-config.js`. The former `/fly-reward-lab/` entry page and its model-notes
page redirect here, preserving query strings and fragments when JavaScript is enabled.
The old static assets and scientific data remain available for existing links.
