# Deploying the circuit/body mode

GitHub Pages serves the viewer; the persistent simulator runs on the existing VPS. The old checksum-pinned `cloud/fly-backend.tar.gz` is a historical release and does not contain this mode.

Build the current `fly-reward-lab/` directory using its Dockerfile. The Python wheel includes `flyreward/assets/cpg_front_v1.json`; check this asset is present in any custom packaging.

## State separation

The circuit/body engine has a separate checkpoint schema and mechanism fingerprint. Do not point it at the older `/state/live` directory. Use `/state/live-cpg-v1` and retain the original state and image for rollback. The new model begins a new run with a distinct `run_id`.

The running service must remain a single worker. No new cloud service, port, credentials or paid resource is required. Caddy and its TLS volume remain unchanged.

## Versioned update

Stage the reviewed source in a versioned directory such as `/opt/fly-reward-lab-cpg-COMMIT`. Build a versioned image before stopping the existing service. Back up the running image tag and gracefully stop the old backend so its final checkpoint is written.

At `/opt/fly-reward-lab/deploy/`, add `compose.cpg.yaml` with an explicit source path and versioned image tag:

```yaml
services:
  backend:
    build:
      context: /opt/fly-reward-lab-cpg-COMMIT
    image: fly-reward-lab:cpg-COMMIT
    command: ["python", "-m", "flyreward.server", "--host", "0.0.0.0", "--port", "8000", "--data-dir", "/state/data", "--state-dir", "/state/live-cpg-v1", "--viewer-dir", "/app/visualizer", "--motor-mode", "cpg"]
```

Use the existing environment file and all three Compose files:

```sh
sudo docker compose --project-directory /opt/fly-reward-lab/deploy \
  --env-file /opt/fly-reward-lab/deploy/actualIP.env \
  -f /opt/fly-reward-lab/deploy/compose.yaml \
  -f /opt/fly-reward-lab/deploy/compose.ip.yaml \
  -f /opt/fly-reward-lab/deploy/compose.cpg.yaml \
  up -d --no-deps --no-build backend
```

Record the third file in the deployment environment's `COMPOSE_FILE` so future routine commands keep the same configuration. The backend creates its new state subdirectory in the existing volume as UID 10001.

Verify `/api/health` is ready, `/api/meta` says `motor_mode: cpg`, multiple `/api/state` values advance with one run ID, raw `/api/neurons` averages equal channel rates, and body geometry is present. An SSE connection alone is not proof of advancement. Verify the visible viewer receives these states too.

## Rollback

Gracefully stop the CPG backend. Start `backend` with the original two Compose files, `--no-deps --no-build`, and the retained original `fly-reward-lab:local` image. Restore the environment's two-file `COMPOSE_FILE`. This resumes `/state/live` and preserves `/state/live-cpg-v1` for analysis. Never delete the state volume to resolve a checkpoint incompatibility.

Subsequent mechanism edits also invalidate the CPG checkpoint. Create a separately versioned state path for those edits or implement an explicit reviewed migration. Do not silently reset an incompatible checkpoint.
