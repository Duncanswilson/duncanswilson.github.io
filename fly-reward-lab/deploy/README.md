# Run the continuing model on one VM

This package prepares the live backend; it does not create cloud resources or enable billing. The cheapest eligible option is **Oracle A1 Always Free: 2 OCPUs, 12 GB RAM, and a 50 GB boot disk at $0/month** within the account's allowance. Oracle signup and available capacity in the account's home region are required. Do not select a paid shape or upgrade the account automatically. See [current prices and limitations](CLOUD_OPTIONS.md).

The backend advances one persistent model even when no browser is connected. It downloads the approximately 1.1 GB official data on its first start, verifies it, and caches the compiled graph. Transfers below 1 KiB/s for 120 seconds time out and retry up to five times; partial downloads remain resumable if those retries fail. The `state` Docker volume stores the data and live checkpoint. Two separate volumes preserve Caddy's certificates and configuration. The simulation process runs as UID/GID 10001; a short-lived initializer sets the state volume's ownership before it starts.

## Prepare the VM

Use an Always Free eligible Ubuntu ARM image on A1. Install Docker Engine and its Compose plugin using [Docker's official Ubuntu instructions](https://docs.docker.com/engine/install/ubuntu/), then ensure Docker starts on boot. These files also build natively on an x86-64 Linux VM; no architecture override is needed.

Copy the whole source package to a directory on the VM, for example `/opt/fly-reward-lab`. Permit inbound TCP ports 80 and 443 in both the cloud network rules and the host firewall, plus the SSH access you use to manage it. Only Caddy publishes ports; Python port 8000 stays inside the Compose network. Choose one of the HTTPS options below.

### Cheapest immediate option: the VM's public IPv4

No domain purchase or DNS account is needed. From the package's `deploy` directory, enter the actual **public IPv4 shown for this VM in Oracle**, then start both Compose files:

```sh
read -r -p 'Actual VM public IPv4 (no https:// or path): ' FLY_API_DOMAIN
printf 'FLY_API_DOMAIN=%s\nCOMPOSE_FILE=compose.yaml:compose.ip.yaml\n' "$FLY_API_DOMAIN" > .env
docker compose --env-file .env -f compose.yaml -f compose.ip.yaml config --quiet
docker compose --env-file .env -f compose.yaml -f compose.ip.yaml up -d --build
docker compose --env-file .env -f compose.yaml -f compose.ip.yaml logs --tail=100 -f backend caddy
```

The override pins Caddy 2.11.4 and uses [Caddyfile.ip](Caddyfile.ip) to obtain a trusted Let's Encrypt IP certificate. Certificates last 160 hours and Caddy renews them automatically; preserve `caddy_data` and leave TCP port 80 reachable for validation. `default_sni` handles IP clients behind VM/container NAT. The saved `COMPOSE_FILE` setting makes subsequent commands below retain both files. [Let's Encrypt IP certificates](https://letsencrypt.org/2026/01/15/6day-and-ip-general-availability), [Caddy release](https://github.com/caddyserver/caddy/releases/tag/v2.11.4)

The merged Compose configuration and Caddy's `adapt --validate` check passed locally using the checksum-verified official Caddy 2.11.4 ARM64 binary. This checks configuration only; no certificate was requested and no public endpoint was started.

### Optional: a hostname you already control

Point the hostname's DNS record at the VM's public IP. From `deploy`, choose the original domain configuration:

```sh
read -r -p 'API hostname (no https:// or path): ' FLY_API_DOMAIN
printf 'FLY_API_DOMAIN=%s\nCOMPOSE_FILE=compose.yaml\n' "$FLY_API_DOMAIN" > .env
docker compose --env-file .env config --quiet
docker compose --env-file .env up -d --build
docker compose --env-file .env logs --tail=100 -f backend caddy
```

This path uses the original [Caddyfile](Caddyfile) and automatic domain certificates. [Caddy HTTPS requirements](https://caddyserver.com/docs/automatic-https)

### Verify either option

These commands are for Bash on the Ubuntu VM. Use `sudo docker` if your account does not have Docker access. `.env` remains on the VM; it contains the public address and Compose file selection, not credentials. The first start needs time to download and load the graph. A proxy response alone does not establish that the neural worker has started.

After startup, stop following the logs with Ctrl-C and check the actual HTTPS API:

```sh
curl --fail --silent --show-error "https://${FLY_API_DOMAIN}/api/health"
curl --fail --silent --show-error "https://${FLY_API_DOMAIN}/api/state"
```

Health returns HTTP 200 with `ready: true` only when the simulation is running. It returns 503 during startup or a model error. The state response includes `sim_time`, `step`, `run_id`, and `stream_id`. Repeat the state request later and confirm that `sim_time` and `step` increased. The same VM serves its viewer at `/viewer/live.html` and named `state` events at `/api/stream`; `/api/meta` provides model provenance and `/api/neurons` includes the individual motor-neuron rates.

Caddy flushes `text/event-stream` responses immediately by default, while retaining normal disconnect handling for observers. [Caddy streaming behavior](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy#streaming)

`FLY_API_DOMAIN` must be this VM's actual public IPv4 for the IP option, or a real hostname pointing at it for the domain option. There is no deployed endpoint or placeholder backend to connect to yet. Keep the existing Pages visualizer's live mode offline until the real backend reports readiness and advancing simulation time, then configure it with that actual HTTPS origin. A browser's closed tab or disconnected stream must not be used to stop the server.

## Restart, stop, and preserve state

```sh
docker compose --env-file .env ps
docker compose --env-file .env restart backend
docker compose --env-file .env stop -t 120
docker compose --env-file .env up -d
```

The backend gets SIGTERM and up to 120 seconds for orderly shutdown and checkpointing. A failed simulation worker makes the command-line server close its streams and exit with status 1, preserving the last good checkpoint. `restart: unless-stopped` then restarts the container; it also resumes containers after a crash or host reboot. A deliberate stop remains stopped until started again. An incompatible or corrupt checkpoint remains an error across restarts and must be investigated; the service never silently resets it. Do not use `down --volumes` or remove the state volume when preserving the running model. Keep only one backend instance per state volume. [Compose dependency order](https://docs.docker.com/compose/how-tos/startup-order/)

Logs use Docker's rotating local driver, limited to three 10 MB files per container. This bounds container logs; the application must independently bound its checkpoint/history files. [Docker local logging](https://docs.docker.com/engine/logging/drivers/local/)

Before relying on the service, check that simulation time advances with all browser clients disconnected, then restart the backend and verify that time resumes from its saved state rather than starting over. The run ID should persist and stream ID should change after restart. Health status alone does not trigger Docker's restart policy: worker exceptions cause an explicit process exit, but an indefinitely stalled computation does not. Inspect backend logs if it remains unhealthy or repeatedly restarts. Keep an independent copy of important checkpoints: a persistent volume survives container replacement but is not protection against losing the VM's disk.

The container has been prepared for the requested CLI:

```sh
python -m flyreward.server --host 0.0.0.0 --port 8000 --data-dir /state/data --state-dir /state/live --viewer-dir /app/visualizer --download
```

It uses Python 3.12 and installs the package's `live` dependencies under `requirements-live-lock.txt`. A pip dry run verified compatible Python 3.12 Linux ARM64 wheels for all 19 runtime dependencies. The full local live service used 1.19 GiB peak resident memory and advanced at about 0.51 simulated seconds per wall second; these are local measurements with a compiled graph cache. See [verification details](../LIVE_VERIFICATION.json). Cloud execution, ARM performance, certificate issuance, and a complete container build must still be verified on the chosen host. A continuing simulation need not run at real-time speed, and fixed inputs can still make its modeled motor output converge to a steady pose.
