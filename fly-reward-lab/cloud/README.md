# Continuous simulator deployment

Deployment package for the fly reward lab. The frontend on the default branch uses the live OVH backend. The unpacked Python source, tests, Dockerfile and Compose configuration are available [one directory above](../README.md) on `master`.

`fly-backend.tar.gz` remains the original checksum-pinned installer artifact. Its Python backend matches the unpacked source; its bundled viewer predates the current FlyPaint design in `../visualizer/`. The Oracle provisioning scripts below are retained as deployment history and alternatives; the current VPS runs on OVH.

The backend runs the connectome rate model continuously and saves checkpoints. Motor readouts drive the viewer; this is an uncalibrated model, not evidence of subjective pleasure.

`discover-cloud.sh` inventories the intended Oracle home region. `provision-free.sh` reviews or creates one A1 instance within a total ceiling of 2 OCPUs, 12 GB RAM and 200 GB combined boot/block storage. Its target boot disk is 50 GB. Preserve the provisioning state journal on retries. Supply an SSH public key and source CIDR locally; credentials and keys are not included here.

After a VM is running, copy `fly-backend.tar.gz` and `install-backend.sh` to it, then run `sudo bash install-backend.sh --ip ACTUAL_PUBLIC_IPV4`. The installer verifies the archive checksum and configures Docker plus Caddy IP HTTPS. Readiness, certificate issuance and checkpoint recovery must be verified before switching the frontend.

The `provision-amd.py` fallback tries one Always Free E2.1.Micro with the already created network. It preserves launch intent in the same journal and does not automatically retry ambiguous requests. The 1 GB model deployment is an unverified slow-run fallback; do not switch the viewer before measuring it.

For this smaller VM, `install-native.sh` installs a Python systemd service, native Caddy, and 4 GiB swap on the existing boot disk. The default does not start services. Transfer the matching compiled graph cache to `/var/lib/flyreward/data/compiled/8dde907611492b22ae0c/` first; the service verifies it before downloading the original public source files. No cache, credentials or private SSH key are committed here.
