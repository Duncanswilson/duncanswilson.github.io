# Continuous simulator deployment

Deployment package for the existing fly reward lab. The frontend on the default branch remains the recorded viewer until the cloud API is verified.

The backend runs the connectome rate model continuously and saves checkpoints. Motor readouts drive the viewer; this is an uncalibrated model, not evidence of subjective pleasure.

`discover-cloud.sh` inventories the intended Oracle home region. `provision-free.sh` reviews or creates one A1 instance within a total ceiling of 2 OCPUs, 12 GB RAM and 200 GB combined boot/block storage. Its target boot disk is 50 GB. Preserve the provisioning state journal on retries. Supply an SSH public key and source CIDR locally; credentials and keys are not included here.

After a VM is running, copy `fly-backend.tar.gz` and `install-backend.sh` to it, then run `sudo bash install-backend.sh --ip ACTUAL_PUBLIC_IPV4`. The installer verifies the archive checksum and configures Docker plus Caddy IP HTTPS. Readiness, certificate issuance and checkpoint recovery must be verified before switching the frontend.
