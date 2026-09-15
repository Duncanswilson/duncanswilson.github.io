#!/usr/bin/env bash
# Prepare the reviewed backend on Ubuntu 24.04 amd64. Run on the VM as root.
# Default: install only. Seed the compiled cache before enabling the services.
# Usage: sudo bash install-native.sh --archive ./fly-backend.tar.gz --public-ip IP [--start]
set -euo pipefail
umask 022

archive=
public_ip=
start=false
while (($#)); do
  case "$1" in
    --archive) archive=${2:?Missing archive}; shift 2 ;;
    --public-ip) public_ip=${2:?Missing public IPv4}; shift 2 ;;
    --start) start=true; shift ;;
    -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ $EUID -eq 0 && -f "$archive" && -n "$public_ip" ]] || {
  echo 'Run as root with --archive FILE and --public-ip IPv4.' >&2; exit 2;
}
[[ $(uname -m) == x86_64 ]] || { echo 'Requires x86_64.' >&2; exit 2; }
. /etc/os-release
[[ $ID == ubuntu && $VERSION_ID == 24.04 ]] || { echo 'Requires Ubuntu 24.04.' >&2; exit 2; }
python3 - "$public_ip" <<'PY'
import ipaddress,sys
address=ipaddress.IPv4Address(sys.argv[1])
if not address.is_global:
    raise SystemExit('Use the actual public IPv4 assigned to the VM.')
PY
backend_sha=131d95c4c94252e5c763f2cb90f37088113c80a047f556bcc62f650739a77d9c
printf '%s  %s\n' "$backend_sha" "$archive" | sha256sum --check --status
if systemctl is-active --quiet flyreward.service || systemctl is-active --quiet caddy.service; then
  echo 'An existing service is running; this installer does not replace running services.' >&2
  exit 2
fi

# Swap lives on the existing boot filesystem; no additional cloud volume.
swapfile=/swapfile-flyreward
if [[ ! -e "$swapfile" ]]; then
  (umask 077; dd if=/dev/zero of="$swapfile" bs=1M count=4096 conv=fsync status=progress)
  mkswap "$swapfile"
fi
[[ -f "$swapfile" && ! -L "$swapfile" && $(stat -c %s "$swapfile") -eq 4294967296 ]] || {
  echo 'Existing flyreward swap file is not a regular 4 GiB file.' >&2; exit 2;
}
chmod 600 "$swapfile"
if ! swapon --show=NAME --noheadings | grep -Fxq "$swapfile"; then swapon "$swapfile"; fi
if ! grep -Eq '^/swapfile-flyreward[[:space:]]' /etc/fstab; then
  printf '/swapfile-flyreward none swap sw 0 0\n' >> /etc/fstab
fi
printf 'vm.swappiness=10\n' > /etc/sysctl.d/90-flyreward-swap.conf
sysctl -p /etc/sysctl.d/90-flyreward-swap.conf

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends python3.12-venv ca-certificates curl

app=/opt/fly-reward-lab
if [[ -e "$app" && ! -f "$app/.native-artifact-sha256" ]]; then
  echo "$app already exists without this installer's artifact marker." >&2; exit 2
fi
if [[ -f "$app/.native-artifact-sha256" ]]; then
  [[ $(cat "$app/.native-artifact-sha256") == "$backend_sha" ]] || {
    echo 'Installed backend artifact differs; preserve its state and review the update.' >&2; exit 2;
  }
else
  tar --extract --gzip --file "$archive" --directory /opt --no-same-owner
  printf '%s\n' "$backend_sha" > "$app/.native-artifact-sha256"
fi
chmod -R a+rX "$app"
python3.12 -m venv "$app/.venv"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
# The local source package builds a pure Python wheel. All external dependencies
# must use binary wheels, avoiding scientific-library compilation on the micro VM.
"$app/.venv/bin/python" -m pip install --no-cache-dir --only-binary=:all: \
  -c "$app/requirements-live-lock.txt" "$app[live]"

getent group flyreward >/dev/null || groupadd --system flyreward
id -u flyreward >/dev/null 2>&1 || useradd --system --gid flyreward \
  --home-dir /var/lib/flyreward --shell /usr/sbin/nologin flyreward
install -d -o flyreward -g flyreward -m 0750 /var/lib/flyreward /var/lib/flyreward/data /var/lib/flyreward/live
install -d -m 0755 /usr/local/libexec
cat > /usr/local/libexec/flyreward-check-cache <<'PY'
#!/usr/bin/python3
"""Require the exact transferred cache, so a missing cache cannot cold-compile."""
from pathlib import Path
import hashlib
root=Path('/var/lib/flyreward/data/compiled/8dde907611492b22ae0c')
expected={
    'connectivity.npz':'9b04a0494fc1065dd34a8e359845aef8f8f1c4c619a3a9f79ea8635d734208b1',
    'neurons.npz':'3a3afd46701fd673df4e320064ecfb5680020f869cceb4fd02447136e71f6ad2',
    'metadata.json':'f21642223f16483a4e681b5a0b1ad3fa04d5ff2a8dd7fefda8f3aefb07a2015c',
}
for name,wanted in expected.items():
    with (root/name).open('rb') as handle:
        if hashlib.file_digest(handle,'sha256').hexdigest()!=wanted:
            raise SystemExit('Compiled cache checksum mismatch: '+name)
PY
chmod 0755 /usr/local/libexec/flyreward-check-cache
cat > /etc/systemd/system/flyreward.service <<'UNIT'
[Unit]
Description=Persistent MaleCNS hypothesis simulator
Wants=network-online.target
After=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=flyreward
Group=flyreward
WorkingDirectory=/opt/fly-reward-lab
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONDONTWRITEBYTECODE=1
Environment=OPENBLAS_NUM_THREADS=1
Environment=OMP_NUM_THREADS=1
Environment=MKL_NUM_THREADS=1
Environment=NUMEXPR_NUM_THREADS=1
ExecStartPre=/usr/local/libexec/flyreward-check-cache
ExecStart=/opt/fly-reward-lab/.venv/bin/python -m flyreward.server --host 127.0.0.1 --port 8000 --data-dir /var/lib/flyreward/data --state-dir /var/lib/flyreward/live --viewer-dir /opt/fly-reward-lab/visualizer --download
Restart=on-failure
RestartSec=30s
TimeoutStartSec=300s
TimeoutStopSec=300s
KillSignal=SIGTERM
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/flyreward
UMask=0027

[Install]
WantedBy=multi-user.target
UNIT

# SHA-512 copied from the official v2.11.4 release checksum asset:
# https://github.com/caddyserver/caddy/releases/download/v2.11.4/caddy_2.11.4_checksums.txt
caddy_sha512=8220d1f013b6f27510247b2360c9e0ca9f018feebd82515f07635318b34ff9777ccc8fd0b6e6f2486ce3a33fe389fbb7db12d05baa474f4587509fb4f5ebf1c9
temp_dir=$(mktemp -d /var/tmp/fly-native.XXXXXX)
trap 'rm -rf -- "$temp_dir"' EXIT
curl --fail --silent --show-error --location --retry 5 --speed-time 120 --speed-limit 1024 \
  https://github.com/caddyserver/caddy/releases/download/v2.11.4/caddy_2.11.4_linux_amd64.tar.gz \
  --output "$temp_dir/caddy.tar.gz"
printf '%s  %s\n' "$caddy_sha512" "$temp_dir/caddy.tar.gz" | sha512sum --check --status
tar --extract --gzip --file "$temp_dir/caddy.tar.gz" --directory "$temp_dir" caddy
install -m 0755 "$temp_dir/caddy" /usr/local/bin/caddy
getent group caddy >/dev/null || groupadd --system caddy
id -u caddy >/dev/null 2>&1 || useradd --system --gid caddy \
  --home-dir /var/lib/caddy --shell /usr/sbin/nologin caddy
install -d -o caddy -g caddy -m 0750 /var/lib/caddy
install -d -m 0755 /etc/caddy
# Keep the reviewed IP-certificate configuration; replace only the upstream
# and the validated public-IP placeholder for this native deployment.
python3 - "$app/deploy/Caddyfile.ip" "$public_ip" <<'PY'
from pathlib import Path
import sys
text=Path(sys.argv[1]).read_text()
if text.count('reverse_proxy backend:8000')!=1:
    raise SystemExit('Unexpected reviewed Caddy configuration')
text=text.replace('reverse_proxy backend:8000','reverse_proxy 127.0.0.1:8000')
text=text.replace('{$FLY_API_DOMAIN}',sys.argv[2])
Path('/etc/caddy/Caddyfile').write_text(text)
PY
cat > /etc/systemd/system/caddy.service <<'UNIT'
[Unit]
Description=Caddy HTTPS proxy for the persistent simulator
Wants=network-online.target
After=network-online.target flyreward.service

[Service]
Type=notify
User=caddy
Group=caddy
Environment=XDG_DATA_HOME=/var/lib/caddy/data
Environment=XDG_CONFIG_HOME=/var/lib/caddy/config
ExecStart=/usr/local/bin/caddy run --config /etc/caddy/Caddyfile --adapter caddyfile
ExecReload=/usr/local/bin/caddy reload --config /etc/caddy/Caddyfile --adapter caddyfile --force
Restart=on-failure
RestartSec=5s
TimeoutStopSec=30s
LimitNOFILE=65536
AmbientCapabilities=CAP_NET_BIND_SERVICE
CapabilityBoundingSet=CAP_NET_BIND_SERVICE
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/caddy
UMask=0027

[Install]
WantedBy=multi-user.target
UNIT
runuser -u caddy -- env XDG_DATA_HOME=/var/lib/caddy/data XDG_CONFIG_HOME=/var/lib/caddy/config \
  /usr/local/bin/caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
systemctl daemon-reload
if "$start"; then
  runuser -u flyreward -- /usr/local/libexec/flyreward-check-cache
  systemctl enable --now flyreward.service caddy.service
else
  echo 'Installed; services remain stopped. Extract the reviewed compiled cache under /var/lib/flyreward/data.'
  echo 'Then: chown -R flyreward:flyreward /var/lib/flyreward/data'
  echo 'Then: systemctl enable --now flyreward caddy'
fi
echo 'Readiness: curl --fail --silent --show-error http://127.0.0.1:8000/api/health'
printf 'Public viewer after startup: https://%s/viewer/live.html\n' "$public_ip"
echo 'Logs: journalctl -u flyreward -u caddy -n 100 --no-pager'
echo 'The startup service downloads pinned raw data. A proxy response alone does not establish simulation readiness.'
