#!/usr/bin/env bash
# Run on the allocated Ubuntu 24.04 VM as root, after allowing public TCP 80/443.
# Official repository instructions: https://docs.docker.com/engine/install/ubuntu/
# Usage: sudo bash install-backend.sh --ip ACTUAL_PUBLIC_IPV4
# --check validates the local artifact and arguments without installation/network.
set -Eeuo pipefail
EXPECTED_ARCHIVE_SHA256="131d95c4c94252e5c763f2cb90f37088113c80a047f556bcc62f650739a77d9c"
INSTALL_DIR=/opt/fly-reward-lab
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ARCHIVE="$SCRIPT_DIR/fly-backend.tar.gz"
PUBLIC_IP=""
CHECK_ONLY=0
usage() {
  echo 'Usage: install-backend.sh --ip ACTUAL_PUBLIC_IPV4 [--archive /path/fly-backend.tar.gz] [--check]'
}
while (($#)); do
  case "$1" in
    --ip) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; PUBLIC_IP=$2; shift 2 ;;
    --archive) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; ARCHIVE=$2; shift 2 ;;
    --check) CHECK_ONLY=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
[[ -n "$PUBLIC_IP" ]] || { echo 'An explicit public IPv4 is required.' >&2; exit 2; }
command -v python3 >/dev/null || { echo 'Python 3 is required (included in Ubuntu 24.04).' >&2; exit 2; }
# Validate before any apt, Docker, network, or destination filesystem changes.
python3 - "$PUBLIC_IP" "$ARCHIVE" "$EXPECTED_ARCHIVE_SHA256" <<'PY'
import hashlib, ipaddress, pathlib, re, sys, tarfile
try:
    address = ipaddress.IPv4Address(sys.argv[1])
    if not address.is_global:
        raise ValueError('The IPv4 must be the VM public address, not a private or reserved address')
    path = pathlib.Path(sys.argv[2])
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != sys.argv[3]:
        raise ValueError('Archive is missing or does not match this installer SHA256')
    exact = {'Dockerfile', '.dockerignore', 'pyproject.toml', 'requirements-live-lock.txt'}
    exact |= {'visualizer/' + n for n in ('live.html', 'live.js', 'live-room.js', 'live.css', 'live-config.js', 'style.css', 'fly.js')}
    exact |= {'deploy/' + n for n in ('compose.yaml', 'Caddyfile', 'compose.ip.yaml', 'Caddyfile.ip', 'README.md')}
    found = set()
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive.getmembers():
            if not member.name.startswith('fly-reward-lab/') or not member.isfile():
                raise ValueError('Unexpected archive type or root')
            name = member.name.removeprefix('fly-reward-lab/')
            if name not in exact and not re.fullmatch(r'flyreward/[A-Za-z_][A-Za-z_0-9]*\.py', name):
                raise ValueError('Unexpected package member: ' + name)
            if name in found:
                raise ValueError('Duplicate package member: ' + name)
            found.add(name)
    if not exact.issubset(found) or not {'flyreward/server.py', 'flyreward/live_model.py', 'flyreward/__init__.py'}.issubset(found):
        raise ValueError('Archive is missing required runtime files')
    print(f'Package verified: {len(found)} files; public endpoint https://{address}')
except (OSError, ValueError, tarfile.TarError) as exc:
    raise SystemExit('Preflight failed: ' + str(exc))
PY
if ((CHECK_ONLY)); then
  echo 'Check complete; no host changes or network requests were made.'
  exit 0
fi
[[ $(id -u) -eq 0 ]] || { echo 'Run this installer with sudo on the Ubuntu VM.' >&2; exit 2; }
[[ -r /etc/os-release ]] || { echo 'Cannot identify host OS.' >&2; exit 2; }
# shellcheck disable=SC1091
. /etc/os-release
[[ ${ID:-} == ubuntu && ${VERSION_ID:-} == 24.04 ]] || { echo 'This installer requires Ubuntu 24.04.' >&2; exit 2; }
DOCKER_ARCH=$(dpkg --print-architecture)
[[ $DOCKER_ARCH == arm64 || $DOCKER_ARCH == amd64 ]] || { echo 'This package targets arm64 or amd64.' >&2; exit 2; }
[[ ! -L "$INSTALL_DIR" ]] || { echo 'Installation directory must not be a symlink.' >&2; exit 2; }
trap 'echo "Installation stopped at line $LINENO; existing Docker volumes were not removed." >&2' ERR
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a
APT=(apt-get -o DPkg::Lock::Timeout=180)

# Remove distribution runtime packages which conflict with Docker's official
# packages. This never purges Docker storage or persistent application volumes.
CONFLICTS=()
for package in docker.io docker-compose docker-compose-v2 docker-doc docker-buildx podman-docker containerd runc; do
  if [[ $(dpkg-query -W -f='${Status}' "$package" 2>/dev/null || true) == 'install ok installed' ]]; then
    CONFLICTS+=("$package")
  fi
done
if ((${#CONFLICTS[@]})); then "${APT[@]}" remove -y "${CONFLICTS[@]}"; fi
"${APT[@]}" update
"${APT[@]}" install -y --no-install-recommends ca-certificates curl
install -m 0755 -d /etc/apt/keyrings
curl --fail --silent --show-error --location --retry 5 --retry-delay 3 \
  https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
cat > /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: noble
Components: stable
Architectures: $DOCKER_ARCH
Signed-By: /etc/apt/keyrings/docker.asc
EOF
"${APT[@]}" update
"${APT[@]}" install -y --no-install-recommends docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
systemctl enable --now docker
docker version
docker compose version

install -d -m 0755 "$INSTALL_DIR"
STAGE=$(mktemp -d /opt/fly-backend-install.XXXXXX)
trap 'rm -rf -- "$STAGE"' EXIT
tar -xzf "$ARCHIVE" -C "$STAGE" --no-same-owner
cp -a "$STAGE/fly-reward-lab/." "$INSTALL_DIR/"
DEPLOY_DIR="$INSTALL_DIR/deploy"
umask 022
printf 'FLY_API_DOMAIN=%s\nCOMPOSE_FILE=compose.yaml:compose.ip.yaml\n' "$PUBLIC_IP" > "$DEPLOY_DIR/actualIP.env"
cp "$DEPLOY_DIR/actualIP.env" "$DEPLOY_DIR/.env"
COMPOSE=(docker compose --project-directory "$DEPLOY_DIR" --env-file "$DEPLOY_DIR/actualIP.env" -f "$DEPLOY_DIR/compose.yaml" -f "$DEPLOY_DIR/compose.ip.yaml")
"${COMPOSE[@]}" config --quiet
"${COMPOSE[@]}" pull caddy state-init
# Validate the actual architecture's Caddy image before starting the public proxy.
"${COMPOSE[@]}" run --rm --no-deps --entrypoint caddy caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
"${COMPOSE[@]}" up -d --build
"${COMPOSE[@]}" ps
cat <<EOF
Backend containers started. First startup downloads and loads the real graph.
This installer has not asserted simulation readiness or successful certificate issuance.
Cloud rules must permit inbound TCP 80 and 443; no DNS record is required.
Check logs: cd $DEPLOY_DIR && docker compose --env-file actualIP.env logs --tail=100 backend caddy
Check readiness: curl --fail https://$PUBLIC_IP/api/health
Check advancement twice: curl --fail https://$PUBLIC_IP/api/state
Viewer: https://$PUBLIC_IP/viewer/live.html
Persistent model/checkpoint and certificate volumes remain managed by Compose.
EOF
