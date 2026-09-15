#!/usr/bin/env python3
"""Launch one free E2 micro using an existing network journal; run only in Cloud Shell."""
import argparse, base64, fcntl, hashlib, ipaddress, json, os, re, subprocess, sys, time, uuid
from pathlib import Path

REGION, AD, SHAPE = "us-sanjose-1", "Cyim:US-SANJOSE-1-AD-1", "VM.Standard.E2.1.Micro"
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--image-id", required=True)
p.add_argument("--ssh-public-key-file", required=True, type=Path)
p.add_argument("--ssh-cidr", help="Optional validation only; existing network ingress is unchanged")
p.add_argument("--state-file", type=Path, default=Path("fly-provision-state.json"))
a = p.parse_args()

class CliError(Exception):
    def __init__(self, code, capacity=False):
        super().__init__(code); self.code, self.capacity = code, capacity

def cli(*args):
    print("OCI " + " ".join(args[:3]), file=sys.stderr, flush=True)
    command = ["oci", "--region", REGION, "--output", "json", "--no-retry", "--connection-timeout", "10", "--read-timeout", "45", *args]
    try: result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired: raise CliError("command_timeout") from None
    if result.returncode:
        match = re.search(r'"code"\s*:\s*"([A-Za-z0-9_.-]+)"', result.stderr)
        raise CliError(match.group(1) if match else "cli_or_transport_error", "out of host capacity" in result.stderr.lower())
    if not result.stdout.strip() and args[2] in ("list", "list-vnics"): return []
    try: return json.loads(result.stdout)["data"]
    except (ValueError, KeyError, TypeError): raise CliError("unexpected_response") from None

def require(condition, message):
    if not condition: raise ValueError(message)

def save():
    temporary = a.state_file.with_suffix(a.state_file.suffix + ".amd.tmp")
    with temporary.open("w") as handle:
        json.dump(state, handle, indent=2, sort_keys=True); handle.flush(); os.fsync(handle.fileno())
    temporary.replace(a.state_file)
    directory = os.open(str(a.state_file.parent), os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)

try:
    tenancy = os.environ.get("OCI_TENANCY", "")
    require(tenancy.startswith("ocid1.tenancy."), "Run inside authenticated OCI Cloud Shell with OCI_TENANCY")
    if a.ssh_cidr: ipaddress.IPv4Network(a.ssh_cidr, strict=True)
    public_key = a.ssh_public_key_file.read_text().strip(); parts = public_key.split()
    require(len(public_key.splitlines()) == 1 and len(parts) >= 2 and parts[0] in ("ssh-ed25519", "ssh-rsa"), "Supply an OpenSSH public key")
    base64.b64decode(parts[1], validate=True)
    os.umask(0o077)
    lock = a.state_file.with_suffix(a.state_file.suffix + ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads(a.state_file.read_text())
    subnet_id = state["resources"]["subnet"]["id"]
    plan = {"region": REGION, "ad": AD, "shape": SHAPE, "image_id": a.image_id, "tenancy": tenancy,
            "subnet_id": subnet_id, "boot_gb": 50, "vpus_per_gb": 10, "key_sha256": hashlib.sha256(public_key.encode()).hexdigest()}
    attempt = state.get("amd_launch")
    if attempt:
        require(attempt.get("plan") == plan, "Existing AMD launch plan differs; inspect journal")
        require(attempt.get("instance_id"), "Prior AMD launch intent is unresolved or failed; no automatic retry")
    else:
        identity = cli("iam", "tenancy", "get", "--tenancy-id", tenancy)
        require(identity.get("name") == "duncanscottwilson", "Unexpected tenancy")
        regions = cli("iam", "region-subscription", "list", "--tenancy-id", tenancy, "--all")
        require(any(r.get("region-name") == REGION and r.get("is-home-region") for r in regions), "Expected Always Free home region not confirmed")
        ads = cli("iam", "availability-domain", "list", "--compartment-id", tenancy, "--all")
        require(AD in {x["name"] for x in ads}, "Fixed AMD availability domain is unavailable")
        image = cli("compute", "image", "get", "--image-id", a.image_id)
        name = str(image.get("display-name", ""))
        require(image.get("compartment-id") is None and image.get("lifecycle-state") == "AVAILABLE"
                and image.get("operating-system") == "Canonical Ubuntu" and name.startswith("Canonical-Ubuntu-24.04-Minimal-")
                and not any(x in name.lower() for x in ("aarch64", "arm64")), "Expected official Ubuntu 24.04 Minimal AMD64 platform image")
        shapes = cli("compute", "shape", "list", "--compartment-id", tenancy, "--availability-domain", AD, "--image-id", a.image_id, "--shape", SHAPE, "--all")
        require(any(x.get("shape") == SHAPE for x in shapes), "Image is incompatible with E2 micro in this AD")
        subnet = cli("network", "subnet", "get", "--subnet-id", subnet_id)
        require(subnet.get("lifecycle-state") == "AVAILABLE" and not subnet.get("prohibit-public-ip-on-vnic"), "Journal subnet is unavailable or prohibits public IPs")
        children = cli("iam", "compartment", "list", "--compartment-id", tenancy, "--access-level", "ANY", "--compartment-id-in-subtree", "true", "--lifecycle-state", "ACTIVE", "--all")
        compartments = {tenancy, *(row["id"] for row in children)}
        instances, disks = [], []
        for cid in sorted(compartments):
            instances += cli("compute", "instance", "list", "--compartment-id", cid, "--all")
            disks += cli("bv", "volume", "list", "--compartment-id", cid, "--all")
            for domain in ads:
                disks += cli("bv", "boot-volume", "list", "--compartment-id", cid, "--availability-domain", domain["name"], "--all")
        active = [x for x in instances if x.get("lifecycle-state") != "TERMINATED"]
        count = sum(x.get("shape") == SHAPE for x in active)
        disk_gb = sum(float(x["size-in-gbs"]) for x in disks if x.get("lifecycle-state") != "TERMINATED")
        require(count <= 1 and disk_gb + 50 <= 200, "Launch would exceed two free E2 micro instances or 200 GB boot/block storage")
        require(not any(x.get("display-name") == "fly-reward-lab-amd" for x in active), "Named AMD instance already exists; inspect before launching")
        attempt = {"intent_id": str(uuid.uuid4()), "status": "intent_recorded", "plan": plan,
                   "guard": {"compartments": len(compartments), "existing_e2": count, "existing_disk_gb": disk_gb}, "created_at": time.time()}
        state["amd_launch"] = attempt; save()  # Durable intent precedes the single create request.
        try:
            instance = cli("compute", "instance", "launch", "--compartment-id", tenancy, "--availability-domain", AD,
                           "--display-name", "fly-reward-lab-amd", "--shape", SHAPE, "--subnet-id", subnet_id, "--assign-public-ip", "true",
                           "--source-details", json.dumps({"sourceType": "image", "imageId": a.image_id, "bootVolumeSizeInGBs": 50, "bootVolumeVpusPerGB": 10}),
                           "--ssh-authorized-keys-file", str(a.ssh_public_key_file.resolve()), "--freeform-tags", json.dumps({"flyreward-amd-intent": attempt["intent_id"]}))
            require(str(instance.get("id", "")).startswith("ocid1.instance."), "Launch response has no instance ID; preserve intent")
            attempt.update(instance_id=instance["id"], status="launched"); save()
        except (CliError, ValueError) as exc:
            attempt.update(status="capacity_error" if isinstance(exc, CliError) and exc.capacity else "ambiguous_launch_error", error=str(exc)); save()
            raise
    # After an ID is recorded, all subsequent commands (including reruns) only read.
    deadline = time.monotonic() + 300
    while True:
        instance = cli("compute", "instance", "get", "--instance-id", attempt["instance_id"])
        status = instance.get("lifecycle-state")
        require(status not in ("TERMINATED", "TERMINATING"), "Instance terminated; preserve journal and inspect")
        vnics = cli("compute", "instance", "list-vnics", "--instance-id", attempt["instance_id"], "--all")
        public_ip = next((v["public-ip"] for v in vnics if v.get("is-primary") and v.get("public-ip")), None)
        if public_ip: ipaddress.IPv4Address(public_ip)
        if status == "RUNNING" and public_ip:
            attempt.update(status="running", public_ip=public_ip); save()
            print(json.dumps({"instance_id": attempt["instance_id"], "public_ip": public_ip, "shape": SHAPE, "state_file": str(a.state_file)})); break
        if time.monotonic() >= deadline:
            print(json.dumps({"instance_id": attempt["instance_id"], "status": status, "public_ip": public_ip, "next": "Rerun the same command for read-only status; no new launch"})); break
        time.sleep(5)
except (OSError, ValueError, KeyError, TypeError, CliError) as exc:
    sys.exit("Stopped: " + str(exc) + ". Preserve the journal; no launch retry was made.")
