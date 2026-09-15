#!/usr/bin/env bash
# Prepare/review first; --apply permits the explicitly described OCI creates.
# Keep the state JSON: it records create intents before requests to prevent duplicates.
# Requires the reviewed JSON output from discover-cloud.sh (at most 30 minutes old).
# No paid shapes, capacity reservations, NAT gateway, block disks or load balancers.
# CLI references: https://docs.oracle.com/en-us/iaas/tools/oci-cli/latest/oci_cli_docs/
# Free limits: https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm
set -euo pipefail
python3 - "$@" <<'PY'
import argparse
import base64
import datetime
import fcntl
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

p = argparse.ArgumentParser(description="Review or provision one Always Free A1 fly-model VM.")
p.add_argument("--tenancy-id", required=True)
p.add_argument("--availability-domain", required=True)
p.add_argument("--image-id", required=True, help="Reviewed official Ubuntu 24.04 aarch64 platform image OCID")
p.add_argument("--ssh-public-key-file", required=True, type=Path)
p.add_argument("--ssh-cidr", required=True, help="Explicit IPv4 source CIDR allowed to reach SSH")
p.add_argument("--inventory", required=True, type=Path, help="Reviewed output of discover-cloud.sh")
p.add_argument("--name", default="fly-reward-lab")
p.add_argument("--state-file", type=Path, default=Path("fly-provision-state.json"))
p.add_argument("--apply", action="store_true")
a = p.parse_args()
REGION, SHAPE, TAG = "us-sanjose-1", "VM.Standard.A1.Flex", "flyreward-deployment"
if a.tenancy_id != os.environ.get("OCI_TENANCY"):
    p.error("--tenancy-id must match the authenticated Cloud Shell OCI_TENANCY")
if not re.fullmatch(r"[a-z][a-z0-9-]{2,39}", a.name):
    p.error("--name must contain 3-40 lowercase letters/digits/hyphens, starting with a letter")
try:
    ssh_cidr = str(ipaddress.IPv4Network(a.ssh_cidr, strict=True))
    public_key = a.ssh_public_key_file.read_text().strip()
    parts = public_key.split()
    if len(public_key.splitlines()) != 1 or len(parts) < 2 or parts[0] not in ("ssh-ed25519", "ssh-rsa"):
        raise ValueError("Use one OpenSSH public key, not a PEM/private key")
    base64.b64decode(parts[1], validate=True)
    inventory = json.loads(a.inventory.read_text())
except (OSError, ValueError) as exc:
    p.error(str(exc))
if (inventory.get("tenancy", {}).get("id") != a.tenancy_id
        or inventory.get("tenancy", {}).get("name") != "duncanscottwilson"
        or inventory.get("target_region") != REGION
        or not inventory.get("inventory_totals", {}).get("complete_for_accessible_compartments")):
    p.error("Inventory must confirm the expected tenancy/region and a complete accessible-resource scan")
if not any(row.get("is-home-region") and row.get("region-name") == REGION
           for row in inventory.get("region_subscriptions") or []):
    p.error("Inventory does not confirm us-sanjose-1 as the home region")
if a.availability_domain not in {row["name"] for row in inventory.get("availability_domains") or []}:
    p.error("Availability domain is absent from the verified inventory")
if a.image_id not in {row["id"] for row in inventory.get("ubuntu_24_04_aarch64_image_candidates") or []}:
    p.error("Image is absent from the reviewed A1-compatible Ubuntu 24.04 ARM candidates")

def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

plan = {"tenancy_id": a.tenancy_id, "region": REGION, "availability_domain": a.availability_domain,
        "image_id": a.image_id, "name": a.name, "shape": SHAPE,
        "attempts": [{"ocpus": 2, "memoryInGBs": 12}, {"ocpus": 1, "memoryInGBs": 6}],
        "fallback_condition": "Only a definite out-of-host-capacity error, after confirming no new instance or boot disk",
        "source_details": {"sourceType": "image", "imageId": a.image_id,
                           "bootVolumeSizeInGBs": 50, "bootVolumeVpusPerGB": 10},
        "vcn_cidr": "10.83.0.0/16", "subnet_cidr": "10.83.1.0/24",
        "ingress_tcp": [{"port": 22, "source": ssh_cidr}, {"port": 80, "source": "0.0.0.0/0"},
                        {"port": 443, "source": "0.0.0.0/0"}],
        "ingress_icmp": "IPv4 fragmentation-needed (type 3/code 4)",
        "egress": "Stateful IPv4 to 0.0.0.0/0", "public_ip": "Ephemeral primary VNIC address",
        "ssh_public_key_sha256": hashlib.sha256(public_key.encode()).hexdigest(),
        "instance_metadata": "ssh_authorized_keys only; no startup script",
        "always_free_allocation_ceiling": {"a1_ocpus_total": 2, "a1_memory_gbs_total": 12, "boot_plus_block_gbs_total": 200},
        "state_file": str(a.state_file.resolve())}
print(json.dumps({"mode": "apply" if a.apply else "review_only", "plan": plan}, indent=2), flush=True)
if not a.apply:
    sys.exit(0)
observed = datetime.datetime.fromisoformat(inventory["observed_at_utc"])
age = (datetime.datetime.now(datetime.timezone.utc) - observed).total_seconds()
if not 0 <= age <= 1800:
    sys.exit("Refresh and review discover-cloud.sh inventory; its maximum age is 30 minutes.")
os.umask(0o077)
a.state_file.parent.mkdir(parents=True, exist_ok=True)
lock = a.state_file.with_suffix(a.state_file.suffix + ".lock").open("a")
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    sys.exit("Another provisioning process owns this state file")
digest = hashlib.sha256(encode(plan).encode()).hexdigest()
if a.state_file.exists():
    state = json.loads(a.state_file.read_text())
    if state.get("plan_sha256") != digest:
        sys.exit("Plan differs from existing state; do not reuse or delete the journal to bypass this check")
else:
    state = {"plan_sha256": digest, "deployment_id": str(uuid.uuid4()), "resources": {}, "attempts": []}

def save():
    temporary = a.state_file.with_suffix(a.state_file.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(state, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(a.state_file)
    directory = os.open(str(a.state_file.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)

save()
tags = {TAG: state["deployment_id"]}

class CliError(Exception):
    def __init__(self, label, code=None, message=None, status=None):
        super().__init__(label + ": " + (code or "CLI/transport failure"))
        self.code, self.message, self.status = code, message, status

def cli(label, *args):
    print(label + "...", file=sys.stderr, flush=True)
    command = ["oci", "--region", REGION, "--output", "json", "--no-retry",
               "--connection-timeout", "10", "--read-timeout", "60", *args]
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=90)
    except subprocess.TimeoutExpired:
        raise CliError(label, "command_timeout") from None
    if result.returncode:
        # Parse service errors without echoing raw diagnostics or request metadata.
        error = {}
        start = result.stderr.find("{")
        if start >= 0:
            try:
                error, _ = json.JSONDecoder().raw_decode(result.stderr[start:])
            except ValueError:
                pass
        raise CliError(label, error.get("code"), error.get("message"), error.get("status"))
    if args[2] == "list" and not result.stdout.strip():
        # OCI CLI can emit no JSON at all for a successful empty list.
        return []
    try:
        return json.loads(result.stdout)["data"]
    except (ValueError, KeyError, TypeError):
        raise CliError(label, "unexpected_response") from None

def listed(group, *args):
    return cli("List " + group, "network", group, "list", "--compartment-id", a.tenancy_id, "--all", *args)

def wait_resource(label, group, id_flag, resource_id, target="AVAILABLE", timeout=180):
    deadline = time.monotonic() + timeout
    while True:
        row = cli("Read " + label, *group, "get", id_flag, resource_id)
        status = row.get("lifecycle-state")
        if status == target:
            return row
        if status in ("TERMINATED", "TERMINATING", "FAULTY", "DELETED"):
            raise RuntimeError(label + " entered " + str(status) + "; preserve state and inspect")
        if time.monotonic() >= deadline:
            raise RuntimeError(label + " still pending; rerun with this same journal to resume reading it")
        time.sleep(5)

def network_resource(key, group, id_flag, *create_args, vcn_id=None):
    saved = state["resources"].get(key)
    if saved and saved.get("id"):
        return wait_resource(key, ("network", group), id_flag, saved["id"])["id"]
    name = a.name + "-" + key
    filters = ["--display-name", name]
    if vcn_id:
        filters += ["--vcn-id", vcn_id]
    matches = [row for row in listed(group, *filters) if row.get("lifecycle-state") not in ("TERMINATED", "DELETED")]
    if matches:
        if len(matches) != 1 or matches[0].get("freeform-tags", {}).get(TAG) != state["deployment_id"]:
            raise RuntimeError("Conflicting " + name + "; existing resources are never adopted by name alone")
        resource_id = matches[0]["id"]
    else:
        if saved:
            raise RuntimeError("Unresolved create intent for " + key + "; inspect it before retrying, even if list is empty")
        state["resources"][key] = {"status": "create_pending", "display_name": name}
        save()
        row = cli("Create " + key, "network", group, "create", "--compartment-id", a.tenancy_id,
                  "--display-name", name, "--freeform-tags", encode(tags), *create_args)
        resource_id = row["id"]
    state["resources"][key] = {"id": resource_id, "display_name": name}
    save()
    return wait_resource(key, ("network", group), id_flag, resource_id)["id"]

def current_allocations():
    instances, boots, blocks = [], [], []
    # ANY enumerates the entire hierarchy. A resource-read permission failure in
    # any compartment stops provisioning instead of treating hidden use as zero.
    children = cli("Refresh all active compartments", "iam", "compartment", "list", "--compartment-id", a.tenancy_id,
                   "--access-level", "ANY", "--compartment-id-in-subtree", "true", "--lifecycle-state", "ACTIVE", "--all")
    compartments = [{"id": a.tenancy_id}] + [row for row in children if row["id"] != a.tenancy_id]
    for compartment in compartments:
        cid = compartment["id"]
        instances.extend(cli("Check current instances", "compute", "instance", "list", "--compartment-id", cid, "--all"))
        blocks.extend(cli("Check current block disks", "bv", "volume", "list", "--compartment-id", cid, "--all"))
        for ad in inventory["availability_domains"]:
            boots.extend(cli("Check current boot disks", "bv", "boot-volume", "list", "--compartment-id", cid,
                             "--availability-domain", ad["name"], "--all"))
    active = lambda rows: [row for row in rows if row.get("lifecycle-state") != "TERMINATED"]
    return active(instances), active(boots), active(blocks)

def free_guard(cpu, memory, allocation):
    instances, boots, blocks = allocation
    a1 = [row["shape-config"] for row in instances if row.get("shape") == SHAPE]
    if (sum(row["ocpus"] for row in a1) + cpu > 2
            or sum(row["memory-in-gbs"] for row in a1) + memory > 12
            or sum(row["size-in-gbs"] for row in boots + blocks) + 50 > 200):
        raise RuntimeError("Requested VM would exceed the current Always Free allocation ceiling; no launch")

try:
    # Confirm input identity/image again with get/list APIs before any create.
    tenancy = cli("Confirm tenancy", "iam", "tenancy", "get", "--tenancy-id", a.tenancy_id)
    if tenancy.get("name") != "duncanscottwilson":
        raise RuntimeError("Unexpected tenancy")
    image = cli("Confirm platform image", "compute", "image", "get", "--image-id", a.image_id)
    if (image.get("compartment-id") is not None or image.get("lifecycle-state") != "AVAILABLE"
            or not str(image.get("display-name", "")).startswith("Canonical-Ubuntu-24.04-")
            or "aarch64" not in str(image.get("display-name", "")).lower()
            or image.get("operating-system") != "Canonical Ubuntu"):
        raise RuntimeError("Expected an available official Ubuntu 24.04 aarch64 platform image")
    shapes = cli("Confirm A1/image/AD compatibility", "compute", "shape", "list", "--compartment-id", a.tenancy_id,
                 "--availability-domain", a.availability_domain, "--image-id", a.image_id, "--shape", SHAPE, "--all")
    if not any(row.get("shape") == SHAPE for row in shapes):
        raise RuntimeError("A1 is not eligible with this image/availability domain")

    instance_record = state["resources"].get("instance")
    if not instance_record:
        if len(state["attempts"]) >= len(plan["attempts"]):
            raise RuntimeError("Both permitted launch attempts are exhausted; no further cloud creates")
        allocation = current_allocations()
        next_config = plan["attempts"][len(state["attempts"])]
        free_guard(next_config["ocpus"], next_config["memoryInGBs"], allocation)
    vcn = network_resource("vcn", "vcn", "--vcn-id", "--cidr-block", plan["vcn_cidr"])
    igw = network_resource("igw", "internet-gateway", "--ig-id", "--vcn-id", vcn, "--is-enabled", "true", vcn_id=vcn)
    routes = [{"destination": "0.0.0.0/0", "destinationType": "CIDR_BLOCK", "networkEntityId": igw}]
    route = network_resource("route", "route-table", "--rt-id", "--vcn-id", vcn, "--route-rules", encode(routes), vcn_id=vcn)
    ingress = [{"protocol": "6", "source": row["source"], "sourceType": "CIDR_BLOCK", "isStateless": False,
                "tcpOptions": {"destinationPortRange": {"min": row["port"], "max": row["port"]}}}
               for row in plan["ingress_tcp"]]
    ingress.append({"protocol": "1", "source": "0.0.0.0/0", "sourceType": "CIDR_BLOCK", "isStateless": False,
                    "icmpOptions": {"type": 3, "code": 4}})
    egress = [{"protocol": "all", "destination": "0.0.0.0/0", "destinationType": "CIDR_BLOCK", "isStateless": False}]
    security = network_resource("security", "security-list", "--security-list-id", "--vcn-id", vcn,
                                "--ingress-security-rules", encode(ingress), "--egress-security-rules", encode(egress), vcn_id=vcn)
    subnet = network_resource("subnet", "subnet", "--subnet-id", "--vcn-id", vcn, "--cidr-block", plan["subnet_cidr"],
                              "--route-table-id", route, "--security-list-ids", encode([security]),
                              "--prohibit-public-ip-on-vnic", "false", vcn_id=vcn)

    if instance_record and not instance_record.get("id"):
        matches = cli("Reconcile pending instance", "compute", "instance", "list", "--compartment-id", a.tenancy_id,
                      "--display-name", a.name, "--all")
        matches = [row for row in matches if row.get("freeform-tags", {}).get(TAG) == state["deployment_id"]]
        if len(matches) != 1:
            raise RuntimeError("Pending instance has no unique reconciliation; do not retry launch or remove the journal")
        instance_record["id"] = matches[0]["id"]
        instance_record.pop("status", None)
        if state["attempts"]:
            state["attempts"][-1]["outcome"] = "created_reconciled"
        save()
    if not instance_record:
        if state["attempts"] and not all(row.get("outcome") == "capacity_error" for row in state["attempts"]):
            raise RuntimeError("Earlier launch outcome is unresolved; no additional attempt")
        for shape_config in plan["attempts"][len(state["attempts"]):]:
            before = current_allocations()
            free_guard(shape_config["ocpus"], shape_config["memoryInGBs"], before)
            same_name = [row for row in before[0] if row.get("display-name") == a.name]
            if same_name:
                raise RuntimeError("Instance name already exists; no duplicate launch")
            attempt = {"shape_config": shape_config, "outcome": "pending"}
            state["attempts"].append(attempt)
            state["resources"]["instance"] = {"status": "create_pending"}
            save()
            try:
                instance = cli("Launch A1 " + str(shape_config["ocpus"]) + " OCPU", "compute", "instance", "launch",
                               "--compartment-id", a.tenancy_id, "--availability-domain", a.availability_domain,
                               "--display-name", a.name, "--shape", SHAPE, "--shape-config", encode(shape_config),
                               "--source-details", encode(plan["source_details"]), "--subnet-id", subnet,
                               "--assign-public-ip", "true", "--ssh-authorized-keys-file", str(a.ssh_public_key_file.resolve()),
                               "--freeform-tags", encode(tags), "--capacity-reservation-id", "",
                               "--instance-options", encode({"areLegacyImdsEndpointsDisabled": True}))
            except CliError as exc:
                capacity_error = (exc.status in (400, 409, 500, 503)
                                  and str(exc.message or "").strip().lower().rstrip(".") == "out of host capacity")
                if not capacity_error:
                    raise
                after = current_allocations()
                if any({row["id"] for row in after[i]} - {row["id"] for row in before[i]} for i in (0, 1)):
                    raise RuntimeError("Capacity error accompanied a new instance/boot disk; preserve journal and inspect")
                attempt["outcome"] = "capacity_error"
                state["resources"].pop("instance")
                save()
                continue
            attempt["outcome"] = "created"
            instance_record = {"id": instance["id"], "shape_config": shape_config}
            state["resources"]["instance"] = instance_record
            save()
            break
        if not instance_record:
            raise RuntimeError("Both permitted A1 sizes returned capacity errors; no further launch attempts")
    instance = wait_resource("instance", ("compute", "instance"), "--instance-id", instance_record["id"], "RUNNING", 600)
    attachments = cli("Read primary VNIC attachment", "compute", "vnic-attachment", "list",
                      "--compartment-id", a.tenancy_id, "--instance-id", instance["id"], "--all")
    vnics = [cli("Read VNIC address", "network", "vnic", "get", "--vnic-id", row["vnic-id"])
             for row in attachments if row.get("lifecycle-state") == "ATTACHED"]
    primary = [row for row in vnics if row.get("is-primary")]
    print(json.dumps({"status": "RUNNING", "region": REGION, "instance_id": instance["id"],
                      "shape": instance["shape"], "shape_config": instance.get("shape-config"),
                      "public_ip": primary[0].get("public-ip") if len(primary) == 1 else None,
                      "ssh_user": "ubuntu", "resources": state["resources"], "state_file": str(a.state_file)}, indent=2))
except (CliError, RuntimeError, KeyError, TypeError) as exc:
    print(json.dumps({"status": "STOPPED_FOR_REVIEW", "reason": str(exc),
                      "state_file": str(a.state_file), "recorded_resources": state["resources"],
                      "attempts": state["attempts"]}, indent=2))
    sys.exit(2)
PY
