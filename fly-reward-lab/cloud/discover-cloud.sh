#!/usr/bin/env bash
# Read-only OCI discovery. Run in the already authenticated OCI Cloud Shell.
# Emits one JSON summary on stdout and progress on stderr. No cloud writes.
# Does not print credentials, instance metadata/user-data, tags, or CLI config.
# Env: https://blogs.oracle.com/autonomous-ai-database/oci-cli-commands-for-newbies
# CLI: https://docs.oracle.com/en-us/iaas/tools/oci-cli/latest/oci_cli_docs/
# Images: https://ubuntu.com/docs/oracle/oracle-how-to/find-ubuntu-images/
set -euo pipefail
python3 - <<'PY'
import datetime
import json
import os
import re
import shutil
import subprocess
import sys

REGION = "us-sanjose-1"
EXPECTED_TENANCY = "duncanscottwilson"
SHAPE = "VM.Standard.A1.Flex"
tenancy_id = os.environ.get("OCI_TENANCY", "")
if not tenancy_id.startswith("ocid1.tenancy."):
    sys.exit("OCI_TENANCY is missing or invalid; run this in authenticated OCI Cloud Shell.")
if not shutil.which("oci"):
    sys.exit("OCI CLI is missing; run this in authenticated OCI Cloud Shell.")

errors = []
summary = {
    "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "target_region": REGION,
    "cloud_shell_region": os.environ.get("OCI_REGION"),
    "errors": errors,
}


def emit():
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


def query(label, *args):
    # Only these explicitly audited read operations are permitted here.
    allowed = {
        ("iam", "tenancy", "get"), ("iam", "region-subscription", "list"),
        ("iam", "availability-domain", "list"), ("iam", "compartment", "list"),
        ("compute", "instance", "list"), ("compute", "shape", "list"),
        ("compute", "image", "list"), ("bv", "boot-volume", "list"),
        ("bv", "volume", "list"), ("limits", "value", "list"),
        ("limits", "resource-availability", "get"),
    }
    if tuple(args[:3]) not in allowed:
        raise ValueError("Discovery attempted an operation outside its read-only allowlist")
    print("Checking " + label + "...", file=sys.stderr, flush=True)
    command = ["oci", "--region", REGION, "--output", "json", "--max-retries", "1",
               "--connection-timeout", "10", "--read-timeout", "45", *args]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=100)
    except subprocess.TimeoutExpired:
        errors.append({"query": label, "error": "command_timeout"})
        return None
    if result.returncode:
        # Never echo raw CLI diagnostic output: it can contain request details.
        code = re.search(r'"code"\s*:\s*"([A-Za-z0-9_.-]+)"', result.stderr)
        errors.append({"query": label, "exit_status": result.returncode,
                       "service_code": code.group(1) if code else None})
        return None
    if args[2] == "list" and not result.stdout.strip():
        # OCI CLI can emit no JSON at all for a successful empty list.
        return []
    try:
        return json.loads(result.stdout)["data"]
    except (ValueError, KeyError, TypeError):
        errors.append({"query": label, "error": "unexpected_json_response"})
        return None


def fields(item, *names):
    return {name: item.get(name) for name in names}


tenancy = query("tenancy identity", "iam", "tenancy", "get", "--tenancy-id", tenancy_id)
if not tenancy:
    emit()
    sys.exit(2)
summary["tenancy"] = fields(tenancy, "id", "name", "home-region-key")
if tenancy.get("name") != EXPECTED_TENANCY:
    errors.append({"query": "tenancy identity", "error": "unexpected_tenancy_name"})
    emit()
    sys.exit(2)
subscriptions = query("region subscriptions", "iam", "region-subscription", "list",
                      "--tenancy-id", tenancy_id, "--all")
summary["region_subscriptions"] = None if subscriptions is None else [
    fields(row, "region-name", "region-key", "is-home-region", "status") for row in subscriptions]
if subscriptions is None or not any(row.get("region-name") == REGION and row.get("is-home-region")
                                    for row in subscriptions):
    errors.append({"query": "region subscriptions", "error": "expected_home_region_not_confirmed"})
    emit()
    sys.exit(2)

ads = query("availability domains", "iam", "availability-domain", "list",
            "--compartment-id", tenancy_id, "--all")
summary["availability_domains"] = None if ads is None else [fields(row, "id", "name") for row in ads]
children = query("accessible active compartments", "iam", "compartment", "list",
                 "--compartment-id", tenancy_id, "--compartment-id-in-subtree", "true",
                 "--access-level", "ACCESSIBLE", "--lifecycle-state", "ACTIVE", "--all")
compartments = [{"id": tenancy_id, "name": EXPECTED_TENANCY, "is_root": True}]
compartments.extend({**fields(row, "id", "name", "compartment-id"), "is_root": False}
                    for row in (children or []) if row.get("id") != tenancy_id)
summary["accessible_compartments"] = compartments
inventory = []
summary["inventory_by_compartment"] = inventory
inventory_complete = children is not None and bool(ads)

for compartment in compartments:
    cid = compartment["id"]
    label = "compartment " + cid[-12:]
    instances = query(label + " instances", "compute", "instance", "list",
                      "--compartment-id", cid, "--all")
    volumes = query(label + " block volumes", "bv", "volume", "list", "--compartment-id", cid, "--all")
    boots = []
    boots_complete = bool(ads)
    for ad in (ads or []):
        part = query(label + " boot volumes in " + ad["name"], "bv", "boot-volume", "list",
                     "--compartment-id", cid, "--availability-domain", ad["name"], "--all")
        if part is None:
            boots_complete = False
        else:
            boots.extend(part)
    active_instances = []
    for row in (instances or []):
        if row.get("lifecycle-state") == "TERMINATED":
            continue
        record = fields(row, "id", "display-name", "availability-domain", "lifecycle-state", "shape", "image-id")
        record["shape-config"] = fields(row.get("shape-config") or {}, "ocpus", "memory-in-gbs")
        active_instances.append(record)
    disk_fields = ("id", "display-name", "availability-domain", "lifecycle-state", "size-in-gbs", "vpus-per-gb")
    item = {"compartment_id": cid,
            "instances": None if instances is None else active_instances,
            "boot_volumes_complete": boots_complete,
            "boot_volumes": [fields(row, *disk_fields) for row in boots if row.get("lifecycle-state") != "TERMINATED"],
            "block_volumes": None if volumes is None else [fields(row, *disk_fields) for row in volumes
                                                            if row.get("lifecycle-state") != "TERMINATED"]}
    inventory.append(item)
    inventory_complete &= instances is not None and volumes is not None and boots_complete

a1 = [row for item in inventory for row in (item["instances"] or []) if row["shape"] == SHAPE]
def observed_sum(rows, field):
    values = [row.get(field) for row in rows]
    return None if any(value is None for value in values) else sum(values)

boot_rows = [row for item in inventory for row in item["boot_volumes"]]
block_rows = [row for item in inventory for row in (item["block_volumes"] or [])]
summary["inventory_totals"] = {
    "complete_for_accessible_compartments": bool(inventory_complete),
    "nonterminated_a1_instances": len(a1),
    "observed_a1_ocpus": observed_sum([row["shape-config"] for row in a1], "ocpus"),
    "observed_a1_memory_in_gbs": observed_sum([row["shape-config"] for row in a1], "memory-in-gbs"),
    "observed_boot_size_in_gbs": observed_sum(boot_rows, "size-in-gbs"),
    "observed_block_size_in_gbs": observed_sum(block_rows, "size-in-gbs"),
    "note": "Counts include stopped/terminating resources. These are observed regional allocations, not a billing calculation. Inaccessible compartments and other regions are excluded; incomplete observations are lower bounds.",
}

shape_results = []
for ad in (ads or []):
    shapes = query("A1 shape catalog in " + ad["name"], "compute", "shape", "list",
                   "--compartment-id", tenancy_id, "--availability-domain", ad["name"], "--shape", SHAPE, "--all")
    shape_results.append({"availability_domain": ad["name"], "shapes": None if shapes is None else [
        fields(row, "shape", "processor-description", "is-flexible", "ocpu-options", "memory-options")
        for row in shapes if row.get("shape") == SHAPE]})
summary["a1_shapes_by_availability_domain"] = shape_results

images = query("Ubuntu images compatible with A1", "compute", "image", "list", "--compartment-id", tenancy_id,
               "--operating-system", "Canonical Ubuntu", "--shape", SHAPE, "--lifecycle-state", "AVAILABLE",
               "--sort-by", "TIMECREATED", "--sort-order", "DESC", "--all")
summary["ubuntu_24_04_aarch64_image_candidates"] = None if images is None else [
    fields(row, "id", "display-name", "operating-system", "operating-system-version", "time-created",
           "compartment-id", "base-image-id", "listing-type", "size-in-mbs", "launch-mode")
    for row in images if str(row.get("display-name", "")).startswith("Canonical-Ubuntu-24.04-")
    and "aarch64" in str(row.get("display-name", "")).lower()]
summary["image_selection_note"] = "Candidates come from the live A1-compatible image listing, which can include custom images. Confirm the selected image ID against Oracle's platform-image catalog before launch; no image is auto-selected."

limits = query("compute service limits", "limits", "value", "list", "--compartment-id", tenancy_id,
               "--service-name", "compute", "--all")
limit_results = []
for row in (limits or []):
    if not re.search(r"(^|-)a1(-|$)", row.get("name", "")):
        continue
    record = fields(row, "name", "scope-type", "availability-domain", "value")
    args = ["limits", "resource-availability", "get", "--compartment-id", tenancy_id,
            "--service-name", "compute", "--limit-name", row["name"]]
    if row.get("scope-type") == "AD":
        if not row.get("availability-domain"):
            record["availability"] = None
            limit_results.append(record)
            continue
        args.extend(["--availability-domain", row["availability-domain"]])
    available = query("quota availability for " + row["name"], *args)
    record["availability"] = None if available is None else fields(
        available, "available", "used", "fractional-availability", "fractional-usage",
        "effective-quota-value", "is-resource-availability-supported")
    limit_results.append(record)
summary["a1_limits_and_quota_availability"] = None if limits is None else limit_results
summary["physical_host_capacity"] = "NOT_QUERIED: shape eligibility and quota headroom do not establish available physical host capacity or reserve it."
summary["cloud_writes_performed"] = False
emit()
sys.exit(2 if errors else 0)
PY
