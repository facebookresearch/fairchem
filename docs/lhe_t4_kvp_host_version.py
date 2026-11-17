#!/usr/bin/env python3
"""Read-only, redacted Hyper-V KVP host-version boundary classifier."""

from __future__ import annotations

import collections
import json
import re
import subprocess
from pathlib import Path
from typing import Any


PREFIX = "META_LHE_FAIRCHEM_KVP"
KVP_ROOT = Path("/var/lib/hyperv")
KEY_SIZE = 512
VALUE_SIZE = 2048
RECORD_SIZE = KEY_SIZE + VALUE_SIZE
MAX_POOL_BYTES = RECORD_SIZE * 4096


def c_string(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")


def key_classes(key: str) -> set[str]:
    lowered = key.casefold()
    classes: set[str] = set()
    if "host" in lowered and "build" in lowered:
        classes.add("host_build")
    if "host" in lowered and ("version" in lowered or "release" in lowered):
        classes.add("host_version")
    if "host" in lowered and "name" in lowered:
        classes.add("host_name")
    if "physical" in lowered and "host" in lowered:
        classes.add("physical_host")
    if any(token in lowered for token in ("vmid", "vm_id", "machineid", "machine_id")):
        classes.add("vm_identifier")
    if "virtualmachine" in lowered and ("id" in lowered or "name" in lowered):
        classes.add("vm_identity")
    if any(token in lowered for token in ("ipaddress", "networkaddress", "subnet", "gateway", "dns")):
        classes.add("network")
    if any(token in lowered for token in ("fabric", "wire", "azure", "goalstate")):
        classes.add("azure_fabric")
    if lowered in {
        "osbuildnumber",
        "osversion",
        "osmajorversion",
        "osminorversion",
        "osname",
        "integrationservicesversion",
    }:
        classes.add("guest_auto_metadata")
    if not classes:
        classes.add("other")
    return classes


def fixed_class(components: list[int]) -> str:
    """Classify only public November 2026 Windows fixed-build thresholds."""
    thresholds = {
        14393: 9688,
        17763: 9487,
        19044: 7846,
        19045: 7846,
        20348: 5870,
        22631: 7711,
        26100: 33822,
        26200: 9974,
        26300: 9974,
        28000: 3501,
    }
    for index, component in enumerate(components[:-1]):
        threshold = thresholds.get(component)
        if threshold is not None:
            return "fixed_or_newer" if components[index + 1] >= threshold else "pre_fix"
    return "unclassified"


def version_observations(value: str) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for match in re.finditer(r"(?<!\d)\d+(?:\.\d+){1,5}(?!\d)", value):
        components = tuple(int(part) for part in match.group(0).split("."))
        if components in seen:
            continue
        seen.add(components)
        observations.append(
            {
                "component_count": len(components),
                "contains_server_2025_build": 26100 in components,
                "november_2026_fix_class": fixed_class(list(components)),
            }
        )
    return observations[:8]


def inspect_pool(index: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = KVP_ROOT / f".kvp_pool_{index}"
    result: dict[str, Any] = {
        "present": path.is_file(),
        "regular_file": path.is_file(),
        "size_within_limit": False,
        "record_aligned": False,
        "nonempty_record_count": 0,
        "key_class_counts": {},
        "value_body_emitted": False,
        "raw_key_emitted": False,
    }
    candidates: list[dict[str, Any]] = []
    if not path.is_file():
        return result, candidates
    try:
        size = path.stat().st_size
        result["size_within_limit"] = size <= MAX_POOL_BYTES
        result["record_aligned"] = size % RECORD_SIZE == 0
        if size > MAX_POOL_BYTES or size % RECORD_SIZE:
            return result, candidates
        data = path.read_bytes()
    except OSError as exc:
        result["read_error_class"] = type(exc).__name__
        return result, candidates

    counts: collections.Counter[str] = collections.Counter()
    for offset in range(0, len(data), RECORD_SIZE):
        record = data[offset : offset + RECORD_SIZE]
        key = c_string(record[:KEY_SIZE])
        if not key:
            continue
        value = c_string(record[KEY_SIZE:])
        result["nonempty_record_count"] += 1
        classes = key_classes(key)
        counts.update(classes)
        if classes & {"host_build", "host_version"}:
            candidates.extend(version_observations(value))
    result["key_class_counts"] = dict(sorted(counts.items()))
    return result, candidates[:8]


def service_state() -> str:
    try:
        completed = subprocess.run(
            ["systemctl", "is-active", "hv-kvp-daemon.service"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    value = completed.stdout.strip()
    return value if value in {"active", "inactive", "failed", "activating"} else "other"


def main() -> int:
    pools: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = []
    for index in range(5):
        pool, observed = inspect_pool(index)
        pools[str(index)] = pool
        candidates.extend(observed)

    class_totals: collections.Counter[str] = collections.Counter()
    for pool in pools.values():
        class_totals.update(pool.get("key_class_counts", {}))

    result = {
        "schema": "meta-lhe-fairchem-kvp-host-version-v1",
        "scope": "read-only local Hyper-V KVP pool classification",
        "kvp_root_present": KVP_ROOT.is_dir(),
        "kvp_service_state": service_state(),
        "pools": pools,
        "key_class_totals": dict(sorted(class_totals.items())),
        "host_version_candidate_count": len(candidates),
        "host_version_candidates": candidates[:8],
        "host_version_key_present": bool(
            class_totals["host_build"] or class_totals["host_version"]
        ),
        "host_name_or_identifier_value_emitted": False,
        "network_value_emitted": False,
        "raw_key_or_value_emitted": False,
        "vmbus_request_sent": False,
        "network_request_sent": False,
        "external_state_changed": False,
    }
    print(f"{PREFIX}_json={json.dumps(result, sort_keys=True, separators=(',', ':'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
