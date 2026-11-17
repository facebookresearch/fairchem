#!/usr/bin/env python3
"""Bounded read-only classifier for FairChem T4 guest/host storage boundaries."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import platform
import re
import subprocess
from typing import Any


PREFIX = "META_LHE_FAIRCHEM_STORAGE"
IMDS_HOST = "169.254.169.254"
MAX_DISKS = 2
WINDOW_SIZE = 8 * 1024 * 1024
WINDOW_COUNT = 16
EDGE_SKIP = 64 * 1024 * 1024
MIN_DISK_SIZE = 256 * 1024 * 1024
PREFIXES = (
    b"github_pat_",
    b"ghp_",
    b"gho_",
    b"ghs_",
    b"ghu_",
    b"hf_",
    b"AKIA",
    b"ASIA",
    b"xoxb-",
    b"xoxp-",
    b"-----BEGIN ",
    b"eyJ0eXAiOiJKV1Qi",
    b"eyJhbGciOi",
)
RESEARCH_MARKER = b"META_LHE_FAIRCHEM_STORAGE_BOUNDARY_V1"
DISK_ID_SALT = b"Meta-LHE/FairChem/T4/storage-boundary/2026-11-17/v1"


def command(args: list[str], timeout: int = 10) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"},
        )
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr_bytes": len(completed.stderr.encode("utf-8")),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error_class": type(exc).__name__, "stdout": ""}


def version_tokens(value: str) -> list[str]:
    return sorted(set(re.findall(r"\b\d+(?:\.\d+){1,4}(?:[-+~][A-Za-z0-9.+:~_-]+)?", value)))


def selected_imds() -> dict[str, Any]:
    connection = http.client.HTTPConnection(IMDS_HOST, 80, timeout=5)
    try:
        connection.request(
            "GET",
            "/metadata/instance?api-version=2021-02-01",
            headers={"Metadata": "true"},
        )
        response = connection.getresponse()
        body = response.read(1024 * 1024)
        if response.status != 200:
            return {"status": response.status, "parsed": False}
        value = json.loads(body)
        compute = value.get("compute", {})
        storage = compute.get("storageProfile", {})
        os_disk = storage.get("osDisk", {})
        diff = os_disk.get("diffDiskSettings") or {}
        managed = os_disk.get("managedDisk") or {}
        security = compute.get("securityProfile") or {}
        network = value.get("network", {})
        interfaces = network.get("interface") or []
        return {
            "status": response.status,
            "parsed": True,
            "vm_size": compute.get("vmSize", ""),
            "os_disk": {
                "create_option": os_disk.get("createOption", ""),
                "caching": os_disk.get("caching", ""),
                "disk_size_gb": os_disk.get("diskSizeGB", ""),
                "storage_account_type": managed.get("storageAccountType", ""),
                "diff_disk_option": diff.get("option", ""),
                "diff_disk_placement": diff.get("placement", ""),
            },
            "data_disk_count": len(storage.get("dataDisks") or []),
            "security": {
                "encryption_at_host": security.get("encryptionAtHost", ""),
                "secure_boot_enabled": security.get("secureBootEnabled", ""),
                "virtual_tpm_enabled": security.get("virtualTpmEnabled", ""),
                "security_type": security.get("securityType", ""),
            },
            "network_interface_count": len(interfaces),
            "identifier_or_tag_emitted": False,
        }
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        return {"parsed": False, "error_class": type(exc).__name__}
    finally:
        connection.close()


def flatten_lsblk(blockdevices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []

    def visit(node: dict[str, Any], parent: str = "") -> None:
        copy = {key: value for key, value in node.items() if key != "children"}
        copy["tree_parent"] = parent
        flattened.append(copy)
        for child in node.get("children") or []:
            visit(child, str(node.get("path") or node.get("name") or ""))

    for device in blockdevices:
        visit(device)
    return flattened


def mount_classes(mountpoints: list[str]) -> list[str]:
    classes: set[str] = set()
    for mountpoint in mountpoints:
        if mountpoint == "/":
            classes.add("root")
        elif mountpoint in ("/boot", "/boot/efi"):
            classes.add("boot")
        elif mountpoint == "[SWAP]":
            classes.add("swap")
        elif mountpoint.startswith("/mnt") or mountpoint.startswith("/media"):
            classes.add("temporary_or_data")
        else:
            classes.add("other")
    return sorted(classes)


def descendants_by_disk(
    disks: list[dict[str, Any]], flat: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for node in flat:
        by_parent.setdefault(str(node.get("tree_parent") or ""), []).append(node)

    def collect(path: str) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for child in by_parent.get(path, []):
            found.append(child)
            child_path = str(child.get("path") or child.get("name") or "")
            found.extend(collect(child_path))
        return found

    for disk in disks:
        path = str(disk.get("path") or disk.get("name") or "")
        result[path] = [disk, *collect(path)]
    return result


def parse_mountpoints(node: dict[str, Any]) -> list[str]:
    value = node.get("mountpoints")
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if value:
        return [str(value)]
    single = node.get("mountpoint")
    return [str(single)] if single else []


def scan_offsets(size: int) -> list[int]:
    low = EDGE_SKIP
    high = size - EDGE_SKIP - WINDOW_SIZE
    if high <= low:
        low = 0
        high = max(0, size - WINDOW_SIZE)
    if WINDOW_COUNT == 1 or high <= low:
        return [low - (low % 4096)]
    return [
        (low + ((high - low) * index // (WINDOW_COUNT - 1))) // 4096 * 4096
        for index in range(WINDOW_COUNT)
    ]


def scan_disk(path: str, size: int) -> dict[str, Any]:
    zero_bytes = 0
    nonzero_bytes = 0
    sampled_bytes = 0
    prefix_counts = {prefix.decode("ascii"): 0 for prefix in PREFIXES}
    marker_matches = 0
    windows_read = 0
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    except OSError as exc:
        return {
            "read_supported": False,
            "error_class": type(exc).__name__,
            "raw_bytes_emitted": False,
        }
    try:
        for offset in scan_offsets(size):
            try:
                data = os.pread(descriptor, WINDOW_SIZE, offset)
            except OSError as exc:
                return {
                    "read_supported": False,
                    "error_class": type(exc).__name__,
                    "windows_read_before_error": windows_read,
                    "raw_bytes_emitted": False,
                }
            windows_read += 1
            sampled_bytes += len(data)
            zeros = data.count(0)
            zero_bytes += zeros
            nonzero_bytes += len(data) - zeros
            marker_matches += data.count(RESEARCH_MARKER)
            for prefix in PREFIXES:
                prefix_counts[prefix.decode("ascii")] += data.count(prefix)
        return {
            "read_supported": True,
            "windows_read": windows_read,
            "sampled_bytes": sampled_bytes,
            "zero_bytes": zero_bytes,
            "nonzero_bytes": nonzero_bytes,
            "all_sampled_bytes_zero": sampled_bytes > 0 and nonzero_bytes == 0,
            "research_marker_matches": marker_matches,
            "credential_prefix_shape_counts": prefix_counts,
            "raw_bytes_emitted": False,
        }
    finally:
        os.close(descriptor)


def storage_inventory() -> dict[str, Any]:
    result = command(
        [
            "lsblk",
            "--json",
            "--bytes",
            "--output",
            "NAME,PATH,TYPE,SIZE,FSTYPE,MOUNTPOINTS,MODEL,SERIAL,WWN,RO,RM",
        ]
    )
    if not result.get("ok"):
        return {
            "supported": False,
            "error_class": result.get("error_class", "LsblkError"),
            "raw_error_emitted": False,
        }
    try:
        tree = json.loads(result["stdout"])
    except json.JSONDecodeError:
        return {"supported": False, "error_class": "LsblkJsonError"}
    flat = flatten_lsblk(tree.get("blockdevices") or [])
    disks = [node for node in flat if node.get("type") == "disk"]
    descendants = descendants_by_disk(disks, flat)
    public_disks: list[dict[str, Any]] = []
    candidates: list[tuple[str, int]] = []
    for disk in disks:
        path = str(disk.get("path") or "")
        nodes = descendants.get(path, [disk])
        mounts = [mount for node in nodes for mount in parse_mountpoints(node)]
        classes = mount_classes(mounts)
        size = int(disk.get("size") or 0)
        contains_root = "root" in classes
        identity_material = "|".join(
            str(disk.get(key) or "") for key in ("model", "serial", "wwn")
        ).encode("utf-8")
        public_disks.append(
            {
                "path": path,
                "size_bytes": size,
                "model_class": str(disk.get("model") or "").strip(),
                "device_identity_salted_sha256_24": (
                    hashlib.sha256(DISK_ID_SALT + identity_material).hexdigest()[:24]
                    if identity_material.strip(b"|")
                    else ""
                ),
                "raw_serial_or_wwn_emitted": False,
                "read_only": bool(disk.get("ro")),
                "removable": bool(disk.get("rm")),
                "mount_classes": classes,
                "filesystem_types": sorted(
                    {str(node.get("fstype")) for node in nodes if node.get("fstype")}
                ),
                "contains_root_filesystem": contains_root,
            }
        )
        if (
            path.startswith("/dev/")
            and not contains_root
            and not bool(disk.get("ro"))
            and size >= MIN_DISK_SIZE
        ):
            candidates.append((path, size))

    candidates.sort(key=lambda item: item[1], reverse=True)
    scans = {
        path: scan_disk(path, size) for path, size in candidates[:MAX_DISKS]
    }
    return {
        "supported": True,
        "disk_count": len(disks),
        "disks": public_disks,
        "non_os_candidate_count": len(candidates),
        "scanned_candidate_count": len(scans),
        "scans": scans,
        "os_disk_raw_scan_performed": False,
        "write_performed": False,
    }


def platform_inventory() -> dict[str, Any]:
    virtualization = command(["systemd-detect-virt"])
    nvidia = command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,compute_mode",
            "--format=csv,noheader,nounits",
        ]
    )
    waagent = command(["waagent", "--version"])
    lspci = command(["lspci", "-nn"])
    nvidia_rows: list[dict[str, str]] = []
    if nvidia.get("ok"):
        for line in nvidia["stdout"].splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 3:
                nvidia_rows.append(
                    {"name": parts[0], "driver_version": parts[1], "compute_mode": parts[2]}
                )
    pci_ids: list[str] = []
    if lspci.get("ok"):
        for line in lspci["stdout"].splitlines():
            if "NVIDIA" in line or "Microsoft" in line or "Hyper-V" in line:
                pci_ids.extend(re.findall(r"\[([0-9a-fA-F]{4}:[0-9a-fA-F]{4})\]", line))
    return {
        "kernel_release": platform.release(),
        "machine": platform.machine(),
        "virtualization": (
            virtualization.get("stdout", "").strip()
            if virtualization.get("ok")
            else "unknown"
        ),
        "nvidia": nvidia_rows,
        "waagent_version_tokens": version_tokens(waagent.get("stdout", "")),
        "selected_pci_ids": sorted(set(pci_ids)),
        "iommu_group_count": (
            len(os.listdir("/sys/kernel/iommu_groups"))
            if os.path.isdir("/sys/kernel/iommu_groups")
            else 0
        ),
        "raw_command_output_emitted": False,
    }


def main() -> int:
    storage = storage_inventory()
    platform_facts = platform_inventory()
    scans = storage.get("scans", {}) if storage.get("supported") else {}
    prefix_matches = 0
    marker_matches = 0
    for scan in scans.values():
        prefix_matches += sum(scan.get("credential_prefix_shape_counts", {}).values())
        marker_matches += int(scan.get("research_marker_matches", 0))
    result = {
        "schema": "meta-lhe-fairchem-t4-storage-boundary-v1",
        "scope": "current VM inventory and bounded read-only non-OS block samples",
        "imds": selected_imds(),
        "platform": platform_facts,
        "storage": storage,
        "conclusions": {
            "credential_shape_match_in_non_os_samples": prefix_matches > 0,
            "research_marker_match_in_non_os_samples": marker_matches > 0,
            "cross_vm_storage_residual_confirmed": False,
            "guest_to_host_escape_confirmed": False,
        },
        "maximum_sampled_bytes_per_disk": WINDOW_SIZE * WINDOW_COUNT,
        "maximum_disks_sampled": MAX_DISKS,
        "raw_block_bytes_emitted": False,
        "network_scan_performed": False,
        "external_state_changed": False,
        "write_performed": False,
    }
    print(f"{PREFIX}_json={json.dumps(result, sort_keys=True, separators=(',', ':'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
