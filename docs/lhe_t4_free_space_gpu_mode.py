#!/usr/bin/env python3
"""Bounded read-only classifier for T4 GPU mode and ext4 free-space residue."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import struct
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any


PREFIX = "META_LHE_FAIRCHEM_FREE_SPACE_GPU_MODE"
SALT = b"Meta-LHE/FairChem/T4/free-space-gpu-mode/2026-11-17/v1"
EXT4_MAGIC = 0xEF53
EXT4_FEATURE_INCOMPAT_META_BG = 0x0010
EXT4_FEATURE_INCOMPAT_64BIT = 0x0080
EXT4_FEATURE_RO_COMPAT_BIGALLOC = 0x0200
EXT4_BG_BLOCK_UNINIT = 0x0002
MAX_SAMPLE_BLOCKS = 16_384
CANDIDATE_POINTS = 65_536
MAX_SAMPLE_BYTES = 64 * 1024 * 1024
SECRET_PREFIXES = (
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
PRIOR_PROBE_MARKERS = (
    b"META_LHE_FAIRCHEM_GPU_CONTEXT_MARKER_V1_",
    b"META_LHE_FAIRCHEM_GPU_RESIDUAL_json=",
    b"META_LHE_FAIRCHEM_HOSTGAPLUGIN_json=",
    b"META_LHE_FAIRCHEM_STORAGE_BOUNDARY_V1",
)


def command(args: list[str], timeout: int = 20) -> dict[str, Any]:
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


def salted_identifier(value: str) -> str:
    if not value.strip():
        return ""
    return hashlib.sha256(SALT + value.strip().encode("utf-8")).hexdigest()[:24]


def uniform_subset(values: list[int], maximum: int) -> list[int]:
    if len(values) <= maximum:
        return values
    if maximum <= 1:
        return values[:maximum]
    return [values[index * (len(values) - 1) // (maximum - 1)] for index in range(maximum)]


def parse_root_device() -> tuple[str, str]:
    result = command(["findmnt", "-n", "-o", "SOURCE,FSTYPE,MAJ:MIN", "/"])
    if not result.get("ok"):
        return "", ""
    fields = result["stdout"].strip().split()
    if len(fields) < 2:
        return "", ""
    source = fields[0].split("[", 1)[0]
    resolved = os.path.realpath(source)
    if (not resolved.startswith("/dev/") or not os.path.exists(resolved)) and len(fields) >= 3:
        devices = command(["lsblk", "-n", "-r", "-p", "-o", "PATH,MAJ:MIN"])
        if devices.get("ok"):
            for line in devices["stdout"].splitlines():
                device_fields = line.split()
                if len(device_fields) == 2 and device_fields[1] == fields[2]:
                    resolved = device_fields[0]
                    break
    return resolved, fields[1]


def root_disk_identity(root_device: str) -> dict[str, Any]:
    result = command(
        [
            "lsblk",
            "--json",
            "--output",
            "PATH,PKNAME,MODEL,SERIAL,WWN",
            root_device,
        ]
    )
    if not result.get("ok"):
        return {"available": False, "raw_identifier_emitted": False}
    try:
        nodes = json.loads(result["stdout"]).get("blockdevices") or []
    except json.JSONDecodeError:
        return {"available": False, "raw_identifier_emitted": False}
    if not nodes:
        return {"available": False, "raw_identifier_emitted": False}
    node = nodes[0]
    parent = str(node.get("pkname") or "")
    identity_node = node
    if parent:
        parent_result = command(
            ["lsblk", "--json", "--output", "MODEL,SERIAL,WWN", f"/dev/{parent}"]
        )
        try:
            parent_nodes = json.loads(parent_result.get("stdout", "")).get("blockdevices") or []
            if parent_nodes:
                identity_node = parent_nodes[0]
        except json.JSONDecodeError:
            pass
    identity = "|".join(
        str(identity_node.get(key) or "") for key in ("model", "serial", "wwn")
    )
    return {
        "available": bool(identity.strip("|")),
        "device_identity_salted_sha256_24": salted_identifier(identity),
        "model_class": str(identity_node.get("model") or "").strip(),
        "raw_identifier_emitted": False,
    }


def bit_is_free(bitmap: bytes, bit: int) -> bool:
    byte_index = bit // 8
    return byte_index < len(bitmap) and not (bitmap[byte_index] & (1 << (bit % 8)))


def ext4_free_space_scan(root_device: str, filesystem_type: str) -> dict[str, Any]:
    base: dict[str, Any] = {
        "supported": False,
        "filesystem_type": filesystem_type,
        "root_device_class": os.path.basename(root_device),
        "allocated_block_content_scanned": False,
        "raw_bytes_emitted": False,
        "write_performed": False,
    }
    if filesystem_type != "ext4" or not os.path.isabs(root_device):
        base["reason"] = "unsupported_root_filesystem_or_device"
        return base
    try:
        descriptor = os.open(root_device, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    except OSError as exc:
        base["reason"] = type(exc).__name__
        return base
    try:
        superblock = os.pread(descriptor, 1024, 1024)
        if len(superblock) != 1024 or struct.unpack_from("<H", superblock, 0x38)[0] != EXT4_MAGIC:
            base["reason"] = "invalid_ext4_superblock"
            return base
        blocks_lo = struct.unpack_from("<I", superblock, 0x04)[0]
        first_data_block = struct.unpack_from("<I", superblock, 0x14)[0]
        block_size = 1024 << struct.unpack_from("<I", superblock, 0x18)[0]
        blocks_per_group = struct.unpack_from("<I", superblock, 0x20)[0]
        incompat = struct.unpack_from("<I", superblock, 0x60)[0]
        ro_compat = struct.unpack_from("<I", superblock, 0x64)[0]
        desc_size_raw = struct.unpack_from("<H", superblock, 0xFE)[0]
        blocks_hi = (
            struct.unpack_from("<I", superblock, 0x150)[0]
            if incompat & EXT4_FEATURE_INCOMPAT_64BIT
            else 0
        )
        blocks_count = blocks_lo | (blocks_hi << 32)
        desc_size = max(32, desc_size_raw if incompat & EXT4_FEATURE_INCOMPAT_64BIT else 32)
        if (
            block_size < 1024
            or block_size > 65536
            or blocks_per_group == 0
            or blocks_count <= first_data_block
            or desc_size > block_size
        ):
            base["reason"] = "unsupported_ext4_geometry"
            return base
        if incompat & EXT4_FEATURE_INCOMPAT_META_BG:
            base["reason"] = "ext4_meta_bg_not_supported"
            return base
        if ro_compat & EXT4_FEATURE_RO_COMPAT_BIGALLOC:
            base["reason"] = "ext4_bigalloc_not_supported"
            return base

        group_count = math.ceil((blocks_count - first_data_block) / blocks_per_group)
        descriptor_table_offset = (first_data_block + 1) * block_size
        descriptor_bytes = os.pread(descriptor, group_count * desc_size, descriptor_table_offset)
        if len(descriptor_bytes) != group_count * desc_size:
            base["reason"] = "short_group_descriptor_table"
            return base

        group_bitmaps: dict[int, tuple[int, bytes]] = {}
        skipped_uninitialized_groups = 0
        for group in range(group_count):
            offset = group * desc_size
            flags = struct.unpack_from("<H", descriptor_bytes, offset + 0x12)[0]
            if flags & EXT4_BG_BLOCK_UNINIT:
                skipped_uninitialized_groups += 1
                continue
            bitmap_lo = struct.unpack_from("<I", descriptor_bytes, offset)[0]
            bitmap_hi = (
                struct.unpack_from("<I", descriptor_bytes, offset + 0x20)[0]
                if desc_size >= 64 and incompat & EXT4_FEATURE_INCOMPAT_64BIT
                else 0
            )
            bitmap_block = bitmap_lo | (bitmap_hi << 32)
            if not (first_data_block <= bitmap_block < blocks_count):
                continue
            bitmap = os.pread(descriptor, block_size, bitmap_block * block_size)
            if len(bitmap) == block_size:
                group_bitmaps[group] = (bitmap_block, bitmap)

        span = blocks_count - first_data_block
        candidate_count = min(CANDIDATE_POINTS, span)
        free_candidates: list[int] = []
        if candidate_count == 1:
            raw_candidates = [first_data_block]
        else:
            raw_candidates = [
                first_data_block + (index * (span - 1) // (candidate_count - 1))
                for index in range(candidate_count)
            ]
        for block_number in raw_candidates:
            relative = block_number - first_data_block
            group = relative // blocks_per_group
            bit = relative % blocks_per_group
            metadata = group_bitmaps.get(group)
            if metadata and bit_is_free(metadata[1], bit):
                free_candidates.append(block_number)
        selected = uniform_subset(free_candidates, min(MAX_SAMPLE_BLOCKS, MAX_SAMPLE_BYTES // block_size))
        by_group: dict[int, list[int]] = defaultdict(list)
        for block_number in selected:
            by_group[(block_number - first_data_block) // blocks_per_group].append(block_number)

        zero_bytes = 0
        nonzero_bytes = 0
        revalidated_blocks = 0
        read_error_blocks = 0
        race_discarded_blocks = 0
        secret_counts = {prefix.decode("ascii"): 0 for prefix in SECRET_PREFIXES}
        marker_counts = {marker.decode("ascii"): 0 for marker in PRIOR_PROBE_MARKERS}
        for group, block_numbers in sorted(by_group.items()):
            bitmap_block, _ = group_bitmaps[group]
            before = os.pread(descriptor, block_size, bitmap_block * block_size)
            observations: dict[int, tuple[int, int, dict[str, int], dict[str, int]]] = {}
            for block_number in block_numbers:
                bit = (block_number - first_data_block) % blocks_per_group
                if not bit_is_free(before, bit):
                    race_discarded_blocks += 1
                    continue
                data = os.pread(descriptor, block_size, block_number * block_size)
                if len(data) != block_size:
                    read_error_blocks += 1
                    continue
                observations[block_number] = (
                    data.count(0),
                    len(data) - data.count(0),
                    {prefix.decode("ascii"): data.count(prefix) for prefix in SECRET_PREFIXES},
                    {marker.decode("ascii"): data.count(marker) for marker in PRIOR_PROBE_MARKERS},
                )
            after = os.pread(descriptor, block_size, bitmap_block * block_size)
            for block_number, observation in observations.items():
                bit = (block_number - first_data_block) % blocks_per_group
                if not bit_is_free(after, bit):
                    race_discarded_blocks += 1
                    continue
                zeros, nonzeros, block_secrets, block_markers = observation
                zero_bytes += zeros
                nonzero_bytes += nonzeros
                revalidated_blocks += 1
                for key, value in block_secrets.items():
                    secret_counts[key] += value
                for key, value in block_markers.items():
                    marker_counts[key] += value

        return {
            **base,
            "supported": True,
            "block_size": block_size,
            "block_count": blocks_count,
            "group_count": group_count,
            "skipped_uninitialized_group_count": skipped_uninitialized_groups,
            "uniform_candidate_points": candidate_count,
            "candidate_points_free_at_first_snapshot": len(free_candidates),
            "selected_free_blocks": len(selected),
            "revalidated_free_blocks": revalidated_blocks,
            "race_discarded_blocks": race_discarded_blocks,
            "read_error_blocks": read_error_blocks,
            "sampled_bytes": revalidated_blocks * block_size,
            "zero_bytes": zero_bytes,
            "nonzero_bytes": nonzero_bytes,
            "all_revalidated_bytes_zero": revalidated_blocks > 0 and nonzero_bytes == 0,
            "secret_prefix_shape_counts": secret_counts,
            "prior_probe_marker_counts": marker_counts,
            "filesystem_uuid_emitted": False,
            "double_bitmap_revalidation": True,
        }
    except (OSError, struct.error, OverflowError, ValueError) as exc:
        base["reason"] = type(exc).__name__
        return base
    finally:
        os.close(descriptor)


def nvidia_xml_inventory() -> dict[str, Any]:
    result = command(["nvidia-smi", "-q", "-x"])
    if not result.get("ok"):
        return {"available": False, "raw_output_emitted": False}
    try:
        root = ET.fromstring(result["stdout"])
    except ET.ParseError:
        return {"available": False, "parse_error": True, "raw_output_emitted": False}
    root_tags = ("driver_version", "cuda_version", "attached_gpus")
    gpu_tags = (
        "product_name",
        "product_brand",
        "product_architecture",
        "display_mode",
        "display_active",
        "persistence_mode",
        "addressing_mode",
        "compute_mode",
        "virtualization_mode",
        "host_vgpu_mode",
        "fractional_multi_vgpu",
        "heterogeneous_multivgpu",
        "multi_vgpu_exclusive",
    )
    inventory: dict[str, Any] = {
        tag: (root.findtext(tag) or "").strip() for tag in root_tags
    }
    gpus: list[dict[str, Any]] = []
    for gpu in root.findall("gpu"):
        item: dict[str, Any] = {}
        for tag in gpu_tags:
            values = sorted(
                {
                    (element.text or "").strip()
                    for element in gpu.iter(tag)
                    if (element.text or "").strip()
                }
            )
            if values:
                item[tag] = values[0] if len(values) == 1 else values
        gpus.append(item)
    uuid_result = command(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"]
    )
    uuid_hashes = (
        [salted_identifier(line) for line in uuid_result["stdout"].splitlines() if line.strip()]
        if uuid_result.get("ok")
        else []
    )
    return {
        "available": True,
        **inventory,
        "gpus": gpus,
        "gpu_uuid_salted_sha256_24": uuid_hashes,
        "raw_uuid_or_xml_emitted": False,
    }


def gpu_pci_inventory() -> dict[str, Any]:
    bus_result = command(
        ["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader,nounits"]
    )
    devices: list[dict[str, Any]] = []
    if not bus_result.get("ok"):
        return {"available": False, "devices": devices}
    for raw_bus in bus_result["stdout"].splitlines():
        parts = raw_bus.strip().split(":")
        bus = ":".join(parts[-2:]) if len(parts) >= 3 else raw_bus.strip()
        bus = (f"0000:{bus}" if bus.count(":") == 1 else bus).lower()
        path = f"/sys/bus/pci/devices/{bus}"
        if not os.path.isdir(path):
            continue
        item: dict[str, Any] = {
            "physfn_present": os.path.exists(os.path.join(path, "physfn")),
            "iommu_group_present": os.path.exists(os.path.join(path, "iommu_group")),
            "mdev_supported_types_present": os.path.exists(
                os.path.join(path, "mdev_supported_types")
            ),
        }
        for name in (
            "vendor",
            "device",
            "subsystem_vendor",
            "subsystem_device",
            "class",
            "numa_node",
            "sriov_numvfs",
            "sriov_totalvfs",
        ):
            try:
                item[name] = open(os.path.join(path, name), encoding="ascii").read().strip()
            except OSError:
                pass
        try:
            item["driver"] = os.path.basename(os.path.realpath(os.path.join(path, "driver")))
        except OSError:
            item["driver"] = ""
        try:
            bar_sizes: list[int] = []
            for line in open(os.path.join(path, "resource"), encoding="ascii"):
                fields = line.split()
                if len(fields) >= 2:
                    start, end = int(fields[0], 16), int(fields[1], 16)
                    bar_sizes.append(end - start + 1 if end >= start and (start or end) else 0)
            item["pci_bar_sizes_bytes"] = bar_sizes
        except (OSError, ValueError):
            pass
        devices.append(item)
    return {
        "available": bool(devices),
        "devices": devices,
        "guest_physical_addresses_emitted": False,
    }


def package_inventory() -> dict[str, Any]:
    result = command(
        ["dpkg-query", "-W", "-f=${db:Status-Abbrev}\t${binary:Package}\t${Version}\n"],
        timeout=30,
    )
    current_kernel = platform.release()
    packages: list[dict[str, str]] = []
    allowed = re.compile(
        r"^(?:walinuxagent|python3-walinuxagent|azurelinuxagent|"
        r"nvidia-(?:driver|kernel|dkms|utils|compute-utils)-\d+(?:-server)?|"
        r"libnvidia-compute-\d+(?:-server)?)$"
    )
    if result.get("ok"):
        for line in result["stdout"].splitlines():
            fields = line.split("\t", 2)
            if len(fields) != 3 or not fields[0].startswith("ii"):
                continue
            name = fields[1].split(":", 1)[0]
            if allowed.match(name) or (
                current_kernel in name
                and name.startswith(("linux-image-", "linux-modules-", "linux-modules-extra-"))
            ):
                packages.append({"name": name, "version": fields[2]})
    modinfo: dict[str, str] = {}
    for field in ("version", "vermagic", "signer"):
        value = command(["modinfo", "-F", field, "nvidia"])
        if value.get("ok"):
            modinfo[field] = value["stdout"].strip()
    waagent = command(["waagent", "--version"])
    waagent_versions: dict[str, str] = {}
    if waagent.get("ok"):
        text = waagent["stdout"]
        patterns = {
            "walinuxagent": r"WALinuxAgent[- ](\d+(?:\.\d+){2,4})",
            "goal_state_agent": r"Goal state agent:\s*(\d+(?:\.\d+){2,4})",
            "python": r"Python:\s*(\d+(?:\.\d+){1,4})",
            "distribution": r"running on\s+([A-Za-z0-9._ -]+)",
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                waagent_versions[key] = match.group(1).strip()
    return {
        "kernel_release": current_kernel,
        "packages": sorted(packages, key=lambda item: item["name"]),
        "nvidia_module": modinfo,
        "waagent": waagent_versions,
        "raw_command_output_emitted": False,
    }


def main() -> int:
    root_device, filesystem_type = parse_root_device()
    free_space = ext4_free_space_scan(root_device, filesystem_type)
    secret_matches = sum(free_space.get("secret_prefix_shape_counts", {}).values())
    marker_matches = sum(free_space.get("prior_probe_marker_counts", {}).values())
    gpu = nvidia_xml_inventory()
    modes = {
        str(item.get("virtualization_mode", "")).lower()
        for item in gpu.get("gpus", [])
        if item.get("virtualization_mode")
    }
    result = {
        "schema": "meta-lhe-fairchem-t4-free-space-gpu-mode-v1",
        "scope": "read-only ext4 unallocated blocks and selected GPU/guest version metadata",
        "root_disk_identity": root_disk_identity(root_device),
        "free_space": free_space,
        "nvidia": gpu,
        "gpu_pci": gpu_pci_inventory(),
        "versions": package_inventory(),
        "conclusions": {
            "free_space_secret_shape_match": secret_matches > 0,
            "prior_probe_marker_match": marker_matches > 0,
            "cross_vm_storage_residual_confirmed": False,
            "nvidia_vgpu_mode_observed": "vgpu" in modes,
            "gpu_direct_or_none_mode_observed": "none" in modes or "passthrough" in modes,
            "guest_to_host_escape_confirmed": False,
        },
        "maximum_free_space_sample_bytes": MAX_SAMPLE_BYTES,
        "raw_file_or_block_bytes_emitted": False,
        "cloud_identifier_emitted": False,
        "network_scan_performed": False,
        "external_state_changed": False,
        "write_performed": False,
    }
    print(f"{PREFIX}_json={json.dumps(result, sort_keys=True, separators=(',', ':'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
