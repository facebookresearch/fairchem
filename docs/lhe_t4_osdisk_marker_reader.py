#!/usr/bin/env python3
"""Search logical ext4 free blocks for one public cross-VM control marker."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import subprocess
from collections import defaultdict
from typing import Any


PREFIX = "META_LHE_FAIRCHEM_OSDISK_READER"
READER_ATTEMPT = "EXACT-E"
SALT = b"Meta-LHE/FairChem/T4/osdisk-cross-vm/2026-11-17/v1"
CONTROL_MARKER = (
    b"META_LHE_FAIRCHEM_OSDISK_CROSS_VM_"
    b"56bd3bcdb69250aeea9f3fa59387207c89a21b195c7977cdc02159a021a64dc7"
)
WRITER_GPU_HASH = "0b3744781bf825ee28606be0"
WRITER_DISK_HASH = "b196f365ee0e2fc0d6a9fe86"
TARGET_EXTENTS_BYTES = ((7_918_845_952, 67_108_864),)
EXT4_MAGIC = 0xEF53
EXT4_FEATURE_INCOMPAT_META_BG = 0x0010
EXT4_FEATURE_INCOMPAT_64BIT = 0x0080
EXT4_FEATURE_RO_COMPAT_BIGALLOC = 0x0200
EXT4_BG_BLOCK_UNINIT = 0x0002
WINDOW_BLOCKS = 16
MAX_SAMPLE_WINDOWS = 1024
MAX_SAMPLE_BYTES = 64 * 1024 * 1024
UNINIT_EDGE_GUARD_BLOCKS = 1024


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
        return {"ok": completed.returncode == 0, "stdout": completed.stdout}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error_class": type(exc).__name__, "stdout": ""}


def salted(value: str) -> str:
    if not value.strip():
        return ""
    return hashlib.sha256(SALT + value.strip().encode("utf-8")).hexdigest()[:24]


def uniform_subset(values: list[int], maximum: int) -> list[int]:
    if len(values) <= maximum:
        return values
    if maximum <= 1:
        return values[:maximum]
    return [values[index * (len(values) - 1) // (maximum - 1)] for index in range(maximum)]


def root_device() -> tuple[str, str]:
    result = command(["findmnt", "-n", "-o", "SOURCE,FSTYPE,MAJ:MIN", "/"])
    fields = result.get("stdout", "").strip().split()
    if not result.get("ok") or len(fields) < 2:
        return "", ""
    source = os.path.realpath(fields[0].split("[", 1)[0])
    if (not source.startswith("/dev/") or not os.path.exists(source)) and len(fields) >= 3:
        devices = command(["lsblk", "-n", "-r", "-p", "-o", "PATH,MAJ:MIN"])
        for line in devices.get("stdout", "").splitlines():
            device_fields = line.split()
            if len(device_fields) == 2 and device_fields[1] == fields[2]:
                source = device_fields[0]
                break
    return source, fields[1]


def disk_identity_hash(device: str) -> str:
    result = command(["lsblk", "--json", "--output", "PKNAME,MODEL,SERIAL,WWN", device])
    try:
        nodes = json.loads(result.get("stdout", "")).get("blockdevices") or []
    except json.JSONDecodeError:
        return ""
    if not nodes:
        return ""
    node = nodes[0]
    parent = str(node.get("pkname") or "")
    if parent:
        parent_result = command(
            ["lsblk", "--json", "--output", "MODEL,SERIAL,WWN", f"/dev/{parent}"]
        )
        try:
            parent_nodes = json.loads(parent_result.get("stdout", "")).get("blockdevices") or []
            if parent_nodes:
                node = parent_nodes[0]
        except json.JSONDecodeError:
            pass
    identity = "|".join(str(node.get(key) or "") for key in ("model", "serial", "wwn"))
    return salted(identity)


def gpu_identity_hashes() -> list[str]:
    result = command(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"])
    if not result.get("ok"):
        return []
    return [salted(line) for line in result["stdout"].splitlines() if line.strip()]


def u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def bit_is_free(bitmap: bytes, bit: int) -> bool:
    byte_index = bit // 8
    return byte_index < len(bitmap) and not (bitmap[byte_index] & (1 << (bit % 8)))


def add_metadata_range(
    intervals: dict[int, list[tuple[int, int]]],
    start: int,
    end: int,
    first_data_block: int,
    blocks_per_group: int,
    blocks_count: int,
) -> None:
    start = max(start, first_data_block)
    end = min(end, blocks_count)
    while start < end:
        relative = start - first_data_block
        group = relative // blocks_per_group
        group_start = first_data_block + group * blocks_per_group
        group_end = min(group_start + blocks_per_group, blocks_count)
        part_end = min(end, group_end)
        intervals[group].append((start - group_start, part_end - group_start))
        start = part_end


def in_metadata_interval(intervals: list[tuple[int, int]], bit: int) -> bool:
    return any(start <= bit < end for start, end in intervals)


def ext4_marker_scan(device: str, filesystem_type: str) -> dict[str, Any]:
    base: dict[str, Any] = {
        "supported": False,
        "filesystem_type": filesystem_type,
        "root_device_class": os.path.basename(device),
        "allocated_block_content_scanned": False,
        "raw_bytes_emitted": False,
        "write_performed": False,
    }
    if filesystem_type != "ext4" or not os.path.isabs(device):
        base["reason"] = "unsupported_root_filesystem_or_device"
        return base
    try:
        descriptor = os.open(device, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    except OSError as exc:
        base["reason"] = type(exc).__name__
        return base
    try:
        superblock = os.pread(descriptor, 1024, 1024)
        if len(superblock) != 1024 or u16(superblock, 0x38) != EXT4_MAGIC:
            base["reason"] = "invalid_ext4_superblock"
            return base
        first_data_block = u32(superblock, 0x14)
        block_size = 1024 << u32(superblock, 0x18)
        blocks_per_group = u32(superblock, 0x20)
        inodes_per_group = u32(superblock, 0x28)
        inode_size = u16(superblock, 0x58) or 128
        incompat = u32(superblock, 0x60)
        ro_compat = u32(superblock, 0x64)
        blocks_count = u32(superblock, 0x04)
        if incompat & EXT4_FEATURE_INCOMPAT_64BIT:
            blocks_count |= u32(superblock, 0x150) << 32
        desc_size = max(
            32,
            u16(superblock, 0xFE) if incompat & EXT4_FEATURE_INCOMPAT_64BIT else 32,
        )
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
        gdt_offset = (first_data_block + 1) * block_size
        descriptors = os.pread(descriptor, group_count * desc_size, gdt_offset)
        if len(descriptors) != group_count * desc_size:
            base["reason"] = "short_group_descriptor_table"
            return base
        inode_table_blocks = math.ceil(inodes_per_group * inode_size / block_size)
        records: dict[int, dict[str, Any]] = {}
        metadata_intervals: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for group in range(group_count):
            offset = group * desc_size
            flags = u16(descriptors, offset + 0x12)
            block_bitmap = u32(descriptors, offset)
            inode_bitmap = u32(descriptors, offset + 0x04)
            inode_table = u32(descriptors, offset + 0x08)
            if desc_size >= 64 and incompat & EXT4_FEATURE_INCOMPAT_64BIT:
                block_bitmap |= u32(descriptors, offset + 0x20) << 32
                inode_bitmap |= u32(descriptors, offset + 0x24) << 32
                inode_table |= u32(descriptors, offset + 0x28) << 32
            records[group] = {
                "flags_offset": gdt_offset + offset + 0x12,
                "initial_flags": flags,
                "block_bitmap": block_bitmap,
            }
            add_metadata_range(
                metadata_intervals,
                block_bitmap,
                block_bitmap + 1,
                first_data_block,
                blocks_per_group,
                blocks_count,
            )
            add_metadata_range(
                metadata_intervals,
                inode_bitmap,
                inode_bitmap + 1,
                first_data_block,
                blocks_per_group,
                blocks_count,
            )
            add_metadata_range(
                metadata_intervals,
                inode_table,
                inode_table + inode_table_blocks,
                first_data_block,
                blocks_per_group,
                blocks_count,
            )
        add_metadata_range(
            metadata_intervals,
            gdt_offset // block_size,
            math.ceil((gdt_offset + len(descriptors)) / block_size),
            first_data_block,
            blocks_per_group,
            blocks_count,
        )

        initial_bitmaps: dict[int, bytes] = {}
        for group, record in records.items():
            if record["initial_flags"] & EXT4_BG_BLOCK_UNINIT:
                continue
            bitmap_block = record["block_bitmap"]
            if first_data_block <= bitmap_block < blocks_count:
                bitmap = os.pread(descriptor, block_size, bitmap_block * block_size)
                if len(bitmap) == block_size:
                    initial_bitmaps[group] = bitmap

        def logically_free(group: int, bit: int, flags: int, bitmap: bytes) -> bool:
            group_start = first_data_block + group * blocks_per_group
            valid = min(blocks_per_group, blocks_count - group_start)
            if bit < 0 or bit >= valid:
                return False
            if flags & EXT4_BG_BLOCK_UNINIT:
                if (
                    bit < UNINIT_EDGE_GUARD_BLOCKS
                    or bit >= valid - UNINIT_EDGE_GUARD_BLOCKS
                    or in_metadata_interval(metadata_intervals[group], bit)
                ):
                    return False
                return True
            return bool(bitmap) and bit_is_free(bitmap, bit)

        target_windows: list[int] = []
        for physical_offset, length in TARGET_EXTENTS_BYTES:
            if physical_offset % block_size or length % block_size:
                base["reason"] = "unaligned_target_extent"
                return base
            start_block = physical_offset // block_size
            end_block = (physical_offset + length) // block_size
            target_windows.extend(
                range(start_block, end_block - WINDOW_BLOCKS + 1, WINDOW_BLOCKS)
            )
        free_window_candidates: list[int] = []
        for block_number in target_windows:
            if not (first_data_block <= block_number < blocks_count):
                continue
            relative = block_number - first_data_block
            group = relative // blocks_per_group
            bit = relative % blocks_per_group
            record = records[group]
            bitmap = initial_bitmaps.get(group, b"")
            if all(
                logically_free(group, bit + offset, record["initial_flags"], bitmap)
                for offset in range(WINDOW_BLOCKS)
            ):
                free_window_candidates.append(block_number)
        selected = uniform_subset(
            free_window_candidates,
            min(MAX_SAMPLE_WINDOWS, MAX_SAMPLE_BYTES // (block_size * WINDOW_BLOCKS)),
        )
        by_group: dict[int, list[int]] = defaultdict(list)
        for block_number in selected:
            by_group[(block_number - first_data_block) // blocks_per_group].append(block_number)

        matches = 0
        revalidated_blocks = 0
        race_discarded_windows = 0
        read_error_windows = 0
        uninitialized_group_blocks = 0
        initialized_group_blocks = 0
        for group, block_numbers in sorted(by_group.items()):
            record = records[group]
            before_flags_raw = os.pread(descriptor, 2, record["flags_offset"])
            if len(before_flags_raw) != 2:
                read_error_windows += len(block_numbers)
                continue
            before_flags = u16(before_flags_raw, 0)
            before_bitmap = b""
            if not before_flags & EXT4_BG_BLOCK_UNINIT:
                before_bitmap = os.pread(
                    descriptor, block_size, record["block_bitmap"] * block_size
                )
            observations: dict[int, int] = {}
            for block_number in block_numbers:
                bit = (block_number - first_data_block) % blocks_per_group
                if not all(
                    logically_free(group, bit + offset, before_flags, before_bitmap)
                    for offset in range(WINDOW_BLOCKS)
                ):
                    race_discarded_windows += 1
                    continue
                data = os.pread(
                    descriptor,
                    block_size * WINDOW_BLOCKS,
                    block_number * block_size,
                )
                if len(data) != block_size * WINDOW_BLOCKS:
                    read_error_windows += 1
                    continue
                observations[block_number] = data.count(CONTROL_MARKER)
            after_bitmap = b""
            after_flags_raw = os.pread(descriptor, 2, record["flags_offset"])
            if len(after_flags_raw) != 2:
                race_discarded_windows += len(observations)
                continue
            after_flags = u16(after_flags_raw, 0)
            if not after_flags & EXT4_BG_BLOCK_UNINIT:
                after_bitmap = os.pread(
                    descriptor, block_size, record["block_bitmap"] * block_size
                )
            for block_number, block_matches in observations.items():
                bit = (block_number - first_data_block) % blocks_per_group
                if not all(
                    logically_free(group, bit + offset, after_flags, after_bitmap)
                    for offset in range(WINDOW_BLOCKS)
                ):
                    race_discarded_windows += 1
                    continue
                matches += block_matches
                revalidated_blocks += WINDOW_BLOCKS
                if before_flags & after_flags & EXT4_BG_BLOCK_UNINIT:
                    uninitialized_group_blocks += WINDOW_BLOCKS
                else:
                    initialized_group_blocks += WINDOW_BLOCKS
        return {
            **base,
            "supported": True,
            "block_size": block_size,
            "block_count": blocks_count,
            "group_count": group_count,
            "target_extents_bytes": [list(item) for item in TARGET_EXTENTS_BYTES],
            "target_window_count": len(target_windows),
            "logical_free_target_windows": len(free_window_candidates),
            "window_blocks": WINDOW_BLOCKS,
            "selected_free_windows": len(selected),
            "revalidated_free_blocks": revalidated_blocks,
            "revalidated_uninitialized_group_blocks": uninitialized_group_blocks,
            "revalidated_initialized_group_blocks": initialized_group_blocks,
            "sampled_bytes": revalidated_blocks * block_size,
            "race_discarded_windows": race_discarded_windows,
            "read_error_windows": read_error_windows,
            "control_marker_matches": matches,
            "control_marker_sha256": hashlib.sha256(CONTROL_MARKER).hexdigest(),
            "double_allocation_state_revalidation": True,
            "uninitialized_group_metadata_excluded": True,
        }
    except (OSError, struct.error, OverflowError, ValueError) as exc:
        base["reason"] = type(exc).__name__
        return base
    finally:
        os.close(descriptor)


def main() -> int:
    device, filesystem_type = root_device()
    disk_hash = disk_identity_hash(device)
    gpu_hashes = gpu_identity_hashes()
    scan = ext4_marker_scan(device, filesystem_type)
    matches = int(scan.get("control_marker_matches", 0))
    result = {
        "schema": "meta-lhe-fairchem-t4-osdisk-exact-reader-v3",
        "reader_attempt": READER_ATTEMPT,
        "scope": "writer FIEMAP extents that remain logical ext4 free blocks",
        "scan": scan,
        "root_disk_identity_salted_sha256_24": disk_hash,
        "gpu_uuid_salted_sha256_24": gpu_hashes,
        "comparisons": {
            "same_gpu_uuid_hash_as_writer": WRITER_GPU_HASH in gpu_hashes,
            "same_virtual_disk_identity_hash_as_writer": disk_hash == WRITER_DISK_HASH,
            "writer_control_marker_found": matches > 0,
            "cross_vm_storage_residual_confirmed": matches > 0,
        },
        "raw_disk_gpu_or_file_identifier_emitted": False,
        "raw_block_or_file_bytes_emitted": False,
        "network_scan_performed": False,
        "external_state_changed": False,
        "write_performed": False,
    }
    print(f"{PREFIX}_json={json.dumps(result, sort_keys=True, separators=(',', ':'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
