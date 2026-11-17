#!/usr/bin/env python3
"""Write and unlink one bounded control file on the current ephemeral OS disk."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import secrets
import subprocess
from typing import Any


PREFIX = "META_LHE_FAIRCHEM_OSDISK_WRITER"
SALT = b"Meta-LHE/FairChem/T4/osdisk-cross-vm/2026-11-17/v1"
WRITE_BYTES = 256 * 1024 * 1024
BLOCK_SIZE = 4096
CONTROL_PATH = "/var/tmp/meta_lhe_fairchem_osdisk_control.bin"


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


def root_device() -> str:
    result = command(["findmnt", "-n", "-o", "SOURCE,MAJ:MIN", "/"])
    fields = result.get("stdout", "").strip().split()
    if not result.get("ok") or not fields:
        return ""
    source = os.path.realpath(fields[0].split("[", 1)[0])
    if (not source.startswith("/dev/") or not os.path.exists(source)) and len(fields) >= 2:
        devices = command(["lsblk", "-n", "-r", "-p", "-o", "PATH,MAJ:MIN"])
        for line in devices.get("stdout", "").splitlines():
            device_fields = line.split()
            if len(device_fields) == 2 and device_fields[1] == fields[1]:
                source = device_fields[0]
                break
    return source


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


def main() -> int:
    marker = "META_LHE_FAIRCHEM_OSDISK_CROSS_VM_" + secrets.token_hex(32)
    filler = secrets.token_bytes(BLOCK_SIZE - len(marker.encode("ascii")))
    block = marker.encode("ascii") + filler
    before = os.statvfs("/var/tmp")
    descriptor = os.open(
        CONTROL_PATH,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    written = 0
    allocated_bytes = 0
    try:
        if hasattr(os, "posix_fallocate"):
            os.posix_fallocate(descriptor, 0, WRITE_BYTES)
        while written < WRITE_BYTES:
            pending = memoryview(block)[: min(BLOCK_SIZE, WRITE_BYTES - written)]
            while pending:
                count = os.write(descriptor, pending)
                if count <= 0:
                    raise OSError("short control write")
                written += count
                pending = pending[count:]
        os.fsync(descriptor)
        allocated_bytes = os.fstat(descriptor).st_blocks * 512
    finally:
        os.close(descriptor)
    os.sync()
    os.unlink(CONTROL_PATH)
    os.sync()
    after = os.statvfs("/var/tmp")
    device = root_device()
    result = {
        "schema": "meta-lhe-fairchem-t4-osdisk-writer-v1",
        "scope": "one bounded control file on the current ephemeral OS disk",
        "research_control_marker": marker,
        "research_control_marker_sha256": hashlib.sha256(marker.encode("ascii")).hexdigest(),
        "marker_is_non_secret_public_control": True,
        "requested_bytes": WRITE_BYTES,
        "written_bytes": written,
        "allocated_bytes_before_unlink": allocated_bytes,
        "path_exists_after_unlink": os.path.exists(CONTROL_PATH),
        "free_bytes_delta_after_unlink": (
            after.f_bavail * after.f_frsize - before.f_bavail * before.f_frsize
        ),
        "root_device_class": os.path.basename(device),
        "root_disk_identity_salted_sha256_24": disk_identity_hash(device),
        "gpu_uuid_salted_sha256_24": gpu_identity_hashes(),
        "kernel_release": platform.release(),
        "raw_disk_or_gpu_identifier_emitted": False,
        "external_state_changed": False,
        "local_control_write_performed": True,
        "official_or_third_party_data_touched": False,
        "control_file_unlinked": True,
    }
    print(f"{PREFIX}_json={json.dumps(result, sort_keys=True, separators=(',', ':'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
