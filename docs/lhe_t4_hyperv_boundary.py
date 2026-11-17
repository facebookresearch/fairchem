#!/usr/bin/env python3
"""Read-only Hyper-V host/interface fingerprint for the FairChem T4 VM."""

from __future__ import annotations

import collections
import json
import os
import re
import socket
import subprocess
import tempfile
from typing import Any


PREFIX = "META_LHE_FAIRCHEM_HYPERV"
CPUID_SOURCE = r"""
#include <cpuid.h>
#include <stdint.h>
#include <stdio.h>

int main(void) {
    unsigned int a, b, c, d;
    __cpuid_count(0x40000000U, 0, a, b, c, d);
    printf("max %08x %08x %08x %08x\n", a, b, c, d);
    if (a < 0x40000000U || a > 0x40010000U) return 0;
    unsigned int last = a < 0x4000000cU ? a : 0x4000000cU;
    for (unsigned int leaf = 0x40000001U; leaf <= last; ++leaf) {
        __cpuid_count(leaf, 0, a, b, c, d);
        printf("leaf %08x %08x %08x %08x %08x\n", leaf, a, b, c, d);
    }
    return 0;
}
"""


def command(
    args: list[str], timeout: int = 20, input_text: str | None = None
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            input=input_text,
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


def decode_ascii_registers(registers: list[int]) -> str:
    raw = b"".join(value.to_bytes(4, "little") for value in registers)
    return raw.rstrip(b"\x00").decode("ascii", errors="replace")


def cpuid_inventory() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="meta-lhe-hyperv-cpuid-") as directory:
        binary = os.path.join(directory, "cpuid")
        compiler = command(
            ["cc", "-O2", "-Wall", "-Wextra", "-x", "c", "-o", binary, "-"],
            timeout=30,
            input_text=CPUID_SOURCE,
        )
        if not compiler.get("ok"):
            return {
                "available": False,
                "compiler_error_class": compiler.get("error_class", "CompilerFailure"),
                "raw_output_emitted": False,
            }
        execution = command([binary])
    if not execution.get("ok"):
        return {"available": False, "raw_output_emitted": False}
    leaves: dict[str, list[str]] = {}
    max_leaf = 0
    vendor = ""
    for line in execution["stdout"].splitlines():
        fields = line.split()
        try:
            if len(fields) == 5 and fields[0] == "max":
                max_leaf = int(fields[1], 16)
                vendor = decode_ascii_registers([int(value, 16) for value in fields[2:5]])
            elif len(fields) == 6 and fields[0] == "leaf":
                leaves[fields[1]] = fields[2:6]
        except ValueError:
            continue
    version = {}
    version_leaf = leaves.get("40000002")
    if version_leaf and vendor == "Microsoft Hv":
        eax, ebx, ecx, edx = [int(value, 16) for value in version_leaf]
        version = {
            "build_number": eax,
            "major_version": (ebx >> 16) & 0xFFFF,
            "minor_version": ebx & 0xFFFF,
            "service_pack": ecx,
            "service_branch": (edx >> 24) & 0xFF,
            "service_number": edx & 0xFFFFFF,
        }
    interface = ""
    interface_leaf = leaves.get("40000001")
    if interface_leaf and vendor == "Microsoft Hv":
        interface = decode_ascii_registers([int(interface_leaf[0], 16)])
    return {
        "available": bool(max_leaf),
        "vendor": vendor,
        "max_leaf_hex": f"{max_leaf:08x}",
        "interface_signature": interface,
        "host_version_leaf": version,
        "registers_hex": leaves,
        "raw_output_emitted": False,
    }


def kernel_log_inventory() -> dict[str, Any]:
    result = command(["dmesg", "--color=never"], timeout=30)
    if not result.get("ok"):
        return {
            "available": False,
            "error_class": result.get("error_class", "DmesgFailure"),
            "raw_lines_emitted": False,
        }
    text = result["stdout"]
    host_builds = sorted(
        set(re.findall(r"Hyper-V:\s*Host Build\s+([0-9]+(?:[.-][0-9]+){1,8})", text))
    )
    vmbus_versions = sorted(
        set(re.findall(r"(?:Vmbus|VMBus|vmbus)[^\n]{0,80}?version\s*[:=]?\s*([0-9.]+)", text))
    )
    isolation_types = sorted(
        set(re.findall(r"Hyper-V:\s*Isolation[^\n]{0,80}?([A-Za-z]+)\s*$", text, re.MULTILINE))
    )
    privilege_pairs = sorted(
        set(
            re.findall(
                r"Hyper-V:\s*privilege flags low\s+(0x[0-9a-fA-F]+)\s+high\s+(0x[0-9a-fA-F]+)",
                text,
            )
        )
    )
    return {
        "available": True,
        "hyperv_line_count": sum(
            1 for line in text.splitlines() if "Hyper-V" in line or "Vmbus" in line
        ),
        "host_builds": host_builds,
        "vmbus_versions": vmbus_versions,
        "isolation_type_tokens": isolation_types,
        "privilege_flag_pairs": [list(pair) for pair in privilege_pairs],
        "raw_lines_emitted": False,
    }


def read_small(path: str) -> str:
    try:
        value = open(path, encoding="utf-8", errors="replace").read(256).strip()
    except OSError:
        return ""
    return value if re.fullmatch(r"[A-Za-z0-9 ._+:/#-]{0,255}", value) else ""


def platform_inventory() -> dict[str, Any]:
    dmi_names = ("bios_vendor", "bios_version", "sys_vendor", "product_name")
    dmi = {
        name: value
        for name in dmi_names
        if (value := read_small(f"/sys/class/dmi/id/{name}"))
    }
    hypervisor_files = (
        "type",
        "version/major",
        "version/minor",
        "version/extra",
        "properties/features",
        "properties/privileges",
        "properties/isolation_type",
    )
    hypervisor = {
        name: value
        for name in hypervisor_files
        if (value := read_small(f"/sys/hypervisor/{name}"))
    }
    lscpu = command(["lscpu", "-J"])
    lscpu_selected: dict[str, str] = {}
    if lscpu.get("ok"):
        try:
            rows = json.loads(lscpu["stdout"]).get("lscpu") or []
            allowed = {
                "Architecture:",
                "Hypervisor vendor:",
                "Virtualization type:",
                "BIOS Vendor ID:",
            }
            lscpu_selected = {
                str(row.get("field")): str(row.get("data"))
                for row in rows
                if row.get("field") in allowed
            }
        except json.JSONDecodeError:
            pass
    return {
        "dmi": dmi,
        "sys_hypervisor": hypervisor,
        "lscpu": lscpu_selected,
        "unique_machine_identifier_emitted": False,
    }


def vmbus_inventory() -> dict[str, Any]:
    class_counts: collections.Counter[str] = collections.Counter()
    driver_counts: collections.Counter[str] = collections.Counter()
    root = "/sys/bus/vmbus/devices"
    if os.path.isdir(root):
        for name in os.listdir(root):
            path = os.path.join(root, name)
            class_id = read_small(os.path.join(path, "class_id")).lower()
            if not class_id:
                try:
                    with open(
                        os.path.join(path, "uevent"),
                        encoding="utf-8",
                        errors="replace",
                    ) as stream:
                        uevent = stream.read(4096)
                except OSError:
                    uevent = ""
                match = re.search(
                    r"^MODALIAS=vmbus:([0-9a-fA-F{}-]{32,38})$", uevent, re.MULTILINE
                )
                if match:
                    class_id = match.group(1).lower()
            if re.fullmatch(r"\{?[0-9a-f-]{36,38}\}?", class_id):
                class_counts[class_id.strip("{}") + ""] += 1
            elif re.fullmatch(r"[0-9a-f]{32}", class_id):
                class_counts[class_id] += 1
            driver_path = os.path.join(path, "driver")
            if os.path.islink(driver_path):
                driver_counts[os.path.basename(os.path.realpath(driver_path))] += 1
    modules: list[str] = []
    try:
        for name in os.listdir("/sys/module"):
            if name.startswith(("hv_", "hyperv", "pci_hyperv")):
                modules.append(name)
    except OSError:
        pass
    return {
        "device_count": sum(class_counts.values()),
        "class_id_counts": dict(sorted(class_counts.items())),
        "driver_counts": dict(sorted(driver_counts.items())),
        "hyperv_modules": sorted(modules),
        "instance_guid_emitted": False,
    }


def guest_device_surface_inventory() -> dict[str, Any]:
    scsi_type_counts: collections.Counter[str] = collections.Counter()
    scsi_driver_counts: collections.Counter[str] = collections.Counter()
    scsi_root = "/sys/class/scsi_device"
    if os.path.isdir(scsi_root):
        for name in os.listdir(scsi_root):
            path = os.path.join(scsi_root, name, "device")
            device_type = read_small(os.path.join(path, "type"))
            if device_type.isdigit():
                scsi_type_counts[device_type] += 1
            driver_path = os.path.join(path, "driver")
            if os.path.islink(driver_path):
                scsi_driver_counts[os.path.basename(os.path.realpath(driver_path))] += 1

    block_type_counts: collections.Counter[str] = collections.Counter()
    block_transport_counts: collections.Counter[str] = collections.Counter()
    lsblk = command(["lsblk", "-J", "-o", "TYPE,TRAN"])
    if lsblk.get("ok"):
        try:
            pending = list(json.loads(lsblk["stdout"]).get("blockdevices") or [])
            while pending:
                row = pending.pop()
                value = str(row.get("type") or "unknown")
                block_type_counts[value] += 1
                transport = str(row.get("tran") or "none")
                block_transport_counts[transport] += 1
                pending.extend(row.get("children") or [])
        except json.JSONDecodeError:
            pass

    pci_driver_counts: collections.Counter[str] = collections.Counter()
    pci_root = "/sys/bus/pci/devices"
    if os.path.isdir(pci_root):
        for name in os.listdir(pci_root):
            driver_path = os.path.join(pci_root, name, "driver")
            if os.path.islink(driver_path):
                pci_driver_counts[os.path.basename(os.path.realpath(driver_path))] += 1

    vsock_socket_create = False
    vsock_error = ""
    try:
        candidate = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
        candidate.close()
        vsock_socket_create = True
    except (AttributeError, OSError) as exc:
        vsock_error = type(exc).__name__

    modinfo = command(["modinfo", "-F", "filename", "hv_sock"])
    return {
        "scsi_type_counts": dict(sorted(scsi_type_counts.items())),
        "scsi_driver_counts": dict(sorted(scsi_driver_counts.items())),
        "block_type_counts": dict(sorted(block_type_counts.items())),
        "block_transport_counts": dict(sorted(block_transport_counts.items())),
        "pci_driver_counts": dict(sorted(pci_driver_counts.items())),
        "virtio_driver_observed": any(
            name.startswith("virtio") for name in pci_driver_counts
        ),
        "hv_sock_module_available": bool(
            modinfo.get("ok") and modinfo.get("stdout", "").strip()
        ),
        "hv_sock_module_loaded": os.path.isdir("/sys/module/hv_sock"),
        "vsock_socket_create_succeeded": vsock_socket_create,
        "vsock_socket_create_error_class": vsock_error,
        "vsock_proc_entry_present": os.path.exists("/proc/net/vsock"),
        "vsock_device_node_present": os.path.exists("/dev/vsock"),
        "vsock_connect_attempted": False,
        "unique_device_identifier_emitted": False,
    }


def main() -> int:
    cpuid = cpuid_inventory()
    kernel_log = kernel_log_inventory()
    cpuid_version = cpuid.get("host_version_leaf", {})
    result = {
        "schema": "meta-lhe-fairchem-t4-hyperv-device-surface-v2",
        "scope": (
            "read-only Hyper-V, VMBus, SCSI, PCI-driver, block, and local "
            "AF_VSOCK capability metadata"
        ),
        "cpuid": cpuid,
        "kernel_log": kernel_log,
        "platform": platform_inventory(),
        "vmbus": vmbus_inventory(),
        "guest_device_surface": guest_device_surface_inventory(),
        "conclusions": {
            "microsoft_hypervisor_observed": cpuid.get("vendor") == "Microsoft Hv",
            "host_build_exposed": bool(kernel_log.get("host_builds") or cpuid_version),
            "root_partition_authority_confirmed": False,
            "guest_to_host_escape_confirmed": False,
        },
        "network_probe_performed": False,
        "raw_kernel_log_emitted": False,
        "cloud_or_machine_identifier_emitted": False,
        "external_state_changed": False,
        "write_performed": False,
    }
    print(f"{PREFIX}_json={json.dumps(result, sort_keys=True, separators=(',', ':'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
