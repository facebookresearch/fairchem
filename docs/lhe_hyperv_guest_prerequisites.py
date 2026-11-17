#!/usr/bin/env python3
"""Emit a bounded, identifier-free Hyper-V guest prerequisite inventory."""

from __future__ import annotations

import json
import os
from pathlib import Path


ALLOWED_NETWORK_DRIVERS = {
    "ena",
    "hv_netvsc",
    "mlx4_core",
    "mlx5_core",
    "virtio_net",
}
ALLOWED_BUS_CLASSES = {"pci", "vmbus", "virtio"}
ALLOWED_TPM_DRIVERS = {"tpm_crb", "tpm_tis", "tpm_tis_core"}


def symlink_basename(path: Path) -> str | None:
    try:
        return path.resolve(strict=True).name
    except (FileNotFoundError, OSError, RuntimeError):
        return None


def allowlisted(value: str | None, allowed: set[str]) -> str:
    if value is None:
        return "none"
    return value if value in allowed else "other"


def default_route_interface() -> str | None:
    try:
        lines = Path("/proc/net/route").read_text().splitlines()[1:]
    except (FileNotFoundError, OSError):
        return None
    for line in lines:
        fields = line.split()
        if len(fields) < 4:
            continue
        try:
            flags = int(fields[3], 16)
        except ValueError:
            continue
        if fields[1] == "00000000" and flags & 0x2:
            return fields[0]
    return None


def bound_device_count(driver_dir: Path) -> int:
    try:
        entries = list(driver_dir.iterdir())
    except (FileNotFoundError, OSError):
        return 0
    return sum(
        entry.is_symlink()
        and entry.name not in {"module", "subsystem"}
        and (entry.resolve(strict=False) / "uevent").exists()
        for entry in entries
    )


def module_present(name: str) -> bool:
    if Path(f"/sys/module/{name}").exists():
        return True
    try:
        return any(
            line.split(maxsplit=1)[0] == name
            for line in Path("/proc/modules").read_text().splitlines()
            if line
        )
    except (FileNotFoundError, OSError):
        return False


def cpu_virtualization_flags() -> tuple[bool, bool]:
    try:
        text = Path("/proc/cpuinfo").read_text()
    except (FileNotFoundError, OSError):
        return False, False
    flags: set[str] = set()
    for line in text.splitlines():
        if line.startswith("flags") and ":" in line:
            flags.update(line.split(":", 1)[1].split())
    return "vmx" in flags, "svm" in flags


def nested_parameter() -> str:
    for path in (
        Path("/sys/module/kvm_intel/parameters/nested"),
        Path("/sys/module/kvm_amd/parameters/nested"),
    ):
        try:
            value = path.read_text().strip().lower()
        except (FileNotFoundError, OSError):
            continue
        if value in {"1", "y", "yes"}:
            return "enabled"
        if value in {"0", "n", "no"}:
            return "disabled"
        return "other"
    return "absent"


def main() -> int:
    interface = default_route_interface()
    net_path = Path("/sys/class/net") / interface if interface else None
    network_driver = (
        symlink_basename(net_path / "device/driver") if net_path is not None else None
    )
    network_bus = (
        symlink_basename(net_path / "device/subsystem") if net_path is not None else None
    )
    netvsc_bound = bound_device_count(Path("/sys/bus/vmbus/drivers/hv_netvsc"))

    tpm_nodes = sorted(Path("/sys/class/tpm").glob("tpm[0-9]*"))
    tpm_driver = symlink_basename(tpm_nodes[0] / "device/driver") if tpm_nodes else None
    vmx, svm = cpu_virtualization_flags()

    payload = {
        "schema": "meta-lhe-fairchem-hyperv-guest-prerequisites-v1",
        "euid_is_root": os.geteuid() == 0,
        "network": {
            "default_route_present": interface is not None,
            "default_route_driver_class": allowlisted(
                network_driver, ALLOWED_NETWORK_DRIVERS
            ),
            "default_route_bus_class": allowlisted(network_bus, ALLOWED_BUS_CLASSES),
            "default_route_is_hv_netvsc": network_driver == "hv_netvsc",
            "default_route_is_vmbus": network_bus == "vmbus",
            "hv_netvsc_driver_present": Path(
                "/sys/bus/vmbus/drivers/hv_netvsc"
            ).exists(),
            "hv_netvsc_bound_device_count": netvsc_bound,
        },
        "vtpm": {
            "sysfs_device_count": len(tpm_nodes),
            "dev_tpm0_present": Path("/dev/tpm0").exists(),
            "dev_tpmrm0_present": Path("/dev/tpmrm0").exists(),
            "driver_class": allowlisted(tpm_driver, ALLOWED_TPM_DRIVERS),
        },
        "nested_virtualization": {
            "dev_kvm_present": Path("/dev/kvm").exists(),
            "cpu_vmx_flag": vmx,
            "cpu_svm_flag": svm,
            "kvm_module_present": module_present("kvm"),
            "kvm_intel_module_present": module_present("kvm_intel"),
            "kvm_amd_module_present": module_present("kvm_amd"),
            "nested_parameter": nested_parameter(),
        },
        "integration_endpoints": {
            "hv_utils_driver_present": Path(
                "/sys/bus/vmbus/drivers/hv_utils"
            ).exists(),
            "hv_fcopy_module_present": module_present("hv_fcopy"),
            "hv_fcopy_device_present": Path("/dev/vmbus/hv_fcopy").exists(),
            "hv_kvp_device_present": Path("/dev/vmbus/hv_kvp").exists(),
            "hv_vss_device_present": Path("/dev/vmbus/hv_vss").exists(),
        },
        "mutation": {
            "device_message_sent": False,
            "network_packet_crafted": False,
            "hypercall_sent": False,
            "device_state_changed": False,
        },
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
