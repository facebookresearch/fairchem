#!/usr/bin/env python3
"""Emit a bounded, read-only NVIDIA PCI/firmware boundary classification.

The output intentionally excludes GPU UUIDs, serial numbers, PCI addresses,
raw PCI configuration bytes, host/VM identifiers, network data and secrets.
No reset, unbind, ROM read-enable, firmware write or device mutation occurs.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def run(argv: list[str], timeout: int = 20) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(argv, 127, "", "command-not-found")
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(argv, 124, "", "timeout")


def read_text(path: Path, limit: int = 4096) -> str | None:
    try:
        return path.read_text(errors="replace")[:limit].strip()
    except OSError:
        return None


def mode_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"present": path.exists()}
    try:
        metadata = path.stat()
    except OSError:
        return record
    record.update(
        {
            "mode": stat.filemode(metadata.st_mode),
            "uid": metadata.st_uid,
            "gid": metadata.st_gid,
            "size": metadata.st_size,
        }
    )
    return record


def cap_enabled(cap_eff: int, number: int) -> bool:
    return bool(cap_eff & (1 << number))


def safe_nvidia_xml() -> dict[str, Any]:
    result = run(["nvidia-smi", "-q", "-x"])
    record: dict[str, Any] = {
        "available": result.returncode == 0,
        "returncode": result.returncode,
    }
    if result.returncode != 0:
        record["stderr_class"] = result.stderr.strip().splitlines()[:1]
        return record
    try:
        root = ET.fromstring(result.stdout)
    except ET.ParseError:
        record["xml_parse"] = False
        return record

    record["xml_parse"] = True
    record["driver_version"] = root.findtext("driver_version")
    record["cuda_version"] = root.findtext("cuda_version")
    gpus: list[dict[str, Any]] = []
    selected_paths = {
        "product_name": "product_name",
        "product_brand": "product_brand",
        "product_architecture": "product_architecture",
        "display_mode": "display_mode",
        "display_active": "display_active",
        "persistence_mode": "persistence_mode",
        "addressing_mode": "addressing_mode",
        "compute_mode": "compute_mode",
        "performance_state": "performance_state",
        "vbios_version": "vbios_version",
        "gsp_firmware_version": "gsp_firmware_version",
        "gpu_operation_mode/current_gom": "gpu_operation_mode_current",
        "gpu_operation_mode/pending_gom": "gpu_operation_mode_pending",
        "virtualization_mode/host_vgpu_mode": "host_vgpu_mode",
        "virtualization_mode/vgpu_device_capability": "vgpu_device_capability",
        "inforom_version/img_version": "inforom_image",
        "inforom_version/oem_object": "inforom_oem",
        "inforom_version/ecc_object": "inforom_ecc",
        "inforom_version/pwr_object": "inforom_power",
        "ecc_mode/current_ecc": "ecc_current",
        "ecc_mode/pending_ecc": "ecc_pending",
        "fb_memory_usage/total": "framebuffer_total",
        "bar1_memory_usage/total": "bar1_total",
        "pci/pci_device_id": "pci_device_id",
        "pci/pci_sub_system_id": "pci_subsystem_id",
        "pci/pci_gpu_link_info/pcie_gen/max_link_gen": "pcie_max_generation",
        "pci/pci_gpu_link_info/pcie_gen/current_link_gen": "pcie_current_generation",
        "pci/pci_gpu_link_info/link_widths/max_link_width": "pcie_max_width",
        "pci/pci_gpu_link_info/link_widths/current_link_width": "pcie_current_width",
        "gpu_reset_status/reset_required": "reset_required",
        "gpu_reset_status/drain_and_reset_recommended": "drain_and_reset_recommended",
        "confidential_compute/gpu_cc_mode/current_cc_mode": "cc_mode_current",
        "confidential_compute/gpu_cc_mode/pending_cc_mode": "cc_mode_pending",
    }
    for gpu in root.findall("gpu"):
        item: dict[str, Any] = {}
        for xml_path, output_name in selected_paths.items():
            value = gpu.findtext(xml_path)
            if value is not None:
                item[output_name] = value.strip()
        gpus.append(item)
    record["gpu_count"] = len(gpus)
    record["gpus"] = gpus
    return record


def pci_capabilities(device_name: str) -> dict[str, Any]:
    result = run(["lspci", "-vv", "-s", device_name])
    capabilities: set[str] = set()
    dsn_present = False
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line.startswith("Capabilities:"):
            continue
        description = line.split("]", 1)[-1].strip() if "]" in line else line
        if description.startswith("Device Serial Number"):
            dsn_present = True
            continue
        capability = description.split(":", 1)[0].strip()
        if capability:
            capabilities.add(capability)
    return {
        "lspci_returncode": result.returncode,
        "capability_classes": sorted(capabilities),
        "device_serial_number_capability_present": dsn_present,
    }


def pci_record(device: Path) -> dict[str, Any]:
    device_name = device.name
    record: dict[str, Any] = {
        "vendor": read_text(device / "vendor"),
        "device": read_text(device / "device"),
        "subsystem_vendor": read_text(device / "subsystem_vendor"),
        "subsystem_device": read_text(device / "subsystem_device"),
        "class": read_text(device / "class"),
        "numa_node": read_text(device / "numa_node"),
        "boot_vga": read_text(device / "boot_vga"),
        "driver": None,
        "hyperv_vmbus_ancestry": "vmbus" in str(device.resolve()).lower(),
    }
    driver = device / "driver"
    if driver.is_symlink():
        record["driver"] = driver.resolve().name

    group = device / "iommu_group"
    record["iommu_group_present"] = group.is_symlink()
    if group.is_symlink():
        members = list((group.resolve() / "devices").iterdir())
        record["iommu_group_member_count"] = len(members)
        record["iommu_group_member_classes"] = sorted(
            filter(None, (read_text(member / "class") for member in members))
        )
        record["iommu_group_member_vendors"] = sorted(
            filter(None, (read_text(member / "vendor") for member in members))
        )

    attributes = [
        "config",
        "resource",
        "rom",
        "reset",
        "remove",
        "enable",
        "driver_override",
        "sriov_numvfs",
        "sriov_totalvfs",
    ]
    record["attribute_access"] = {
        name: mode_record(device / name) for name in attributes
    }
    record["reset_method"] = read_text(device / "reset_method")

    bar_sizes: list[int] = []
    resource = read_text(device / "resource")
    if resource:
        for line in resource.splitlines():
            fields = line.split()
            if len(fields) < 2:
                continue
            try:
                start, end = int(fields[0], 16), int(fields[1], 16)
            except ValueError:
                continue
            if end >= start and (start or end):
                bar_sizes.append(end - start + 1)
    record["nonzero_bar_sizes"] = bar_sizes
    record.update(pci_capabilities(device_name))
    return record


def main() -> int:
    status = read_text(Path("/proc/self/status")) or ""
    cap_eff = 0
    for line in status.splitlines():
        if line.startswith("CapEff:"):
            cap_eff = int(line.split()[1], 16)
            break

    pci_devices = []
    for device in sorted(Path("/sys/bus/pci/devices").glob("*")):
        if read_text(device / "vendor") != "0x10de":
            continue
        device_class = read_text(device / "class") or ""
        if not device_class.startswith(("0x0300", "0x0302")):
            continue
        pci_devices.append(pci_record(device))

    parameters: dict[str, str | None] = {}
    for name in (
        "EnableGpuFirmware",
        "PreserveVideoMemoryAllocations",
        "RmLogonRC",
        "UsePageAttributeTable",
        "DynamicPowerManagement",
    ):
        parameters[name] = read_text(Path("/sys/module/nvidia/parameters") / name)

    modinfo = run(["modinfo", "nvidia"])
    module_info: dict[str, str] = {}
    for line in modinfo.stdout.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() in {"version", "vermagic", "signer", "sig_id"}:
            module_info[key.strip()] = value.strip()

    tool_names = (
        "nvidia-smi",
        "nvflash",
        "nvflash64",
        "nvidia-installer",
        "nvidia-persistenced",
        "lspci",
        "setpci",
        "mokutil",
    )
    devices = {
        path.name: mode_record(path)
        for path in sorted(Path("/dev").glob("nvidia*"))
        if path.name in {"nvidia0", "nvidiactl", "nvidia-uvm", "nvidia-uvm-tools"}
    }
    firmware_files = list(Path("/lib/firmware/nvidia").rglob("*")) if Path(
        "/lib/firmware/nvidia"
    ).exists() else []

    payload = {
        "schema": "meta-lhe-fairchem-t4-gpu-firmware-boundary-v1",
        "mutation_policy": "read_only_no_reset_no_unbind_no_rom_enable_no_firmware_write",
        "identity_redaction": {
            "gpu_uuid": "omitted",
            "gpu_serial": "omitted",
            "pci_address": "omitted",
            "raw_pci_configuration": "omitted",
        },
        "uid": os.getuid(),
        "effective_capabilities": {
            "CAP_SYS_ADMIN": cap_enabled(cap_eff, 21),
            "CAP_SYS_RAWIO": cap_enabled(cap_eff, 17),
        },
        "kernel_lockdown": read_text(Path("/sys/kernel/security/lockdown")),
        "nvidia_xml": safe_nvidia_xml(),
        "nvidia_pci_device_count": len(pci_devices),
        "nvidia_pci_devices": pci_devices,
        "nvidia_module": module_info,
        "nvidia_module_parameters": parameters,
        "device_nodes": devices,
        "driver_control_files": {
            "unbind": mode_record(Path("/sys/bus/pci/drivers/nvidia/unbind")),
            "bind": mode_record(Path("/sys/bus/pci/drivers/nvidia/bind")),
            "pci_rescan": mode_record(Path("/sys/bus/pci/rescan")),
        },
        "tools_present": {name: shutil.which(name) is not None for name in tool_names},
        "nvidia_firmware_file_count": sum(path.is_file() for path in firmware_files),
    }
    print("META_LHE_FAIRCHEM_GPU_FIRMWARE_JSON_BEGIN")
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    print("META_LHE_FAIRCHEM_GPU_FIRMWARE_JSON_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
