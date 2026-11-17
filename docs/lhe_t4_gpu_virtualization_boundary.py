#!/usr/bin/env python3
"""Classify the T4 virtualization boundary without changing device state.

Only fixed enums and counts are emitted. GPU UUIDs, serials, PCI addresses,
VM identifiers, credentials and raw command output are intentionally omitted.
"""

from __future__ import annotations

import json
import subprocess
import xml.etree.ElementTree as ET


def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return subprocess.CompletedProcess(argv, 127, "", "unavailable")


def fixed(value: str | None) -> str:
    normalized = (value or "absent").strip().lower().replace("-", "_")
    allowed = {
        "none",
        "pass_through",
        "vgpu",
        "host_vgpu",
        "host_vsga",
        "n/a",
        "absent",
        "default",
        "sriov",
        "non_sriov",
        "supported",
        "not_supported",
    }
    return normalized if normalized in allowed else "other"


def main() -> int:
    result = run(["nvidia-smi", "-q", "-x"])
    payload: dict[str, object] = {
        "schema": "meta-lhe-fairchem-t4-gpu-virtualization-v1",
        "nvidia_smi_success": result.returncode == 0,
        "gpu_count": 0,
        "modes": [],
        "host_vgpu_modes": [],
        "vgpu_device_capability_blocks": 0,
        "licensed_product_blocks": 0,
        "pass_through_count": 0,
        "vgpu_guest_count": 0,
        "host_vgpu_count": 0,
        "network_requests": 0,
        "device_writes": 0,
        "reset_unbind_or_firmware_operations": 0,
        "identifier_or_credential_emitted": False,
    }
    if result.returncode != 0:
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 1

    try:
        root = ET.fromstring(result.stdout)
    except ET.ParseError:
        payload["xml_parse"] = False
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 1

    modes: list[str] = []
    host_modes: list[str] = []
    gpus = root.findall("gpu")
    for gpu in gpus:
        block = gpu.find("gpu_virtualization_mode")
        mode = fixed(block.findtext("virtualization_mode") if block is not None else None)
        host_mode = fixed(block.findtext("host_vgpu_mode") if block is not None else None)
        modes.append(mode)
        host_modes.append(host_mode)

    payload.update(
        {
            "xml_parse": True,
            "gpu_count": len(gpus),
            "modes": sorted(modes),
            "host_vgpu_modes": sorted(host_modes),
            "vgpu_device_capability_blocks": len(root.findall(".//vgpu_device_capability")),
            "licensed_product_blocks": len(root.findall(".//vgpu_software_licensed_product")),
            "pass_through_count": sum(mode == "pass_through" for mode in modes),
            "vgpu_guest_count": sum(mode == "vgpu" for mode in modes),
            "host_vgpu_count": sum(mode in {"host_vgpu", "host_vsga"} for mode in modes),
        }
    )
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
