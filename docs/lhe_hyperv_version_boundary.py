#!/usr/bin/env python3
"""Read-only Hyper-V system-version classifier for the bounded LHE control."""

from __future__ import annotations

import ctypes
import json
import mmap
import platform


PREFIX = "META_LHE_FAIRCHEM_HYPERV_VERSION"
HYPERV_VENDOR = b"Microsoft Hv"
HYPERV_INTERFACE = 0x31237648  # "Hv#1"

# Microsoft November 2026 fixed-build floors. The same base build can map to
# more than one product with different service floors, so the probe reports
# every candidate comparison and deliberately does not infer the host product.
NOVEMBER_2026_FIXED_FLOORS_BY_BUILD = {
    14393: {"windows_10_1607_or_server_2016": 9688},
    17763: {"windows_10_1809_or_server_2019": 9487},
    19044: {"windows_10_21h2": 7846},
    19045: {"windows_10_22h2": 7846},
    20348: {"windows_server_2022": 5870},
    22631: {"windows_11_23h2": 7711},
    26100: {"windows_11_24h2": 9974, "windows_server_2025": 33822},
    26200: {"windows_11_25h2": 9974},
    26300: {"windows_11_26h2": 9974},
    28000: {"windows_11_26h1": 3501},
}


def make_cpuid():
    # System V AMD64: (leaf=EDI, subleaf=ESI, out=RDX). Preserve RBX and copy
    # the output pointer to R8 before CPUID overwrites EDX.
    code = bytes.fromhex(
        "53"              # push rbx
        "89f8"            # mov eax, edi
        "89f1"            # mov ecx, esi
        "4989d0"          # mov r8, rdx
        "0fa2"            # cpuid
        "418900"          # mov [r8], eax
        "41895804"        # mov [r8+4], ebx
        "41894808"        # mov [r8+8], ecx
        "4189500c"        # mov [r8+12], edx
        "5b"              # pop rbx
        "c3"              # ret
    )
    page = mmap.mmap(
        -1,
        len(code),
        prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
    )
    page.write(code)
    address = ctypes.addressof(ctypes.c_char.from_buffer(page))
    function = ctypes.CFUNCTYPE(
        None,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    )(address)
    return page, function


def cpuid(function, leaf: int, subleaf: int = 0) -> tuple[int, int, int, int]:
    output = (ctypes.c_uint32 * 4)()
    function(leaf, subleaf, output)
    return tuple(int(value) for value in output)


def classify() -> dict[str, object]:
    result: dict[str, object] = {
        "schema": "meta-lhe-fairchem-hyperv-version-v1",
        "architecture": platform.machine(),
        "cpuid_attempted": False,
        "hypervisor_present": False,
        "microsoft_hyperv_vendor": False,
        "hv1_interface": False,
        "version_leaf_available": False,
        "version": None,
        "known_november_2026_floor": False,
        "candidate_product_floor_checks": {},
        "below_all_candidate_floors": None,
        "at_or_above_all_candidate_floors": None,
        "vmbus_request_sent": False,
        "host_or_guest_state_changed": False,
        "identifier_or_credential_emitted": False,
    }
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        result["error_class"] = "unsupported_architecture"
        return result

    page = None
    try:
        page, function = make_cpuid()
        result["cpuid_attempted"] = True
        _, _, leaf1_ecx, _ = cpuid(function, 1)
        result["hypervisor_present"] = bool(leaf1_ecx & (1 << 31))

        max_leaf, vendor_ebx, vendor_ecx, vendor_edx = cpuid(function, 0x40000000)
        vendor = b"".join(
            value.to_bytes(4, "little")
            for value in (vendor_ebx, vendor_ecx, vendor_edx)
        )
        result["microsoft_hyperv_vendor"] = vendor == HYPERV_VENDOR
        result["max_hypervisor_leaf"] = max_leaf
        if max_leaf < 0x40000002:
            result["error_class"] = "version_leaf_unavailable"
            return result

        interface_eax, _, _, _ = cpuid(function, 0x40000001)
        result["hv1_interface"] = interface_eax == HYPERV_INTERFACE
        version_eax, version_ebx, version_ecx, version_edx = cpuid(
            function, 0x40000002
        )
        result["version_leaf_available"] = True
        version = {
            "major": (version_ebx >> 16) & 0xFFFF,
            "minor": version_ebx & 0xFFFF,
            "build": version_eax,
            "service_number": version_edx & 0xFFFFFF,
            "service_pack": version_ecx,
            "service_branch": (version_edx >> 24) & 0xFF,
        }
        result["version"] = version
        floors = NOVEMBER_2026_FIXED_FLOORS_BY_BUILD.get(version["build"], {})
        result["known_november_2026_floor"] = bool(floors)
        if floors:
            checks = {
                product: {
                    "fixed_service_floor": floor,
                    "at_or_above": version["service_number"] >= floor,
                }
                for product, floor in sorted(floors.items())
            }
            result["candidate_product_floor_checks"] = checks
            result["below_all_candidate_floors"] = all(
                not check["at_or_above"] for check in checks.values()
            )
            result["at_or_above_all_candidate_floors"] = all(
                check["at_or_above"] for check in checks.values()
            )
        return result
    except (BufferError, OSError, TypeError, ValueError):
        result["error_class"] = "cpuid_execution_error"
        return result
    finally:
        if page is not None:
            page.close()


def main() -> int:
    print(f"{PREFIX}_json={json.dumps(classify(), sort_keys=True, separators=(',', ':'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
