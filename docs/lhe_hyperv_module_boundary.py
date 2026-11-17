#!/usr/bin/env python3
"""Classify whether the ephemeral T4 guest can run raw VMBus kernel code.

The probe builds a GPL no-op module that has a relocation against
vmbus_send_tl_connect_request, loads it, and immediately unloads it. It does
not call that function, send a VMBus message, open a socket, or touch a device.
Output is deliberately limited to booleans and coarse error classes.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path


SYMBOL = "vmbus_send_tl_connect_request"
MODULE = "fairchem_lhe_probe"


def run(argv: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
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


def read_text(path: Path, limit: int = 1 << 20) -> str | None:
    try:
        return path.read_text(errors="replace")[:limit]
    except OSError:
        return None


def matching_lines(path: Path, predicate: Callable[[str], bool]) -> list[str]:
    result: list[str] = []
    try:
        with path.open(errors="replace") as stream:
            for line in stream:
                if predicate(line):
                    result.append(line.rstrip())
    except OSError:
        pass
    return result


def read_config(release: str) -> dict[str, str]:
    text: str | None = None
    compressed = Path("/proc/config.gz")
    if compressed.exists():
        try:
            with gzip.open(compressed, "rt", errors="replace") as stream:
                text = stream.read(1 << 20)
        except OSError:
            text = None
    for candidate in (
        Path("/boot") / f"config-{release}",
        Path("/lib/modules") / release / "build" / ".config",
    ):
        if text is None:
            text = read_text(candidate)
    wanted = {
        "CONFIG_HYPERV",
        "CONFIG_HYPERV_VSOCKETS",
        "CONFIG_MODULES",
        "CONFIG_MODULE_SIG",
        "CONFIG_MODULE_SIG_FORCE",
        "CONFIG_SECURITY_LOCKDOWN_LSM",
    }
    result: dict[str, str] = {}
    for line in (text or "").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in wanted:
            result[key] = value
    return result


def error_class(stderr: str) -> str:
    value = stderr.lower()
    if "unknown symbol" in value:
        return "unknown-symbol"
    if "key was rejected" in value or "required key" in value or "signature" in value:
        return "signature-policy"
    if "operation not permitted" in value or "permission denied" in value:
        return "permission-policy"
    if "invalid module format" in value:
        return "module-format"
    if "command-not-found" in value:
        return "tool-missing"
    if "timeout" in value:
        return "timeout"
    return "other" if stderr.strip() else "none"


def cap_eff() -> int:
    status = read_text(Path("/proc/self/status"), 65536) or ""
    for line in status.splitlines():
        if line.startswith("CapEff:"):
            try:
                return int(line.split()[1], 16)
            except (IndexError, ValueError):
                return 0
    return 0


def main() -> int:
    release = os.uname().release
    build = Path("/lib/modules") / release / "build"
    matching = [
        line.split()
        for line in matching_lines(
            Path("/proc/kallsyms"),
            lambda line: line.rstrip().endswith(f" {SYMBOL}"),
        )
    ]
    symbol_present = bool(matching)
    symbol_address_visible = any(parts and int(parts[0], 16) != 0 for parts in matching)

    symvers_lines = matching_lines(
        build / "Module.symvers",
        lambda line: f"\t{SYMBOL}\t" in line,
    )
    symbol_exported = bool(symvers_lines)
    symbol_gpl_export = any("EXPORT_SYMBOL_GPL" in line for line in symvers_lines)

    lockdown = (read_text(Path("/sys/kernel/security/lockdown"), 4096) or "").strip()
    modules_disabled = (read_text(Path("/proc/sys/kernel/modules_disabled"), 64) or "").strip()
    sig_enforce = (read_text(Path("/proc/sys/kernel/module_sig_enforce"), 64) or "").strip()

    result: dict[str, object] = {
        "schema": 1,
        "euid_is_root": os.geteuid() == 0,
        "cap_sys_module": bool(cap_eff() & (1 << 16)),
        "hyperv_guest": Path("/sys/bus/vmbus").exists(),
        "hv_vmbus_loaded": Path("/sys/module/hv_vmbus").exists(),
        "hv_sock_loaded": Path("/sys/module/hv_sock").exists(),
        "build_tree_present": build.is_dir(),
        "compiler_present": shutil.which("cc") is not None,
        "make_present": shutil.which("make") is not None,
        "insmod_present": shutil.which("insmod") is not None,
        "rmmod_present": shutil.which("rmmod") is not None,
        "btf_present": Path("/sys/kernel/btf/vmlinux").is_file(),
        "symbol_in_kallsyms": symbol_present,
        "symbol_address_visible": symbol_address_visible,
        "symbol_in_module_symvers": symbol_exported,
        "symbol_is_gpl_export": symbol_gpl_export,
        "modules_disabled": modules_disabled == "1",
        "module_sig_enforced_runtime": sig_enforce == "1",
        "lockdown_active": bool(lockdown and "[none]" not in lockdown),
        "selected_kernel_config": read_config(release),
        "module_build_attempted": False,
        "module_built": False,
        "module_load_attempted": False,
        "module_loaded": False,
        "module_removed": False,
        "module_load_error_class": "not-attempted",
    }

    can_build = build.is_dir() and shutil.which("make") is not None and shutil.which("cc") is not None
    can_load = shutil.which("insmod") is not None and shutil.which("rmmod") is not None
    if can_build:
        result["module_build_attempted"] = True
        with tempfile.TemporaryDirectory(prefix="fairchem-lhe-module-") as temporary:
            directory = Path(temporary)
            (directory / f"{MODULE}.c").write_text(
                """#include <linux/init.h>
#include <linux/module.h>
#include <linux/hyperv.h>
static volatile unsigned long lhe_symbol_sink;
static int __init lhe_init(void)
{
    lhe_symbol_sink = (unsigned long)&vmbus_send_tl_connect_request;
    return lhe_symbol_sink ? 0 : -ENODEV;
}
static void __exit lhe_exit(void) { }
module_init(lhe_init);
module_exit(lhe_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Meta LHE no-op VMBus symbol reachability probe");
"""
            )
            (directory / "Makefile").write_text(f"obj-m += {MODULE}.o\n")
            built = run(["make", "-s", "-C", str(build), f"M={directory}", "modules"])
            module_path = directory / f"{MODULE}.ko"
            result["module_built"] = built.returncode == 0 and module_path.is_file()

            if result["module_built"] and can_load and not Path(f"/sys/module/{MODULE}").exists():
                result["module_load_attempted"] = True
                loaded = run(["insmod", str(module_path)], timeout=30)
                result["module_loaded"] = loaded.returncode == 0 and Path(f"/sys/module/{MODULE}").exists()
                result["module_load_error_class"] = error_class(loaded.stderr)
                if result["module_loaded"]:
                    removed = run(["rmmod", MODULE], timeout=30)
                    result["module_removed"] = (
                        removed.returncode == 0 and not Path(f"/sys/module/{MODULE}").exists()
                    )

    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
