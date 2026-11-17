#!/usr/bin/env python3
"""Probe only documented Hyper-V host service ports without sending data.

The probe loads the standard Linux Hyper-V socket transport when needed and
attempts one short stream connection to each of four Microsoft hcsshim ports.
Every successful socket is closed immediately.  No application bytes are sent,
no response bytes are read, and no VM/SystemId other than VMADDR_CID_HOST is
addressed.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import socket
import subprocess
from pathlib import Path


PORTS = (
    ("hcsshim-entropy", 1),
    ("hcsshim-log", 109),
    ("hcsshim-gcs", 0x40000000),
    ("hcsshim-first-io", 0x40000001),
)


def run(argv: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
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


def error_class(number: int | None) -> str:
    classes = {
        0: "connected",
        errno.ECONNREFUSED: "refused",
        errno.ETIMEDOUT: "timeout",
        errno.ENODEV: "transport-unavailable",
        errno.ENETUNREACH: "unreachable",
        errno.EHOSTUNREACH: "unreachable",
        errno.EACCES: "permission-policy",
        errno.EPERM: "permission-policy",
        errno.EINVAL: "invalid-address",
    }
    return classes.get(number, "other")


def main() -> int:
    module_path = Path("/sys/module/hv_sock")
    initially_loaded = module_path.exists()
    modprobe = shutil.which("modprobe")
    load_attempted = False
    load_succeeded = initially_loaded
    load_error = "not-attempted"

    if not initially_loaded and modprobe:
        load_attempted = True
        loaded = run([modprobe, "hv_sock"])
        load_succeeded = loaded.returncode == 0 and module_path.exists()
        if loaded.returncode == 0:
            load_error = "none"
        elif "not found" in loaded.stderr.lower():
            load_error = "module-missing"
        elif "permission" in loaded.stderr.lower() or "not permitted" in loaded.stderr.lower():
            load_error = "permission-policy"
        else:
            load_error = "other"

    af_vsock = getattr(socket, "AF_VSOCK", None)
    host_cid = getattr(socket, "VMADDR_CID_HOST", 2)
    observations: list[dict[str, object]] = []
    if af_vsock is not None and load_succeeded:
        for name, port in PORTS:
            outcome = "other"
            connected = False
            sock: socket.socket | None = None
            try:
                sock = socket.socket(af_vsock, socket.SOCK_STREAM)
                sock.settimeout(2.0)
                sock.connect((host_cid, port))
                connected = True
                outcome = "connected"
            except TimeoutError:
                outcome = "timeout"
            except OSError as exc:
                outcome = error_class(exc.errno)
            finally:
                if sock is not None:
                    sock.close()
            observations.append(
                {
                    "service": name,
                    "port": port,
                    "connected": connected,
                    "outcome": outcome,
                    "application_bytes_sent": 0,
                    "application_bytes_read": 0,
                }
            )

    removed = False
    if not initially_loaded and load_succeeded and modprobe:
        unloaded = run([modprobe, "-r", "hv_sock"])
        removed = unloaded.returncode == 0 and not module_path.exists()

    result = {
        "schema": 1,
        "euid_is_root": os.geteuid() == 0,
        "hyperv_guest": Path("/sys/bus/vmbus").exists(),
        "af_vsock_supported_by_python": af_vsock is not None,
        "target_cid_class": "VMADDR_CID_HOST",
        "target_count": len(PORTS),
        "documented_hcsshim_ports_only": True,
        "module_initially_loaded": initially_loaded,
        "module_load_attempted": load_attempted,
        "module_load_succeeded": load_succeeded,
        "module_load_error_class": load_error,
        "module_removed": removed,
        "observations": observations,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
