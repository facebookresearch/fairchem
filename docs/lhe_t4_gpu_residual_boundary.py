#!/usr/bin/env python3
"""Bounded Meta LHE probe for CUDA device-memory isolation.

The probe never emits device-memory bytes.  It first records aggregate
zero/non-zero and credential-prefix counts, then writes a fixed research
marker in a short-lived CUDA context and checks for only that marker from a
new process/context.  A cross-job test is intentionally not performed unless
the same-VM context boundary demonstrably retains the controlled marker.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import struct
import subprocess
import sys
from typing import Any


PREFIX = "META_LHE_FAIRCHEM_GPU_RESIDUAL"
GIB = 1024 * 1024 * 1024
MIB = 1024 * 1024
SCAN_LIMIT = 4 * GIB
SCAN_ALLOCATION = 256 * MIB
COPY_WINDOW = 64 * MIB
CONTROL_SIZE = 512 * MIB
CONTROL_STRIDE = 8 * MIB
CONTROL_WORD = 0x4C48454D  # ASCII-ish fixed LHEM research fill word.
CONTROL_FILL_RESIDUAL_THRESHOLD = 1024
CONTROL_MARKER = (
    b"META_LHE_FAIRCHEM_GPU_CONTEXT_MARKER_V1_"
    b"4f7c1b9e55a24b68934fd38c39a490ac"
)
UUID_SALT = b"Meta-LHE/FairChem/T4/GPU-boundary/2026-11-17/v1"
TOKEN_PREFIXES = (
    b"github_pat_",
    b"ghp_",
    b"gho_",
    b"ghs_",
    b"ghu_",
    b"hf_",
    b"AKIA",
    b"ASIA",
    b"eyJ",
)


class CudaError(RuntimeError):
    pass


class Cuda:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcuda.so.1")
        self._bind()
        self._check(self.lib.cuInit(0), "cuInit")
        count = ctypes.c_int()
        self._check(self.lib.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
        if count.value < 1:
            raise CudaError("no CUDA device")
        self.device = ctypes.c_int()
        self._check(self.lib.cuDeviceGet(ctypes.byref(self.device), 0), "cuDeviceGet")
        self.context = ctypes.c_void_p()
        self._check(
            self.lib.cuCtxCreate_v2(ctypes.byref(self.context), 0, self.device),
            "cuCtxCreate_v2",
        )

    def _bind(self) -> None:
        c_uint = ctypes.c_uint
        c_int = ctypes.c_int
        c_size = ctypes.c_size_t
        c_void_p = ctypes.c_void_p
        c_device = ctypes.c_int
        c_ptr = ctypes.POINTER

        self.lib.cuInit.argtypes = [c_uint]
        self.lib.cuInit.restype = c_int
        self.lib.cuDeviceGetCount.argtypes = [c_ptr(c_int)]
        self.lib.cuDeviceGetCount.restype = c_int
        self.lib.cuDeviceGet.argtypes = [c_ptr(c_device), c_int]
        self.lib.cuDeviceGet.restype = c_int
        self.lib.cuCtxCreate_v2.argtypes = [c_ptr(c_void_p), c_uint, c_device]
        self.lib.cuCtxCreate_v2.restype = c_int
        self.lib.cuCtxDestroy_v2.argtypes = [c_void_p]
        self.lib.cuCtxDestroy_v2.restype = c_int
        self.lib.cuDriverGetVersion.argtypes = [c_ptr(c_int)]
        self.lib.cuDriverGetVersion.restype = c_int
        self.lib.cuDeviceTotalMem_v2.argtypes = [c_ptr(c_size), c_device]
        self.lib.cuDeviceTotalMem_v2.restype = c_int
        self.lib.cuMemGetInfo_v2.argtypes = [c_ptr(c_size), c_ptr(c_size)]
        self.lib.cuMemGetInfo_v2.restype = c_int
        self.lib.cuMemAlloc_v2.argtypes = [c_ptr(ctypes.c_uint64), c_size]
        self.lib.cuMemAlloc_v2.restype = c_int
        self.lib.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        self.lib.cuMemFree_v2.restype = c_int
        self.lib.cuMemcpyDtoH_v2.argtypes = [c_void_p, ctypes.c_uint64, c_size]
        self.lib.cuMemcpyDtoH_v2.restype = c_int
        self.lib.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, c_void_p, c_size]
        self.lib.cuMemcpyHtoD_v2.restype = c_int
        self.lib.cuMemsetD32_v2.argtypes = [ctypes.c_uint64, c_uint, c_size]
        self.lib.cuMemsetD32_v2.restype = c_int

    @staticmethod
    def _check(result: int, operation: str) -> None:
        if result != 0:
            raise CudaError(f"{operation} failed with CUDA status {result}")

    def close(self) -> None:
        if getattr(self, "context", None) and self.context.value:
            self.lib.cuCtxDestroy_v2(self.context)
            self.context = ctypes.c_void_p()

    def driver_version(self) -> int:
        value = ctypes.c_int()
        self._check(self.lib.cuDriverGetVersion(ctypes.byref(value)), "cuDriverGetVersion")
        return value.value

    def total_memory(self) -> int:
        value = ctypes.c_size_t()
        self._check(
            self.lib.cuDeviceTotalMem_v2(ctypes.byref(value), self.device),
            "cuDeviceTotalMem_v2",
        )
        return value.value

    def memory_info(self) -> tuple[int, int]:
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        self._check(
            self.lib.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)),
            "cuMemGetInfo_v2",
        )
        return free.value, total.value

    def uuid_hash(self) -> str:
        uuid = (ctypes.c_ubyte * 16)()
        fn = getattr(self.lib, "cuDeviceGetUuid_v2", None)
        if fn is None:
            fn = self.lib.cuDeviceGetUuid
        fn.argtypes = [ctypes.c_void_p, ctypes.c_int]
        fn.restype = ctypes.c_int
        self._check(fn(ctypes.byref(uuid), self.device), "cuDeviceGetUuid")
        return hashlib.sha256(UUID_SALT + bytes(uuid)).hexdigest()[:24]

    def alloc(self, size: int) -> int:
        ptr = ctypes.c_uint64()
        self._check(self.lib.cuMemAlloc_v2(ctypes.byref(ptr), size), "cuMemAlloc_v2")
        return ptr.value

    def free(self, ptr: int) -> None:
        self._check(self.lib.cuMemFree_v2(ptr), "cuMemFree_v2")

    def copy_from_device(self, ptr: int, size: int) -> bytes:
        buf = ctypes.create_string_buffer(size)
        self._check(
            self.lib.cuMemcpyDtoH_v2(ctypes.byref(buf), ptr, size),
            "cuMemcpyDtoH_v2",
        )
        return bytes(buf.raw)

    def copy_to_device(self, ptr: int, data: bytes) -> None:
        buf = ctypes.create_string_buffer(data)
        self._check(
            self.lib.cuMemcpyHtoD_v2(ptr, ctypes.byref(buf), len(data)),
            "cuMemcpyHtoD_v2",
        )

    def fill_word(self, ptr: int, size: int, word: int) -> None:
        self._check(
            self.lib.cuMemsetD32_v2(ptr, word, size // 4),
            "cuMemsetD32_v2",
        )


def identity(cuda: Cuda) -> dict[str, Any]:
    free, total = cuda.memory_info()
    return {
        "driver_version": cuda.driver_version(),
        "gpu_uuid_salted_sha256_24": cuda.uuid_hash(),
        "total_memory_bytes": cuda.total_memory(),
        "free_memory_before_bytes": free,
        "memory_info_total_bytes": total,
    }


def scan_allocations(cuda: Cuda, requested: int) -> dict[str, Any]:
    allocations: list[tuple[int, int]] = []
    remaining = requested
    try:
        while remaining > 0:
            size = min(SCAN_ALLOCATION, remaining)
            try:
                allocations.append((cuda.alloc(size), size))
            except CudaError:
                break
            remaining -= size

        zero_bytes = 0
        nonzero_bytes = 0
        marker_matches = 0
        prefix_counts = {prefix.decode("ascii"): 0 for prefix in TOKEN_PREFIXES}
        for ptr, size in allocations:
            offset = 0
            while offset < size:
                window_size = min(COPY_WINDOW, size - offset)
                data = cuda.copy_from_device(ptr + offset, window_size)
                zeros = data.count(0)
                zero_bytes += zeros
                nonzero_bytes += len(data) - zeros
                marker_matches += data.count(CONTROL_MARKER)
                for prefix in TOKEN_PREFIXES:
                    prefix_counts[prefix.decode("ascii")] += data.count(prefix)
                offset += window_size

        return {
            "requested_bytes": requested,
            "allocated_and_scanned_bytes": sum(size for _, size in allocations),
            "zero_bytes": zero_bytes,
            "nonzero_bytes": nonzero_bytes,
            "fixed_marker_matches": marker_matches,
            "credential_prefix_shape_counts": prefix_counts,
            "raw_bytes_emitted": False,
        }
    finally:
        for ptr, _ in reversed(allocations):
            try:
                cuda.free(ptr)
            except CudaError:
                pass


def initial_scan() -> dict[str, Any]:
    cuda = Cuda()
    try:
        result = {"supported": True, **identity(cuda)}
        free, _ = cuda.memory_info()
        requested = min(SCAN_LIMIT, free // 3)
        requested -= requested % COPY_WINDOW
        if requested < COPY_WINDOW:
            raise CudaError("insufficient bounded scan capacity")
        result["scan"] = scan_allocations(cuda, requested)
        return result
    finally:
        cuda.close()


def write_control() -> dict[str, Any]:
    # Deliberately do not free the allocation or destroy this context.  Process
    # exit is the isolation boundary under test; the CUDA driver owns cleanup.
    cuda = Cuda()
    ptr = cuda.alloc(CONTROL_SIZE)
    cuda.fill_word(ptr, CONTROL_SIZE, CONTROL_WORD)
    copies = 0
    for offset in range(0, CONTROL_SIZE - len(CONTROL_MARKER) + 1, CONTROL_STRIDE):
        cuda.copy_to_device(ptr + offset, CONTROL_MARKER)
        copies += 1
    return {
        "supported": True,
        "gpu_uuid_salted_sha256_24": cuda.uuid_hash(),
        "allocated_bytes": CONTROL_SIZE,
        "fixed_marker_copies_written": copies,
        "raw_bytes_emitted": False,
    }


def read_control() -> dict[str, Any]:
    cuda = Cuda()
    ptr = 0
    try:
        ptr = cuda.alloc(CONTROL_SIZE)
        marker_matches = 0
        fill_word_matches = 0
        fill = struct.pack("<I", CONTROL_WORD)
        offset = 0
        while offset < CONTROL_SIZE:
            size = min(COPY_WINDOW, CONTROL_SIZE - offset)
            data = cuda.copy_from_device(ptr + offset, size)
            marker_matches += data.count(CONTROL_MARKER)
            fill_word_matches += data.count(fill)
            offset += size
        return {
            "supported": True,
            "gpu_uuid_salted_sha256_24": cuda.uuid_hash(),
            "allocated_and_checked_bytes": CONTROL_SIZE,
            "fixed_marker_matches": marker_matches,
            "fixed_fill_word_matches": fill_word_matches,
            "controlled_residual_threshold_words": CONTROL_FILL_RESIDUAL_THRESHOLD,
            "allocation_was_cleared": (
                marker_matches == 0
                and fill_word_matches < CONTROL_FILL_RESIDUAL_THRESHOLD
            ),
            "raw_bytes_emitted": False,
        }
    finally:
        if ptr:
            try:
                cuda.free(ptr)
            except CudaError:
                pass
        cuda.close()


def child_result(mode: str) -> dict[str, Any]:
    try:
        if mode == "initial":
            return initial_scan()
        if mode == "write-control":
            return write_control()
        if mode == "read-control":
            return read_control()
        raise ValueError("unknown mode")
    except (CudaError, OSError, AttributeError) as exc:
        return {
            "supported": False,
            "error_class": type(exc).__name__,
            "raw_error_emitted": False,
        }


def run_child(mode: str) -> dict[str, Any]:
    env = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")}
    if "LD_LIBRARY_PATH" in os.environ:
        env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    try:
        completed = subprocess.run(
            [sys.executable, os.path.abspath(__file__), f"--{mode}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {
            "supported": False,
            "error_class": "ChildTimeout",
            "raw_error_emitted": False,
        }
    try:
        value = json.loads(completed.stdout)
        if not isinstance(value, dict):
            raise ValueError
        return value
    except (json.JSONDecodeError, ValueError):
        return {
            "supported": False,
            "error_class": "ChildProtocolError",
            "child_returncode": completed.returncode,
            "child_stderr_bytes": len(completed.stderr.encode("utf-8")),
            "raw_error_emitted": False,
        }


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1].startswith("--"):
        print(json.dumps(child_result(sys.argv[1][2:]), sort_keys=True))
        return 0

    initial = run_child("initial")
    writer = run_child("write-control")
    reader = run_child("read-control")
    residual = bool(
        writer.get("supported")
        and reader.get("supported")
        and (
            int(reader.get("fixed_marker_matches", 0)) > 0
            or int(reader.get("fixed_fill_word_matches", 0))
            >= CONTROL_FILL_RESIDUAL_THRESHOLD
        )
    )
    same_gpu = bool(
        writer.get("gpu_uuid_salted_sha256_24")
        and writer.get("gpu_uuid_salted_sha256_24")
        == reader.get("gpu_uuid_salted_sha256_24")
    )
    result = {
        "schema": "meta-lhe-fairchem-t4-gpu-residual-boundary-v1",
        "scope": "current T4 job and fresh local CUDA processes only",
        "initial_allocation_scan": initial,
        "controlled_writer": writer,
        "fresh_context_reader": reader,
        "writer_and_reader_same_gpu": same_gpu,
        "cross_context_control_residual_confirmed": residual and same_gpu,
        "cross_job_probe_authorized_by_this_result": residual and same_gpu,
        "cross_job_probe_performed": False,
        "raw_device_bytes_emitted": False,
        "external_state_changed": False,
    }
    print(f"{PREFIX}_json={json.dumps(result, sort_keys=True, separators=(',', ':'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
