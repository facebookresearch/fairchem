#!/usr/bin/env python3
"""Search for rolling marker G, then seed rolling marker H."""

from __future__ import annotations

import ctypes
import hashlib
import json
import mmap
import subprocess


ALLOCATION_BYTES = 512 * 1024 * 1024
PREVIOUS_MARKER = (
    b"META_LHE_FAIRCHEM_T4_VRAM_ROLLING_G_"
    b"8c917762f6604be7a5c2ca5e45dc2dbe_V1"
)
NEXT_MARKER = (
    b"META_LHE_FAIRCHEM_T4_VRAM_ROLLING_H_"
    b"b1af7350ce95452684d769bd069beaf6_V1"
)
SALT = b"Meta-LHE/FairChem/T4/vram-cross-vm/2026-11-17/v1"
PREVIOUS_WRITER_GPU_HASH = "2545d2cec3f8f5b008b5b625"
ROLLING_STEP = "H"


def checked(function: ctypes._CFuncPtr, *arguments: object) -> None:
    result = int(function(*arguments))
    if result != 0:
        raise RuntimeError(f"CUDA call failed with result class {result}")


def configure_cuda() -> ctypes.CDLL:
    cuda = ctypes.CDLL("libcuda.so.1")
    signatures = {
        "cuInit": ([ctypes.c_uint], ctypes.c_int),
        "cuDeviceGet": (
            [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
            ctypes.c_int,
        ),
        "cuCtxCreate_v2": (
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int],
            ctypes.c_int,
        ),
        "cuMemGetInfo_v2": (
            [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)],
            ctypes.c_int,
        ),
        "cuMemAlloc_v2": (
            [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t],
            ctypes.c_int,
        ),
        "cuMemcpyDtoH_v2": (
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t],
            ctypes.c_int,
        ),
        "cuMemcpyHtoD_v2": (
            [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t],
            ctypes.c_int,
        ),
        "cuCtxSynchronize": ([], ctypes.c_int),
        "cuMemFree_v2": ([ctypes.c_uint64], ctypes.c_int),
        "cuCtxDestroy_v2": ([ctypes.c_void_p], ctypes.c_int),
    }
    for name, (argument_types, result_type) in signatures.items():
        function = getattr(cuda, name)
        function.argtypes = argument_types
        function.restype = result_type
    return cuda


def salted_gpu_hash() -> str:
    process = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    uuid = process.stdout.strip().splitlines()[0]
    return hashlib.sha256(SALT + uuid.encode()).hexdigest()[:24]


def main() -> None:
    cuda = configure_cuda()
    checked(cuda.cuInit, 0)
    device = ctypes.c_int()
    checked(cuda.cuDeviceGet, ctypes.byref(device), 0)
    context = ctypes.c_void_p()
    checked(cuda.cuCtxCreate_v2, ctypes.byref(context), 0, device)

    free_bytes = ctypes.c_size_t()
    total_bytes = ctypes.c_size_t()
    checked(
        cuda.cuMemGetInfo_v2,
        ctypes.byref(free_bytes),
        ctypes.byref(total_bytes),
    )
    if free_bytes.value < ALLOCATION_BYTES + 256 * 1024 * 1024:
        raise RuntimeError("bounded allocation would leave insufficient headroom")

    host = mmap.mmap(-1, ALLOCATION_BYTES)
    device_pointer = ctypes.c_uint64()
    checked(
        cuda.cuMemAlloc_v2,
        ctypes.byref(device_pointer),
        ALLOCATION_BYTES,
    )
    host_pointer = ctypes.c_void_p(
        ctypes.addressof(ctypes.c_char.from_buffer(host))
    )
    checked(
        cuda.cuMemcpyDtoH_v2,
        host_pointer,
        device_pointer,
        ALLOCATION_BYTES,
    )
    checked(cuda.cuCtxSynchronize)

    previous_marker_present = host.find(PREVIOUS_MARKER) != -1
    current_gpu_hash = salted_gpu_hash()

    # Overwrite the entire guest buffer with the next public marker, copy it
    # back to the same allocation, and intentionally leave normal process/VM
    # teardown to release the context for the next rolling step.
    chunk_bytes = 1024 * 1024
    next_chunk = (
        NEXT_MARKER
        * ((chunk_bytes + len(NEXT_MARKER) - 1) // len(NEXT_MARKER))
    )[:chunk_bytes]
    for offset in range(0, ALLOCATION_BYTES, chunk_bytes):
        host[offset : offset + chunk_bytes] = next_chunk
    checked(
        cuda.cuMemcpyHtoD_v2,
        device_pointer,
        host_pointer,
        ALLOCATION_BYTES,
    )
    checked(cuda.cuCtxSynchronize)

    # Do not retain any device bytes in guest RAM after the bounded check.
    zero_chunk = b"\x00" * (1024 * 1024)
    for offset in range(0, ALLOCATION_BYTES, len(zero_chunk)):
        host[offset : offset + len(zero_chunk)] = zero_chunk
    host.close()

    result = {
        "schema": "meta-lhe-fairchem-t4-vram-rolling-reader-writer-v1",
        "rolling_step": ROLLING_STEP,
        "allocation_bytes": ALLOCATION_BYTES,
        "previous_writer_gpu_uuid_salted_sha256_24": PREVIOUS_WRITER_GPU_HASH,
        "current_gpu_uuid_salted_sha256_24": current_gpu_hash,
        "same_physical_gpu_as_previous_writer": (
            current_gpu_hash == PREVIOUS_WRITER_GPU_HASH
        ),
        "previous_marker_sha256": hashlib.sha256(PREVIOUS_MARKER).hexdigest(),
        "previous_marker_present": previous_marker_present,
        "next_marker_sha256": hashlib.sha256(NEXT_MARKER).hexdigest(),
        "next_marker_repetitions": ALLOCATION_BYTES // len(NEXT_MARKER),
        "next_marker_written": True,
        "allocation_freed_by_script": False,
        "context_destroyed_by_script": False,
        "host_buffer_zeroed_after_search": True,
        "raw_gpu_uuid_emitted": False,
        "raw_device_bytes_emitted": False,
        "raw_device_bytes_persisted": False,
        "credential_or_identifier_emitted": False,
    }
    print(
        "META_LHE_FAIRCHEM_T4_VRAM_ROLLING="
        + json.dumps(result, sort_keys=True)
    )


if __name__ == "__main__":
    main()
