#!/usr/bin/env python3
"""Search a bounded CUDA allocation only for the public writer marker."""

from __future__ import annotations

import ctypes
import hashlib
import json
import mmap
import subprocess


ALLOCATION_BYTES = 512 * 1024 * 1024
MARKER = (
    b"META_LHE_FAIRCHEM_T4_VRAM_CROSS_VM_"
    b"2f05ab0b7be74bbd9f61862f42e53a58_V1"
)
SALT = b"Meta-LHE/FairChem/T4/vram-cross-vm/2026-11-17/v1"
WRITER_GPU_HASH = "b0fa312bdc154f0fb240d948"
READER_ATTEMPT = "E"


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

    marker_present = host.find(MARKER) != -1
    reader_gpu_hash = salted_gpu_hash()

    # Do not retain any non-marker device bytes in guest RAM after the check.
    zero_chunk = b"\x00" * (1024 * 1024)
    for offset in range(0, ALLOCATION_BYTES, len(zero_chunk)):
        host[offset : offset + len(zero_chunk)] = zero_chunk
    checked(cuda.cuMemFree_v2, device_pointer)
    checked(cuda.cuCtxDestroy_v2, context)
    host.close()

    result = {
        "schema": "meta-lhe-fairchem-t4-vram-reader-v1",
        "reader_attempt": READER_ATTEMPT,
        "allocation_bytes": ALLOCATION_BYTES,
        "writer_gpu_uuid_salted_sha256_24": WRITER_GPU_HASH,
        "reader_gpu_uuid_salted_sha256_24": reader_gpu_hash,
        "same_physical_gpu_as_writer": reader_gpu_hash == WRITER_GPU_HASH,
        "marker_sha256": hashlib.sha256(MARKER).hexdigest(),
        "marker_present": marker_present,
        "host_buffer_zeroed_after_search": True,
        "raw_gpu_uuid_emitted": False,
        "raw_device_bytes_emitted": False,
        "raw_device_bytes_persisted": False,
        "credential_or_identifier_emitted": False,
    }
    print(f"META_LHE_FAIRCHEM_T4_VRAM_READER={json.dumps(result, sort_keys=True)}")


if __name__ == "__main__":
    main()
