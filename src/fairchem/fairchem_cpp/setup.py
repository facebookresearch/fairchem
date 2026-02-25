# Copyright (c) Meta Platforms, Inc.
# All rights reserved.

import os
import glob
import torch

from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

library_name = "fairchem_cpp"

# IMPORTANT: pybind11 does NOT support the limited Python C-API.
# Build a normal (non-limited) extension/wheel.
py_limited_api = False


def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda_env = os.getenv("USE_CUDA", "1") == "1"
    print("DEBUG:", debug_mode)
    print("USE_CUDA (env):", use_cuda_env)
    use_cuda = use_cuda_env and torch.cuda.is_available() and (CUDA_HOME is not None)
    print("USE_CUDA (effective):", use_cuda, "| torch.cuda.is_available():", torch.cuda.is_available(), "| CUDA_HOME:", CUDA_HOME)

    extension = CUDAExtension if use_cuda else CppExtension

    # Common compile options
    cxx_flags = [
        "-O3" if not debug_mode else "-O0",
        "-fdiagnostics-color=always",
        "-std=c++17",
        # Ensure we are NOT in limited API mode for pybind11
        "-UPy_LIMITED_API",
        "-U_Py_LIMITED_API",
    ]

    nvcc_flags = [
        "-O3" if not debug_mode else "-O0",
        "--expt-relaxed-constexpr",
        "-std=c++17",
        # Ensure limited API is not set from any inherited defines
        "-UPy_LIMITED_API",
        "-U_Py_LIMITED_API",
        # PIC for shared objects
        "-Xcompiler", "-fPIC",
    ]

    extra_compile_args = {"cxx": cxx_flags}
    if use_cuda:
        extra_compile_args["nvcc"] = nvcc_flags

    extra_link_args = []
    if debug_mode:
        extra_link_args.extend(["-O0", "-g"])

    # Resolve paths robustly
    this_dir = os.path.abspath(os.path.dirname(__file__))
    extensions_dir = os.path.join(this_dir, library_name, "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))

    if use_cuda:
        cuda_dir = os.path.join(extensions_dir, "cuda")
        cuda_sources = list(glob.glob(os.path.join(cuda_dir, "*.cu")))
        sources += cuda_sources

    print("Building with sources:")
    for s in sources:
        print("  -", s)

    ext_modules = [
        extension(
            name=f"{library_name}._C",
            sources=sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            # Do NOT pass py_limited_api=True; pybind11 needs the full C-API.
            py_limited_api=False,
        )
    ]
    return ext_modules


setup(
    name=library_name,
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch"],
    description="PyTorch C++/CUDA extensions for fairchem_cpp",
    cmdclass={"build_ext": BuildExtension},
    # Do not request a limited-API wheel.
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if False else {},
)

