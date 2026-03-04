"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

"""Debug script to compare balance_channels inputs/outputs between commits."""

import sys
import torch
import numpy as np
from pathlib import Path

# Monkey-patch to capture balance function I/O
_captured_io = {"input": None, "output": None, "charge": None, "spin": None, "natoms": None, "batch": None}

def run_benchmark_and_capture(output_prefix: str):
    """Run the benchmark and capture balance function I/O."""
    from fairchem.core.models.uma import escn_md
    
    # Store original function
    original_balance_channels = escn_md.eSCNMDBackbone.balance_channels
    
    def patched_balance_channels(self, x_message_prime, charge, spin, natoms, batch):
        # Capture inputs
        _captured_io["input"] = x_message_prime.detach().cpu().clone()
        _captured_io["charge"] = charge.detach().cpu().clone()
        _captured_io["spin"] = spin.detach().cpu().clone()
        _captured_io["natoms"] = natoms.detach().cpu().clone()
        _captured_io["batch"] = batch.detach().cpu().clone()
        
        # Call original
        result = original_balance_channels(self, x_message_prime, charge, spin, natoms, batch)
        
        # Capture output
        _captured_io["output"] = result.detach().cpu().clone()
        
        return result
    
    # Patch
    escn_md.eSCNMDBackbone.balance_channels = patched_balance_channels
    
    # Import and run benchmark components
    from ase import Atoms
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    
    checkpoint = "/checkpoint/ocp/shared/bwood/prelim_1_2_chkpt/uma-s-1p2-v1.pt"
    natoms = 2000
    
    # Create random structure (same seed as benchmark)
    torch.manual_seed(42)
    np.random.seed(42)
    
    positions = np.random.rand(natoms, 3) * 10
    symbols = ["Cu"] * natoms
    cell = np.eye(3) * 12
    
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    
    # Load model and run inference
    calc = FAIRChemCalculator.from_model_checkpoint(
        checkpoint,
        task_name="oc20",
    )
    atoms.calc = calc
    
    # Run to capture I/O
    forces = atoms.get_forces()
    
    # Save captured data
    output_dir = Path("benchmark_logs")
    output_dir.mkdir(exist_ok=True)
    
    torch.save({
        "input": _captured_io["input"],
        "output": _captured_io["output"],
        "charge": _captured_io["charge"],
        "spin": _captured_io["spin"],
        "natoms": _captured_io["natoms"],
        "batch": _captured_io["batch"],
    }, output_dir / f"{output_prefix}_balance_io.pt")
    
    print(f"Saved I/O to {output_dir / f'{output_prefix}_balance_io.pt'}")
    print(f"  Input shape: {_captured_io['input'].shape}")
    print(f"  Output shape: {_captured_io['output'].shape}")
    print(f"  Charge: {_captured_io['charge']}")
    print(f"  Spin: {_captured_io['spin']}")
    
    return forces

def compare_io(prefix1: str, prefix2: str):
    """Compare I/O between two saved captures."""
    data1 = torch.load(f"benchmark_logs/{prefix1}_balance_io.pt")
    data2 = torch.load(f"benchmark_logs/{prefix2}_balance_io.pt")
    
    print(f"\nComparison: {prefix1} vs {prefix2}")
    print("=" * 60)
    
    for key in ["input", "output", "charge", "spin", "natoms", "batch"]:
        t1, t2 = data1[key], data2[key]
        if t1.shape != t2.shape:
            print(f"  {key}: SHAPE MISMATCH {t1.shape} vs {t2.shape}")
        else:
            diff = (t1.float() - t2.float()).abs()
            print(f"  {key}: MAE={diff.mean().item():.6e}, Max={diff.max().item():.6e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python debug_balance.py capture <prefix>  # Capture I/O with given prefix")
        print("  python debug_balance.py compare <p1> <p2> # Compare two captures")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "capture":
        prefix = sys.argv[2] if len(sys.argv) > 2 else "debug"
        run_benchmark_and_capture(prefix)
    elif cmd == "compare":
        p1 = sys.argv[2]
        p2 = sys.argv[3]
        compare_io(p1, p2)
    else:
        print(f"Unknown command: {cmd}")
