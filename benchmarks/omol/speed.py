"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Measure ASE molecular-dynamics step times for OMol models.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from ase import units
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet

HERE = Path(__file__).parent
STRUCTURE = HERE / "ice_Ih_1296.extxyz"
MODELS = {
    "pet_omol_s": "PET-OMol-S",
    "pet_omol_m": "PET-OMol-M",
    "pet_omol_l": "PET-OMol-L",
    "orbmol_v2": "OrbMol-v2",
    "mace_mh_1": "MACE-MH-1",
    "uma_s": "UMA-S 1.2.1",
    "uma_m": "UMA-M 1.1",
    "mace_omol_0": "MACE-OMOL-0",
    "mace_polar_1_m": "MACE-POLAR-1-M",
}
PET_CHECKPOINTS = {
    "pet_omol_s": "pet-omol-s-v1.0.0.ckpt",
    "pet_omol_m": "pet-omol-m-v1.0.0.ckpt",
    "pet_omol_l": "pet-omol-l-v1.0.0.ckpt",
}
WARMUP_STEPS = 200
MEASURED_STEPS = 100


def model_file(checkpoint_dir: Path, name: str) -> str:
    path = checkpoint_dir / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def pet_calculator(model: str, checkpoint_dir: Path):
    """Create a PET calculator with conservative forces in IEEE FP32."""
    from upet.calculator import UPETCalculator

    # TF32 caused rare outliers in PET's Cartesian attention.
    torch.set_float32_matmul_precision("highest")
    return UPETCalculator(
        checkpoint_path=model_file(checkpoint_dir, PET_CHECKPOINTS[model]),
        device="cuda",
        dtype="float32",
        non_conservative=False,
        check_consistency=False,
    )


def uma_calculator(model: str):
    """Create an official FairChem ASE calculator."""
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    model_name = "uma-s-1p2p1" if model == "uma_s" else "uma-m-1p1"
    predictor = pretrained_mlip.get_predict_unit(
        model_name,
        device="cuda",
        inference_settings="turbo",
        workers=1,
        seed=41,
    )
    return FAIRChemCalculator(predictor, task_name="omol")


def mace_calculator(model: str, checkpoint_dir: Path, for_speed: bool):
    """Create a MACE calculator with CuEq acceleration."""
    from mace.calculators import mace_mp, mace_omol, mace_polar

    if model == "mace_mh_1":
        return mace_mp(
            model=model_file(checkpoint_dir, "mace-mh-1.model"),
            head="omol",
            device="cuda",
            default_dtype="float32",
            dispersion=False,
            enable_cueq=True,
            compile_mode="reduce-overhead" if for_speed else None,
        )
    if model == "mace_omol_0":
        return mace_omol(
            model=model_file(checkpoint_dir, "MACE-omol-0-extra-large-1024.model"),
            device="cuda",
            default_dtype="float32",
            enable_cueq=True,
        )
    return mace_polar(
        model=model_file(checkpoint_dir, "MACE-POLAR-1-M.model"),
        device="cuda",
        default_dtype="float32",
        enable_cueq=True,
    )


def make_calculator(model: str, checkpoint_dir: Path, for_speed: bool = True):
    """Create the official ASE calculator for one model."""
    if model in PET_CHECKPOINTS:
        return pet_calculator(model, checkpoint_dir)
    if model.startswith("uma_"):
        return uma_calculator(model)
    if model.startswith("mace_"):
        return mace_calculator(model, checkpoint_dir, for_speed)
    if model == "orbmol_v2":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.inference.calculator import ORBCalculator

        orb, adapter = pretrained.orbmol_v2(
            weights_path=model_file(checkpoint_dir, "orbmol-v2-teqabfhg-20260523.ckpt"),
            device="cuda",
            precision="float32-high",
            compile=True,
        )
        orb.disable_stress()
        return ORBCalculator(
            orb,
            atoms_adapter=adapter,
            device="cuda",
            edge_method="knn_alchemi",
            half_supercell=None,
        )
    raise ValueError(model)


def timed_step(steps) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    next(steps)
    torch.cuda.synchronize()
    return (time.perf_counter_ns() - start) / 1e6


def benchmark(model: str, checkpoint_dir: Path) -> dict:
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(41)
    atoms = read(STRUCTURE)
    atoms.info.update(charge=0, spin=1)
    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=1.0,
        force_temp=True,
        rng=np.random.default_rng(201_337),
    )
    Stationary(atoms, preserve_temperature=True)
    atoms.calc = make_calculator(model, checkpoint_dir)

    dynamics = VelocityVerlet(atoms, timestep=units.fs, logfile=None)
    atoms.get_forces(md=True)  # Initialize and compile; excluded from timing.
    steps = dynamics.irun(steps=WARMUP_STEPS + MEASURED_STEPS)
    next(steps)  # Initialize ASE's generator; also excluded.

    warmup = [timed_step(steps) for _ in range(WARMUP_STEPS)]
    samples = [timed_step(steps) for _ in range(MEASURED_STEPS)]
    return {
        "model": MODELS[model],
        "atoms": len(atoms),
        "gpu": torch.cuda.get_device_name(),
        "warmup_steps": len(warmup),
        "samples_ms": samples,
        "mean_ms_per_step": statistics.mean(samples),
        "standard_deviation_ms": statistics.stdev(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    result = benchmark(args.model, args.checkpoint_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['model']}: {result['mean_ms_per_step']:.3f} ms/step")


if __name__ == "__main__":
    main()
