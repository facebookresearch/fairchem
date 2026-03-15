"""
Shared utilities for UMA speed benchmarks.

Provides model loading, system creation, and force/energy comparison.
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ase import Atoms, units
from ase.md.langevin import Langevin

from fairchem.core import FAIRChemCalculator, pretrained_mlip
from fairchem.core.datasets.common_structures import get_fcc_crystal_by_num_atoms
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

MODEL_NAME = "uma-s-1p2"
TASK_NAME = "omat"
NATOMS = 100

# Correctness tolerances (same as GPU benchmark)
FORCES_ATOL = 5e-3
FORCES_RTOL = 5e-3
ENERGY_ATOL_EV = 0.050  # 50 meV
ENERGY_RTOL = 1e-4

GOLD_PKL = Path(__file__).parent / "gold_forces.pkl"


def make_inference_settings(
    backend: str = "general",
    compile: bool = False,
    device: str = "cpu",
) -> InferenceSettings:
    """Build InferenceSettings for a given backend."""
    return InferenceSettings(
        tf32=(device == "cuda"),
        activation_checkpointing=False,
        merge_mole=True,
        compile=compile,
        external_graph_gen=False,
        internal_graph_gen_version=2,
        execution_mode=backend,
    )


def load_predictor(
    backend: str = "general",
    compile: bool = False,
    device: str = "cpu",
) -> pretrained_mlip:
    """Load UMA-S predict unit with given backend and device."""
    settings = make_inference_settings(backend, compile, device)
    log.info("Loading %s on %s  backend=%s  compile=%s", MODEL_NAME, device, backend, compile)
    predictor = pretrained_mlip.get_predict_unit(
        MODEL_NAME,
        inference_settings=settings,
        device=device,
    )
    return predictor


def make_system(natoms: int = NATOMS) -> Atoms:
    """Create a non-PBC FCC carbon system."""
    atoms = get_fcc_crystal_by_num_atoms(natoms)
    atoms.pbc = False
    return atoms


def attach_calculator(atoms: Atoms, predictor) -> Atoms:
    """Attach FAIRChemCalculator to atoms."""
    calc = FAIRChemCalculator(predictor, task_name=TASK_NAME)
    atoms.calc = calc
    return atoms


@dataclass
class GoldStandard:
    """Reference forces and energy from the general backend."""
    energy: float
    forces: np.ndarray
    natoms: int


def save_gold(energy: float, forces: np.ndarray, natoms: int, path: Path = GOLD_PKL) -> None:
    """Save gold-standard reference to pickle."""
    gold = GoldStandard(energy=energy, forces=forces, natoms=natoms)
    with open(path, "wb") as f:
        pickle.dump(gold, f)
    log.info("Saved gold standard to %s  (energy=%.6f eV, natoms=%d)", path, energy, natoms)


def load_gold(path: Path = GOLD_PKL) -> GoldStandard:
    """Load gold-standard reference from pickle."""
    with open(path, "rb") as f:
        gold = pickle.load(f)
    log.info("Loaded gold standard from %s  (energy=%.6f eV, natoms=%d)", path, gold.energy, gold.natoms)
    return gold


def compare(
    energy: float,
    forces: np.ndarray,
    gold: GoldStandard,
) -> tuple[bool, dict]:
    """
    Compare energy and forces against gold standard.

    Returns (passed, details_dict).
    """
    e_abs = abs(energy - gold.energy)
    e_rel = e_abs / max(abs(gold.energy), 1e-12)
    e_pass = e_abs < ENERGY_ATOL_EV and e_rel < ENERGY_RTOL

    f_diff = np.abs(forces - gold.forces)
    f_abs_max = f_diff.max()
    f_abs_mean = f_diff.mean()
    f_close = np.allclose(forces, gold.forces, atol=FORCES_ATOL, rtol=FORCES_RTOL)

    passed = e_pass and f_close

    details = {
        "energy_ref": gold.energy,
        "energy_test": energy,
        "energy_abs_err": e_abs,
        "energy_rel_err": e_rel,
        "energy_pass": e_pass,
        "forces_abs_max_err": f_abs_max,
        "forces_abs_mean_err": f_abs_mean,
        "forces_pass": f_close,
        "overall_pass": passed,
    }
    return passed, details


def print_comparison(details: dict) -> None:
    """Pretty-print comparison results."""
    status = "PASS" if details["overall_pass"] else "FAIL"
    log.info("=" * 60)
    log.info("  Correctness check: %s", status)
    log.info("=" * 60)
    log.info("  Energy ref:       %.6f eV", details["energy_ref"])
    log.info("  Energy test:      %.6f eV", details["energy_test"])
    log.info("  Energy abs err:   %.6e eV  (tol %.1e)", details["energy_abs_err"], ENERGY_ATOL_EV)
    log.info("  Energy rel err:   %.6e     (tol %.1e)", details["energy_rel_err"], ENERGY_RTOL)
    log.info("  Energy:           %s", "PASS" if details["energy_pass"] else "FAIL")
    log.info("  Forces max err:   %.6e     (atol %.1e, rtol %.1e)", details["forces_abs_max_err"], FORCES_ATOL, FORCES_RTOL)
    log.info("  Forces mean err:  %.6e", details["forces_abs_mean_err"])
    log.info("  Forces:           %s", "PASS" if details["forces_pass"] else "FAIL")
    log.info("=" * 60)
    print(status)
