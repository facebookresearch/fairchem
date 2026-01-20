from __future__ import annotations

import time
from typing import TYPE_CHECKING

import ase
from ase import units
from ase.md.langevin import Langevin

from fairchem.core import FAIRChemCalculator
from fairchem.core.calculate import pretrained_mlip
from fairchem.core.components.runner import Runner

if TYPE_CHECKING:
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings


# A simple example of running Langevin Dynamics with ASE using fairchem + Ray framework
class ASELangevinUMARunner(Runner):
    def __init__(
        self,
        atoms_list: list[ase.Atoms],  # Changed to list of atoms
        model_name: str = "uma-s-1p1",
        settings: InferenceSettings | str = "default",
        task_name: str = "omat",
        timestep_fs: float = 1.0,
        temp_k: float = 300.0,
        friction_ps_inv: float = 0.001,
        steps_total: int = 1000,
        warmup_steps: int = 10,
        workers: int = 0,
    ):
        self.atoms_list = atoms_list
        self.timestep_fs = timestep_fs
        self.temp_k = temp_k
        self.friction_ps_inv = friction_ps_inv
        self.steps_total = steps_total
        self.warmup_steps = warmup_steps
        self.settings = settings
        self.model_name = model_name
        self.workers = workers
        self.task_name = task_name

    def run(self):
        print(f"Running MD simulations on {len(self.atoms_list)} systems")
        print(f"Steps: warmup={self.warmup_steps}, production={self.steps_total}")
        print("-" * 80)

        results = []

        for i, atoms in enumerate(self.atoms_list):
            predictor = pretrained_mlip.get_predict_unit(
                self.model_name,
                inference_settings=self.settings,
                device="cuda",
                workers=self.workers,
            )
            calc = FAIRChemCalculator(predictor, task_name=self.task_name)
            natoms = len(atoms)
            print(f"\nSystem {i+1}/{len(self.atoms_list)}: {natoms} atoms")

            # Set calculator for this system
            atoms.calc = calc

            # Create dynamics
            dyn = Langevin(
                atoms,
                timestep=self.timestep_fs * units.fs,
                temperature_K=self.temp_k,
                friction=self.friction_ps_inv / units.fs,
            )

            # Warmup
            print(f"  Warming up ({self.warmup_steps} steps)...")
            warmup_start = time.time()
            dyn.run(steps=self.warmup_steps)
            warmup_time = time.time() - warmup_start
            warmup_qps = self.warmup_steps / warmup_time

            # Attach progress reporter for production run
            dyn.attach(
                lambda atoms=atoms, dyn=dyn: print(
                    f"    Step: {dyn.get_number_of_steps()}, "
                    f"E: {atoms.get_potential_energy():.3f} eV"
                ),
                interval=max(self.steps_total // 10, 1),
            )

            # Production run
            print(f"  Production run ({self.steps_total} steps)...")
            start_time = time.time()
            dyn.run(steps=self.steps_total)
            total_time = time.time() - start_time
            production_qps = self.steps_total / total_time

            # Store results
            result = {
                "system_idx": i,
                "natoms": natoms,
                "warmup_qps": warmup_qps,
                "production_qps": production_qps,
                "total_time": total_time,
                "warmup_time": warmup_time,
            }
            results.append(result)

            print(
                f"  Results: warmup_qps={warmup_qps:.2f}, production_qps={production_qps:.2f}"
            )
            del atoms

        # Summary report
        self._print_summary(results)
        return results

    def _print_summary(self, results):
        """Print a summary table of all results"""
        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY")
        print("=" * 80)
        print(
            f"{'System':<8} {'Atoms':<8} {'Warmup QPS':<12} {'Prod QPS':<12} {'Total Time':<12}"
        )
        print("-" * 80)

        for result in results:
            print(
                f"{result['system_idx']+1:<8} "
                f"{result['natoms']:<8} "
                f"{result['warmup_qps']:<12.2f} "
                f"{result['production_qps']:<12.2f} "
                f"{result['total_time']:<12.2f}"
            )

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        pass

    def load_state(self, checkpoint_location: str | None) -> None:
        pass
