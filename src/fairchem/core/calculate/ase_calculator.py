"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
from functools import partial
from typing import TYPE_CHECKING, Literal

import numpy as np
from ase.calculators.calculator import Calculator
from ase.stress import full_3x3_to_voigt_6_stress

from fairchem.core.calculate import pretrained_mlip
from fairchem.core.datasets import data_list_collater
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit.api.inference import (
    CHARGE_RANGE,
    DEFAULT_CHARGE,
    DEFAULT_SPIN,
    DEFAULT_SPIN_OMOL,
    SPIN_RANGE,
    InferenceSettings,
    UMATask,
)

if TYPE_CHECKING:
    from ase import Atoms

    from fairchem.core.units.mlip_unit import MLIPPredictUnit


class FAIRChemCalculator(Calculator):
    def __init__(
        self,
        predict_unit: MLIPPredictUnit,
        task_name: UMATask | str | None = None,
        seed: int | None = None,  # deprecated
    ):
        """
        Initialize the FAIRChemCalculator from a model MLIPPredictUnit

        Args:
            predict_unit (MLIPPredictUnit): A pretrained MLIPPredictUnit.
            task_name (UMATask or str, optional): Name of the task to use if using a UMA checkpoint.
                Determines default key names for energy, forces, and stress.
                Can be one of 'omol', 'omat', 'oc20', 'odac', or 'omc'.
        Notes:
            - For models that require total charge and spin multiplicity (currently UMA models on omol mode), `charge`
              and `spin` (corresponding to `spin_multiplicity`) are pulled from `atoms.info` during calculations.
                - `charge` must be an integer representing the total charge on the system and can range from -100 to 100.
                - `spin` must be an integer representing the spin multiplicity and can range from 0 to 100.
                - If `task_name="omol"`, and `charge` or `spin` are not set in `atoms.info`, they will default to
                charge=`0` and spin=`1`.
        """

        super().__init__()

        if seed is not None:
            logging.warning(
                "The 'seed' argument is deprecated and will be removed in future versions. "
                "Please set the seed in the MLIPPredictUnit configuration instead."
            )

        if isinstance(task_name, UMATask):
            task_name = task_name.value

        valid_datasets = list(predict_unit.dataset_to_tasks.keys())
        if task_name is not None:
            assert (
                task_name in valid_datasets
            ), f"Given: {task_name}, Valid options are {valid_datasets}"
            self._task = UMATask(task_name)
        elif len(valid_datasets) == 1:
            self._task = UMATask(valid_datasets[0])
        else:
            raise RuntimeError(
                f"A task name must be provided. Valid options are {valid_datasets}"
            )

        self.implemented_properties = [
            task.property for task in predict_unit.dataset_to_tasks[self.task_name]
        ]
        if "energy" in self.implemented_properties:
            self.implemented_properties.append(
                "free_energy"
            )  # Free energy is a copy of energy, see docstring above

        self.predictor = predict_unit

        self.a2g = partial(
            AtomicData.from_ase,
            task_name=self.task_name,
            r_edges=False,
            r_data_keys=["spin", "charge"],
        )

    @property
    def task_name(self) -> str:
        return self._task.value

    @classmethod
    def from_model_checkpoint(
        cls,
        name_or_path: str,
        task_name: UMATask | None = None,
        inference_settings: InferenceSettings | str = "default",
        overrides: dict | None = None,
        device: Literal["cuda", "cpu"] | None = None,
        seed: int = 41,
    ) -> FAIRChemCalculator:
        """Instantiate a FAIRChemCalculator from a checkpoint file.

        Args:
            cls: The class reference
            name_or_path: A model name from fairchem.core.pretrained.available_models or a path to the checkpoint
                file
            task_name: Task name
            inference_settings: Settings for inference. Can be "default" (general purpose) or "turbo"
                (optimized for speed but requires fixed atomic composition). Advanced use cases can
                use a custom InferenceSettings object.
            overrides: Optional dictionary of settings to override default inference settings.
            device: Optional torch device to load the model onto.
            seed: Random seed for reproducibility.
        """

        if name_or_path in pretrained_mlip.available_models:
            predict_unit = pretrained_mlip.get_predict_unit(
                name_or_path,
                inference_settings=inference_settings,
                overrides=overrides,
                device=device,
            )
        elif os.path.isfile(name_or_path):
            predict_unit = pretrained_mlip.load_predict_unit(
                name_or_path,
                inference_settings=inference_settings,
                overrides=overrides,
                device=device,
            )
        else:
            raise ValueError(
                f"{name_or_path=} is not a valid model name or checkpoint path"
            )
        return cls(predict_unit=predict_unit, task_name=task_name, seed=seed)

    def check_state(self, atoms: Atoms, tol: float = 1e-15) -> list:
        """
        Check for any system changes since the last calculation.

        Args:
            atoms (ase.Atoms): The atomic structure to check.
            tol (float): Tolerance for detecting changes.

        Returns:
            list: A list of changes detected in the system.
        """
        state = super().check_state(atoms, tol=tol)
        if (not state) and (self.atoms.info != atoms.info):
            state.append("info")
        return state

    def calculate(
        self, atoms: Atoms, properties: list[str], system_changes: list[str]
    ) -> None:
        """
        Perform the calculation for the given atomic structure.

        Args:
            atoms (Atoms): The atomic structure to calculate properties for.
            properties (list[str]): The list of properties to calculate.
            system_changes (list[str]): The list of changes in the system.

        Notes:
            - `charge` must be an integer representing the total charge on the system and can range from -100 to 100.
            - `spin` must be an integer representing the spin multiplicity and can range from 0 to 100.
            - If `task_name="omol"`, and `charge` or `spin` are not set in `atoms.info`, they will default to `0`.
            - `charge` and `spin` are currently only used for the `omol` head.
            - The `free_energy` is simply a copy of the `energy` and is not the actual electronic free energy.
              It is only set for ASE routines/optimizers that are hard-coded to use this rather than the `energy` key.
        """

        # Our calculators won't work if natoms=0
        if len(atoms) == 0:
            raise ValueError("Atoms object has no atoms inside.")

        # Check if the atoms object has periodic boundary conditions (PBC) set correctly
        self._check_atoms_pbc(atoms)

        # Validate that charge/spin are set correctly for omol, or default to 0 otherwise
        self._validate_charge_and_spin(atoms)

        # Standard call to check system_changes etc
        Calculator.calculate(self, atoms, properties, system_changes)

        if len(atoms) == 1 and sum(atoms.pbc) == 0:
            self.results = self._get_single_atom_energies(atoms)
        else:
            # Convert using the current a2g object
            data_object = self.a2g(atoms)

            # Batch and predict
            batch = data_list_collater([data_object], otf_graph=True)
            pred = self.predictor.predict(batch)

            # Collect the results into self.results
            self.results = {}
            for calc_key in self.implemented_properties:
                if calc_key == "energy":
                    energy = float(pred[calc_key].detach().cpu().numpy()[0])

                    self.results["energy"] = self.results["free_energy"] = (
                        energy  # Free energy is a copy of energy
                    )
                if calc_key == "forces":
                    forces = pred[calc_key].detach().cpu().numpy()
                    self.results["forces"] = forces
                if calc_key == "stress":
                    stress = pred[calc_key].detach().cpu().numpy().reshape(3, 3)
                    stress_voigt = full_3x3_to_voigt_6_stress(stress)
                    self.results["stress"] = stress_voigt

    def _get_single_atom_energies(self, atoms) -> dict:
        """
        Populate output with single atom energies
        """
        if self.predictor.atom_refs is None:
            raise ValueError(
                "Single atom system but no atomic references present. "
                "Please call fairchem.core.pretrained_mlip.get_predict_unit() "
                "with an appropriate checkpoint name."
            )
        logging.warning(
            "Single atom systems are not handled by the model; "
            "the precomputed DFT result is returned. "
            "Spin multiplicity is ignored for monoatomic systems."
        )
        elt = atoms.get_atomic_numbers()[0]
        results = {}

        atom_refs = self.predictor.atom_refs[self.task_name]
        try:
            energy = atom_refs.get(int(elt), {}).get(atoms.info["charge"])
        except AttributeError:
            energy = atom_refs[int(elt)]
        if energy is None:
            raise ValueError("This model has not stored this element with this charge.")
        results["energy"] = energy
        results["forces"] = np.array([[0.0] * 3])
        results["stress"] = np.array([0.0] * 6)
        return results

    def _check_atoms_pbc(self, atoms) -> None:
        """
        Check for invalid PBC conditions

        Args:
            atoms (ase.Atoms): The atomic structure to check.
        """
        if np.all(atoms.pbc) and np.allclose(atoms.cell, 0):
            raise AllZeroUnitCellError
        if np.any(atoms.pbc) and not np.all(atoms.pbc):
            raise MixedPBCError

    def _validate_charge_and_spin(self, atoms: Atoms) -> None:
        """
        Validate and set default values for charge and spin.

        Args:
            atoms (Atoms): The atomic structure containing charge and spin information.
        """

        if "charge" not in atoms.info:
            if self.task_name == UMATask.OMOL.value:
                logging.warning(
                    "task_name='omol' detected, but charge is not set in atoms.info. Defaulting to charge=0. "
                    "Ensure charge is an integer representing the total charge on the system and is within the range -100 to 100."
                )
            atoms.info["charge"] = DEFAULT_CHARGE

        if "spin" not in atoms.info:
            if self.task_name == UMATask.OMOL.value:
                atoms.info["spin"] = DEFAULT_SPIN_OMOL
                logging.warning(
                    "task_name='omol' detected, but spin multiplicity is not set in atoms.info. Defaulting to spin=1. "
                    "Ensure spin is an integer representing the spin multiplicity from 0 to 100."
                )
            else:
                atoms.info["spin"] = DEFAULT_SPIN

        # Validate charge
        charge = atoms.info["charge"]
        if not isinstance(charge, int):
            raise TypeError(
                f"Invalid type for charge: {type(charge)}. Charge must be an integer representing the total charge on the system."
            )
        if not (CHARGE_RANGE[0] <= charge <= CHARGE_RANGE[1]):
            raise ValueError(
                f"Invalid value for charge: {charge}. Charge must be within the range {CHARGE_RANGE[0]} to {CHARGE_RANGE[1]}."
            )

        # Validate spin
        spin = atoms.info["spin"]
        if not isinstance(spin, int):
            raise TypeError(
                f"Invalid type for spin: {type(spin)}. Spin must be an integer representing the spin multiplicity."
            )
        if not (SPIN_RANGE[0] <= spin <= SPIN_RANGE[1]):
            raise ValueError(
                f"Invalid value for spin: {spin}. Spin must be within the range {SPIN_RANGE[0]} to {SPIN_RANGE[1]}."
            )


class MixedPBCError(ValueError):
    """Specific exception example."""

    def __init__(
        self,
        message="Attempted to guess PBC for an atoms object, but the atoms object has PBC set to True for some"
        "dimensions but not others. Please ensure that the atoms object has PBC set to True for all dimensions.",
    ):
        self.message = message
        super().__init__(self.message)


class AllZeroUnitCellError(ValueError):
    """Specific exception example."""

    def __init__(
        self,
        message="Atoms object claims to have PBC set, but the unit cell is identically 0. Please ensure that the atoms"
        "object has a non-zero unit cell.",
    ):
        self.message = message
        super().__init__(self.message)
