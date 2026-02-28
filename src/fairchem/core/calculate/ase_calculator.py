"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from functools import partial
from typing import TYPE_CHECKING, Literal

import numpy as np
from ase.calculators.calculator import Calculator
from ase.stress import full_3x3_to_voigt_6_stress

from fairchem.core.calculate import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit.api.inference import (
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
            if task_name not in valid_datasets:
                raise ValueError(
                    f"Invalid task_name '{task_name}'. Valid options are {valid_datasets}"
                )
            self._task_name = task_name
        elif len(valid_datasets) == 1:
            self._task_name = valid_datasets[0]
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

        if predict_unit.inference_settings.external_graph_gen is True:
            r_edges = True
            max_neigh = 300
            radius = 6.0  # Default radius for edge generation
            logging.warning(
                "External graph generation is enabled, limiting neighbors to 300."
            )
        else:
            r_edges = False
            max_neigh = None
            radius = 6.0  # Still need radius even for internal graph gen

        a2g_kwargs = {
            "task_name": self.task_name,
            "r_edges": r_edges,
            "r_data_keys": ["spin", "charge"],
            "max_neigh": max_neigh,
            "radius": radius,
            "target_dtype": predict_unit.inference_settings.base_precision_dtype,
        }

        self.a2g = partial(AtomicData.from_ase, **a2g_kwargs)

    @property
    def task_name(self) -> str:
        return self._task_name

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

        # Validate input data
        self.predictor.validate_atoms_data(atoms, self.task_name)

        # Standard call to check system_changes etc
        Calculator.calculate(self, atoms, properties, system_changes)

        # Convert using the current a2g object
        data = self.a2g(atoms)

        # Batch and predict
        pred = self.predictor.predict(data)

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


class FormationEnergyCalculator(Calculator):
    def __init__(
        self,
        calculator: Calculator,
        element_references: dict | None = None,
        apply_corrections: bool | None = None,
        correction_type: Literal["MP2020", "OMat24"] = "OMat24",
    ):
        """
        A calculator wrapper that computes formation energies. Assumes task naming matches UMA.

        Args:
            calculator (Calculator): The base calculator to wrap.
            element_references (dict, optional): Dictionary of formation reference energies for each element.
                If None and calculator is FAIRChemCalculator, uses default references from the predictor.
            apply_corrections (bool, optional): Whether to apply MP style corrections to formation energies.
                Only relevant for OMat task. Defaults to True for OMat task if calculator is FAIRChemCalculator.
            correction_type (Literal["MP2020", "OMat24"], optional): Type of corrections to apply. Defaults to "OMat24".
        """
        super().__init__()
        self.calculator = calculator

        if element_references is None:
            if isinstance(calculator, FAIRChemCalculator):
                element_references = calculator.predictor.form_elem_refs[
                    calculator.task_name
                ]
            else:
                raise ValueError("element_references must be provided")
        self.element_references = element_references

        if apply_corrections is True:
            if isinstance(calculator, FAIRChemCalculator):
                if calculator.task_name != UMATask.OMAT.value:
                    raise ValueError(
                        "MP style corrections can only be applied for the OMat task."
                    )
            else:
                logging.warning(
                    "apply_corrections=True specified for non-FAIRChemCalculator. "
                    "Corrections will be attempted."
                )

        if (
            apply_corrections is None
            and isinstance(calculator, FAIRChemCalculator)
            and calculator.task_name == UMATask.OMAT.value
        ):
            apply_corrections = True

        self.apply_corrections = (
            apply_corrections if apply_corrections is not None else False
        )
        self._correction_type = correction_type

        if hasattr(calculator, "implemented_properties"):
            self.implemented_properties = calculator.implemented_properties

    def calculate(
        self, atoms: Atoms, properties: list[str], system_changes: list[str]
    ) -> None:
        """
        Calculate formation energy by wrapping the base calculator.

        Args:
            atoms (Atoms): The atomic structure to calculate properties for.
            properties (list[str]): The list of properties to calculate.
            system_changes (list[str]): The list of changes in the system.
        """
        self.calculator.calculate(atoms, properties, system_changes)

        self.results = self.calculator.results.copy()

        if "energy" in self.results:
            total_energy = self.results["energy"]

            if self.apply_corrections:
                try:
                    from fairchem.data.omat.entries.compatibility import (
                        apply_mp_style_corrections,
                    )
                except ImportError as err:
                    raise ImportError(
                        "fairchem.data.omat is required to apply MP style corrections. Please install it."
                    ) from err
                total_energy = apply_mp_style_corrections(
                    total_energy, atoms, correction_type=self._correction_type
                )

            element_symbols = atoms.get_chemical_symbols()
            element_counts = Counter(element_symbols)

            missing_elements = set(element_symbols) - set(
                self.element_references.keys()
            )
            if missing_elements:
                raise ValueError(
                    f"Missing reference energies for elements: {missing_elements}"
                )

            total_ref_energy = sum(
                self.element_references[element] * count
                for element, count in element_counts.items()
            )

            formation_energy = total_energy - total_ref_energy

            self.results["energy"] = formation_energy

            if "free_energy" in self.results:
                self.results["free_energy"] = formation_energy


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
