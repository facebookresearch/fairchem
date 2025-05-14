"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import numpy as np
import torch
from ase.calculators.calculator import Calculator
from ase.stress import full_3x3_to_voigt_6_stress
from huggingface_hub import hf_hub_download

from fairchem.core.datasets import data_list_collater
from fairchem.core.preprocessing.atoms_to_graphs import AtomsToGraphs
from fairchem.core.units.mlip_unit.api.inference import (
    CHARGE_RANGE,
    DEFAULT_CHARGE,
    DEFAULT_SPIN,
    DEFAULT_SPIN_OMOL,
    SPIN_RANGE,
    InferenceSettings,
    UMATask,
    guess_inference_settings,
)
from fairchem.core.units.mlip_unit.mlip_unit import MLIPPredictUnit

if TYPE_CHECKING:
    from pathlib import Path

    from ase import Atoms


class FAIRChemCalculator(Calculator):
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        hf_hub_repo_id: str | None = "facebook/UMA",
        hf_hub_filename: str | None = None,
        task_name: UMATask | None = None,
        device: str = "cuda",
        inference_settings: InferenceSettings | str = "default",
        seed: int = 42,
        max_neighbors: int | None = 100,
    ):
        """
        Initialize the FAIRChemCalculator, downloading model checkpoints as necessary.

        Args:
            checkpoint_path (str | Path | None): Path to the inference checkpoint file on the local disk. Ignored if
            `hf_hub_repo_id` and `hf_hub_filename` are provided.
            hf_hub_repo_id (str | None): Hugging Face Hub repository ID to download the checkpoint from.
            hf_hub_filename (str | None): Filename of the checkpoint in the Hugging Face Hub repository.
            task_name (UMATask | None): Name of the task to use if using a UMA checkpoint. Determines default key names
            for energy, forces, and stress. Can be one of 'omol', 'omat', 'oc20', 'odac', or 'omc'
            device (str): Device to run the calculations on (e.g., "cuda" or "cpu"). Default is "cuda".
            inference_settings (InferenceSettings | str): Defines the inference flags for the Calculator, currently the
            acceptable modes are "default" (general purpose but not the fastest), or "turbo" which optimizes for speed
            for running simulations but the user must keep the atomic composition fixed. Advanced users can also pass in
            custom settings by passing an InferenceSettings object.
            seed (int | None): Random seed for reproducibility. Default is 42.
            max_neighbors (int | None): define a custom max neighbors per atom limit, defaults to 100, typically
            fairchem models are trained with 300, but we find 100 is sufficient for most applications
        Notes:
            - For models that require total charge and spin multiplicity (currently UMA models on omol mode), `charge`
              and `spin` (corresponding to `spin_multiplicity`) are pulled from `atoms.info` during calculations.
                - `charge` must be an integer representing the total charge on the system and can range from -100 to 100.
                - `spin` must be an integer representing the spin multiplicity and can range from 0 to 100.
                - If `task_name="omol"`, and `charge` or `spin` are not set in `atoms.info`, they will default to
                charge=`0` and spin=`1`.
            - The `free_energy` is simply a copy of the `energy` and is not the actual electronic free energy.
            It is only set for ASE routines/optimizers that are hard-coded to use this rather than the `energy` key.
        """

        super().__init__()
        self.implemented_properties = []

        # Handle checkpoint download
        if hf_hub_repo_id and hf_hub_filename:
            logging.info(
                f"Downloading checkpoint from Hugging Face Hub: repo_id={hf_hub_repo_id}, filename={hf_hub_filename}"
            )
            checkpoint_path = hf_hub_download(
                repo_id=hf_hub_repo_id, filename=hf_hub_filename
            )
        elif not checkpoint_path:
            raise ValueError(
                "Either `checkpoint_path` or both `hf_hub_repo_id` and `hf_hub_filename` must be provided."
            )
        self.inference_settings_obj = guess_inference_settings(inference_settings)
        if self.inference_settings_obj.external_graph_gen:
            logging.warning(
                "inference_settings.external_graph_gen not supported in the FAIRChemCalculator, this is always set to false here"
            )

        self.inference_settings_obj.external_graph_gen = False

        self.predictor = MLIPPredictUnit(
            checkpoint_path,
            device=device,
            inference_settings=self.inference_settings_obj,
            overrides={"backbone": {"always_use_pbc": False}},
        )

        self.calc_property_to_model_key_mapping = {}
        logging.debug(f"Available task names: {self.available_tasks}")

        self.max_neighbors = min(
            max_neighbors, self.predictor.model.module.backbone.max_neighbors
        )
        assert self.max_neighbors > 0

        self.cutoff = self.predictor.model.module.backbone.cutoff
        self.direct_force = self.predictor.model.module.backbone.direct_forces
        self.device = device

        if task_name is not None:
            assert (
                task_name in self.available_tasks
            ), f"Given: {task_name}, Valid options are {self.available_tasks}"
            self._task_name = task_name
        elif len(self.available_tasks) == 1:
            self._task_name = self.available_tasks[0]
        else:
            raise RuntimeError(
                f"A task name must be provided. Valid options are {self.available_tasks}"
            )

        self._reset_calc_key_mapping(self._task_name)
        self.seed = seed

        # Even when our models may not use the charge/spin keys from atoms.info, they should still pull it
        a2g_kwargs = {"r_data_keys": ["spin", "charge"]}
        self.a2g = AtomsToGraphs(
            max_neigh=self.max_neighbors,
            radius=self.cutoff,
            r_energy=False,
            r_forces=False,
            r_distances=False,
            r_edges=False,
            r_pbc=True,
            **a2g_kwargs,
        )

        if self.direct_force:
            logging.warning(
                "This inference checkpoint is a direct-force model. This may lead to discontinuities in the potential "
                "energy surface and energy conservation errors. Use with caution."
            )

    @property
    def task_name(self) -> str:
        return self._task_name

    @property
    def available_tasks(self) -> list[str]:
        return self.predictor.model.module.backbone.dataset_list

    def _reset_calc_key_mapping(self, task_name: str) -> None:
        """
        Create a map of calculator keys to predictor output keys based on whats available in the model.

        Args:
            task_name (str): The name of the task to use.
        """
        implemented_properties = set()
        self.calc_property_to_model_key_mapping.clear()

        for model_task_name, model_task in self.predictor.tasks.items():
            if task_name in model_task.datasets:
                for calc_key in ["energy", "forces", "stress"]:
                    if calc_key == model_task.property:
                        self.calc_property_to_model_key_mapping[calc_key] = (
                            model_task_name
                        )
                        implemented_properties.add(calc_key)
                        if calc_key == "energy":
                            implemented_properties.add("free_energy")
        self.implemented_properties = list(implemented_properties)

    @property
    def seed(self) -> int:
        """
        Get the current random seed.

        Returns:
            int: The current random seed.
        """
        return self._seed

    @seed.setter
    def seed(self, seed: int) -> None:
        logging.info(f"Setting random seed to {seed}")
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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
            - The `free_energy` is simply a copy of the `energy` and is not the actual electronic free energy. It is only set for ASE routines/optimizers that are hard-coded to use this rather than the `energy` key.
        """
        assert (
            self.task_name is not None
        ), "You must set a task name before attempting to use the calculator"

        # Our calculators won't work if natoms=0
        if len(atoms) == 0:
            raise NoAtoms

        # Check if the atoms object has periodic boundary conditions (PBC) set correctly
        self._check_atoms_pbc(atoms)

        # Validate that charge/spin are set correctly for omol, or default to 0 otherwise
        self._validate_charge_and_spin(atoms)

        # Standard call to check system_changes etc
        Calculator.calculate(self, atoms, properties, system_changes)

        # Convert using the current a2g object
        data_object = self.a2g.convert(atoms)
        data_object.dataset = self.task_name

        # Batch and predict
        batch = data_list_collater([data_object], otf_graph=True)
        pred = self.predictor.predict(
            batch,
        )

        # Collect the results into self.results
        self.results = {}
        for calc_key, predictor_key in self.calc_property_to_model_key_mapping.items():
            if calc_key == "energy":
                energy = float(pred[predictor_key].detach().cpu().numpy()[0])

                self.results["energy"] = self.results["free_energy"] = (
                    energy  # Free energy is a copy of energy
                )
            if calc_key == "forces":
                forces = pred[predictor_key].detach().cpu().numpy()
                self.results["forces"] = forces
            if calc_key == "stress":
                stress = pred[predictor_key].detach().cpu().numpy().reshape(3, 3)
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


class MixedPBCError(ValueError):
    """Specific exception example."""

    def __init__(
        self,
        message="Attempted to guess PBC for an atoms object, but the atoms object has PBC set to True for some dimensions but not others. Please ensure that the atoms object has PBC set to True for all dimensions.",
    ):
        self.message = message
        super().__init__(self.message)


class AllZeroUnitCellError(ValueError):
    """Specific exception example."""

    def __init__(
        self,
        message="Atoms object claims to have PBC set, but the unit cell is identically 0. Please ensure that the atoms object has a non-zero unit cell.",
    ):
        self.message = message
        super().__init__(self.message)


class NoAtoms(ValueError):
    """Specific exception example."""

    def __init__(
        self,
        message="Atoms object has no atoms inside.",
    ):
        self.message = message
        super().__init__(self.message)
