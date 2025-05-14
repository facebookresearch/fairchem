from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np
import torch
from ase.calculators.calculator import Calculator
from ase.stress import full_3x3_to_voigt_6_stress
from huggingface_hub import hf_hub_download

from fairchem.core.datasets import data_list_collater
from fairchem.core.preprocessing.atoms_to_graphs import AtomsToGraphs
from fairchem.core.units.mlip_unit.api.inference import (
    InferenceSettings,
    guess_inference_settings,
)
from fairchem.core.units.mlip_unit.mlip_unit import MLIPPredictUnit

if TYPE_CHECKING:
    from pathlib import Path

    from ase import Atoms


class FAIRChemCalculator(Calculator):
    implemented_properties: ClassVar = ["energy", "forces", "stress", "free_energy"]

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        hf_hub_repo_id: str | None = "facebook/UMA",
        hf_hub_filename: str | None = None,
        task_name: Literal["omol", "omat", "oc20", "odac", "omc"] | None = None,
        device: str = "cuda",
        inference_settings: InferenceSettings | str = "default",
        seed: int | None = 42,
        max_neighbors: int | None = 100,
    ):
        """
        Initialize the FAIRChemCalculator, downloading model checkpoints as necessary.

        Args:
            checkpoint_path (str | Path | None): Path to the inference checkpoint file on the local disk. Ignored if `hf_hub_repo_id` and `hf_hub_filename` are provided.
            hf_hub_repo_id (str | None): Hugging Face Hub repository ID to download the checkpoint from.
            hf_hub_filename (str | None): Filename of the checkpoint in the Hugging Face Hub repository.
            task_name (Literal["omol", "omat", "oc20", "odac", "omc"] | None): Name of the task to use if using a UMA checkpoint. Determines default key names for energy, forces, and stress. Can be one of 'omol', 'omat', 'oc20', 'odac', or 'omc'.
            device (str): Device to run the calculations on (e.g., "cuda" or "cpu"). Default is "cuda".
            inference_settings (InferenceSettings): Defines the inference flags for the Calculator, currently the acceptable modes are "default" (general purpose but not the fastest), or "turbo" which optimizes for speed for running simulations but the user must keep the atomic composition fixed. Advanced users can also pass in custom settings by passing an InferenceSettings object.
            seed (int | None): Random seed for reproducibility. Default is 42.
            max_neighbors (int | None): define a custom max neighbors per atom limit, defaults to 100, typically fairchem models are trained with 300, but we find 100 is sufficient for most applications
        Notes:
            - For models that require total charge and spin multiplicity (currently UMA models on omol mode), `charge` and `spin` (corresponding to `spin_multiplicity`) are pulled from `atoms.info` during calculations.
                - `charge` must be an integer representing the total charge on the system and can range from -100 to 100.
                - `spin` must be an integer representing the spin multiplicity and can range from 0 to 100.
                - If `task_name="omol"`, and `charge` or `spin` are not set in `atoms.info`, they will default to `0`.
            - The `free_energy` is simply a copy of the `energy` and is not the actual electronic free energy. It is only set for ASE routines/optimizers that are hard-coded to use this rather than the `energy` key.
        """

        super().__init__()

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

        # TODO: move these to a separate function retrieve these properties
        self.available_datasets = self.predictor.model.module.backbone.dataset_list
        self.available_output_keys = list(self.predictor.tasks.keys())
        logging.info(f"Available task names: {self.available_datasets}")
        logging.info(f"Available output keys: {self.available_output_keys}")

        self.max_neighbors = min(
            max_neighbors, self.predictor.model.module.backbone.max_neighbors
        )
        assert self.max_neighbors > 0

        self.cutoff = self.predictor.model.module.backbone.cutoff
        self.direct_force = self.predictor.model.module.backbone.direct_forces

        self.device = device

        self.task_name = task_name
        if self.task_name is None and len(self.available_datasets) == 1:
            self.energy_key = (
                "energy" if "energy" in self.available_output_keys else None
            )
            self.forces_key = (
                "forces" if "forces" in self.available_output_keys else None
            )
            self.stress_key = (
                "stress" if "stress" in self.available_output_keys else None
            )

        if seed is not None:
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

        self.print_warnings()

    @property
    def task_name(self) -> str:
        """
        Get the current task name.

        Returns:
            str: The current task name.
        """

        if (len(self.available_datasets) > 1) and self._task_name is None:
            logging.warning(
                "You are using a UMA model, but task_name is not set. Please set it before using the calculator!"
            )

        return self._task_name

    @task_name.setter
    def task_name(self, task_name: str) -> None:
        """
        Set the task name for the calculator and automatically set energy, forces, and stress keys.

        Args:
            task_name (str): The name of the task to use.
        """
        self._task_name = task_name
        self.energy_key = (
            f"{task_name}_energy"
            if f"{task_name}_energy" in self.available_output_keys
            else None
        )
        self.forces_key = (
            f"{task_name}_forces"
            if f"{task_name}_forces" in self.available_output_keys
            else None
        )
        self.stress_key = (
            f"{task_name}_stress"
            if f"{task_name}_stress" in self.available_output_keys
            else None
        )

        if self.energy_key not in self.available_output_keys:
            logging.warning(
                f"energy_key: <{self.energy_key}> not found in predictor. available keys: {self.available_output_keys}"
            )

        if self.forces_key not in self.available_output_keys:
            logging.warning(
                f"forces_key: <{self.forces_key}> not found in predictor. available keys: {self.available_output_keys}"
            )

        if self.stress_key not in self.available_output_keys:
            logging.warning(
                f"stress_key: <{self.stress_key}> not found in predictor. available keys: {self.available_output_keys}"
            )

    def print_warnings(self) -> None:
        """
        Print warnings related to the model configuration.
        """
        if self.direct_force:
            logging.warning(
                "This inference checkpoint is a direct-force model. This may lead to discontinuities in the potential energy surface and energy conservation errors. Use with caution."
            )

        if self.max_neighbors < 0.5 * (self.cutoff**3):
            logging.warning(
                f"The limit on maximum number of neighbors: <{self.max_neighbors}> is less than 0.5 * cutoff^3: <{0.1 * (self.cutoff**3)}>. This may lead to discontinuities in the potential energy surface and energy conservation errors. Use with caution."
            )

        if not hasattr(self, "seed"):
            logging.warning(
                "The random seed is not set. This may lead to non-deterministic behavior. Use <self.seed = seed> to set the random seed."
            )

        if self.task_name is None and len(self.available_datasets) > 1:
            logging.warning(
                f"task_name is not set. If you are using a UMA model, call <self.set_task_name(task_name)> before using the calculator. Available task names: {self.available_datasets}."
            )

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
            atoms.info["charge"] = 0
            logging.warning(
                "task_name='omol' detected, but charge is not set in atoms.info. Defaulting to charge=0. "
                "Ensure charge is an integer representing the total charge on the system and is within the range -100 to 100."
            )
        if "spin" not in atoms.info:
            atoms.info["spin"] = 0
            logging.warning(
                "task_name='omol' detected, but spin multiplicity is not set in atoms.info. Defaulting to spin=0. "
                "Ensure spin is an integer representing the spin multiplicity from 0 to 100."
            )

        # Validate charge
        if not isinstance(atoms.info["charge"], int):
            raise TypeError(
                f"Invalid type for charge: {type(atoms.info['charge'])}. Charge must be an integer representing the total charge on the system."
            )
        if not (-100 <= atoms.info["charge"] <= 100):
            raise ValueError(
                f"Invalid value for charge: {atoms.info['charge']}. Charge must be within the range -100 to 100."
            )

        # Validate spin
        if not isinstance(atoms.info["spin"], int):
            raise TypeError(
                f"Invalid type for spin: {type(atoms.info['spin'])}. Spin must be an integer representing the spin multiplicity."
            )
        if not (0 <= atoms.info["spin"] <= 100):
            raise ValueError(
                f"Invalid value for spin: {atoms.info['spin']}. Spin must be within the range 0 to 100."
            )

    def calculate(
        self, atoms: Atoms, properties: list[str], system_changes: list[str]
    ) -> None:
        """
        Perform the calculation for the given atomic structure.

        Args:
            atoms (Atoms): The atomic structure to calculate properties for. `charge             properties (list[str]): The list of properties to calculate.
            system_changes (list[str]): The list of changes in the system.

        Notes:
            - `charge` must be an integer representing the total charge on the system and can range from -100 to 100.
            - `spin` must be an integer representing the spin multiplicity and can range from 0 to 100.
            - If `task_name="omol"`, and `charge` or `spin` are not set in `atoms.info`, they will default to `0`.
            - `charge` and `spin` are currently only used for the `omol` head.
            - The `free_energy` is simply a copy of the `energy` and is not the actual electronic free energy. It is only set for ASE routines/optimizers that are hard-coded to use this rather than the `energy` key.
        """

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
        if self.energy_key is not None:
            energy = float(pred[self.energy_key].detach().cpu().numpy()[0])

            self.results["energy"] = self.results["free_energy"] = (
                energy  # Free energy is a copy of energy
            )
        if self.forces_key is not None:
            forces = pred[self.forces_key].detach().cpu().numpy()
            self.results["forces"] = forces
        if self.stress_key is not None:
            stress = pred[self.stress_key].detach().cpu().numpy().reshape(3, 3)
            stress_voigt = full_3x3_to_voigt_6_stress(stress)
            self.results["stress"] = stress_voigt

    @property
    def seed(self) -> int:
        """
        Get the current random seed.

        Returns:
            int: The current random seed.
        """
        return self._seed if hasattr(self, "_seed") else None

    @seed.setter
    def seed(self, seed: int) -> None:
        logging.info(f"Setting random seed to {seed}")
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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
