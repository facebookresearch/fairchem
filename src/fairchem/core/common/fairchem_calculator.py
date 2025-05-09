from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
import torch
from ase.calculators.calculator import Calculator
from ase.stress import full_3x3_to_voigt_6_stress
from huggingface_hub import hf_hub_download

from fairchem.core.datasets import data_list_collater
from fairchem.core.preprocessing.atoms_to_graphs import AtomsToGraphs
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
        task_name: Literal["omol", "omat", "oc20", "odac", "osc"] | None = None,
        device: str = "cuda",
        act_ckpt: bool = True,
        a2g_kwargs: dict[str, Any] | None = None,
        seed: int | None = 42,
        pbc_vacuum_buffer_for_aperiodic_atoms: float | None = None,
    ):
        """
        Initialize the FAIRChemCalculator, downloading model checkpoints as necessary.

        Args:
            checkpoint_path (str | Path | None): Path to the inference checkpoint file on the local disk. Ignored if `hf_hub_repo_id` and `hf_hub_filename` are provided.
            hf_hub_repo_id (str | None): Hugging Face Hub repository ID to download the checkpoint from.
            hf_hub_filename (str | None): Filename of the checkpoint in the Hugging Face Hub repository.
            task_name (Literal["omol", "omat", "oc20", "odac", "osc"] | None): Name of the task to use if using a UMA checkpoint. Determines default key names for energy, forces, and stress. Can be one of 'omol', 'omat', 'oc20', 'odac', or 'osc' (where osc corresponds to the OMC dataset).
            device (str): Device to run the calculations on (e.g., "cuda" or "cpu"). Default is "cuda".
            act_ckpt (bool): Whether to enable activation checkpointing for memory efficiency. Default is True.
                Setting `act_ckpt=False` will make the model run faster (approximately 20-30%) at the expense of using more memory.
            a2g_kwargs (dict[str, Any] | None): Additional arguments for the AtomsToGraphs conversion.
            seed (int | None): Random seed for reproducibility. Default is 42.
            pbc_vacuum_buffer_for_aperiodic_atoms (float | None): Vacuum size for guessing PBC for aperiodic atoms if not None. Default is None, which raises errors if the atoms object does not have PBC set.
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

        self.predictor = MLIPPredictUnit(checkpoint_path, device=device)

        # TODO: clean up config loading after a2g refactor.
        self.available_datasets = self.predictor.model.module.backbone.dataset_list
        self.available_output_keys = list(self.predictor.tasks.keys())
        logging.info(f"Available task names: {self.available_datasets}")
        logging.info(f"Available output keys: {self.available_output_keys}")

        self.max_neighbors = self.predictor.model.module.backbone.max_neighbors
        self.cutoff = self.predictor.model.module.backbone.cutoff
        self.direct_force = self.predictor.model.module.backbone.direct_forces
        self.otf_graph = self.predictor.model.module.backbone.otf_graph = True

        self.device = device
        self.pbc_vacuum_buffer_for_aperiodic_atoms = (
            pbc_vacuum_buffer_for_aperiodic_atoms
        )

        self.act_ckpt = act_ckpt
        if self.act_ckpt:
            self.predictor.model.module.backbone.activation_checkpointing = True

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
        if a2g_kwargs is None:
            a2g_kwargs = {}
        a2g_kwargs.update({"r_data_keys": ["spin", "charge"]})
        logging.info(f"setting a2g_kwargs to {a2g_kwargs}")

        self.a2g = AtomsToGraphs(
            max_neigh=self.max_neighbors,
            radius=self.cutoff,
            r_energy=False,
            r_forces=False,
            r_distances=False,
            r_edges=not self.otf_graph,
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
        atoms = self._check_or_set_atoms_pbc(atoms)

        # Validate that charge/spin are set correctly for omol, or default to 0 otherwise
        self._validate_charge_and_spin(atoms)

        # Standard call to check system_changes etc
        Calculator.calculate(self, atoms, properties, system_changes)

        # Convert using the current a2g object
        data_object = self.a2g.convert(atoms)
        data_object.dataset = self.task_name

        # Batch and predict
        batch = data_list_collater([data_object], otf_graph=self.otf_graph)
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

    def _check_or_set_atoms_pbc(self, atoms) -> Atoms:
        """
        Check if the atoms object has periodic boundary conditions (PBC) set or set it if self.pbc_vacuum_buffer_for_aperiodic_atoms is not None.

        Args:
            atoms (ase.Atoms): The atomic structure to check.

        Returns:
            atoms: unchanged atoms, or atoms with guessed PBC if self.add_pbc_for_aperiodic_atoms=True.
        """
        if hasattr(atoms, "pbc") and np.all(atoms.pbc):
            if not np.allclose(atoms.cell, 0):
                return atoms
            else:
                raise AllZeroUnitCellError
        elif self.pbc_vacuum_buffer_for_aperiodic_atoms:
            return guess_pbc(atoms, vacuum=self.pbc_vacuum_buffer_for_aperiodic_atoms)
        else:
            raise MissingPBCError


def guess_pbc(atoms, vacuum: float) -> Atoms:
    if hasattr(atoms, "pbc") and np.all(atoms.pbc):
        # PBC is already set, return the atoms object
        return atoms
    elif hasattr(atoms, "pbc") and not np.all(~atoms.pbc):
        # We have mixed PBC, and we really can't know what to do. It's on the user to figure this out.
        raise MixedPBCError

    # if we've gotten here, we need to guess the box and apply pbc!
    logging.info(
        "Guessed a sufficiently large unit cell for the atoms object. Setting pbc=True."
    )
    atoms_copy = atoms.copy()
    atoms_copy.center(vacuum=vacuum)
    atoms_copy.pbc = True
    return atoms_copy


class MixedPBCError(ValueError):
    """Specific exception example."""

    def __init__(
        self,
        message="Attempted to guess PBC for an atoms object, but the atoms object has PBC set to True for some dimensions but not others. Please ensure that the atoms object has PBC set to True for all dimensions.",
    ):
        self.message = message
        super().__init__(self.message)


class MissingPBCError(ValueError):
    """Specific exception example."""

    def __init__(
        self,
        message="The atoms object does not have periodic boundary conditions (PBC) in all directions. Please ensure that the atoms object has PBC set to True for all dimensions, or use the `fairchem.core.common.calculator.guess_pbc` to guess a sufficiently large unit cell if you have an uncharged aperiodic system or set add_pbc_for_aperiodic_atoms=True in the calculator.",
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
