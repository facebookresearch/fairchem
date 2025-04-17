from __future__ import annotations

import warnings
from typing import ClassVar

from ase.calculators.calculator import Calculator
from ase.stress import full_3x3_to_voigt_6_stress

# multi gpu inference
from fairchem.core.datasets import data_list_collater
from fairchem.core.preprocessing.atoms_to_graphs import AtomsToGraphs
from fairchem.core.units.mlip_unit.mlip_unit import MLIPPredictUnit


class MLIPCalculator(Calculator):
    implemented_properties: ClassVar = ["energy", "forces", "stress", "free_energy"]

    def __init__(
        self,
        inference_ckpt_path,
        # consider using another name for this, as model trained on one dataset can be used for another
        dataset_name=None,
        energy_key=None,
        forces_key=None,
        stress_key=None,
        device="cuda",
        regress_stress=True,
        act_ckpt=False,
        a2g_kwargs=None,
    ):
        if a2g_kwargs is None:
            a2g_kwargs = {}
        super().__init__()
        self.predictor = MLIPPredictUnit(inference_ckpt_path, device=device)

        # TODO: clean up config loading after a2g refactor.
        self.available_datasets = self.predictor.model.module.backbone.dataset_list
        self.available_output_keys = list(self.predictor.tasks.keys())
        print("Available dataset names:", self.available_datasets)
        print("Available output keys:", self.available_output_keys)

        self.max_neighbors = self.predictor.model.module.backbone.max_neighbors
        self.cutoff = self.predictor.model.module.backbone.cutoff
        self.direct_force = self.predictor.model.module.backbone.direct_forces
        self.internal_graph = self.predictor.model.module.backbone.otf_graph
        self.device = device

        self.predictor.model.module.backbone.regress_stress = regress_stress

        self.act_ckpt = act_ckpt
        if self.act_ckpt:
            self.predictor.model.module.backbone.activation_checkpointing = True

        # TODO molecule cell size?
        # TODO infer whether charge/spin is required by looking at the inference ckpt.
        self.a2g = AtomsToGraphs(
            max_neigh=self.max_neighbors,
            radius=self.cutoff,
            r_energy=False,
            r_forces=False,
            r_distances=False,
            r_edges=not self.internal_graph,
            r_pbc=True,
            # TODO raise warning if atoms contain this info but not used.
            # TODO having to set molecule_cell_size is particularly error-prone.
            # if r_edges is False and atoms doesn't have a cell, must do molecule_cell_size process.
            # because the gpu graph code must use pbc.
            # TODO this is a bit hacky, thing such as r_data_key is important and it is not very good to hide under this.
            **a2g_kwargs,
        )

        if dataset_name is None:
            warnings.warn(
                f"dataset_name not set. call <self.set_dataset_name(dataset_name)> before using the calculator. available dataset names: {self.available_datasets}"
            )
        if dataset_name not in self.available_datasets:
            raise KeyError(
                f"dataset_name: <{dataset_name}> not found in predictor. available dataset names: {self.available_datasets}"
            )

        self.dataset_name = dataset_name
        self.energy_key = energy_key
        self.forces_key = forces_key
        self.stress_key = stress_key

        if energy_key is None:
            warnings.warn(
                f"energy_key not set. call <self.set_energy_key(energy_key)> before using the calculator or energy will not be calculated. available keys: {self.available_output_keys}"
            )
        elif energy_key not in self.available_output_keys:
            raise KeyError(
                f"energy_key: <{energy_key}> not set or not found in predictor. available keys: {self.available_output_keys}"
            )

        if forces_key is None:
            warnings.warn(
                f"forces_key not set. call <self.set_forces_key(forces_key)> before using the calculator or forces will not be calculated. available keys: {self.available_output_keys}"
            )
        elif forces_key not in self.available_output_keys:
            raise KeyError(
                f"forces_key: <{forces_key}> not set or not found in predictor. available keys: {self.available_output_keys}"
            )

        # stress could be optional...
        if stress_key is None:
            warnings.warn(
                f"stress_key not set. call <self.set_stress_key(energy_key)> before using the calculator or stress will not be calculated. available keys: {self.available_output_keys}"
            )
        elif stress_key not in self.available_output_keys:
            raise KeyError(
                f"stress_key: <{stress_key}> not set or not found in predictor. available keys: {self.available_output_keys}"
            )

        self.print_warnings()

    # TODO use @property if we are going down this route of selecting dataset name and keys.
    def set_dataset_name(self, dataset_name):
        self.dataset_name = dataset_name

    def set_energy_key(self, energy_key):
        self.energy_key = energy_key

    def set_forces_key(self, forces_key):
        self.forces_key = forces_key

    def set_stress_key(self, stress_key):
        self.stress_key = stress_key

    def print_warnings(self):
        if self.direct_force:
            warnings.warn(
                "This inference checkpoint is a direct-force model. This may lead to discontinuities in the potential energy surface and energy conservation errors. Use with caution."
            )

        if self.max_neighbors < 0.5 * (self.cutoff**3):
            warnings.warn(
                f"The limit on maximum number of neighbors: <{self.max_neighbors}> is less than 0.5 * cutoff^3: <{0.1 * (self.cutoff ** 3)}>. This may lead to discontinuities in the potential energy surface and energy conservation errors. Use with caution."
            )

    def check_state(self, atoms, tol=1e-15):
        """
        Check for any system changes since last calculation, including atoms.info which contains global charge+spin.
        """
        state = super().check_state(atoms, tol=tol)
        # Ensure atoms.info is different for things ASE says a calculation is not required.
        if (not state) and (self.atoms.info != atoms.info):
            state.append("info")
        return state

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        data_object = self.a2g.convert(atoms)

        if not hasattr(data_object, "charge"):
            data_object.charge = 0
        if not hasattr(data_object, "spin"):
            data_object.spin = 0
        data_object.dataset = self.dataset_name

        batch = data_list_collater([data_object], otf_graph=self.internal_graph)
        pred = self.predictor.predict(batch)
        self.results = {}
        if self.energy_key is not None:
            energy = pred[self.energy_key].detach().cpu().numpy()
            self.results["energy"] = self.results["free_energy"] = float(energy)
        if self.forces_key is not None:
            forces = pred[self.forces_key].detach().cpu().numpy()
            self.results["forces"] = forces
        if self.stress_key is not None:
            stress = pred[self.stress_key].detach().cpu().numpy().reshape(3, 3)
            stress_voigt = full_3x3_to_voigt_6_stress(stress)
            self.results["stress"] = stress_voigt
