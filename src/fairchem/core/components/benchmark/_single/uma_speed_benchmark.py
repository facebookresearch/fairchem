"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import functools
import logging
import os
import random
import timeit
import uuid
from collections import defaultdict

import numpy as np
import torch
from ase import build
from torch.profiler import ProfilerActivity, profile

from fairchem.core.common.profiler_utils import get_profile_schedule
from fairchem.core.components.runner import Runner
from fairchem.core.datasets import data_list_collater
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit import MLIPPredictUnit
from fairchem.core.units.mlip_unit.api.inference import (
    InferenceSettings,
    inference_settings_default,
)


def seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ase_to_graph(atoms, neighbors: int, cutoff: float, external_graph=True):
    data_object = AtomicData.from_ase(
        atoms,
        max_neigh=neighbors,
        radius=cutoff,
        r_edges=external_graph,
    )
    data_object.natoms = torch.tensor(len(atoms))
    data_object.charge = torch.LongTensor([0])
    data_object.spin = torch.LongTensor([0])
    data_object.dataset = "omat"
    data_object.pos.requires_grad = True
    data_loader = torch.utils.data.DataLoader(
        [data_object],
        collate_fn=data_list_collater,
        batch_size=1,
        shuffle=False,
    )
    return next(iter(data_loader))


def get_fcc_carbon_xtal(
    neighbors: int,
    radius: float,
    num_atoms: int,
    lattice_constant: float = 3.8,
    external_graph: bool = True,
):
    # lattice_constant = 3.8, fcc generates a supercell with ~50 edges/atom
    atoms = build.bulk("C", "fcc", a=lattice_constant)
    n_cells = int(np.ceil(np.cbrt(num_atoms)))
    atoms = atoms.repeat((n_cells, n_cells, n_cells))
    indices = np.random.choice(len(atoms), num_atoms, replace=False)
    sampled_atoms = atoms[indices]
    return ase_to_graph(sampled_atoms, neighbors, radius, external_graph)


def get_qps(data, predictor, warmups: int = 10, timeiters: int = 100):
    def timefunc():
        predictor.predict(data)
        torch.cuda.synchronize()

    for _ in range(warmups):
        timefunc()
        logging.info(f"memory allocated: {torch.cuda.memory_allocated()/(1024**3)}")

    result = timeit.timeit(timefunc, number=timeiters)
    qps = timeiters / result
    ns_per_day = qps * 24 * 3600 / 1e6
    return qps, ns_per_day


def trace_handler(p, name, save_loc):
    trace_name = f"{name}.pt.trace.json"
    output_path = os.path.join(save_loc, trace_name)
    logging.info(f"Saving trace in {output_path}")
    p.export_chrome_trace(output_path)


def make_profile(data, predictor, name, save_loc):
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    profile_schedule, total_profile_steps = get_profile_schedule()
    tc = functools.partial(trace_handler, name=name, save_loc=save_loc)

    with profile(
        activities=activities,
        schedule=profile_schedule,
        on_trace_ready=tc,
    ) as p:
        for _ in range(total_profile_steps):
            predictor.predict(data)
            torch.cuda.synchronize()
            p.step()


class InferenceBenchRunner(Runner):
    def __init__(
        self,
        run_dir_root,
        natoms_list: list[int],
        model_checkpoints: dict[str, str],
        timeiters: int = 10,
        seed: int = 1,
        device="cuda",
        overrides: dict | None = None,
        inference_settings: InferenceSettings = inference_settings_default(),  # noqa B008
        generate_traces: bool = False,  # takes additional memory and time
    ):
        self.natoms_list = natoms_list
        self.device = device
        self.seed = seed
        self.timeiters = timeiters
        self.model_checkpoints = model_checkpoints
        self.run_dir = os.path.join(run_dir_root, uuid.uuid4().hex.upper()[0:8])
        self.overrides = overrides
        self.inference_settings = inference_settings
        self.generate_traces = generate_traces
        os.makedirs(self.run_dir, exist_ok=True)

    def run(self) -> None:
        seed_everywhere(self.seed)

        model_to_qps_data = defaultdict(list)

        for name, model_checkpoint in self.model_checkpoints.items():
            logging.info(
                f"Loading model: {model_checkpoint}, inference_settings: {self.inference_settings}"
            )
            predictor = MLIPPredictUnit(
                model_checkpoint,
                self.device,
                overrides=self.overrides,
                inference_settings=self.inference_settings,
            )
            max_neighbors = predictor.model.module.backbone.max_neighbors
            cutoff = predictor.model.module.backbone.cutoff
            logging.info(f"Model's max_neighbors: {max_neighbors}, cutoff: {cutoff}")

            # benchmark all cell sizes
            for natoms in self.natoms_list:
                data = get_fcc_carbon_xtal(
                    max_neighbors,
                    cutoff,
                    natoms,
                    external_graph=self.inference_settings.external_graph_gen,
                )
                num_atoms = data.natoms.item()

                print_info = f"Starting profile: model: {model_checkpoint}, num_atoms: {num_atoms}"
                if self.inference_settings.external_graph_gen:
                    num_edges = data.edge_index.shape[1]
                    print_info += f" num edges compute on: {num_edges}"
                logging.info(print_info)
                if self.generate_traces:
                    make_profile(data, predictor, name=name, save_loc=self.run_dir)
                qps, ns_per_day = get_qps(data, predictor, timeiters=self.timeiters)
                model_to_qps_data[name].append([num_atoms, ns_per_day])
                logging.info(
                    f"Profile results: model: {model_checkpoint}, num_atoms: {num_atoms}, qps: {qps}, ns_per_day: {ns_per_day}"
                )

    def save_state(self, _):
        return

    def load_state(self, _):
        return
