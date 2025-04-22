from __future__ import annotations

import functools
import logging
import os
import random
import timeit
import uuid
from collections import defaultdict
from functools import partial

import numpy as np
import torch
from ase import build
from torch.profiler import ProfilerActivity, profile

from fairchem.core.common.profiler_utils import get_profile_schedule
from fairchem.core.components.runner import Runner
from fairchem.core.datasets import data_list_collater
from fairchem.core.preprocessing import AtomsToGraphs
from fairchem.core.units.mlip_unit.mlip_unit import MLIPPredictUnit


def seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_diamond_tg_data(neighbors: int, cutoff: float, size: int):
    # get torch geometric data object for diamond
    atoms = build.bulk("C", "diamond", a=3.567, cubic=True)
    atoms = atoms.repeat((size, size, size))
    a2g = AtomsToGraphs(
        max_neigh=neighbors, radius=cutoff, r_edges=True, r_distances=True
    )
    data_object = a2g.convert(atoms)
    data_object.natoms = len(atoms)
    data_object.charge = 0
    data_object.spin = 0
    data_object.dataset = "omol"
    data_object.pos.requires_grad = True
    data_loader = torch.utils.data.DataLoader(
        [data_object],
        collate_fn=partial(data_list_collater, otf_graph=True),
        batch_size=1,
        shuffle=False,
    )
    return next(iter(data_loader))


def get_qps(data, predictor, warmups: int = 10, timeiters: int = 100):
    def timefunc():
        predictor.predict(data)
        torch.cuda.synchronize()

    for _ in range(warmups):
        timefunc()
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
        model_checkpoints: dict[str, str],
        timeiters: int = 100,
        seed: int = 1,
        sizes_to_bench: list[int] | None = None,
        device="cuda",
    ):
        if sizes_to_bench is None:
            sizes_to_bench = [3, 4]
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        self.device = device
        self.seed = seed
        self.timeiters = timeiters
        self.model_checkpoints = model_checkpoints
        self.sizes_to_bench = sizes_to_bench
        self.run_dir = os.path.join(run_dir_root, uuid.uuid4().hex.upper()[0:8])
        os.makedirs(self.run_dir, exist_ok=True)

    def run(self) -> None:
        seed_everywhere(self.seed)

        model_to_qps_data = defaultdict(list)

        # benchmark all models
        for name, model_checkpoint in self.model_checkpoints.items():
            logging.info(f"Loading model: {model_checkpoint}")
            predictor = MLIPPredictUnit(model_checkpoint, self.device)
            max_neighbors = predictor.model.module.backbone.max_neighbors
            cutoff = predictor.model.module.backbone.cutoff
            max_neighbors = 120

            predictor.model.module.backbone.otf_graph = False

            # benchmark all cell sizes
            for size in self.sizes_to_bench:
                diamond_data = get_diamond_tg_data(
                    neighbors=max_neighbors, cutoff=cutoff, size=size
                ).to(self.device)
                make_profile(diamond_data, predictor, name=name, save_loc=self.run_dir)

                qps, ns_per_day = get_qps(
                    diamond_data, predictor, timeiters=self.timeiters
                )
                num_atoms = diamond_data.natoms.item()
                num_edges = diamond_data.edge_index.shape[1]
                model_to_qps_data[name].append([num_atoms, ns_per_day])
                logging.info(
                    f"model: {model_checkpoint}, num_atoms: {num_atoms}, num_edges: {num_edges}, qps: {qps}"
                )

    def save_state(self, _):
        return

    def load_state(self, _):
        return
