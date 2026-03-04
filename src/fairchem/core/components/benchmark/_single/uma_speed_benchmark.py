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
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch._dynamo as dynamo
from ase.build import make_supercell
from ase.io import read
from torch.profiler import ProfilerActivity, profile, record_function

from fairchem.core.common import distutils
from fairchem.core.common.profiler_utils import get_profile_schedule
from fairchem.core.components.runner import Runner
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.datasets.common_structures import get_fcc_crystal_by_num_atoms
from fairchem.core.units.mlip_unit import MLIPPredictUnit
from fairchem.core.units.mlip_unit.api.inference import (
    InferenceSettings,
    inference_settings_default,
)

# Configure logging to INFO level
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ase_to_graph(
    atoms, neighbors: int, cutoff: float, external_graph=True, dataset_name="omat"
):
    data_object = AtomicData.from_ase(
        atoms,
        max_neigh=neighbors,
        radius=cutoff,
        r_edges=external_graph,
        task_name=dataset_name,
    )
    data_object.natoms = torch.tensor(len(atoms))
    data_object.charge = torch.LongTensor([0])
    data_object.spin = torch.LongTensor([0])
    data_object.pos.requires_grad = True
    return atomicdata_list_to_batch([data_object])


def get_qps(data, predictor, warmups: int = 10, timeiters: int = 10, repeats: int = 5):
    def timefunc():
        predictor.predict(data)
        # torch.cuda.synchronize()
        torch.distributed.barrier()

    for _ in range(warmups):
        timefunc()
        logging.info(f"memory allocated: {torch.cuda.memory_allocated()/(1024**3)}")

    result = timeit.repeat(timefunc, number=timeiters, repeat=repeats)
    logging.info(
        f"Timing results over {repeats} repeats: {result}, mean: {np.mean(result)}, std: {np.std(result)}"
    )
    qps = timeiters / np.mean(result)
    ns_per_day = qps * 24 * 3600 / 1e6
    return qps, ns_per_day


def trace_handler(p, name, save_loc):
    trace_name = f"{name}.{distutils.get_rank()}.pt.trace.json"
    output_path = os.path.join(save_loc, trace_name)
    logging.info(f"Saving trace in {output_path}")
    p.export_chrome_trace(output_path)


def make_profile(data, predictor, name, save_loc):
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    profile_schedule, total_profile_steps = get_profile_schedule(active=5)
    tc = functools.partial(trace_handler, name=name, save_loc=save_loc)

    with profile(
        activities=activities,
        schedule=profile_schedule,
        on_trace_ready=tc,
    ) as p:
        torch.distributed.barrier()
        for i in range(total_profile_steps):
            predictor.predict(data)
            logging.info(f"done step {i}")
            with record_function(f"final_barrier_{i}"):
                torch.distributed.barrier()
            p.step()


@dataclass
class CompileDebugSettings:
    """
    Settings for debugging torch.compile graph breaks and compilation issues.

    Attributes:
        enabled: Enable compile debugging (default: False)
        log_graph_breaks: Log graph break reasons (default: True)
        run_dynamo_explain: Run torch._dynamo.explain() for detailed analysis
        verbose: Enable verbose dynamo logging
        fullgraph: Test with fullgraph=True to find breaks (will error if breaks exist)
    """

    enabled: bool = False
    log_graph_breaks: bool = True
    run_dynamo_explain: bool = False
    verbose: bool = False
    fullgraph: bool = False


def _setup_compile_debug_logging(settings: CompileDebugSettings) -> None:
    """
    Configure torch._dynamo logging for debugging compilation issues.

    Args:
        settings: CompileDebugSettings instance with debug configuration
    """
    if not settings.enabled:
        return

    logging.info("Enabling torch.compile debug logging...")

    # Enable dynamo verbose mode
    if settings.verbose:
        dynamo.config.verbose = True
        logging.getLogger("torch._dynamo").setLevel(logging.DEBUG)

    # Log graph breaks
    if settings.log_graph_breaks:
        # Set environment-like logging via torch internal config
        # This is equivalent to TORCH_LOGS="graph_breaks"
        torch._logging.set_logs(graph_breaks=True)
        logging.info("Graph break logging enabled")

    logging.info(f"Compile debug settings: {settings}")


def _run_dynamo_explain(model, data, name: str) -> None:
    """
    Run torch._dynamo.explain() to analyze graph breaks.

    Args:
        model: The model to analyze
        data: Sample input data
        name: Name for logging
    """
    logging.info(f"Running torch._dynamo.explain() for {name}...")
    try:
        explanation = dynamo.explain(model)(data)
        logging.info("=" * 60)
        logging.info(f"DYNAMO EXPLAIN RESULTS for {name}")
        logging.info("=" * 60)
        logging.info(f"Graph count: {explanation.graph_count}")
        logging.info(f"Graph break count: {explanation.graph_break_count}")
        logging.info(f"Ops per graph: {explanation.ops_per_graph}")

        if explanation.break_reasons:
            logging.info("-" * 60)
            logging.info("GRAPH BREAK REASONS:")
            logging.info("-" * 60)
            for i, reason in enumerate(explanation.break_reasons):
                logging.info(f"\n[Break {i + 1}]")
                logging.info(str(reason))
        else:
            logging.info("No graph breaks detected!")
        logging.info("=" * 60)
    except Exception as e:
        logging.error(f"Error during dynamo.explain(): {e}")
        import traceback

        traceback.print_exc()


def _to_compile_debug_settings(
    value: CompileDebugSettings | dict | None,
) -> CompileDebugSettings:
    """
    Convert a dict or DictConfig to CompileDebugSettings.

    Args:
        value: CompileDebugSettings, dict, or None

    Returns:
        CompileDebugSettings instance with defaults for missing fields
    """
    if value is None:
        return CompileDebugSettings()
    if isinstance(value, CompileDebugSettings):
        return value
    # Handle dict or DictConfig from Hydra
    return CompileDebugSettings(
        enabled=value.get("enabled", False),
        log_graph_breaks=value.get("log_graph_breaks", True),
        run_dynamo_explain=value.get("run_dynamo_explain", False),
        verbose=value.get("verbose", False),
        fullgraph=value.get("fullgraph", False),
    )


class InferenceBenchRunner(Runner):
    def __init__(
        self,
        model_checkpoints: dict[str, str],
        natoms_list: list[int] | None = None,
        input_system: dict | None = None,
        timeiters: int = 10,
        repeats: int = 5,
        seed: int = 1,
        device="cuda",
        overrides: dict | None = None,
        inference_settings: InferenceSettings = inference_settings_default(),  # noqa B008
        generate_traces: bool = False,  # takes additional memory and time
        expand_supercells: int | None = None,
        dataset_name: str = "omat",
        debug_compile: CompileDebugSettings | None = None,
    ):
        self.natoms_list = natoms_list
        self.input_system = input_system
        assert (natoms_list is None) ^ (
            input_system is None
        ), "input must be either list of natoms or dict names: input system files"
        self.device = device
        self.seed = seed
        self.timeiters = timeiters
        self.model_checkpoints = model_checkpoints
        self.overrides = overrides
        self.inference_settings = inference_settings
        self.generate_traces = generate_traces
        self.expand_supercells = expand_supercells
        self.dataset_name = dataset_name
        self.repeats = repeats
        self.debug_compile = _to_compile_debug_settings(debug_compile)

    def run(self) -> None:
        self.run_dir = self.job_config.metadata.results_dir
        os.makedirs(self.run_dir, exist_ok=True)
        seed_everywhere(self.seed)

        # Setup compile debug logging if enabled
        _setup_compile_debug_logging(self.debug_compile)

        model_to_qps_data = defaultdict(list)

        for model_name, model_checkpoint in self.model_checkpoints.items():
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

            def yield_inputs(max_neighbors=max_neighbors, cutoff=cutoff):
                if self.natoms_list is not None:
                    for natoms in self.natoms_list:
                        atoms = get_fcc_crystal_by_num_atoms(natoms)
                        data = ase_to_graph(
                            atoms,
                            max_neighbors,
                            cutoff,
                            external_graph=self.inference_settings.external_graph_gen,
                            dataset_name=self.dataset_name,
                        )
                        yield data.natoms.item(), data
                else:
                    for k, v in self.input_system.items():
                        atoms = read(v)
                        if self.expand_supercells is not None:
                            size = self.expand_supercells
                            supercell_size = [[size, 0, 0], [0, size, 0], [0, 0, size]]
                            atoms = make_supercell(atoms, supercell_size)

                        data = ase_to_graph(
                            atoms,
                            max_neighbors,
                            cutoff,
                            external_graph=self.inference_settings.external_graph_gen,
                            dataset_name=self.dataset_name,
                        )
                        yield k, data

            # benchmark all models or number of atoms
            for name, data in yield_inputs():
                num_atoms = data.natoms.item()
                print_info = f"Starting profile: model: {model_checkpoint}, input: {name}, num_atoms: {num_atoms}"
                if self.inference_settings.external_graph_gen:
                    num_edges = data.edge_index.shape[1]
                    print_info += f" num edges compute on: {num_edges}"
                logging.info(print_info)
                inp = data.clone()

                # Run dynamo.explain() if debug enabled
                if self.debug_compile.enabled and self.debug_compile.run_dynamo_explain:
                    _run_dynamo_explain(predictor.model, inp, f"{model_name}_{name}")

                if self.generate_traces:
                    make_profile(inp, predictor, name=name, save_loc=self.run_dir)
                qps, ns_per_day = get_qps(
                    inp, predictor, timeiters=self.timeiters, repeats=self.repeats
                )
                model_to_qps_data[model_name].append([num_atoms, ns_per_day])
                logging.info(
                    f"Profile results: model: {model_checkpoint}, num_atoms: {num_atoms}, qps: {qps}, ns_per_day: {ns_per_day}"
                )

    def save_state(self, _):
        return

    def load_state(self, _):
        return
