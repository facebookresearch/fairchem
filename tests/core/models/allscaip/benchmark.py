"""
Benchmark script for AllScAIP model performance testing.
Tests speed, memory usage, and correctness across different system sizes.

Usage:
    # Full benchmark with graph generation
    python benchmark.py --num_atoms 100 --mode full --num_warmup 5 --num_iterations 20

    # Benchmark without graph generation (precomputes graph once)
    python benchmark.py --num_atoms 100 --mode no_graph_gen --num_warmup 5 --num_iterations 20

    # Preprocess and save data for later benchmarking (fastest, reusable across model sizes)
    python benchmark.py --preprocess_data --atom_sizes 500,1000,2000,5000 --data_dir ./preprocessed_data

    # Benchmark using pre-saved data (no graph construction overhead)
    python benchmark.py --mode from_preprocessed --num_atoms 1000 --data_dir ./preprocessed_data --hidden_size 512 --num_layers 6

    # Profile mode - detailed profiling of each component (disables compile automatically)
    python benchmark.py --num_atoms 1000 --mode full --md_mode --profile
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import time
from functools import partial

import numpy as np
import torch

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.collaters.simple_collater import data_list_collater
from fairchem.core.datasets.common_structures import get_fcc_carbon_xtal
from fairchem.core.models.allscaip.AllScAIP import (
    AllScAIPBackbone,
    AllScAIPDirectForceHead,
    AllScAIPEnergyHead,
)
from fairchem.core.models.base import HydraModelV2

# Configuration from md_NeNo.yaml
DATASET_LIST = ["oc20", "omol", "osc", "omat", "odac"]


def seed_everywhere(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_backbone_config(
    cutoff: float,
    use_compile: bool,
    max_atoms: int,
    max_neighbors: int = 50,
    max_neighbors_pad_size: int = 60,
    use_sincx_mask: bool = False,
    use_freq_mask: bool = False,
    use_node_path: bool = True,
    single_system_no_padding: bool = False,
    use_chunked_graph: bool = False,
    graph_chunk_size: int = 512,
    preprocess_on_cpu: bool = False,
    hidden_size: int = 512,
    num_layers: int = 6,
    atten_num_heads: int = 8,
):
    """Get backbone config based on sm_NeNo.yaml"""
    # Compute frequency list to match head_dim = hidden_size / atten_num_heads
    head_dim = hidden_size // atten_num_heads
    # Default frequency list that sums to 64, scale if needed
    base_freq_list = [20, 10, 4, 10, 20]  # sums to 64
    if head_dim != 64:
        # Scale proportionally
        scale = head_dim / 64
        freq_list = [max(1, int(f * scale)) for f in base_freq_list]
        # Adjust last element to ensure sum matches head_dim
        freq_list[-1] = head_dim - sum(freq_list[:-1])
    else:
        freq_list = base_freq_list

    return {
        # Global Configs
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "direct_forces": True,  # Always direct forces for inference benchmark
        "regress_forces": True,
        "regress_stress": False,
        "use_compile": use_compile,
        "use_padding": not single_system_no_padding,  # Disable padding in MD mode
        "dataset_list": DATASET_LIST,
        "use_node_path": use_node_path,
        "use_freq_mask": use_freq_mask,
        "use_sincx_mask": use_sincx_mask,
        "use_residual_scale": False,
        # MD simulation mode: enables various optimizations
        "single_system_no_padding": single_system_no_padding,
        # Molecular Graph Configs
        "max_num_elements": 110,
        "max_batch_size": 1,  # Single system per batch
        "max_atoms": max_atoms,
        "max_radius": cutoff,
        "knn_k": max_neighbors,
        "knn_soft": True,
        "knn_sigmoid_scale": 0.2,
        "knn_lse_scale": 0.1,
        "knn_use_low_mem": True,
        "knn_pad_size": max_neighbors_pad_size,
        # Chunked graph construction
        "use_chunked_graph": use_chunked_graph,
        "graph_chunk_size": graph_chunk_size,
        # CPU preprocessing
        "preprocess_on_cpu": preprocess_on_cpu,
        # GNN Configs
        "atten_name": "memory_efficient",
        "atten_num_heads": atten_num_heads,
        # Frequency list: must sum to head_dim = hidden_size / atten_num_heads
        "freequency_list": freq_list,
    }


def get_sample_data(num_atoms: int, no_pbc: bool = False):
    """Generate sample FCC carbon crystal data."""
    samples = get_fcc_carbon_xtal(num_atoms)
    if no_pbc:
        samples.pbc = False
        samples.cell = None
    data_object = AtomicData.from_ase(samples)
    data_object.natoms = torch.tensor(len(samples))
    data_object.charge = torch.LongTensor([0])
    data_object.spin = torch.LongTensor([0])
    data_object.dataset = "omol"
    data_object.pos.requires_grad = False  # Direct forces, no grad needed
    data_loader = torch.utils.data.DataLoader(
        [data_object],
        collate_fn=partial(data_list_collater, otf_graph=True),
        batch_size=1,
        shuffle=False,
    )
    return next(iter(data_loader))


def get_allscaip_model(
    cutoff: float,
    use_compile: bool,
    max_atoms: int,
    max_neighbors: int,
    max_neighbors_pad_size: int,
    use_sincx_mask: bool = False,
    use_freq_mask: bool = False,
    use_node_path: bool = True,
    single_system_no_padding: bool = False,
    use_chunked_graph: bool = False,
    graph_chunk_size: int = 512,
    preprocess_on_cpu: bool = False,
    hidden_size: int = 512,
    num_layers: int = 6,
    atten_num_heads: int = 8,
    device: str = "cuda",
):
    """Build the AllScAIP model with direct force and energy heads."""
    backbone_config = get_backbone_config(
        cutoff=cutoff,
        use_compile=use_compile,
        max_atoms=max_atoms,
        max_neighbors=max_neighbors,
        max_neighbors_pad_size=max_neighbors_pad_size,
        use_sincx_mask=use_sincx_mask,
        use_freq_mask=use_freq_mask,
        use_node_path=use_node_path,
        single_system_no_padding=single_system_no_padding,
        use_chunked_graph=use_chunked_graph,
        graph_chunk_size=graph_chunk_size,
        preprocess_on_cpu=preprocess_on_cpu,
        hidden_size=hidden_size,
        num_layers=num_layers,
        atten_num_heads=atten_num_heads,
    )
    backbone = AllScAIPBackbone(**backbone_config)
    heads = {
        "energy_head": AllScAIPEnergyHead(backbone),
        "force_head": AllScAIPDirectForceHead(backbone),
    }
    model = HydraModelV2(backbone, heads).to(device)
    model.eval()
    return model


def clear_cuda_cache():
    """Clear CUDA cache and run garbage collection."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def benchmark_full_forward(
    model: HydraModelV2,
    data: AtomicData,
    num_warmup: int,
    num_iterations: int,
    device: str = "cuda",
):
    """
    Benchmark full forward pass including graph generation (data preprocessing).
    """
    # Warmup
    for _ in range(num_warmup):
        data_copy = data.clone().to(device)
        with torch.no_grad():
            _ = model(data_copy)
        torch.cuda.synchronize()

    clear_cuda_cache()

    # Benchmark
    times = []
    for _ in range(num_iterations):
        data_copy = data.clone().to(device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            output = model(data_copy)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    # Get memory stats
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB

    return {
        "times": times,
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "peak_memory_mb": peak_memory,
        "output": output,
    }


def benchmark_no_graph_gen(
    model: HydraModelV2,
    data: AtomicData,
    num_warmup: int,
    num_iterations: int,
    device: str = "cuda",
):
    """
    Benchmark forward pass without graph generation.
    Pre-computes the graph data and only benchmarks the model forward.
    """
    from fairchem.core.models.allscaip.utils.data_preprocess import unpad_results

    # Pre-compute graph data
    data = data.clone().to(device)

    print("Pre-computing graph data...")
    # Run one full forward to get the preprocessed data
    with torch.no_grad():
        # First, call the backbone data_preprocess manually
        data["atomic_numbers"] = data["atomic_numbers"].long()
        data["atomic_numbers_full"] = data["atomic_numbers"]
        data["batch_full"] = data["batch"]

        from fairchem.core.models.escaip.utils.graph_utils import (
            get_displacement_and_cell,
        )

        displacement, orig_cell = get_displacement_and_cell(
            data,
            model.backbone.regress_stress,
            model.backbone.regress_forces,
            model.backbone.direct_forces,
        )
        preprocessed_data = model.backbone.data_preprocess(data)
    print("Graph data pre-computed")

    print("Starting forward pass...")
    # Get the compiled forward function ready
    forward_fn = (
        torch.compile(model.backbone.compiled_forward)
        if model.backbone.global_cfg.use_compile
        else model.backbone.compiled_forward
    )

    # Get head forward functions
    energy_forward_fn = (
        torch.compile(model.output_heads["energy_head"].compiled_forward)
        if model.backbone.global_cfg.use_compile
        else model.output_heads["energy_head"].compiled_forward
    )
    force_forward_fn = (
        torch.compile(model.output_heads["force_head"].compiled_forward)
        if model.backbone.global_cfg.use_compile
        else model.output_heads["force_head"].compiled_forward
    )

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            emb = forward_fn(preprocessed_data)
            emb["displacement"] = displacement
            emb["orig_cell"] = orig_cell
            _ = energy_forward_fn(emb)
            _ = force_forward_fn(emb)
        torch.cuda.synchronize()

    clear_cuda_cache()

    # Benchmark
    times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            emb = forward_fn(preprocessed_data)
            emb["displacement"] = displacement
            emb["orig_cell"] = orig_cell
            energy = energy_forward_fn(emb)
            forces = force_forward_fn(emb)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    # Get memory stats
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB

    # Unpad outputs
    if len(energy.shape) == 0:
        energy = energy.unsqueeze(0)
    output_raw = {
        "energy": energy,
        "forces": forces,
    }
    output_unpadded = unpad_results(output_raw, preprocessed_data)

    # Construct output dict similar to full forward
    output = {
        "energy_head": {"energy": output_unpadded["energy"]},
        "force_head": {"forces": output_unpadded["forces"]},
    }

    return {
        "times": times,
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "peak_memory_mb": peak_memory,
        "output": output,
    }


def verify_correctness(output1: dict, output2: dict, atol: float = 1e-4):
    """Verify that two outputs are close."""
    results = {}

    for head_name in output1:
        if head_name not in output2:
            results[head_name] = {"match": False, "error": "Missing in output2"}
            continue

        for key in output1[head_name]:
            if key not in output2[head_name]:
                results[f"{head_name}/{key}"] = {
                    "match": False,
                    "error": "Missing in output2",
                }
                continue

            val1 = output1[head_name][key]
            val2 = output2[head_name][key]

            if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
                match = torch.allclose(val1, val2, atol=atol)
                max_diff = (val1 - val2).abs().max().item()
                results[f"{head_name}/{key}"] = {
                    "match": match,
                    "max_diff": max_diff,
                }
            else:
                results[f"{head_name}/{key}"] = {"match": val1 == val2}

    return results


def profile_model(
    model: HydraModelV2,
    data: AtomicData,
    num_warmup: int = 2,
    num_profile_steps: int = 3,
    device: str = "cuda",
    output_dir: str = "./profile_results",
):
    """
    Profile model forward pass with detailed component breakdown.

    Profiles:
    - data_preprocess: Graph construction
    - input_block: Input embeddings
    - layer_X_neighbor_att: Neighborhood attention for layer X
    - layer_X_edge_ffn: Edge FFN for layer X
    - layer_X_node_att: Node attention for layer X
    - layer_X_node_ffn: Node FFN for layer X
    - energy_head: Energy head forward
    - force_head: Force head forward
    """
    from torch.profiler import ProfilerActivity, profile, schedule

    os.makedirs(output_dir, exist_ok=True)

    # Move data to device
    data = data.clone().to(device)

    print("Running profiler...")
    print(f"  Warmup steps: {num_warmup}")
    print(f"  Profile steps: {num_profile_steps}")

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(data.clone())
        torch.cuda.synchronize()

    clear_cuda_cache()

    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(
            wait=0,
            warmup=0,
            active=num_profile_steps,
            repeat=1,
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(num_profile_steps):
            data_copy = data.clone()
            with torch.no_grad():
                _ = model(data_copy)
            torch.cuda.synchronize()
            prof.step()

    # Print summary
    print("\n" + "=" * 80)
    print("PROFILER RESULTS (sorted by CUDA time)")
    print("=" * 80)

    # Get key averages
    key_averages = prof.key_averages()

    # Filter for our record_function annotations
    our_annotations = [
        "data_preprocess",
        "get_displacement_and_cell",
        "input_block",
        "backbone_compile_forward",
        "energy_head",
        "force_head",
    ]

    # Add layer-specific annotations
    num_layers = model.backbone.global_cfg.num_layers
    for i in range(num_layers):
        our_annotations.extend(
            [
                f"layer_{i}_neighbor_att",
                f"layer_{i}_edge_ffn",
                f"layer_{i}_node_att",
                f"layer_{i}_node_ffn",
            ]
        )

    # Print table header
    print(
        f"\n{'Component':<40} {'CUDA Time (ms)':<15} {'CPU Time (ms)':<15} {'Calls':<8}"
    )
    print("-" * 80)

    # Collect and print our annotations
    results = {}
    for event in key_averages:
        if event.key in our_annotations:
            device_time = (
                event.self_device_time_total / 1000 / num_profile_steps
            )  # us to ms, average
            cpu_time = event.self_cpu_time_total / 1000 / num_profile_steps
            calls = event.count // num_profile_steps
            results[event.key] = {
                "cuda_time_ms": device_time,
                "cpu_time_ms": cpu_time,
                "calls": calls,
            }
            print(f"{event.key:<40} {device_time:<15.3f} {cpu_time:<15.3f} {calls:<8}")

    # Print aggregated summary
    print("\n" + "=" * 80)
    print("AGGREGATED SUMMARY")
    print("=" * 80)

    # Aggregate by component type
    aggregates = {
        "data_preprocess": 0,
        "input_block": 0,
        "neighbor_att": 0,
        "edge_ffn": 0,
        "node_att": 0,
        "node_ffn": 0,
        "energy_head": 0,
        "force_head": 0,
    }

    for key, val in results.items():
        if "data_preprocess" in key:
            aggregates["data_preprocess"] += val["cuda_time_ms"]
        elif "input_block" in key:
            aggregates["input_block"] += val["cuda_time_ms"]
        elif "neighbor_att" in key:
            aggregates["neighbor_att"] += val["cuda_time_ms"]
        elif "edge_ffn" in key:
            aggregates["edge_ffn"] += val["cuda_time_ms"]
        elif "node_att" in key:
            aggregates["node_att"] += val["cuda_time_ms"]
        elif "node_ffn" in key:
            aggregates["node_ffn"] += val["cuda_time_ms"]
        elif "energy_head" in key:
            aggregates["energy_head"] += val["cuda_time_ms"]
        elif "force_head" in key:
            aggregates["force_head"] += val["cuda_time_ms"]

    total = sum(aggregates.values())
    print(f"\n{'Component':<25} {'CUDA Time (ms)':<15} {'Percentage':<10}")
    print("-" * 50)
    for comp, time_ms in sorted(aggregates.items(), key=lambda x: -x[1]):
        if time_ms > 0:
            pct = time_ms / total * 100 if total > 0 else 0
            print(f"{comp:<25} {time_ms:<15.3f} {pct:<10.1f}%")
    print("-" * 50)
    print(f"{'TOTAL':<25} {total:<15.3f}")

    # Save chrome trace
    trace_file = os.path.join(output_dir, "chrome_trace.json")
    prof.export_chrome_trace(trace_file)
    print(f"\nChrome trace saved to: {trace_file}")
    print("Open in chrome://tracing or https://ui.perfetto.dev/")

    # Save text summary
    summary_file = os.path.join(output_dir, "profile_summary.txt")
    with open(summary_file, "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
    print(f"Summary saved to: {summary_file}")

    # Save JSON with component breakdown
    import json

    # Build per-layer breakdown
    per_layer = {}
    for i in range(num_layers):
        layer_key = f"layer_{i}"
        per_layer[layer_key] = {
            "neighbor_att_ms": results.get(f"layer_{i}_neighbor_att", {}).get(
                "cuda_time_ms", 0
            ),
            "edge_ffn_ms": results.get(f"layer_{i}_edge_ffn", {}).get(
                "cuda_time_ms", 0
            ),
            "node_att_ms": results.get(f"layer_{i}_node_att", {}).get(
                "cuda_time_ms", 0
            ),
            "node_ffn_ms": results.get(f"layer_{i}_node_ffn", {}).get(
                "cuda_time_ms", 0
            ),
        }
        per_layer[layer_key]["layer_total_ms"] = sum(per_layer[layer_key].values())

    # Build JSON output
    json_output = {
        "num_atoms": data.pos.shape[0],
        "num_layers": num_layers,
        "hidden_size": model.backbone.global_cfg.hidden_size,
        "num_warmup": num_warmup,
        "num_profile_steps": num_profile_steps,
        "summary": {
            "data_preprocess_ms": aggregates["data_preprocess"],
            "input_block_ms": aggregates["input_block"],
            "neighbor_att_total_ms": aggregates["neighbor_att"],
            "edge_ffn_total_ms": aggregates["edge_ffn"],
            "node_att_total_ms": aggregates["node_att"],
            "node_ffn_total_ms": aggregates["node_ffn"],
            "energy_head_ms": aggregates["energy_head"],
            "force_head_ms": aggregates["force_head"],
            "total_ms": total,
        },
        "percentages": {
            comp: (time_ms / total * 100 if total > 0 else 0)
            for comp, time_ms in aggregates.items()
        },
        "per_layer": per_layer,
        "raw_results": results,
    }

    json_file = os.path.join(output_dir, "profile_results.json")
    with open(json_file, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"JSON results saved to: {json_file}")

    return results, aggregates


def profile_from_preprocessed(
    model: HydraModelV2,
    preprocessed_data,
    displacement: torch.Tensor,
    orig_cell: torch.Tensor | None,
    num_warmup: int = 2,
    num_profile_steps: int = 3,
    device: str = "cuda",
    output_dir: str = "./profile_results",
):
    """
    Profile model forward pass using preprocessed data (no graph construction).

    Profiles:
    - input_block: Input embeddings
    - layer_X_neighbor_att: Neighborhood attention for layer X
    - layer_X_edge_ffn: Edge FFN for layer X
    - layer_X_node_att: Node attention for layer X
    - layer_X_node_ffn: Node FFN for layer X
    - energy_head: Energy head forward
    - force_head: Force head forward
    """
    import json

    from torch.profiler import ProfilerActivity, profile, schedule

    os.makedirs(output_dir, exist_ok=True)

    print("Running profiler (from_preprocessed mode)...")
    print(f"  Warmup steps: {num_warmup}")
    print(f"  Profile steps: {num_profile_steps}")

    # Get forward functions (no compile for profiling)
    forward_fn = model.backbone.compiled_forward
    energy_forward_fn = model.output_heads["energy_head"].compiled_forward
    force_forward_fn = model.output_heads["force_head"].compiled_forward

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            emb = forward_fn(preprocessed_data)
            emb["displacement"] = displacement
            emb["orig_cell"] = orig_cell
            _ = energy_forward_fn(emb)
            _ = force_forward_fn(emb)
        torch.cuda.synchronize()

    clear_cuda_cache()

    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(
            wait=0,
            warmup=0,
            active=num_profile_steps,
            repeat=1,
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(num_profile_steps):
            with torch.no_grad():
                emb = forward_fn(preprocessed_data)
                emb["displacement"] = displacement
                emb["orig_cell"] = orig_cell
                _ = energy_forward_fn(emb)
                _ = force_forward_fn(emb)
            torch.cuda.synchronize()
            prof.step()

    # Print summary
    print("\n" + "=" * 80)
    print("PROFILER RESULTS (sorted by Device time)")
    print("=" * 80)

    # Get key averages
    key_averages = prof.key_averages()

    # Filter for our record_function annotations (no data_preprocess in this mode)
    our_annotations = [
        "input_block",
        "energy_head_compile_forward",
        "force_head_compile_forward",
    ]

    # Add layer-specific annotations
    num_layers = model.backbone.global_cfg.num_layers
    for i in range(num_layers):
        our_annotations.extend(
            [
                f"layer_{i}_neighbor_att",
                f"layer_{i}_edge_ffn",
                f"layer_{i}_node_att",
                f"layer_{i}_node_ffn",
            ]
        )

    # Print table header
    print(
        f"\n{'Component':<40} {'Device Time (ms)':<15} {'CPU Time (ms)':<15} {'Calls':<8}"
    )
    print("-" * 80)

    # Collect and print our annotations
    results = {}
    for event in key_averages:
        if event.key in our_annotations:
            device_time = (
                event.self_device_time_total / 1000 / num_profile_steps
            )  # us to ms, average
            cpu_time = event.self_cpu_time_total / 1000 / num_profile_steps
            calls = event.count // num_profile_steps
            results[event.key] = {
                "cuda_time_ms": device_time,
                "cpu_time_ms": cpu_time,
                "calls": calls,
            }
            print(f"{event.key:<40} {device_time:<15.3f} {cpu_time:<15.3f} {calls:<8}")

    # Print aggregated summary
    print("\n" + "=" * 80)
    print("AGGREGATED SUMMARY")
    print("=" * 80)

    # Aggregate by component type
    aggregates = {
        "input_block": 0,
        "neighbor_att": 0,
        "edge_ffn": 0,
        "node_att": 0,
        "node_ffn": 0,
        "energy_head": 0,
        "force_head": 0,
    }

    for key, val in results.items():
        if "input_block" in key:
            aggregates["input_block"] += val["cuda_time_ms"]
        elif "neighbor_att" in key:
            aggregates["neighbor_att"] += val["cuda_time_ms"]
        elif "edge_ffn" in key:
            aggregates["edge_ffn"] += val["cuda_time_ms"]
        elif "node_att" in key:
            aggregates["node_att"] += val["cuda_time_ms"]
        elif "node_ffn" in key:
            aggregates["node_ffn"] += val["cuda_time_ms"]
        elif "energy_head" in key:
            aggregates["energy_head"] += val["cuda_time_ms"]
        elif "force_head" in key:
            aggregates["force_head"] += val["cuda_time_ms"]

    total = sum(aggregates.values())
    print(f"\n{'Component':<25} {'Device Time (ms)':<15} {'Percentage':<10}")
    print("-" * 50)
    for comp, time_ms in sorted(aggregates.items(), key=lambda x: -x[1]):
        if time_ms > 0:
            pct = time_ms / total * 100 if total > 0 else 0
            print(f"{comp:<25} {time_ms:<15.3f} {pct:<10.1f}%")
    print("-" * 50)
    print(f"{'TOTAL':<25} {total:<15.3f}")

    # Save chrome trace
    trace_file = os.path.join(output_dir, "chrome_trace.json")
    prof.export_chrome_trace(trace_file)
    print(f"\nChrome trace saved to: {trace_file}")
    print("Open in chrome://tracing or https://ui.perfetto.dev/")

    # Save text summary
    summary_file = os.path.join(output_dir, "profile_summary.txt")
    with open(summary_file, "w") as f:
        f.write(
            prof.key_averages().table(sort_by="self_device_time_total", row_limit=50)
        )
    print(f"Summary saved to: {summary_file}")

    # Build per-layer breakdown
    per_layer = {}
    for i in range(num_layers):
        layer_key = f"layer_{i}"
        per_layer[layer_key] = {
            "neighbor_att_ms": results.get(f"layer_{i}_neighbor_att", {}).get(
                "cuda_time_ms", 0
            ),
            "edge_ffn_ms": results.get(f"layer_{i}_edge_ffn", {}).get(
                "cuda_time_ms", 0
            ),
            "node_att_ms": results.get(f"layer_{i}_node_att", {}).get(
                "cuda_time_ms", 0
            ),
            "node_ffn_ms": results.get(f"layer_{i}_node_ffn", {}).get(
                "cuda_time_ms", 0
            ),
        }
        per_layer[layer_key]["layer_total_ms"] = sum(per_layer[layer_key].values())

    # Build JSON output
    json_output = {
        "num_atoms": preprocessed_data.num_nodes,
        "num_layers": num_layers,
        "hidden_size": model.backbone.global_cfg.hidden_size,
        "num_warmup": num_warmup,
        "num_profile_steps": num_profile_steps,
        "mode": "from_preprocessed",
        "summary": {
            "data_preprocess_ms": 0,  # Not applicable for preprocessed mode
            "input_block_ms": aggregates["input_block"],
            "neighbor_att_total_ms": aggregates["neighbor_att"],
            "edge_ffn_total_ms": aggregates["edge_ffn"],
            "node_att_total_ms": aggregates["node_att"],
            "node_ffn_total_ms": aggregates["node_ffn"],
            "energy_head_ms": aggregates["energy_head"],
            "force_head_ms": aggregates["force_head"],
            "total_ms": total,
        },
        "percentages": {
            comp: (time_ms / total * 100 if total > 0 else 0)
            for comp, time_ms in aggregates.items()
        },
        "per_layer": per_layer,
        "raw_results": results,
    }

    json_file = os.path.join(output_dir, "profile_results.json")
    with open(json_file, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"JSON results saved to: {json_file}")

    return results, aggregates


def save_reference_output(output: dict, num_atoms: int, output_dir: str):
    """Save reference output for later comparison."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"reference_output_{num_atoms}.pt")

    # Convert to CPU tensors for saving
    save_dict = {}
    for head_name in output:
        save_dict[head_name] = {}
        for key, val in output[head_name].items():
            if isinstance(val, torch.Tensor):
                save_dict[head_name][key] = val.detach().cpu()
            else:
                save_dict[head_name][key] = val

    torch.save(save_dict, filepath)
    print(f"Saved reference output to {filepath}")


def load_reference_output(num_atoms: int, output_dir: str, device: str = "cuda"):
    """Load reference output for comparison."""
    filepath = os.path.join(output_dir, f"reference_output_{num_atoms}.pt")
    if not os.path.exists(filepath):
        return None

    save_dict = torch.load(filepath)

    # Move to device
    for head_name in save_dict:
        for key, val in save_dict[head_name].items():
            if isinstance(val, torch.Tensor):
                save_dict[head_name][key] = val.to(device)

    return save_dict


# ============================================================================
# Preprocessing functions for benchmarking without graph construction
# ============================================================================


def preprocess_and_save_data(
    atom_sizes: list[int],
    output_dir: str,
    cutoff: float = 6.0,
    max_neighbors: int = 50,
    max_neighbors_pad_size: int = 60,
    use_sincx_mask: bool = False,
    use_freq_mask: bool = False,
    use_chunked_graph: bool = False,
    graph_chunk_size: int = 512,
    device: str = "cuda",
    seed: int = 42,
    no_pbc: bool = False,
):
    """
    Preprocess and save graph data for multiple system sizes.

    The saved data is independent of model size (hidden_size, num_layers) and can be
    reused for benchmarking different model configurations.

    Args:
        atom_sizes: List of atom counts to preprocess
        output_dir: Directory to save preprocessed data
        cutoff: Cutoff radius for graph construction
        max_neighbors: Maximum number of neighbors (knn_k)
        max_neighbors_pad_size: Padded size for neighbors
        use_sincx_mask: Whether to compute sincx mask for node attention
        use_chunked_graph: Whether to use chunked graph construction
        graph_chunk_size: Chunk size for chunked graph construction
        device: Device to use for preprocessing
        seed: Random seed
        no_pbc: Whether to disable periodic boundary conditions
    """
    from dataclasses import asdict

    from fairchem.core.models.allscaip.configs import (
        GlobalConfigs,
        GraphNeuralNetworksConfigs,
        MolecularGraphConfigs,
    )
    from fairchem.core.models.allscaip.utils.data_preprocess import (
        data_preprocess_radius_graph,
    )
    from fairchem.core.models.escaip.utils.graph_utils import get_displacement_and_cell

    os.makedirs(output_dir, exist_ok=True)
    seed_everywhere(seed)

    # Save preprocessing parameters for verification during loading
    params = {
        "cutoff": cutoff,
        "max_neighbors": max_neighbors,
        "max_neighbors_pad_size": max_neighbors_pad_size,
        "use_sincx_mask": use_sincx_mask,
        "use_freq_mask": use_freq_mask,
        "no_pbc": no_pbc,
    }
    torch.save(params, os.path.join(output_dir, "preprocess_params.pt"))

    for num_atoms in atom_sizes:
        print(f"\n{'='*60}")
        print(f"Preprocessing {num_atoms} atoms...")
        print(f"{'='*60}")

        # Generate data
        data = get_sample_data(num_atoms, no_pbc=no_pbc)
        actual_num_atoms = data.pos.shape[0]
        print(f"Actual number of atoms: {actual_num_atoms}")

        # Move to device
        data = data.to(device)

        # Prepare data
        data["atomic_numbers"] = data["atomic_numbers"].long()
        data["atomic_numbers_full"] = data["atomic_numbers"]
        data["batch_full"] = data["batch"]

        # Get displacement and cell (needed for force computation)
        displacement, orig_cell = get_displacement_and_cell(
            data,
            regress_stress=False,
            regress_forces=True,
            direct_forces=True,
        )

        # Create minimal configs for preprocessing
        # Note: hidden_size and num_layers don't affect preprocessing
        global_cfg = GlobalConfigs(
            regress_forces=True,
            direct_forces=True,
            hidden_size=512,  # Doesn't affect preprocessing
            num_layers=6,  # Doesn't affect preprocessing
            use_padding=False,  # MD mode: no padding
            use_node_path=True,
            single_system_no_padding=True,
        )

        molecular_graph_cfg = MolecularGraphConfigs(
            max_num_elements=110,
            max_atoms=actual_num_atoms,
            max_batch_size=1,
            max_radius=cutoff,
            knn_k=max_neighbors,
            knn_pad_size=max_neighbors_pad_size,
            use_chunked_graph=use_chunked_graph,
            graph_chunk_size=graph_chunk_size,
        )

        gnn_cfg = GraphNeuralNetworksConfigs(
            atten_name="memory_efficient",
            atten_num_heads=8,
            use_sincx_mask=use_sincx_mask,
            use_freq_mask=use_freq_mask,
        )

        # Run preprocessing
        print("Running graph construction...")
        start_time = time.perf_counter()
        with torch.no_grad():
            preprocessed_data = data_preprocess_radius_graph(
                data, global_cfg, gnn_cfg, molecular_graph_cfg
            )
        torch.cuda.synchronize()
        preprocess_time = time.perf_counter() - start_time
        print(f"Preprocessing time: {preprocess_time*1000:.2f} ms")

        # Save to CPU to reduce file size and allow loading on any device
        save_dict = {
            "preprocessed_data": asdict(preprocessed_data.to(torch.device("cpu"))),
            "displacement": displacement.cpu() if displacement is not None else None,
            "orig_cell": orig_cell.cpu() if orig_cell is not None else None,
            "num_atoms": actual_num_atoms,
        }

        filepath = os.path.join(output_dir, f"preprocessed_{actual_num_atoms}.pt")
        torch.save(save_dict, filepath)
        print(f"Saved to {filepath}")

        # Clear memory
        del data, preprocessed_data, save_dict
        clear_cuda_cache()

    print(f"\n{'='*60}")
    print(f"Preprocessing complete! Data saved to {output_dir}")
    print(f"{'='*60}")


def load_preprocessed_data(num_atoms: int, data_dir: str, device: str = "cuda"):
    """
    Load preprocessed graph data.

    Args:
        num_atoms: Number of atoms to load
        data_dir: Directory containing preprocessed data
        device: Device to load data to

    Returns:
        Tuple of (preprocessed_data, displacement, orig_cell) or None if not found
    """
    from fairchem.core.models.allscaip.custom_types import GraphAttentionData

    filepath = os.path.join(data_dir, f"preprocessed_{num_atoms}.pt")
    if not os.path.exists(filepath):
        # Try to find closest match
        import glob

        files = glob.glob(os.path.join(data_dir, "preprocessed_*.pt"))
        if files:
            available = [int(f.split("_")[-1].replace(".pt", "")) for f in files]
            print(f"Available atom counts: {sorted(available)}")
        return None

    print(f"Loading preprocessed data from {filepath}...")
    save_dict = torch.load(filepath)

    # Reconstruct GraphAttentionData
    data_dict = save_dict["preprocessed_data"]
    # Move tensors to device
    for key, val in data_dict.items():
        if isinstance(val, torch.Tensor):
            data_dict[key] = val.to(device)

    preprocessed_data = GraphAttentionData(**data_dict)
    displacement = (
        save_dict["displacement"].to(device)
        if save_dict["displacement"] is not None
        else None
    )
    orig_cell = (
        save_dict["orig_cell"].to(device)
        if save_dict["orig_cell"] is not None
        else None
    )

    return preprocessed_data, displacement, orig_cell


def benchmark_from_preprocessed(
    preprocessed_data,
    displacement: torch.Tensor,
    orig_cell: torch.Tensor | None,
    hidden_size: int,
    num_layers: int,
    atten_num_heads: int,
    num_warmup: int,
    num_iterations: int,
    use_compile: bool = False,
    device: str = "cuda",
):
    """
    Benchmark model forward pass using preprocessed data.

    This is the fastest benchmarking mode as it skips all graph construction.

    Args:
        preprocessed_data: GraphAttentionData from load_preprocessed_data
        displacement: Displacement tensor
        orig_cell: Original cell tensor
        hidden_size: Model hidden size
        num_layers: Number of layers
        atten_num_heads: Number of attention heads
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
        use_compile: Whether to use torch.compile
        device: Device to use
    """
    from fairchem.core.models.allscaip.utils.data_preprocess import unpad_results

    num_atoms = preprocessed_data.num_nodes

    # Build model with specified size
    print(f"Building model (hidden_size={hidden_size}, num_layers={num_layers})...")
    model = get_allscaip_model(
        cutoff=6.0,  # Not used since we're using preprocessed data
        use_compile=use_compile,
        max_atoms=num_atoms,
        max_neighbors=50,
        max_neighbors_pad_size=60,
        use_sincx_mask=False,
        single_system_no_padding=True,
        hidden_size=hidden_size,
        num_layers=num_layers,
        atten_num_heads=atten_num_heads,
        device=device,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")

    # Get forward functions
    forward_fn = (
        torch.compile(model.backbone.compiled_forward)
        if use_compile
        else model.backbone.compiled_forward
    )
    energy_forward_fn = (
        torch.compile(model.output_heads["energy_head"].compiled_forward)
        if use_compile
        else model.output_heads["energy_head"].compiled_forward
    )
    force_forward_fn = (
        torch.compile(model.output_heads["force_head"].compiled_forward)
        if use_compile
        else model.output_heads["force_head"].compiled_forward
    )

    # Warmup
    print(f"Warming up ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        with torch.no_grad():
            emb = forward_fn(preprocessed_data)
            emb["displacement"] = displacement
            emb["orig_cell"] = orig_cell
            _ = energy_forward_fn(emb)
            _ = force_forward_fn(emb)
        torch.cuda.synchronize()

    clear_cuda_cache()

    # Benchmark
    print(f"Benchmarking ({num_iterations} iterations)...")
    times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            emb = forward_fn(preprocessed_data)
            emb["displacement"] = displacement
            emb["orig_cell"] = orig_cell
            energy = energy_forward_fn(emb)
            forces = force_forward_fn(emb)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    # Get memory stats
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB

    # Unpad outputs
    if len(energy.shape) == 0:
        energy = energy.unsqueeze(0)
    output_raw = {"energy": energy, "forces": forces}
    output_unpadded = unpad_results(output_raw, preprocessed_data)

    output = {
        "energy_head": {"energy": output_unpadded["energy"]},
        "force_head": {"forces": output_unpadded["forces"]},
    }

    return {
        "times": times,
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "peak_memory_mb": peak_memory,
        "output": output,
        "num_params": num_params,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark AllScAIP model")
    parser.add_argument("--num_atoms", type=int, help="Number of atoms in the system")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "no_graph_gen", "from_preprocessed"],
        default="full",
        help="Benchmark mode: 'full' includes graph generation, 'no_graph_gen' precomputes once, 'from_preprocessed' loads saved data",
    )
    parser.add_argument(
        "--num_warmup", type=int, default=5, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--num_iterations", type=int, default=20, help="Number of benchmark iterations"
    )
    parser.add_argument("--cutoff", type=float, default=6.0, help="Cutoff radius")
    parser.add_argument(
        "--max_neighbors",
        type=int,
        default=50,
        help="Maximum number of neighbors (knn_k)",
    )
    parser.add_argument(
        "--max_neighbors_pad_size",
        type=int,
        default=60,
        help="Padded size for neighbors",
    )
    parser.add_argument("--use_compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save_reference", action="store_true", help="Save output as reference"
    )
    parser.add_argument(
        "--check_reference", action="store_true", help="Check against saved reference"
    )
    parser.add_argument(
        "--reference_dir",
        type=str,
        default="./benchmark_references",
        help="Directory for reference outputs",
    )
    parser.add_argument(
        "--use_sincx_mask",
        action="store_true",
        help="Enable sincx mask in node attention (Si)",
    )
    parser.add_argument(
        "--use_freq_mask",
        action="store_true",
        help="Enable frequency mask in neighbor attention (An)",
    )
    parser.add_argument(
        "--use_node_path",
        action="store_true",
        default=True,
        help="Enable node attention path (No)",
    )
    parser.add_argument(
        "--no_node_path", action="store_true", help="Disable node attention path"
    )
    parser.add_argument(
        "--md_mode",
        action="store_true",
        help="MD simulation mode: single system, no padding (enables optimizations)",
    )
    parser.add_argument(
        "--use_chunked_graph",
        action="store_true",
        help="Enable chunked graph construction to reduce peak memory",
    )
    parser.add_argument(
        "--graph_chunk_size",
        type=int,
        default=512,
        help="Chunk size for chunked graph construction",
    )
    parser.add_argument(
        "--preprocess_on_cpu",
        action="store_true",
        help="Run graph preprocessing on CPU to reduce GPU memory",
    )

    # Model size parameters (for benchmarking different model sizes)
    parser.add_argument(
        "--hidden_size", type=int, default=512, help="Model hidden size"
    )
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers")
    parser.add_argument(
        "--atten_num_heads", type=int, default=8, help="Number of attention heads"
    )

    # Preprocessing mode
    parser.add_argument(
        "--preprocess_data",
        action="store_true",
        help="Preprocess and save data for multiple system sizes",
    )
    parser.add_argument(
        "--atom_sizes",
        type=str,
        default="500,1000,2000,5000",
        help="Comma-separated list of atom sizes to preprocess",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./preprocessed_data",
        help="Directory for preprocessed data",
    )
    parser.add_argument(
        "--no_pbc", action="store_true", help="Disable periodic boundary conditions"
    )

    # Profile mode
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run profiler to get detailed component breakdown (disables compile)",
    )
    parser.add_argument(
        "--profile_dir",
        type=str,
        default="./profile_results",
        help="Directory for profile outputs",
    )

    args = parser.parse_args()

    # Handle preprocessing mode
    if args.preprocess_data:
        atom_sizes = [int(x.strip()) for x in args.atom_sizes.split(",")]
        os.environ["CUDA_VISIBLE_DEVICES"] = (
            args.device.split(":")[-1] if ":" in args.device else "0"
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"

        preprocess_and_save_data(
            atom_sizes=atom_sizes,
            output_dir=args.data_dir,
            cutoff=args.cutoff,
            max_neighbors=args.max_neighbors,
            max_neighbors_pad_size=args.max_neighbors_pad_size,
            use_sincx_mask=args.use_sincx_mask,
            use_freq_mask=args.use_freq_mask,
            use_chunked_graph=args.use_chunked_graph,
            graph_chunk_size=args.graph_chunk_size,
            device=device,
            seed=args.seed,
            no_pbc=args.no_pbc,
        )
        return

    # Check required args for benchmark modes
    if args.num_atoms is None:
        parser.error("--num_atoms is required for benchmark modes")

    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        args.device.split(":")[-1] if ":" in args.device else "0"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("AllScAIP Benchmark")
    print("=" * 60)
    print(f"Number of atoms: {args.num_atoms}")
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"Use compile: {args.use_compile}")
    if args.mode != "from_preprocessed":
        print(f"Cutoff: {args.cutoff}")
        print(f"Max neighbors: {args.max_neighbors}")
    print(f"Model hidden_size: {args.hidden_size}")
    print(f"Model num_layers: {args.num_layers}")
    print(f"Model atten_num_heads: {args.atten_num_heads}")
    print(f"Warmup iterations: {args.num_warmup}")
    print(f"Benchmark iterations: {args.num_iterations}")
    # Handle node path flag
    use_node_path = not args.no_node_path

    if args.mode != "from_preprocessed":
        print(f"Use node path (No): {use_node_path}")
        print(f"Use freq mask (An): {args.use_freq_mask}")
        print(f"Use sincx mask (Si): {args.use_sincx_mask}")
        print(f"MD mode (single system, no padding): {args.md_mode}")
        print(f"Use chunked graph: {args.use_chunked_graph}")
        if args.use_chunked_graph:
            print(f"Graph chunk size: {args.graph_chunk_size}")
        print(f"Preprocess on CPU: {args.preprocess_on_cpu}")
    else:
        print(f"Data directory: {args.data_dir}")
    print("=" * 60)

    # Set seed
    seed_everywhere(args.seed)

    # Handle from_preprocessed mode separately
    if args.mode == "from_preprocessed":
        # Load preprocessed data
        loaded = load_preprocessed_data(args.num_atoms, args.data_dir, device)
        if loaded is None:
            print(
                f"Error: No preprocessed data found for {args.num_atoms} atoms in {args.data_dir}"
            )
            return

        preprocessed_data, displacement, orig_cell = loaded
        actual_num_atoms = preprocessed_data.num_nodes
        print(f"Loaded preprocessed data for {actual_num_atoms} atoms")

        # Disable compile when profiling
        use_compile = args.use_compile
        if args.profile:
            if use_compile:
                print(
                    "WARNING: Disabling compile for profiling (compile fuses operations)"
                )
            use_compile = False

        # Build model for profiling or benchmarking
        print("\nBuilding model...")
        model = get_allscaip_model(
            cutoff=6.0,  # Not used since we're using preprocessed data
            use_compile=use_compile,
            max_atoms=actual_num_atoms,
            max_neighbors=50,
            max_neighbors_pad_size=60,
            use_sincx_mask=False,
            use_freq_mask=False,
            use_node_path=True,
            single_system_no_padding=True,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            atten_num_heads=args.atten_num_heads,
            device=device,
        )

        num_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {num_params:,}")

        # Clear cache before benchmark
        clear_cuda_cache()

        # Run profiler if requested
        if args.profile:
            print("\nRunning profiler (from_preprocessed mode)...")
            profile_from_preprocessed(
                model=model,
                preprocessed_data=preprocessed_data,
                displacement=displacement,
                orig_cell=orig_cell,
                num_warmup=args.num_warmup,
                num_profile_steps=args.num_iterations,
                device=device,
                output_dir=args.profile_dir,
            )
            return

        # Run benchmark
        print(f"\nRunning benchmark ({args.mode} mode)...")
        results = benchmark_from_preprocessed(
            preprocessed_data=preprocessed_data,
            displacement=displacement,
            orig_cell=orig_cell,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            atten_num_heads=args.atten_num_heads,
            num_warmup=args.num_warmup,
            num_iterations=args.num_iterations,
            use_compile=use_compile,
            device=device,
        )
    else:
        # Get data
        print("\nGenerating sample data...")
        data = get_sample_data(args.num_atoms, no_pbc=args.no_pbc)
        actual_num_atoms = data.pos.shape[0]
        print(f"Actual number of atoms: {actual_num_atoms}")

        # Compute max_atoms - no buffer in MD mode
        if args.md_mode:
            max_atoms = actual_num_atoms
            print("MD mode enabled: single system optimizations active")
        else:
            max_atoms = actual_num_atoms + 10

        # Disable compile when profiling (compile fuses operations)
        use_compile = args.use_compile
        if args.profile:
            if use_compile:
                print(
                    "WARNING: Disabling compile for profiling (compile fuses operations)"
                )
            use_compile = False

        # Build model
        print("\nBuilding model...")
        model = get_allscaip_model(
            cutoff=args.cutoff,
            use_compile=use_compile,
            max_atoms=max_atoms,
            max_neighbors=args.max_neighbors,
            max_neighbors_pad_size=args.max_neighbors_pad_size,
            use_sincx_mask=args.use_sincx_mask,
            use_freq_mask=args.use_freq_mask,
            use_node_path=use_node_path,
            single_system_no_padding=args.md_mode,
            use_chunked_graph=args.use_chunked_graph,
            graph_chunk_size=args.graph_chunk_size,
            preprocess_on_cpu=args.preprocess_on_cpu,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            atten_num_heads=args.atten_num_heads,
            device=device,
        )

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {num_params:,}")

        # Clear cache before benchmark
        clear_cuda_cache()

        # Run profile mode
        if args.profile:
            print("\nRunning profiler...")
            profile_model(
                model=model,
                data=data,
                num_warmup=args.num_warmup,
                num_profile_steps=args.num_iterations,
                device=device,
                output_dir=args.profile_dir,
            )
            return

        # Run benchmark
        print(f"\nRunning benchmark ({args.mode} mode)...")
        if args.mode == "full":
            results = benchmark_full_forward(
                model=model,
                data=data,
                num_warmup=args.num_warmup,
                num_iterations=args.num_iterations,
                device=device,
            )
        else:  # no_graph_gen
            results = benchmark_no_graph_gen(
                model=model,
                data=data,
                num_warmup=args.num_warmup,
                num_iterations=args.num_iterations,
                device=device,
            )

    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Mean time: {results['mean_time']*1000:.3f} ms")
    print(f"Std time: {results['std_time']*1000:.3f} ms")
    print(f"Min time: {min(results['times'])*1000:.3f} ms")
    print(f"Max time: {max(results['times'])*1000:.3f} ms")
    print(f"Peak memory: {results['peak_memory_mb']:.2f} MB")
    print(f"Throughput: {1/results['mean_time']:.2f} forward/s")

    # Show output shapes
    print("\nOutput shapes:")
    for head_name, head_output in results["output"].items():
        for key, val in head_output.items():
            if isinstance(val, torch.Tensor):
                print(f"  {head_name}/{key}: {val.shape}")

    # Get actual_num_atoms for reference operations (already set in from_preprocessed mode)
    if args.mode != "from_preprocessed":
        pass  # actual_num_atoms already set above

    # Save reference if requested
    if args.save_reference:
        save_reference_output(results["output"], actual_num_atoms, args.reference_dir)

    # Check against reference if requested
    if args.check_reference:
        reference = load_reference_output(actual_num_atoms, args.reference_dir, device)
        if reference is not None:
            print("\nChecking against reference:")
            verification = verify_correctness(results["output"], reference)
            all_match = True
            for key, result in verification.items():
                status = "✓" if result["match"] else "✗"
                print(f"  {key}: {status}", end="")
                if "max_diff" in result:
                    print(f" (max_diff: {result['max_diff']:.2e})", end="")
                print()
                all_match = all_match and result["match"]

            if all_match:
                print("\n✓ All outputs match reference!")
            else:
                print("\n✗ Some outputs differ from reference!")
        else:
            print(f"\nNo reference found at {args.reference_dir}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
