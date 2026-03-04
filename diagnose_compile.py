#!/usr/bin/env python
"""Diagnose torch.compile graph breaks for umas_fast_gpu backend."""
from __future__ import annotations

import torch
import torch._dynamo as dynamo
import warnings

warnings.filterwarnings('ignore')

# Enable dynamo logging
torch._dynamo.config.verbose = True

from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.datasets.common_structures import get_fcc_crystal_by_num_atoms
from fairchem.core.units.mlip_unit import MLIPPredictUnit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

CHECKPOINT = "/checkpoint/ocp/shared/uma/release/uma_sm_osc_name_fix.pt"
DEVICE = "cuda"  # Not cuda:0
NATOMS = 2000

print("=" * 60)
print("Creating test structure...")
print("=" * 60)
atoms = get_fcc_crystal_by_num_atoms(NATOMS)
atoms.pbc = [False, False, False]
print(f"Created {len(atoms)} atoms")

# Create inference settings matching run_benchmarks.sh
settings = InferenceSettings(
    tf32=True,
    activation_checkpointing=False,
    merge_mole=True,
    compile=False,  # Disable compile initially for explain
    external_graph_gen=True,
    internal_graph_gen_version=2,
    execution_mode="umas_fast_gpu",
)

print("\n" + "=" * 60)
print("Loading model with umas_fast_gpu backend...")
print("=" * 60)
print(f"Settings: {settings}")

predictor = MLIPPredictUnit(
    CHECKPOINT,
    DEVICE,
    inference_settings=settings,
)

print("\nModel loaded successfully")
model = predictor.model

# Create input data
print("\n" + "=" * 60)
print("Creating input data...")
print("=" * 60)
max_neighbors = predictor.model.module.backbone.max_neighbors
cutoff = predictor.model.module.backbone.cutoff
print(f"Building graph: max_neighbors={max_neighbors}, cutoff={cutoff}")

data_obj = AtomicData.from_ase(
    atoms,
    max_neigh=max_neighbors,
    radius=cutoff,
    r_edges=True,
    task_name="omat",
)
data_obj.natoms = torch.tensor(len(atoms))
data_obj.charge = torch.LongTensor([0])
data_obj.spin = torch.LongTensor([0])
data_obj.pos.requires_grad = True
data = atomicdata_list_to_batch([data_obj])
data = data.to("cuda")
print(f"Graph edges: {data.edge_index.shape[1]}")

# Warmup
print("\n" + "=" * 60)
print("Warmup run...")
print("=" * 60)
with torch.no_grad():
    _ = model(data)
print("Warmup done")

# Run dynamo.explain
print("\n" + "=" * 60)
print("Running torch._dynamo.explain()...")
print("=" * 60)

try:
    explanation = dynamo.explain(model)(data)
    
    print("\n" + "=" * 60)
    print("DYNAMO EXPLAIN RESULTS")
    print("=" * 60)
    print(f"Graph count: {explanation.graph_count}")
    print(f"Graph break count: {explanation.graph_break_count}")
    print(f"Ops per graph: {explanation.ops_per_graph}")
    
    if explanation.break_reasons:
        print("\n" + "-" * 60)
        print("GRAPH BREAK REASONS:")
        print("-" * 60)
        for i, reason in enumerate(explanation.break_reasons):
            print(f"\n[Break {i+1}]")
            print(reason)
    else:
        print("\nNo graph breaks detected!")
        
except Exception as e:
    print(f"\nError during explain: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Done")
print("=" * 60)
