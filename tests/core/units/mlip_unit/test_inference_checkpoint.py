"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

run using, 
pytest -s --inference-checkpoint inference_ckpt.pt tests/core/units/mlip_unit/test_inference_checkpoint.py
"""
import os
import pytest
import torch
from fairchem.core.datasets.ase_datasets import AseDBDataset
from fairchem.core.datasets.lmdb_dataset import data_list_collater
from fairchem.core.preprocessing.atoms_to_graphs import AtomsToGraphs
from fairchem.core.units.mlip_unit.mlip_unit import MLIPPredictUnit


@pytest.mark.inference_check
def test_inference_checkpoint_direct(
    command_line_inference_checkpoint, fake_puma_dataset, torch_deterministic
):

    predictor = MLIPPredictUnit(command_line_inference_checkpoint, device="cpu")
    # predictor.model.module.backbone.regress_stress = True

    db=AseDBDataset(
                config={"src": os.path.join(fake_puma_dataset,'oc20')}
            )

    a2g = AtomsToGraphs(
            max_neigh=10,
            radius=100,
            r_energy=False,
            r_forces=False,
            r_distances=False,
            r_edges=True,
            r_pbc=True,
            r_data_keys= ['spin', 'charge']
    )


    energies=[]
    forces=[]

    sample_idx=0
    while sample_idx<min(5,len(db)):
        sample = a2g.convert(db.get_atoms(sample_idx))
        sample['dataset']='oc20'
        batch = data_list_collater([sample], otf_graph=False)

        out = predictor.predict(batch)
        energies.append(out['oc20_energy'])
        forces.append(out['forces'] if 'forces' in out else out['oc20_forces'])
        sample_idx+=1
    forces=torch.vstack(forces)
    energies=torch.stack(energies)
    
    print(f"oc20_energies_abs_mean: {energies.abs().mean().item()}, oc20_forces_abs_mean: {forces.abs().mean().item()}")

