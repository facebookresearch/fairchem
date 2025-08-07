import os
import pytest
import torch
import numpy as np
import shutil
import glob
from pathlib import Path
from ase.build import bulk
from ase.db import connect
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit.mlip_unit import initialize_finetuning_model
from fairchem.core.datasets.atomic_data import AtomicData

@pytest.fixture(scope="function")
def cleanup_finetune_directories():
    """Cleanup finetune directories after each test."""
    yield  # Run the test
    
    # Clean up common temporary directories created during fine-tuning tests
    cleanup_paths = [
        "/tmp/uma_finetune_runs/",
        "/tmp/finetune_run/"
    ]
    
    for path in cleanup_paths:
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                print(f"Cleaned up directory: {path}")
            except Exception as e:
                print(f"Warning: Could not clean up {path}: {e}")

@pytest.mark.integration
class TestIntegrationWithRealModel:
    def test_finetune_and_infer_uma_with_5_heads(self, tmp_path, cleanup_finetune_directories):
        import subprocess
        from fairchem.core.scripts.create_uma_finetune_dataset import create_yaml
        # 1. Create a Cu bulk structure and write to ASE database
        db_path = tmp_path / "cu_bulk.db"
        atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
        db = connect(str(db_path))
        db.write(atoms, data={
            'energy': -3.0,
            'natoms': len(atoms),
            'metadata': {'natoms': len(atoms)}
        })

        # 1b. Create minimal metadata.npz file for UMA
        import numpy as np
        np.savez(db_path.parent / "metadata.npz", natoms=np.array([len(atoms)]))

        # 2. Generate UMA finetune config using create_yaml, then patch for shallow ensemble/multi-head
        train_path = str(db_path)
        val_path = str(db_path)
        output_dir = tmp_path
        dataset_name = "omat"
        regression_tasks = "e"
        base_model_name = "uma-s-1p1"
        force_rms = 1.0
        linref_coeff = [0.0]*100
        from fairchem.core.scripts.create_uma_finetune_dataset import create_yaml
        create_yaml(
            train_path=train_path,
            val_path=val_path,
            force_rms=force_rms,
            linref_coeff=linref_coeff,
            output_dir=output_dir,
            dataset_name=dataset_name,
            regression_tasks=regression_tasks,
            base_model_name=base_model_name,
        )

        import yaml
        # Patch the generated data yaml for multi-head/shallow ensemble
        data_yaml_path = tmp_path / "data" / "uma_conserving_data_task_energy.yaml"
        with open(data_yaml_path) as f:
            data_yaml = yaml.safe_load(f)

        num_heads = 5
        heads = {
            f"energy_{i}": {
                "module": "fairchem.core.models.uma.escn_moe.DatasetSpecificSingleHeadWrapper",
                "head_cls": "fairchem.core.models.uma.escn_md.MLP_EFS_Head",
                "dataset_names": [dataset_name],
                "wrap_property": False
            } for i in range(num_heads)
        }
        # Patch tasks_list for shallow ensemble
        task = {
            "_target_": "fairchem.core.units.mlip_unit.mlip_unit.Task",
            "name": "energy",
            "level": "system",
            "property": "energy",
            "loss_fn": {
                "_target_": "fairchem.core.modules.loss.DDPMTLoss",
                "loss_fn": {"_target_": "fairchem.core.modules.loss.PerAtomMAELoss"},
                "coefficient": 20
            },
            "out_spec": {"dim": [1], "dtype": "float32"},
            "normalizer": {
                "_target_": "fairchem.core.modules.normalization.normalizer.Normalizer",
                "mean": 0.0,
                "rmsd": 1.0
            },
            "element_references": {
                "_target_": "fairchem.core.modules.normalization.element_references.ElementReferences",
                "element_references": {"_target_": "torch.DoubleTensor", "_args_": [[0.0]*100]}
            },
            "datasets": [dataset_name],
            "metrics": ["mae", "per_atom_mae"],
            "shallow_ensemble": True
        }
        data_yaml["heads"] = heads
        data_yaml["tasks_list"] = [task]

        # Write back patched data yaml
        with open(data_yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

        # Patch the finetune template yaml to point to the correct data yaml and use tmp_path
        finetune_yaml_path = tmp_path / "uma_sm_finetune_template.yaml"
        with open(finetune_yaml_path) as f:
            finetune_yaml = yaml.safe_load(f)
        finetune_yaml["defaults"][0]["data"] = "uma_conserving_data_task_energy"
        
        # Fix the run_dir to use the test's temporary directory instead of /tmp/uma_finetune_runs/
        finetune_yaml["job"]["run_dir"] = str(tmp_path / "finetune_run")
        
        with open(finetune_yaml_path, "w") as f:
            yaml.dump(finetune_yaml, f, default_flow_style=False, sort_keys=False)

        # 3. Run UMA fine-tuning using fairchem CLI
        try:
            subprocess.run([
                "fairchem", "-c", str(finetune_yaml_path)
            ], check=True)
        except subprocess.CalledProcessError as e:
            # Clean up on failure too
            if (tmp_path / "finetune_run").exists():
                shutil.rmtree(tmp_path / "finetune_run")
            raise e

        # 4. Find the latest UMA fine-tune checkpoint in tmp_path/finetune_run/*/checkpoints/*/inference_ckpt.pt
        ckpt_candidates = glob.glob(str(tmp_path / "finetune_run/*/checkpoints/*/inference_ckpt.pt"))
        assert ckpt_candidates, f"No UMA fine-tune checkpoint found in {tmp_path}/finetune_run/*/checkpoints/*/inference_ckpt.pt"
        ckpt_path = sorted(ckpt_candidates)[-1]
        print(f"Using checkpoint: {ckpt_path}")
        predictor = load_predict_unit(str(ckpt_path))

        # 5. Directly access energy from the model output for assertion
        data = AtomicData.from_ase(atoms)
        data.dataset = ["omat"]
        model_output = predictor.model(data)
        # For shallow ensemble, expect energy_0, energy_1, ...
        for i in range(num_heads):
            key = f"energy_{i}"
            assert key in model_output, f"Model output keys: {list(model_output.keys())}"
            energy = model_output[key]["energy"] if isinstance(model_output[key], dict) else model_output[key]
            assert isinstance(energy, torch.Tensor)
            assert energy.numel() > 0
        
        # 6. Manual cleanup at the end of the test for the test-specific directory
        try:
            if (tmp_path / "finetune_run").exists():
                shutil.rmtree(tmp_path / "finetune_run")
                print(f"Cleaned up test-specific directory: {tmp_path / 'finetune_run'}")
        except Exception as e:
            print(f"Warning: Could not clean up test directory: {e}")
