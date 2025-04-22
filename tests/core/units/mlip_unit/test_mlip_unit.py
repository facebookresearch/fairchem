from __future__ import annotations

import math
import os
import pickle
import shutil
import tempfile
from collections import namedtuple
from typing import TYPE_CHECKING

import pytest
import torch
from torchtnt.framework.callback import Callback

from tests.core.testing_utils import launch_main

if TYPE_CHECKING:
    from torchtnt.framework.state import State
    from torchtnt.framework.unit import TEvalUnit, TTrainUnit


class TrainEndCallback(Callback):
    def __init__(
        self,
        expected_loss: float | None,
        expected_max_steps: int | None,
        expected_max_epochs: int | None,
    ):
        self.expected_loss = expected_loss
        self.expected_max_steps = expected_max_steps
        self.expected_max_epochs = expected_max_epochs

    def on_train_end(self, state: State, unit: TTrainUnit) -> None:
        if self.expected_loss is None:
            return
        if self.expected_max_epochs is not None:
            assert (
                unit.train_progress.num_steps_completed
                == len(state.train_state.dataloader)
                * unit.train_progress.num_epochs_completed
            )
            assert unit.train_progress.num_epochs_completed == self.expected_max_epochs
        if self.expected_max_steps is not None:
            assert unit.train_progress.num_steps_completed == self.expected_max_steps
        assert math.isclose(unit.last_loss, self.expected_loss, rel_tol=1e-4)


class EvalEndCallback(Callback):
    def __init__(
        self, total_atoms: int, oc20_energy_mae: float, omol_energy_mae: float
    ):
        self.total_atoms = total_atoms
        self.oc20_energy_mae = oc20_energy_mae
        self.omol_energy_mae = omol_energy_mae

    def on_eval_end(
        self,
        state: State,
        unit: TEvalUnit,
    ) -> None:
        num_samples = len(state.eval_state.dataloader)
        assert unit.eval_progress.num_steps_completed == num_samples
        assert unit.total_atoms == self.total_atoms  # 285
        assert unit.running_metrics["oc20_energy"]["oc20"][
            "mae"
        ].metric == pytest.approx(self.oc20_energy_mae, rel=1e-4)
        assert unit.running_metrics["omol_energy"]["omol"][
            "mae"
        ].metric == pytest.approx(self.omol_energy_mae, rel=1e-4)

        # check that all force metrics are recorded as well
        assert unit.running_metrics["forces"]["oc20"].keys() == {
            "per_atom_mae",
            "cosine_similarity",
            "magnitude_error",
        }


class EvalEndCallbackASELMDB(Callback):
    def __init__(
        self, total_atoms: int, oc20_energy_mae: float, omol_energy_mae: float
    ):
        self.total_atoms = total_atoms
        self.oc20_energy_mae = oc20_energy_mae
        self.omol_energy_mae = omol_energy_mae

    def on_eval_end(
        self,
        state: State,
        unit: TEvalUnit,
    ) -> None:
        num_samples = len(state.eval_state.dataloader)
        assert unit.eval_progress.num_steps_completed == num_samples
        assert unit.total_atoms == self.total_atoms  # 285
        assert unit.running_metrics["oc20_energy"]["oc20.train"][
            "mae"
        ].metric == pytest.approx(self.oc20_energy_mae, rel=1e-4)
        assert unit.running_metrics["omol_energy"]["omol.train"][
            "mae"
        ].metric == pytest.approx(self.omol_energy_mae, rel=1e-4)


def pickle_data_loader(pickle_path: str, steps: int):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    # TODO: regenerate this pickle so it has these names
    data.dataset_name = ["oc20", "oc20", "oc20", "omol"]
    _DummyDataset = namedtuple("_DummyDataset", field_names="dataset_names")

    class _DummyLoader:
        dataset = _DummyDataset(dataset_names=list(set(data.dataset_name)))
        batches = [data] * steps

        def __len__(self):
            return len(self.batches)

        def __iter__(self):
            yield from self.batches

    return _DummyLoader()


def test_full_eval_from_cli():
    sys_args = [
        "--config",
        "tests/core/units/mlip_unit/test_mlip_eval.yaml",
    ]
    launch_main(sys_args)


# test to make sure moe eval heads are in fact giving
# different outputs , would be nice to have to extended
# so that it checks specifically one example at a time and
# not an aggregate MAE
def test_full_conserving_moe_eval_from_cli(fake_puma_dataset):
    sys_args = [
        "--config",
        "tests/core/units/mlip_unit/test_mlip_conserving_eval.yaml",
        f"datasets.data_root_dir={fake_puma_dataset}",
        "oc20_energy_mae=16.582261562347412",
        "omol_energy_mae=0.2785639584064484",
    ]
    launch_main(sys_args)

    sys_args = [
        "--config",
        "tests/core/units/mlip_unit/test_mlip_conserving_eval.yaml",
        f"datasets.data_root_dir={fake_puma_dataset}",
        "oc20_energy_mae=16.582261562347412",
        "omol_energy_mae=0.2785639584064484",
        f"datasets.oc20_val.splits.train.src=[{fake_puma_dataset}/oc20/oc20_val.aselmdb]",
        f"datasets.omol_val.splits.train.src=[{fake_puma_dataset}/omol/omol_val.aselmdb]",
    ]
    launch_main(sys_args)


def test_full_train_from_cli():
    sys_args = [
        "--config",
        "tests/core/units/mlip_unit/test_mlip_train.yaml",
        "+expected_loss=8.070940971374512",
    ]
    launch_main(sys_args)
    
    sys_args = [
        "--config",
        "tests/core/units/mlip_unit/test_mlip_train.yaml",
        "backbone=K2L2_gate",
        "+expected_loss=10.157896995544434",
    ]
    launch_main(sys_args)

    sys_args = [
        "--config",
        "tests/core/units/mlip_unit/test_mlip_train.yaml",
        "backbone=K2L2_gate",
        "backbone.mmax=1",
        "+expected_loss=23.392169952392578",
    ]
    launch_main(sys_args)
    
def test_full_train_eval_from_cli_aselmdb(fake_puma_dataset):
    sys_args = [
        "--config",
        "tests/core/units/mlip_unit/test_mlip_train.yaml",
        "datasets=aselmdb",
        f"datasets.data_root_dir={fake_puma_dataset}",
        "+expected_loss=13.662849426269531",
    ]
    launch_main(sys_args)


def test_grad_train_from_cli_aselmdb_no_lr(fake_puma_dataset):
    with tempfile.TemporaryDirectory() as tmpdirname:
        run1_path = os.path.join(tmpdirname, "run1")
        run2_path = os.path.join(tmpdirname, "run2")
        os.makedirs(run1_path, exist_ok=True)
        os.makedirs(run2_path, exist_ok=True)

        sys_args = [
            "--config",
            "tests/core/units/mlip_unit/test_mlip_train.yaml",
            "datasets=aselmdb",
            f"datasets.data_root_dir={fake_puma_dataset}",
            "optimizer=savegrad",
        ]

        no_gp_args = sys_args.copy()
        no_gp_args.append(f"optimizer.save_path={run1_path}")
        launch_main(no_gp_args)

        no_gp_args = sys_args.copy()
        no_gp_args.append(f"optimizer.save_path={run2_path}")
        launch_main(no_gp_args)

        for step in range(3):
            run1_params_and_grads = torch.load(
                os.path.join(run1_path, f"ddp1.0_gp0.0_step{step}.pt")
            )
            run2_params_and_grads = torch.load(
                os.path.join(run2_path, f"ddp1.0_gp0.0_step{step}.pt")
            )
            relative_diffs = [
                (
                    run1_params_and_grads["grad"][idx]
                    - run2_params_and_grads["grad"][idx]
                ).abs()
                / torch.tensor(
                    [
                        run1_params_and_grads["grad"][idx].abs().max(),
                        run2_params_and_grads["grad"][idx].abs().max(),
                        1e-7,
                    ]
                ).max()
                for idx in range(len(run1_params_and_grads["grad"]))
            ]

            percent_within_tolerance = (
                (torch.tensor([x.mean() for x in relative_diffs]) < 0.0001)
                .to(float)
                .mean()
            )
            assert percent_within_tolerance > 0.999, "Failed percent withing tolerance"


@pytest.mark.parametrize(
    "train_config, dataset_config",
    [
        ("tests/core/units/mlip_unit/test_mlip_train.yaml", "aselmdb"),
        (
            "tests/core/units/mlip_unit/test_mlip_train_conserving.yaml",
            "aselmdb_conserving",
        ),
    ],
)
def test_grad_train_from_cli_aselmdb_no_lr_gp_vs_nongp(
    train_config, dataset_config, fake_puma_dataset
):
    with tempfile.TemporaryDirectory() as tmpdirname:
        no_gp_save_path = os.path.join(tmpdirname, "no_gp")
        gp_save_path = os.path.join(tmpdirname, "gp")
        os.makedirs(no_gp_save_path, exist_ok=True)
        os.makedirs(gp_save_path, exist_ok=True)

        sys_args = [
            "--config",
            train_config,
            f"datasets={dataset_config}",
            f"datasets.data_root_dir={fake_puma_dataset}",
            "optimizer=savegrad",
            "runner.max_steps=1"
        ]

        no_gp_args = sys_args.copy()
        no_gp_args.append(f"optimizer.save_path={no_gp_save_path}")
        no_gp_args += ["+job.scheduler.ranks_per_node=2"]

        launch_main(no_gp_args)

        gp_args = sys_args.copy()
        gp_args.append(f"optimizer.save_path={gp_save_path}")
        gp_args += [
            "+job.scheduler.ranks_per_node=4",
            "+job.graph_parallel_group_size=2",
        ]
        launch_main(gp_args)

        for step in range(1):   
            for ddp_rank in range(4):
                gp_rank = ddp_rank // 2
                compare_to_non_gp_ddp_rank = gp_rank
                gp_params_and_grads = torch.load(
                    os.path.join(
                        gp_save_path, f"ddp4.{ddp_rank}_gp2.{gp_rank}_step{step}.pt"
                    )
                )
                non_gp_params_and_grads = torch.load(
                    os.path.join(
                        no_gp_save_path,
                        f"ddp2.{compare_to_non_gp_ddp_rank}_gp0.0_step{step}.pt",
                    )
                )
                relative_diffs = [
                    (
                        gp_params_and_grads["grad"][idx]
                        - non_gp_params_and_grads["grad"][idx]
                    ).abs()
                    / torch.tensor(
                        [
                            gp_params_and_grads["grad"][idx].abs().max(),
                            non_gp_params_and_grads["grad"][idx].abs().max(),
                            1e-7,
                        ]
                    ).max()
                    for idx in range(len(gp_params_and_grads["grad"]))
                ]

                percent_within_tolerance = (
                    (torch.tensor([x.mean() for x in relative_diffs]) < 0.0001)
                    .to(float)
                    .mean()
                )
                assert (
                    percent_within_tolerance > 0.9
                ), "Failed percent withing 10% tolerance"


@pytest.mark.parametrize("mode", ["gp", "no_gp"])
def test_conserve_train_from_cli_aselmdb(mode, fake_puma_dataset):
    sys_args = [
        "--config",
        "tests/core/units/mlip_unit/test_mlip_train_conserving.yaml",
        "datasets=aselmdb_conserving",
        f"datasets.data_root_dir={fake_puma_dataset}",
        "+expected_loss=33.96360397338867",
    ]
    if mode == "gp":
        sys_args += [
            "+job.scheduler.ranks_per_node=1",
            "+job.graph_parallel_group_size=1",
        ]
    launch_main(sys_args)


@pytest.mark.parametrize(
    "checkpoint_step, max_epochs, expected_loss",
    [
        (3, 2, 6.207723140716553),
        (6, 2, 6.207660675048828),
        (5, 2, 6.207639217376709),
        (6, 3, 43.085227966308594),
        (14, 3, 43.08521270751953),
    ],
)
def test_train_and_resume_max_epochs(
    checkpoint_step, max_epochs, expected_loss, fake_puma_dataset
):
    # first train to completion
    temp_dir = tempfile.mkdtemp()
    timestamp_id = "12345"
    sys_args = [
        "--config",
        "tests/core/units/mlip_unit/test_mlip_train_checkpoint_resume.yaml",
        "datasets=aselmdb",
        f"datasets.data_root_dir={fake_puma_dataset}",
        f"+job.run_dir={temp_dir}",
        f"+job.timestamp_id={timestamp_id}",
        f"max_epochs={max_epochs}",
        f"+expected_loss={expected_loss}",
        f"runner.callbacks.1.checkpoint_every_n_steps={checkpoint_step}",
    ]
    launch_main(sys_args)

    # Now resume from checkpoint_step and should get the same result
    # TODO, should get the run config and get checkpoint location from there
    checkpoin_dir = os.path.join(
        temp_dir, timestamp_id, "checkpoints", f"step_{checkpoint_step}"
    )
    checkpoint_state_yaml = os.path.join(checkpoin_dir, "train_state.yaml")
    assert os.path.isdir(checkpoin_dir)
    assert os.path.isfile(checkpoint_state_yaml)
    # next do a manual restart from a checkpoint on step 3
    sys_args = ["--config", checkpoint_state_yaml]
    launch_main(sys_args)
    shutil.rmtree(temp_dir)


@pytest.mark.parametrize(
    "checkpoint_step, max_steps, expected_loss",
    [
        (4, 7, 29.3492488861084),
        (6, 7, 29.349267959594727),
    ],
)
def test_train_and_resume_max_steps(
    checkpoint_step, max_steps, expected_loss, fake_puma_dataset
):
    # first train to completion
    temp_dir = tempfile.mkdtemp()
    timestamp_id = "12345"
    sys_args = [
        "--config",
        "tests/core/units/mlip_unit/test_mlip_train_checkpoint_resume.yaml",
        "datasets=aselmdb",
        f"datasets.data_root_dir={fake_puma_dataset}",
        f"+job.run_dir={temp_dir}",
        f"+job.timestamp_id={timestamp_id}",
        "max_epochs=null",
        f"max_steps={max_steps}",
        f"+expected_loss={expected_loss}",
        f"runner.callbacks.1.checkpoint_every_n_steps={checkpoint_step}",
    ]
    launch_main(sys_args)

    # Now resume from checkpoint_step and should get the same result
    # TODO, should get the run config and get checkpoint location from there
    checkpoin_dir = os.path.join(
        temp_dir, timestamp_id, "checkpoints", f"step_{checkpoint_step}"
    )
    checkpoint_state_yaml = os.path.join(checkpoin_dir, "train_state.yaml")
    assert os.path.isdir(checkpoin_dir)
    assert os.path.isfile(checkpoint_state_yaml)
    # next do a manual restart from a checkpoint on step 3
    sys_args = ["--config", checkpoint_state_yaml]
    launch_main(sys_args)
    shutil.rmtree(temp_dir)
