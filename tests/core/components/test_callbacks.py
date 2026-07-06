"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests:  CPU-only unit tests for TrainCheckpointCallback best-model
        checkpoint saving. Uses minimal dataclass stubs (no mock)
        instead of real TorchTNT training loops.
Models: none (no pretrained model needed). No GPU required.
CI:     test (core shard) — base CPU partition.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from fairchem.core.components.callbacks import TrainCheckpointCallback

# ---------------------------------------------------------------------------
# Minimal stubs for TorchTNT types (no mock library)
# ---------------------------------------------------------------------------


@dataclass
class _FakeProgress:
    num_steps_completed: int = 0


@dataclass
class _FakeTrainUnit:
    train_progress: _FakeProgress = field(default_factory=_FakeProgress)


@dataclass
class _FakeEvalUnit:
    train_progress: _FakeProgress = field(default_factory=_FakeProgress)
    last_eval_metrics: dict[str, Any] | None = None


@dataclass
class _FakeState:
    """Minimal State stub — only what the callback touches."""
    eval_state: Any = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_save_callback():
    """
    Return a ``save_callback`` that writes a marker file (simulating a
    checkpoint save) so the test can verify which paths were "saved".
    """
    def _save(path: str) -> None:
        os.makedirs(path, exist_ok=True)
        (Path(path) / ".checkpoint_marker").write_text("")
    return _save


def _make_load_callback():
    return lambda _: None


def _setup_callback(
    callback: TrainCheckpointCallback,
    tmp_path: Path,
    *,
    save_callback_override=None,
) -> TrainCheckpointCallback:
    """Attach runner callbacks to the callback under test."""
    checkpoint_dir = str(tmp_path / "checkpoints")
    callback.set_runner_callbacks(
        save_callback=save_callback_override or _make_save_callback(),
        load_callback=_make_load_callback(),
        checkpoint_dir=checkpoint_dir,
    )
    return callback


def _count_checkpoints(tmp_path: Path) -> int:
    """Return the number of non-empty checkpoint directories."""
    ckpt_dir = tmp_path / "checkpoints"
    if not ckpt_dir.exists():
        return 0
    return len([
        d for d in ckpt_dir.iterdir()
        if d.is_dir() and (d / ".checkpoint_marker").exists()
    ])


def _best_model_dirs(tmp_path: Path) -> list[Path]:
    """Return best_model checkpoint directories sorted by name."""
    ckpt_dir = tmp_path / "checkpoints"
    if not ckpt_dir.exists():
        return []
    return sorted(
        d for d in ckpt_dir.iterdir()
        if d.is_dir() and d.name.startswith("best_model_")
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTrainCheckpointCallbackBestModel:
    """Best-model checkpoint saving (the new feature in PR #2060)."""

    def test_save_best_model_with_monitor(self, tmp_path: Path):
        """
        When ``monitor`` is set and a metric improves, the callback saves
        a best_model checkpoint.
        """
        callback = TrainCheckpointCallback(
            checkpoint_every_n_steps=None,
            monitor="val/loss",
            mode="min",
            save_top_k=2,  # keep 2 so improvements don't remove older ones
        )
        _setup_callback(callback, tmp_path)

        state = _FakeState()
        unit = _FakeEvalUnit(
            train_progress=_FakeProgress(num_steps_completed=10),
            last_eval_metrics={"val/loss": 0.5},
        )

        # First eval — should save (len < save_top_k)
        callback.on_eval_end(state, unit)  # type: ignore[arg-type]
        assert _count_checkpoints(tmp_path) == 1, "Should save on first evaluation"

        # Better metric — should save (improving)
        unit.last_eval_metrics = {"val/loss": 0.3}
        callback.on_eval_end(state, unit)  # type: ignore[arg-type]
        assert _count_checkpoints(tmp_path) == 2, "Should save on improvement"

        # Worse metric — should NOT save
        unit.last_eval_metrics = {"val/loss": 0.4}
        callback.on_eval_end(state, unit)  # type: ignore[arg-type]
        assert _count_checkpoints(tmp_path) == 2, "Should NOT save on regression"

    def test_save_best_model_with_save_top_k(self, tmp_path: Path):
        """
        ``save_top_k > 1`` keeps the top-k best checkpoints and removes
        the worst when the buffer exceeds ``save_top_k``.
        """
        callback = TrainCheckpointCallback(
            checkpoint_every_n_steps=None,
            monitor="val/mae",
            mode="min",
            save_top_k=2,
        )
        _setup_callback(callback, tmp_path)

        state = _FakeState()
        unit = _FakeEvalUnit(
            train_progress=_FakeProgress(num_steps_completed=10),
            last_eval_metrics={"val/mae": 0.5},
        )

        # Eval 1: metric=0.5 → saved (first entry)
        callback.on_eval_end(state, unit)  # type: ignore[arg-type]
        assert _count_checkpoints(tmp_path) == 1

        # Eval 2: metric=0.3 → saved (improving, now 2 entries)
        unit.train_progress.num_steps_completed = 20
        unit.last_eval_metrics = {"val/mae": 0.3}
        callback.on_eval_end(state, unit)  # type: ignore[arg-type]
        assert _count_checkpoints(tmp_path) == 2

        # Eval 3: metric=0.2 → saved (improving, should push out worst=0.5)
        unit.train_progress.num_steps_completed = 30
        unit.last_eval_metrics = {"val/mae": 0.2}
        callback.on_eval_end(state, unit)  # type: ignore[arg-type]

        # Still 2 checkpoints because save_top_k=2
        assert _count_checkpoints(tmp_path) == 2

        # The two survivors should be the best two: 0.2 and 0.3
        best_dirs = _best_model_dirs(tmp_path)
        assert len(best_dirs) == 2
        # Metric values embedded in directory names
        dir_names = [d.name for d in best_dirs]
        assert any("0.2000" in n for n in dir_names)
        assert any("0.3000" in n for n in dir_names)
        assert not any("0.5000" in n for n in dir_names)

    def test_mode_max(self, tmp_path: Path):
        """
        ``mode="max"`` saves checkpoints when the metric increases.
        """
        callback = TrainCheckpointCallback(
            checkpoint_every_n_steps=None,
            monitor="val/accuracy",
            mode="max",
            save_top_k=2,
        )
        _setup_callback(callback, tmp_path)

        state = _FakeState()
        unit = _FakeEvalUnit(
            train_progress=_FakeProgress(num_steps_completed=10),
            last_eval_metrics={"val/accuracy": 0.8},
        )

        # First eval — saved (len < save_top_k)
        callback.on_eval_end(state, unit)  # type: ignore[arg-type]
        assert _count_checkpoints(tmp_path) == 1

        # Better — saved
        unit.last_eval_metrics = {"val/accuracy": 0.9}
        callback.on_eval_end(state, unit)  # type: ignore[arg-type]
        assert _count_checkpoints(tmp_path) == 2

        # Worse — NOT saved
        unit.last_eval_metrics = {"val/accuracy": 0.85}
        callback.on_eval_end(state, unit)  # type: ignore[arg-type]
        assert _count_checkpoints(tmp_path) == 2

    def test_no_monitor_does_not_save_best(self, tmp_path: Path):
        """
        When ``monitor`` is None (backward-compatible mode), calling
        ``on_eval_end`` does nothing — no best_model checkpoints.
        """
        callback = TrainCheckpointCallback(
            checkpoint_every_n_steps=None,
            monitor=None,
        )
        _setup_callback(callback, tmp_path)

        state = _FakeState()
        # Periodic saving is also None, so nothing is saved
        unit = _FakeEvalUnit(
            train_progress=_FakeProgress(num_steps_completed=10),
            last_eval_metrics={"val/loss": 0.5},
        )

        callback.on_eval_end(state, unit)  # type: ignore[arg-type]
        assert _count_checkpoints(tmp_path) == 0

    def test_missing_metric_logs_and_skips(self, tmp_path: Path):
        """
        When the monitored metric is not present in ``last_eval_metrics``,
        the callback skips saving (no crash).
        """
        callback = TrainCheckpointCallback(
            checkpoint_every_n_steps=None,
            monitor="val/loss",
            mode="min",
            save_top_k=1,
        )
        _setup_callback(callback, tmp_path)

        state = _FakeState()
        unit = _FakeEvalUnit(
            train_progress=_FakeProgress(num_steps_completed=10),
            last_eval_metrics={"some_other_metric": 0.5},
        )

        # Should not raise, should not save
        callback.on_eval_end(state, unit)  # type: ignore[arg-type]
        assert _count_checkpoints(tmp_path) == 0

    def test_no_metrics_on_unit_skips(self, tmp_path: Path):
        """
        When ``last_eval_metrics`` is None (not yet set), the callback
        skips saving (no crash).
        """
        callback = TrainCheckpointCallback(
            checkpoint_every_n_steps=None,
            monitor="val/loss",
            mode="min",
            save_top_k=1,
        )
        _setup_callback(callback, tmp_path)

        state = _FakeState()
        unit = _FakeEvalUnit(
            train_progress=_FakeProgress(num_steps_completed=10),
            last_eval_metrics=None,
        )

        callback.on_eval_end(state, unit)  # type: ignore[arg-type]
        assert _count_checkpoints(tmp_path) == 0


class TestTrainCheckpointCallbackPeriodic:
    """Periodic checkpoint saving (existing behavior, must remain intact)."""

    def test_periodic_saving(self, tmp_path: Path):
        """
        With ``checkpoint_every_n_steps=5``, checkpoints are saved every
        5 steps.
        """
        callback = TrainCheckpointCallback(
            checkpoint_every_n_steps=5,
            max_saved_checkpoints=10,
            monitor=None,  # best-model disabled
        )
        _setup_callback(callback, tmp_path)

        state = _FakeState()
        unit = _FakeTrainUnit()

        for step in range(1, 21):
            unit.train_progress.num_steps_completed = step
            callback.on_train_step_start(state, unit)  # type: ignore[arg-type]

        # Steps 5, 10, 15, 20 → 4 periodic checkpoints
        assert _count_checkpoints(tmp_path) == 4

    def test_periodic_cleanup_skips_best_model(self, tmp_path: Path):
        """
        When both periodic and best-model saving are active, periodic
        cleanup does NOT remove ``best_model_*`` directories.
        """
        callback = TrainCheckpointCallback(
            checkpoint_every_n_steps=2,
            max_saved_checkpoints=3,
            monitor="val/loss",
            mode="min",
            save_top_k=2,
        )
        save_callback = _make_save_callback()
        _setup_callback(callback, tmp_path, save_callback_override=save_callback)

        state = _FakeState()
        train_unit = _FakeTrainUnit()

        # Pre-create some best_model directories (simulating prior saves)
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for name in ("best_model_step_5_metric_0.5000",
                     "best_model_step_10_metric_0.3000",
                     "best_model_step_15_metric_0.1000"):
            save_callback(str(ckpt_dir / name))

        # Run periodic steps — these should NOT remove best_model_* dirs
        for step in range(1, 13):
            train_unit.train_progress.num_steps_completed = step
            callback.on_train_step_start(state, train_unit)  # type: ignore[arg-type]

        # Verify all best_model dirs still exist
        best_dirs = _best_model_dirs(tmp_path)
        assert len(best_dirs) == 3, (
            "Periodic cleanup should not remove best_model directories"
        )

    def test_periodic_max_saved_checkpoints(self, tmp_path: Path):
        """
        ``max_saved_checkpoints`` limits the number of periodic checkpoints
        kept on disk.
        """
        callback = TrainCheckpointCallback(
            checkpoint_every_n_steps=3,
            max_saved_checkpoints=2,
            monitor=None,
        )
        _setup_callback(callback, tmp_path)

        state = _FakeState()
        unit = _FakeTrainUnit()

        for step in range(1, 31):
            unit.train_progress.num_steps_completed = step
            callback.on_train_step_start(state, unit)  # type: ignore[arg-type]

        # At most 2 periodic checkpoints survive cleanup
        assert _count_checkpoints(tmp_path) <= 2


class TestTrainCheckpointCallbackInit:
    """Initialization edge cases."""

    def test_invalid_mode_raises(self):
        """``mode`` must be 'min' or 'max' when ``monitor`` is set."""
        with pytest.raises(ValueError, match="mode must be 'min' or 'max'"):
            TrainCheckpointCallback(
                checkpoint_every_n_steps=None,
                monitor="val/loss",
                mode="invalid",
            )

    def test_no_monitor_no_periodic_logs_warning(self, caplog):
        """Neither periodic nor best-model enabled — warning, not error."""
        import logging

        caplog.set_level(logging.WARNING)
        TrainCheckpointCallback(
            checkpoint_every_n_steps=None,
            monitor=None,
        )
        assert "Neither 'checkpoint_every_n_steps' nor 'monitor'" in caplog.text

    def test_set_runner_callbacks(self, tmp_path: Path):
        """set_runner_callbacks correctly wires up save/load/checkpoint_dir."""
        callback = TrainCheckpointCallback(
            checkpoint_every_n_steps=None,
            monitor=None,
        )
        save_cb = _make_save_callback()
        load_cb = _make_load_callback()

        callback.set_runner_callbacks(save_cb, load_cb, str(tmp_path / "my_checkpoints"))
        assert callback.save_callback is save_cb
        assert callback.load_callback is load_cb
        assert callback.checkpoint_dir == str(tmp_path / "my_checkpoints")

    def test_uninitialized_callback_raises(self):
        """Calling hooks before set_runner_callbacks raises AssertionError."""
        callback = TrainCheckpointCallback(
            checkpoint_every_n_steps=10,
            monitor=None,
        )
        state = _FakeState()
        unit = _FakeTrainUnit()
        with pytest.raises(AssertionError, match="Must initialize"):
            callback.on_train_step_start(state, unit)  # type: ignore[arg-type]
