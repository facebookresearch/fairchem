"""
Test RayCluster functionality by mocking out submitit to avoid actual SLURM job submission.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest
import submitit

from fairchem.core.launchers.cluster.ray_cluster import (
    CheckpointableRayJob,
    HeadInfo,
    RayCluster,
    RayClusterState,
    mk_symlinks,
)


@dataclass
class MockJobPaths:
    """Mock submitit job paths"""

    stderr: Path
    stdout: Path


class MockJob:
    """Mock submitit job for testing"""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.paths = MockJobPaths(
            stderr=Path(f"/tmp/mock_{job_id}.err"),
            stdout=Path(f"/tmp/mock_{job_id}.out"),
        )

    def result(self):
        return "mock_result"

    def state(self):
        return "COMPLETED"


class MockAutoExecutor:
    """Mock submitit AutoExecutor for testing"""

    def __init__(self, folder: str, cluster: str):
        self.folder = folder
        self.cluster = cluster
        self.parameters = {}
        self._job_counter = 0

    def update_parameters(self, **kwargs):
        self.parameters.update(kwargs)

    def submit(self, callable_func, *args, **kwargs):
        self._job_counter += 1
        job_id = f"mock_job_{self._job_counter}"
        return MockJob(job_id)

    def batch(self):
        return self  # Return self for context manager

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TestRayClusterState:
    """Test RayClusterState functionality"""

    def test_init_with_defaults(self):
        state = RayClusterState()
        assert state.cluster_id is not None
        assert len(state.cluster_id) == 32  # UUID hex string length
        assert state.rendezvous_dir.name == state.cluster_id

    def test_init_with_custom_params(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rdv_dir = Path(temp_dir)
            cluster_id = "test_cluster_123"

            state = RayClusterState(rdv_dir=rdv_dir, cluster_id=cluster_id)
            assert state.cluster_id == cluster_id
            assert state.rendezvous_dir == rdv_dir / cluster_id

    def test_head_info_operations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = RayClusterState(rdv_dir=Path(temp_dir))

            # Initially no head info
            assert not state.is_head_ready()
            assert state.head_info() is None

            # Save head info
            head_info = HeadInfo(hostname="test_host", port=8080, temp_dir="/tmp/ray")
            state.save_head_info(head_info)

            # Check head info is available
            assert state.is_head_ready()
            retrieved_info = state.head_info()
            assert retrieved_info is not None
            assert retrieved_info.hostname == "test_host"
            assert retrieved_info.port == 8080
            assert retrieved_info.temp_dir == "/tmp/ray"

    def test_job_management(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = RayClusterState(rdv_dir=Path(temp_dir))

            # Add mock jobs
            job1 = MockJob("job_123")
            job2 = MockJob("job_456")

            state.add_job(job1)
            state.add_job(job2)

            # Check job IDs are listed
            job_ids = state.list_job_ids()
            assert "job_123" in job_ids
            assert "job_456" in job_ids

    def test_reset_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = RayClusterState(rdv_dir=Path(temp_dir))

            # Create head info
            head_info = HeadInfo(hostname="test", port=8080)
            state.save_head_info(head_info)
            assert state.is_head_ready()

            # Reset state
            state.reset_state()
            assert not state.is_head_ready()


class TestRayCluster:
    """Test RayCluster functionality with mocked submitit"""

    def setup_method(self, method):
        """Reset mock state before each test"""
        MockAutoExecutor._job_counter = 0

    @patch(
        "fairchem.core.launchers.cluster.ray_cluster.submitit.AutoExecutor",
        MockAutoExecutor,
    )
    @patch("fairchem.core.launchers.cluster.ray_cluster.mk_symlinks")
    def test_start_head_and_workers(self, mock_mk_symlinks):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            # Use temp_dir as rendezvous dir to isolate tests
            cluster = RayCluster(log_dir=log_dir, rdv_dir=Path(temp_dir) / "rdv")

            # Test payload function
            def test_payload(output_path: str, workers: int):
                return f"payload_result_{workers}"

            # Start head and workers
            requirements = {
                "nodes": 2,
                "gpus_per_task": 8,
                "cpus_per_task": 192,
                "timeout_min": 60,
            }

            job_id = cluster.start_head_and_workers(
                requirements=requirements,
                name="test_cluster",
                payload=test_payload,
                output_path=str(log_dir),
                workers=16,
            )

            # Check that job was created
            assert job_id == "mock_job_1"
            assert cluster.head_started
            assert len(cluster.jobs) == 1
            assert len(cluster.state.list_job_ids()) == 1

            # Check symlinks were created
            mock_mk_symlinks.assert_called_once()

    @patch(
        "fairchem.core.launchers.cluster.ray_cluster.submitit.AutoExecutor",
        MockAutoExecutor,
    )
    @patch("fairchem.core.launchers.cluster.ray_cluster.mk_symlinks")
    def test_start_head_then_workers(self, mock_mk_symlinks):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            # Use temp_dir as rendezvous dir to isolate tests
            cluster = RayCluster(log_dir=log_dir, rdv_dir=Path(temp_dir) / "rdv")

            # Start head
            head_requirements = {
                "nodes": 1,
                "gpus_per_task": 8,
                "cpus_per_task": 192,
            }

            def test_head_payload(**kwargs):
                return "head_result"

            head_job_id = cluster.start_head(
                requirements=head_requirements,
                name="test_head",
                payload=test_head_payload,
            )

            assert head_job_id == "mock_job_1"
            assert cluster.head_started

            # Start workers
            worker_requirements = {
                "nodes": 2,
                "gpus_per_task": 8,
                "cpus_per_task": 192,
            }

            worker_job_ids = cluster.start_workers(
                num_workers=4, requirements=worker_requirements, name="test_workers"
            )

            assert len(worker_job_ids) == 4
            assert all(job_id.startswith("mock_job_") for job_id in worker_job_ids)
            assert len(cluster.jobs) == 5  # 1 head + 4 workers

            # Check symlinks were created for all jobs
            assert mock_mk_symlinks.call_count == 5

    @patch("fairchem.core.launchers.cluster.ray_cluster.scancel")
    @patch("fairchem.core.launchers.cluster.ray_cluster.kill_proc_tree")
    def test_shutdown(self, mock_kill_proc, mock_scancel):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"

            # Use a specific cluster_id so we can check directory cleanup
            cluster = RayCluster(
                log_dir=log_dir,
                rdv_dir=Path(temp_dir) / "rdv",
                cluster_id="test_cluster_shutdown",
            )

            # Add some mock jobs to state
            cluster.state.add_job(MockJob("job_1"))
            cluster.state.add_job(MockJob("job_2"))

            # Verify rendezvous dir exists
            assert cluster.state.rendezvous_dir.exists()

            # Shutdown cluster
            cluster.shutdown()

            # Check that scancel was called with job IDs
            mock_scancel.assert_called_once()
            called_job_ids = mock_scancel.call_args[0][0]
            assert "job_1" in called_job_ids
            assert "job_2" in called_job_ids

            # Check that kill_proc_tree was called
            mock_kill_proc.assert_called_once()

            # Check that rendezvous dir was cleaned up
            assert not cluster.state.rendezvous_dir.exists()
            assert cluster.is_shutdown

    def test_context_manager(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"

            with patch("fairchem.core.launchers.cluster.ray_cluster.scancel"), patch(
                "fairchem.core.launchers.cluster.ray_cluster.kill_proc_tree"
            ):
                with RayCluster(log_dir=log_dir, cluster_id="test_ctx") as cluster:
                    assert not cluster.is_shutdown

                # After context exit, cluster should be shutdown
                assert cluster.is_shutdown


class TestCheckpointableRayJob:
    """Test CheckpointableRayJob functionality"""

    def test_init(self):
        state = RayClusterState(cluster_id="test")

        def test_payload(**kwargs):
            return "test_result"

        job = CheckpointableRayJob(
            cluster_state=state,
            worker_wait_timeout_seconds=60,
            payload=test_payload,
            test_arg="test_value",
        )

        assert job.cluster_state == state
        assert job.worker_wait_timeout_seconds == 60
        assert job.payload == test_payload
        assert job.kwargs["test_arg"] == "test_value"

    @patch("fairchem.core.launchers.cluster.ray_cluster.os_environ_get_or_throw")
    @patch("fairchem.core.launchers.cluster.ray_cluster._ray_head_script")
    @patch("fairchem.core.launchers.cluster.ray_cluster.worker_script")
    def test_call_head_node(self, mock_worker, mock_head, mock_env):
        """Test job execution on head node (SLURM_NODEID=0)"""
        mock_env.return_value = "0"  # Head node

        state = RayClusterState(cluster_id="test")

        def test_payload():
            return "test"

        job = CheckpointableRayJob(
            cluster_state=state,
            worker_wait_timeout_seconds=60,
            payload=test_payload,
            test_kwarg="value",
        )

        job()

        # Head script should be called, not worker script
        mock_head.assert_called_once_with(
            cluster_state=state,
            worker_wait_timeout_seconds=60,
            payload=job.payload,
            test_kwarg="value",
        )
        mock_worker.assert_not_called()

    @patch("fairchem.core.launchers.cluster.ray_cluster.os_environ_get_or_throw")
    @patch("fairchem.core.launchers.cluster.ray_cluster._ray_head_script")
    @patch("fairchem.core.launchers.cluster.ray_cluster.worker_script")
    def test_call_worker_node(self, mock_worker, mock_head, mock_env):
        """Test job execution on worker node (SLURM_NODEID>0)"""
        mock_env.return_value = "1"  # Worker node

        state = RayClusterState(cluster_id="test")

        def test_payload():
            return "test"

        job = CheckpointableRayJob(
            cluster_state=state, worker_wait_timeout_seconds=60, payload=test_payload
        )

        job()

        # Worker script should be called, not head script
        mock_worker.assert_called_once_with(
            cluster_state=state, worker_wait_timeout_seconds=60
        )
        mock_head.assert_not_called()

    def test_checkpoint(self):
        """Test checkpointing functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            state = RayClusterState(rdv_dir=Path(temp_dir), cluster_id="test")

            # Create head info to test reset
            head_info = HeadInfo(hostname="test", port=8080)
            state.save_head_info(head_info)
            assert state.is_head_ready()

            def test_payload():
                return "test"

            job = CheckpointableRayJob(
                cluster_state=state,
                worker_wait_timeout_seconds=60,
                payload=test_payload,
                test_arg="value",
            )

            # Trigger checkpoint
            delayed_submission = job.checkpoint()

            # State should be reset
            assert not state.is_head_ready()

            # Should return DelayedSubmission with new job
            assert isinstance(delayed_submission, submitit.core.utils.DelayedSubmission)


class TestUtilityFunctions:
    """Test utility functions"""

    def test_mk_symlinks(self):
        """Test symlink creation function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_dir = Path(temp_dir) / "target"
            target_dir.mkdir()

            # Create mock stderr/stdout files
            stderr_file = Path(temp_dir) / "job.err"
            stdout_file = Path(temp_dir) / "job.out"
            stderr_file.write_text("error content")
            stdout_file.write_text("output content")

            mock_paths = MockJobPaths(stderr=stderr_file, stdout=stdout_file)

            # Create symlinks
            mk_symlinks(target_dir, "test_job", mock_paths)

            # Check symlinks were created
            err_link = target_dir / "test_job.err"
            out_link = target_dir / "test_job.out"

            assert err_link.is_symlink()
            assert out_link.is_symlink()
            assert err_link.resolve() == stderr_file
            assert out_link.resolve() == stdout_file


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
