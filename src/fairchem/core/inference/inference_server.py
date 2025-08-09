"""
Sequential request server with parallel model execution
Usage: python server.py --workers 4 --port 8000
"""

from __future__ import annotations

import logging
import os
import pickle
import socket
import time
import traceback
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Protocol

import hydra
import torch.multiprocessing as mp

if TYPE_CHECKING:
    from omegaconf import DictConfig
from torch.distributed.elastic.utils.distributed import get_free_port

from fairchem.core.common import distutils, gp_utils
from fairchem.core.common.distutils import (
    get_device_for_local_rank,
    setup_env_local_multi_gpu,
)
from fairchem.core.common.utils import detach_dict_tensors

logging.basicConfig(level=logging.INFO)


def worker_process(
    worker_id,
    input_shm_name,
    output_shm_name,
    input_size_shm_name,
    output_size_shm_name,
    input_ready_event,
    output_ready_event,
    shutdown_event,
    ready_queue,
    predict_config,
    master_port,
    world_size,
):
    """Worker process - waits for input data and processes it"""
    setup_env_local_multi_gpu(worker_id, master_port)
    device = predict_config.get("device")
    backend = "gloo" if device == "cpu" else "nccl"
    dist_config = {
        "distributed_backend": backend,
        "world_size": world_size,
        "cpu": device == "cpu",
        "submit": False,
    }
    distutils.setup(dist_config)
    gp_utils.setup_graph_parallel_groups(world_size, backend)
    predict_unit = hydra.utils.instantiate(predict_config)
    logging.debug(
        f"Worker {worker_id} loaded predict unit: {predict_unit}, "
        f"on port {master_port}, with device: {get_device_for_local_rank()}, config: {predict_config}"
    )
    ready_queue.put(worker_id)

    # Connect to shared memory
    input_shm = shared_memory.SharedMemory(name=input_shm_name)
    output_shm = shared_memory.SharedMemory(name=output_shm_name)
    input_size_shm = shared_memory.SharedMemory(name=input_size_shm_name)
    output_size_shm = shared_memory.SharedMemory(name=output_size_shm_name)

    try:
        while True:
            # Wait for input data signal
            logging.debug(f"Worker {worker_id} waiting for input")
            if shutdown_event.is_set():
                break

            input_ready_event.wait()
            if shutdown_event.is_set():
                break

            try:
                # Read input size
                input_size = int.from_bytes(input_size_shm.buf[:4], "big")

                # Read and deserialize input data
                request_data = bytes(input_shm.buf[:input_size])
                atomic_data_input = pickle.loads(request_data)
                logging.debug(f"Worker {worker_id} received input: {atomic_data_input}")

                # Run prediction
                result = predict_unit.predict_step(None, atomic_data_input)
                result = detach_dict_tensors(result)
                logging.debug(f"Worker {worker_id} predicted result: {result}")

                # Serialize result
                response_data = {
                    "worker_id": worker_id,
                    "result": result,
                    "error": None,
                }
                pickled_result = pickle.dumps(response_data)

                # Write result size and data
                output_size_shm.buf[:4] = len(pickled_result).to_bytes(4, "big")
                output_shm.buf[: len(pickled_result)] = pickled_result

            except Exception as e:
                logging.error(f"Worker {worker_id} encountered error: {e}")
                traceback.print_exc()

                # Write error response
                error_response = {
                    "worker_id": worker_id,
                    "result": None,
                    "error": str(e),
                }
                pickled_error = pickle.dumps(error_response)
                output_size_shm.buf[:4] = len(pickled_error).to_bytes(4, "big")
                output_shm.buf[: len(pickled_error)] = pickled_error

            # Signal that output is ready
            output_ready_event.set()

    finally:
        distutils.cleanup()
        input_shm.close()
        output_shm.close()
        input_size_shm.close()
        output_size_shm.close()


class InferenceServerProtocol(Protocol):
    def run(self) -> None: ...

    def ready(self) -> bool: ...

    def shutdown(self) -> None: ...


class MLIPInferenceServerMP(InferenceServerProtocol):
    """Inference server using python multiprocessing for parallel model execution, not designed for multi-node use"""

    def __init__(
        self,
        num_workers: int,
        port: int,
        predict_config: DictConfig,
        start_method: str = "spawn",
        max_data_size: int = 100 * 1024 * 1024,  # 100MB default
    ):
        # note this is a global setting
        mp.set_start_method(start_method)
        self.num_workers = num_workers
        self.port = port
        self.predict_config = predict_config
        self.max_data_size = max_data_size

        # Create shared memory segments for each worker
        self.server_socket = None
        self.input_shms = []
        self.output_shms = []
        self.input_size_shms = []
        self.output_size_shms = []
        self.input_ready_events = []
        self.output_ready_events = []
        self.shutdown_event = mp.Event()
        self.ready_queue = mp.Queue()
        self.workers = []

        # Create shared memory segments
        for _ in range(num_workers):
            # Input data shared memory
            input_shm = shared_memory.SharedMemory(create=True, size=max_data_size)
            self.input_shms.append(input_shm)

            # Output data shared memory
            output_shm = shared_memory.SharedMemory(create=True, size=max_data_size)
            self.output_shms.append(output_shm)

            # Size information shared memory (4 bytes each)
            input_size_shm = shared_memory.SharedMemory(create=True, size=4)
            self.input_size_shms.append(input_size_shm)

            output_size_shm = shared_memory.SharedMemory(create=True, size=4)
            self.output_size_shms.append(output_size_shm)

            # Synchronization events
            self.input_ready_events.append(mp.Event())
            self.output_ready_events.append(mp.Event())

    def start_workers(self):
        """Start all worker processes"""
        port = get_free_port()
        for i in range(self.num_workers):
            worker = mp.Process(
                target=worker_process,
                args=(
                    i,
                    self.input_shms[i].name,
                    self.output_shms[i].name,
                    self.input_size_shms[i].name,
                    self.output_size_shms[i].name,
                    self.input_ready_events[i],
                    self.output_ready_events[i],
                    self.shutdown_event,
                    self.ready_queue,
                    self.predict_config,
                    port,
                    self.num_workers,
                ),
            )
            worker.start()
            self.workers.append(worker)
        logging.info(f"MLIP Inference Server: Started {self.num_workers} workers")

    def process_request(self, request_data):
        """Send request to all workers and collect results"""
        start_time = time.time()

        # Clear all output ready events
        for event in self.output_ready_events:
            event.clear()

        # Write input data to all workers' shared memory
        for i in range(self.num_workers):
            if len(request_data) > self.max_data_size:
                raise ValueError(
                    f"Request data size {len(request_data)} exceeds maximum {self.max_data_size}"
                )

            # Write size and data
            self.input_size_shms[i].buf[:4] = len(request_data).to_bytes(4, "big")
            self.input_shms[i].buf[: len(request_data)] = request_data

            # Signal input is ready
            self.input_ready_events[i].set()

        # Wait for and collect results from all workers
        results = []
        for i in range(self.num_workers):
            self.output_ready_events[i].wait()

            # Read result size and data
            output_size = int.from_bytes(self.output_size_shms[i].buf[:4], "big")
            result_data = bytes(self.output_shms[i].buf[:output_size])
            result = pickle.loads(result_data)
            results.append(result)

            # Clear events for next request
            self.input_ready_events[i].clear()
            self.output_ready_events[i].clear()

        inference_time = (time.time() - start_time) * 1000

        # All responses should not error
        if all(r["error"] is None for r in results):
            response_data = {
                "predictions": results[0]["result"],
                "error": None,
                "inference_time": inference_time,
            }
        else:
            # Handle errors
            response_data = {
                "predictions": [],
                "error": [result.get("error", "") for result in results],
                "inference_time": inference_time,
            }

        return response_data

    def handle_client(self, client_socket, addr):
        """Handle single client connection - process requests sequentially"""
        logging.info(f"Connection from {addr}")
        try:
            while True:
                # Read request length
                length_data = client_socket.recv(4)
                if not length_data:
                    break

                msg_length = int.from_bytes(length_data, "big")

                # Read request data
                data = client_socket.recv(msg_length, socket.MSG_WAITALL)
                if not data or len(data) != msg_length:
                    break

                # Process request (blocks until all workers complete)
                # The requested data is in a pickled AtomicData for now
                response_data = self.process_request(data)
                pickled_response = pickle.dumps(response_data)

                # Send response back in pickled format as well
                length = len(pickled_response).to_bytes(4, "big")
                client_socket.sendall(length + pickled_response)

        except Exception as e:
            logging.error(f"Client {addr} encountered error: {e}")
            traceback.print_exc()
        finally:
            client_socket.close()
            logging.info(f"Client {addr} disconnected")

    def ready(self):
        """Check if all workers are ready"""
        ready_workers = []
        while len(ready_workers) < self.num_workers:
            try:
                worker_id = self.ready_queue.get(timeout=1)
                ready_workers.append(worker_id)
            except Exception:
                break
        return len(ready_workers) == self.num_workers

    def run(self):
        """Start server - handles one request at a time"""
        # Start worker processes
        self.start_workers()

        # Start server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(("localhost", self.port))
        self.server_socket.listen(1)  # Only accept 1 connection at a time

        logging.info(f"MLIP Inference Server: Listening on port {self.port}")

        try:
            while True:
                # Accept one client at a time
                client_socket, addr = self.server_socket.accept()
                # Handle this client completely before accepting next one
                self.handle_client(client_socket, addr)
        except Exception:
            logging.info("Server shutting down")
        finally:
            self.shutdown()

    def shutdown(self):
        """Shutdown the server and all workers"""
        self.shutdown_event.set()

        # Signal all workers to wake up and check shutdown event
        for event in self.input_ready_events:
            event.set()

        # Ensure all workers are terminated
        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                logging.warning(f"Worker {worker.pid} still alive, terminating...")
                worker.terminate()
                worker.join()

        # Clean up shared memory
        for shm in (
            self.input_shms
            + self.output_shms
            + self.input_size_shms
            + self.output_size_shms
        ):
            try:
                shm.close()
                # Check if shared memory still exists before unlinking
                shm_path = f"/dev/shm/{shm.name}"
                if os.path.exists(shm_path):
                    shm.unlink()
            except FileNotFoundError:
                # Shared memory already cleaned up, this is fine
                pass
            except Exception as e:
                logging.warning(f"Error cleaning up shared memory {shm.name}: {e}")

        if self.server_socket:
            self.server_socket.close()


@hydra.main(
    version_base=None,
    config_path=".",
    config_name="server_config",
)
def main(cfg: DictConfig):
    # for single-node use, we can just use a backend of python multiprocessing
    server: InferenceServerProtocol = MLIPInferenceServerMP(
        num_workers=cfg.server.workers,
        port=cfg.server.port,
        predict_config=cfg.predict_unit,
    )
    server.run()
    # otherwise use ray for multi-node inference ... To be implemented


if __name__ == "__main__":
    main()
