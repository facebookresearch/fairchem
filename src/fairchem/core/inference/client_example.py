from __future__ import annotations

import pickle
import socket

from ase import build

from fairchem.core.datasets.atomic_data import AtomicData


def main():
    # Create an AtomicData object
    h2o = build.molecule("H2O")
    atomic_data = AtomicData.from_ase(h2o)
    atomic_data.task_name = ["omol"]

    # Serialize the AtomicData object using pickle
    serialized_data = pickle.dumps(atomic_data)

    # Connect to the inference server
    server_address = ("localhost", 8001)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect(server_address)

        # Send the length of the serialized data
        length = len(serialized_data).to_bytes(4, "big")
        client_socket.sendall(length + serialized_data)

        # Receive the response from the server
        length_data = client_socket.recv(4)
        msg_length = int.from_bytes(length_data, "big")
        response_data = client_socket.recv(msg_length)

        # Deserialize the response data
        response = pickle.loads(response_data)
        print("Response from server:", response)


if __name__ == "__main__":
    main()
