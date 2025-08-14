"""
Socket utility functions for robust data transmission
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import socket

SOCKET_TIMEOUT = 120


def recv_all(sock: socket.socket, length: int) -> bytes:
    """
    Receive exactly 'length' bytes from socket, handling partial receives.

    Args:
        sock: Socket to receive from
        length: Number of bytes to receive

    Returns:
        bytes: Received data

    Raises:
        ConnectionError: If connection is closed before all data is received
    """
    data = b""
    remaining = length
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Connection closed while receiving data")
        data += chunk
        remaining -= len(chunk)
    return data


def recv_message(sock: socket.socket) -> bytes:
    """
    Receive a length-prefixed message from socket.
    First 4 bytes contain the message length in big-endian format.

    Args:
        sock: Socket to receive from

    Returns:
        bytes: Received message data

    Raises:
        ConnectionError: If connection is closed
        ValueError: If received data length doesn't match expected length
    """
    # Read message length (4 bytes)
    length_data = recv_all(sock, 4)
    if not length_data:
        raise ConnectionError("Connection closed while reading message length")

    msg_length = int.from_bytes(length_data, "big")

    # Read message data
    data = recv_all(sock, msg_length)

    if len(data) != msg_length:
        raise ValueError(
            f"Received data length {len(data)} does not match expected length {msg_length}"
        )

    return data


def send_message(sock: socket.socket, data: bytes) -> None:
    """
    Send a length-prefixed message to socket.
    First 4 bytes contain the message length in big-endian format.

    Args:
        sock: Socket to send to
        data: Data to send
    """
    length = len(data).to_bytes(4, "big")
    sock.sendall(length + data)
