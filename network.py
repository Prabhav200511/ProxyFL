"""Length-prefixed, JSON-only transport for the ProxyFL simulation.

Messages previously used ``pickle.loads`` directly on network bytes.  That is
unsafe because arbitrary code can run before the certificate-less verifier gets
a chance to reject a forged message.  The outer envelope is now a constrained
JSON structure; model bytes remain opaque until their AES-GCM tag and
signature are accepted by the receiving RSU/server.
"""

from __future__ import annotations

import base64
import json
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from config import MAX_MESSAGE_BYTES
from metrics import metrics_tracker


_BYTES_MARKER = "__proxyfl_wire_type__"


def _to_wire(value: Any) -> Any:
    if isinstance(value, bytes):
        return {_BYTES_MARKER: "bytes", "base64": base64.b64encode(value).decode("ascii")}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_wire(item) for item in value]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("network message dictionaries require string keys")
        return {key: _to_wire(item) for key, item in value.items()}
    raise TypeError(f"unsupported network message value: {type(value)!r}")


def _from_wire(value: Any) -> Any:
    if isinstance(value, list):
        return [_from_wire(item) for item in value]
    if isinstance(value, dict):
        if value.get(_BYTES_MARKER) == "bytes":
            if set(value) != {_BYTES_MARKER, "base64"} or not isinstance(value["base64"], str):
                raise ValueError("malformed bytes value")
            return base64.b64decode(value["base64"], validate=True)
        return {key: _from_wire(item) for key, item in value.items()}
    return value


def encode_message(msg: Mapping[str, Any]) -> bytes:
    if not isinstance(msg, Mapping):
        raise TypeError("network message must be a mapping")
    return json.dumps(_to_wire(msg), sort_keys=True, separators=(",", ":")).encode("utf-8")


def decode_message(data: bytes) -> dict:
    try:
        decoded = _from_wire(json.loads(data.decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise ValueError("malformed JSON network message") from exc
    if not isinstance(decoded, dict) or not isinstance(decoded.get("type"), str):
        raise ValueError("network message is not a typed object")
    return decoded


@dataclass(frozen=True)
class SendResult:
    success: bool
    payload_bytes: int
    send_time_ms: float

    def __bool__(self) -> bool:
        return self.success


def send_msg(addr, msg: Mapping[str, Any], metric_node: str | None = None, round_num: int | None = None) -> SendResult:
    """Send one safe envelope and return its actual byte overhead and latency."""
    started_at = time.perf_counter()
    try:
        data = encode_message(msg)
        if len(data) > MAX_MESSAGE_BYTES:
            raise ValueError(f"message exceeds {MAX_MESSAGE_BYTES} byte limit")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(10)
            sock.connect(addr)
            sock.sendall(struct.pack(">I", len(data)) + data)
        elapsed = (time.perf_counter() - started_at) * 1000.0
        metric_round = msg.get("round", round_num)
        if metric_node is not None and isinstance(metric_round, int):
            metrics_tracker.record_bytes(metric_node, metric_round, "tx", len(data))
            metrics_tracker.record_duration(metric_node, metric_round, "communication_tx", elapsed / 1000.0)
        return SendResult(True, len(data), elapsed)
    except ConnectionRefusedError:
        elapsed = (time.perf_counter() - started_at) * 1000.0
        print(f"[NET] Connection refused to {addr[1]} (target may be offline)")
        return SendResult(False, 0, elapsed)
    except Exception as exc:
        elapsed = (time.perf_counter() - started_at) * 1000.0
        print(f"[NET] Send failed to {addr[1]}: {exc}")
        return SendResult(False, 0, elapsed)


class Receiver:
    """TCP listener that only dispatches a validated JSON envelope."""

    def __init__(self, port: int, callback: Callable[[dict], None], metric_node: str | None = None):
        self.port = port
        self.callback = callback
        self.metric_node = metric_node
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", self.port))
        self.sock.listen()

    def start(self) -> None:
        threading.Thread(target=self._listen, daemon=True).start()

    def shutdown(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

    def _listen(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
                threading.Thread(target=self._handle, args=(conn,), daemon=True).start()
            except OSError:
                break

    def _handle(self, conn: socket.socket) -> None:
        started_at = time.perf_counter()
        try:
            with conn:
                raw_msglen = self._recvall(conn, 4)
                if raw_msglen is None:
                    return
                msglen = struct.unpack(">I", raw_msglen)[0]
                if msglen <= 0 or msglen > MAX_MESSAGE_BYTES:
                    raise ValueError("invalid network message length")
                data = self._recvall(conn, msglen)
                if data is None:
                    return
                msg = decode_message(data)
            elapsed = time.perf_counter() - started_at
            round_num = msg.get("round")
            if self.metric_node is not None and isinstance(round_num, int):
                metrics_tracker.record_bytes(self.metric_node, round_num, "rx", len(data))
                metrics_tracker.record_duration(self.metric_node, round_num, "communication_rx", elapsed)
            self.callback(msg)
        except Exception as exc:
            print(f"[NET] Receive error on port {self.port}: {exc}")

    @staticmethod
    def _recvall(conn: socket.socket, size: int) -> bytes | None:
        data = bytearray()
        while len(data) < size:
            packet = conn.recv(size - len(data))
            if not packet:
                return None
            data.extend(packet)
        return bytes(data)
