# network.py — TCP message passing for the ProxyFL VANET simulation
import socket
import struct
import threading
import time
from config import MAX_NETWORK_MESSAGE_BYTES
from metrics import metrics_tracker
from wire_codec import decode_message, encode_message
from vanet_channel import WirelessLink, link_capacity_bps


def send_msg(addr, msg, sender_name=None, round_num=None, wireless_link=None):
    """Send a safely encoded message with a 4-byte length prefix.

    Measures TX bytes and communication latency when sender/round are available.
    Returns True on success, False on failure (logged to stderr).
    """
    sender = sender_name or (msg.get("sender") if isinstance(msg, dict) else None)
    r = round_num if round_num is not None else (msg.get("round") if isinstance(msg, dict) else None)

    t0 = time.perf_counter()
    try:
        data = encode_message(msg)
        n_bytes = len(data)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(10)
            s.connect(addr)
            s.sendall(struct.pack('>I', n_bytes) + data)
        duration = time.perf_counter() - t0
        if sender and r is not None and isinstance(r, int) and r >= 0:
            try:
                metrics_tracker.record_bytes(sender, r, "tx", n_bytes + 4)
                metrics_tracker.record_duration(
                    sender, r, "communication_tx", duration)
                if wireless_link is not None:
                    if not isinstance(wireless_link, WirelessLink):
                        raise TypeError(
                            "wireless_link must be a WirelessLink")
                    metrics_tracker.record_wireless_delivery(
                        sender, r, n_bytes + 4,
                        link_capacity_bps(wireless_link.distance_m),
                    )
            except Exception as measurement_error:
                print(f"[NET] Measurement failed after delivery: "
                      f"{measurement_error}")
        return True
    except ConnectionRefusedError:
        print(f"[NET] Connection refused to {addr[1]} "
              f"(target may be offline)")
        return False
    except Exception as e:
        print(f"[NET] Send failed to {addr[1]}: {e}")
        return False


class Receiver:
    """TCP server that deserializes incoming messages and dispatches them
    to a callback function.
    """

    def __init__(self, port, callback, node_name=None,
                 max_message_bytes=MAX_NETWORK_MESSAGE_BYTES):
        self.port = port
        self.callback = callback
        self.node_name = node_name
        self.max_message_bytes = max_message_bytes
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", self.port))
        self.sock.listen()

    def start(self):
        threading.Thread(target=self._listen, daemon=True).start()

    def shutdown(self):
        """Close the listening socket so the port is freed."""
        try:
            self.sock.close()
        except Exception:
            pass

    def _listen(self):
        while True:
            try:
                conn, _ = self.sock.accept()
                threading.Thread(
                    target=self._handle, args=(conn,), daemon=True).start()
            except OSError:
                break

    def _handle(self, conn):
        t0 = time.perf_counter()
        try:
            raw_msglen = self._recvall(conn, 4)
            if not raw_msglen:
                return
            msglen = struct.unpack('>I', raw_msglen)[0]
            if msglen <= 0 or msglen > self.max_message_bytes:
                raise ValueError(f"invalid frame length: {msglen}")
            data = self._recvall(conn, msglen)
            duration = time.perf_counter() - t0
            if data:
                msg = decode_message(data)
                if self.node_name and isinstance(msg, dict):
                    r = msg.get("round")
                    if isinstance(r, int) and r >= 0:
                        metrics_tracker.record_bytes(self.node_name, r, "rx", msglen + 4)
                        metrics_tracker.record_duration(self.node_name, r, "communication_rx", duration)
                self.callback(msg)
        except Exception as e:
            print(f"[NET] Receive error on port {self.port}: {e}")
        finally:
            conn.close()

    def _recvall(self, conn, n):
        data = bytearray()
        while len(data) < n:
            packet = conn.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data
