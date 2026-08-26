import json
import socket
import struct
import threading
import unittest

from network import Receiver
from wire_codec import WireCodecError, decode_message, encode_message


class WireCodecTests(unittest.TestCase):
    def test_round_trip_preserves_nested_bytes(self):
        message = {
            "type": "LOCAL_UPDATE",
            "sender": "C0_V1",
            "round": 3,
            "ciphertext": b"\x00\x01payload",
            "pk": {"aid": {"token": b"token"}},
        }
        self.assertEqual(decode_message(encode_message(message)), message)

    def test_rejects_unsupported_tensor_like_value(self):
        with self.assertRaises(WireCodecError):
            encode_message({"type": "BAD", "weights": object()})

    def test_rejects_malformed_bytes_tag(self):
        malformed = json.dumps({
            "type": "BAD",
            "payload": {"__proxyfl_type__": "bytes", "base64": "%%%"},
        }).encode("utf-8")
        with self.assertRaises(WireCodecError):
            decode_message(malformed)

    def test_receiver_rejects_oversized_frame_before_callback(self):
        received = []
        receiver = Receiver(0, received.append, max_message_bytes=32)
        port = receiver.sock.getsockname()[1]
        receiver.start()
        with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
            client.sendall(struct.pack(">I", 33))
        threading.Event().wait(0.1)
        receiver.shutdown()
        self.assertEqual(received, [])


if __name__ == "__main__":
    unittest.main()
