"""Ensure failed authentication cannot enter the RSU aggregation buffer."""

from __future__ import annotations

import unittest
import threading
from unittest.mock import patch

import torch

from crypto_protocol import Authority, build_envelope, encrypt_payload, message_aad
from rsu import RSU
from network import Receiver
from metrics import metrics_tracker
from model_codec import serialize_weights


class RSUSecurityTests(unittest.TestCase):
    def test_duplicate_vehicle_report_is_buffered_once(self):
        rsu = RSU(
            "RSU_0_Central", 0, [6000, 6001], 8000,
            vehicle_names=["C0_V1", "C0_V2"], security_enabled=False,
        )
        message = {
            "type": "LOCAL_UPDATE",
            "sender": "C0_V1",
            "recipient": "RSU_0_Central",
            "round": 2,
            "weights": serialize_weights({"weight": torch.tensor(1.0)}),
        }
        try:
            rsu.on_receive(message)
            rsu.on_receive(message)
            self.assertEqual(len(rsu.round_buffers[2]), 1)
            self.assertEqual(rsu.round_reported[2], {"C0_V1"})
        finally:
            with rsu._lock:
                rsu._cancel_timer_locked(2)
            rsu.shutdown()

    def test_empty_cluster_reports_no_cluster_update_to_server(self):
        received = []
        received_event = threading.Event()

        def capture(message):
            received.append(message)
            received_event.set()

        server_receiver = Receiver(0, capture, node_name="ServerTest")
        server_port = server_receiver.sock.getsockname()[1]
        server_receiver.start()
        rsu = RSU(
            "RSU_0_Central", 0, [6000], server_port,
            vehicle_names=["C0_V1"], security_enabled=False,
        )
        try:
            rsu.on_receive({
                "type": "NO_UPDATE",
                "sender": "C0_V1",
                "recipient": "RSU_0_Central",
                "round": 4,
            })
            self.assertTrue(received_event.wait(2.0))
            self.assertEqual(received[0]["type"], "NO_CLUSTER_UPDATE")
            self.assertEqual(received[0]["sender"], "RSU_0_Central")
            self.assertEqual(received[0]["round"], 4)
        finally:
            rsu.shutdown()
            server_receiver.shutdown()

    def test_explicit_no_security_mode_accepts_plain_update(self):
        authority = Authority()
        authority.enroll_mvd("RSU_0_Central")
        rsu_identity = authority.register("RSU_0_Central")
        rsu = RSU(
            "RSU_0_Central", 0, [6000, 6001], 8000,
            vehicle_names=["C0_V1", "C0_V2"],
            security_authority=authority,
            security_identity=rsu_identity,
            security_enabled=False,
        )
        try:
            rsu.on_receive({
                "type": "LOCAL_UPDATE",
                "sender": "C0_V1",
                "recipient": "RSU_0_Central",
                "round": 1,
                "weights": serialize_weights({"weight": torch.tensor(1.0)}),
            })
            self.assertEqual(len(rsu.round_buffers[1]), 1)
        finally:
            with rsu._lock:
                rsu._cancel_timer_locked(1)
            rsu.shutdown()

    def test_corrupt_update_never_reaches_average_weights(self) -> None:
        authority = Authority()
        authority.enroll_mvd("C1_V1")
        authority.enroll_mvd("Cluster_1")
        authority.enroll_mvd("Server")
        device = authority.register("C1_V1")
        rsu_identity = authority.register("Cluster_1")
        authority.register("Server")
        rsu = RSU(
            "Cluster_1", 0, [6000], 8000, vehicle_names=["C1_V1"],
            security_authority=authority, security_identity=rsu_identity,
        )
        try:
            payload = b"tensor-only state would be here"
            aad = message_aad("LOCAL_UPDATE", "C1_V1", "Cluster_1", 1)
            signature = device.sign(payload)
            ciphertext, nonce, tag = encrypt_payload(
                device.shared_secret_for(rsu_identity.get_public_info()), payload, aad)
            corrupt = build_envelope(
                "LOCAL_UPDATE", device, "Cluster_1", 1, signature,
                bytes([ciphertext[0] ^ 1]) + ciphertext[1:], nonce, tag,
            )
            with patch("rsu.average_weights") as average_weights:
                rsu.on_receive(corrupt)
                average_weights.assert_not_called()
            self.assertNotIn(1, rsu.round_buffers)
        finally:
            rsu.shutdown()

    def test_authenticated_no_update_completes_an_empty_cluster(self) -> None:
        metrics_tracker.reset()
        authority = Authority()
        authority.enroll_mvd("C1_V1")
        authority.enroll_mvd("Cluster_1")
        authority.enroll_mvd("Server")
        device = authority.register("C1_V1")
        rsu_identity = authority.register("Cluster_1")
        authority.register("Server")
        rsu = RSU(
            "Cluster_1", 0, [6000], 1, vehicle_names=["C1_V1"],
            security_authority=authority, security_identity=rsu_identity,
        )
        try:
            payload = b"NO_UPDATE"
            aad = message_aad("NO_UPDATE", "C1_V1", "Cluster_1", 1)
            signature = device.sign(payload)
            ciphertext, nonce, tag = encrypt_payload(
                device.shared_secret_for(rsu_identity.get_public_info()), payload, aad)
            rsu.on_receive(build_envelope(
                "NO_UPDATE", device, "Cluster_1", 1, signature, ciphertext, nonce, tag))
            self.assertIn(1, rsu.completed_rounds)
            vehicle_row = next(
                row for row in metrics_tracker.rows()
                if row["node"] == "C1_V1" and row["round"] == 1
            )
            self.assertGreater(vehicle_row["signature_verification_ms"], 0.0)
        finally:
            rsu.shutdown()


if __name__ == "__main__":
    unittest.main(verbosity=2)
