import copy
import queue
import socket
import unittest
from unittest.mock import patch

from crypto_protocol import Authority, build_envelope, encrypt_payload, message_aad, verify_envelope
from network import Receiver, WirelessRouter, send_msg
from metrics import MetricsTracker
from routing_sim import RoutingSimulator
from vanet_sim import VanetTopology


class RoutedTransportTests(unittest.TestCase):
    def setUp(self):
        self.received = queue.Queue()
        self.receiver = Receiver(0, self.received.put)
        self.port = self.receiver.sock.getsockname()[1]
        self.receiver.start()
        self.topology = VanetTopology(1)
        self.topology.register_vehicle("A", 1250, 0, 0, 0)
        self.topology.register_vehicle("B", 950, 0, 0, 0)
        self.topology.register_rsu("C", 0, 0)
        self.router = WirelessRouter(self.topology, RoutingSimulator(capacity=lambda d: 1e6))
        self.router.register("A", 60001)
        self.router.register("B", 60002)
        self.router.register("C", self.port)

    def tearDown(self):
        self.receiver.shutdown()

    def message(self, kind="LOCAL_UPDATE"):
        return dict(type=kind, sender="A", recipient="C", round=1, weights=b"opaque")

    def test_multihop_hands_off_one_identical_message(self):
        message = self.message()
        self.assertFalse(self.topology.can_reach_rsu("A", "C"))
        self.assertTrue(send_msg(("127.0.0.1", self.port), message, router=self.router))
        self.assertEqual(self.received.get(timeout=3), message)
        self.assertTrue(self.received.empty())
        self.assertEqual(self.router.simulator.ledger.rows()[0]["host_handoffs_succeeded"], 1)

    def test_link_capacity_uses_real_forwarding_hops_not_end_to_end_distance(self):
        tracker = MetricsTracker()
        with patch("network.metrics_tracker", tracker):
            self.assertTrue(send_msg(("127.0.0.1", self.port), self.message(), router=self.router))
        rows = {row["node"]: row for row in tracker.rows()}
        self.assertEqual(rows["A"]["vanet_link_capacity_bps"], 1e6)
        self.assertEqual(rows["B"]["vanet_link_capacity_bps"], 1e6)
        self.assertEqual(rows["A"]["vanet_wireless_bits"], rows["B"]["vanet_wireless_bits"])
        self.assertNotIn("C", rows)

    def test_unregistered_relay_cannot_create_a_phantom_route(self):
        self.router = WirelessRouter(self.topology, RoutingSimulator(capacity=lambda d: 1e6))
        self.router.register("A", 60001)
        self.router.register("C", self.port)
        self.assertFalse(send_msg(("127.0.0.1", self.port), self.message(), router=self.router))
        self.assertTrue(self.received.empty())

    def test_disconnected_no_update_and_global_never_bypass_to_tcp(self):
        self.topology.register_vehicle("B", 5000, 0, 0, 0)
        for kind in ("NO_UPDATE", "GLOBAL_UPDATE"):
            self.assertFalse(send_msg(("127.0.0.1", self.port), self.message(kind), router=self.router))
        self.assertTrue(self.received.empty())
        row = self.router.simulator.ledger.rows()[0]
        self.assertEqual(row["data_packets_delivered"], 0)
        self.assertEqual(row["host_handoffs_succeeded"], 0)

    def test_unregistered_destination_fails_closed_and_backhaul_remains_direct(self):
        self.router = WirelessRouter(self.topology)
        self.assertFalse(send_msg(("127.0.0.1", self.port), self.message(), router=self.router))
        self.assertTrue(send_msg(("127.0.0.1", self.port), self.message()))
        self.assertEqual(self.received.get(timeout=3), self.message())
        self.assertEqual(self.router.simulator.ledger.rows(), [])

    def test_host_failure_is_separate_from_network_arrival(self):
        with socket.socket() as closed:
            closed.bind(("127.0.0.1", 0))
            port = closed.getsockname()[1]
        self.router = WirelessRouter(self.topology, RoutingSimulator(capacity=lambda d: 1e6))
        self.router.register("A", 60001)
        self.router.register("B", 60002)
        self.router.register("C", port)
        self.assertFalse(send_msg(("127.0.0.1", port), self.message(), router=self.router))
        row = self.router.simulator.ledger.rows()[0]
        self.assertEqual(row["messages_network_delivered"], 1)
        self.assertEqual(row["host_handoffs_failed"], 1)
        self.assertEqual(row["host_handoffs_succeeded"], 0)

    def test_post_delivery_measurement_error_does_not_report_failure(self):
        with patch.object(self.router.simulator.ledger, "host_handoff", side_effect=OSError("metrics")):
            self.assertTrue(send_msg(("127.0.0.1", self.port), self.message(), router=self.router))
        self.assertEqual(self.received.get(timeout=3), self.message())

    def test_secure_envelope_verifies_unchanged_and_tampering_still_rejected(self):
        authority = Authority()
        for name in ("A", "C"):
            authority.enroll_mvd(name)
        first = authority.register("A", real_id="A")
        last = authority.register("C", real_id="C")
        payload = b"opaque" * 300
        aad = message_aad("LOCAL_UPDATE", "A", "C", 1)
        ciphertext, nonce, tag = encrypt_payload(first.shared_secret_for(last.get_public_info()), payload, aad)
        envelope = build_envelope("LOCAL_UPDATE", first, "C", 1, first.sign(payload), ciphertext, nonce, tag)
        baseline = {**self.message(), "weights": payload}
        self.assertTrue(send_msg(("127.0.0.1", self.port), envelope, router=self.router, unsecured_msg=baseline))
        delivered = self.received.get(timeout=3)
        self.assertEqual(delivered, envelope)
        self.assertEqual(verify_envelope(authority, last, delivered, "LOCAL_UPDATE")[0], payload)
        tampered = copy.deepcopy(delivered)
        tampered["ciphertext"] = b"x" + tampered["ciphertext"][1:]
        self.assertIsNone(verify_envelope(authority, last, tampered, "LOCAL_UPDATE"))
        self.assertIsNone(verify_envelope(authority, last, baseline, "LOCAL_UPDATE"))
        self.assertGreater(self.router.simulator.ledger.rows()[0]["security_bytes_tx"], 0)

    def test_snapshot_contains_no_rsu_to_rsu_edges(self):
        self.topology.register_rsu("D", 1, 0)
        snapshot = self.topology.routing_snapshot()
        self.assertNotIn("D", snapshot.adjacency["C"])
        self.assertEqual(snapshot.adjacency["A"], {"B": 300.0})

    def test_actual_rsu_global_forwarding_still_verifies_after_multihop(self):
        import torch
        from device import Device
        from model_codec import serialize_weights
        from rsu import RSU
        authority = Authority()
        signers = {}
        for name in ("A", "C", "Server"):
            authority.enroll_mvd(name)
            signers[name] = authority.register(name, real_id=name)
        router = WirelessRouter(self.topology, RoutingSimulator(capacity=lambda d: 1e6))
        rsu = RSU("C", 0, [self.port], 60004, topology=self.topology, vehicle_names=["A"],
                  security_authority=authority, security_identity=signers["C"],
                  security_enabled=True, router=router)
        router.register("A", self.port)
        router.register("B", 60002)
        router.register("C", rsu.receiver.sock.getsockname()[1])
        weights = {"weight": torch.tensor([1.0, 2.0])}
        payload = serialize_weights(weights)
        ciphertext, nonce, tag = encrypt_payload(signers["Server"].shared_secret_for(signers["C"].get_public_info()),
                                                 payload, message_aad("GLOBAL_UPDATE", "Server", "C", 1))
        incoming = build_envelope("GLOBAL_UPDATE", signers["Server"], "C", 1,
                                  signers["Server"].sign(payload), ciphertext, nonce, tag)
        try:
            rsu.on_receive(incoming)
            outgoing = self.received.get(timeout=3)
            vehicle = Device.__new__(Device)
            vehicle.name, vehicle.rsu_name = "A", "C"
            vehicle.security_enabled = True
            vehicle.security_authority, vehicle.signer = authority, signers["A"]
            decoded = vehicle._decode_rsu_global(outgoing)
            self.assertTrue(torch.equal(decoded["weight"], weights["weight"]))
            self.assertEqual(outgoing["sender"], "C")
            result = next(event for event in router.simulator.ledger.events if event["event"] == "network_result")
            self.assertEqual(result["path"], ("C", "B", "A"))
        finally:
            rsu.shutdown()

    def test_actual_device_no_update_cannot_bypass_disconnection(self):
        from device import Device
        self.topology.register_vehicle("B", 5000, 0, 0, 0)
        device = Device.__new__(Device)
        device.name, device.rsu_name, device.rsu_port = "A", "C", self.port
        device.topology, device.router = self.topology, self.router
        device.security_enabled = False
        device.signer = device.security_authority = None
        self.assertFalse(device._send_no_update(1))
        self.assertTrue(self.received.empty())


if __name__ == "__main__":
    unittest.main()
