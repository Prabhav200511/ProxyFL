import queue
import threading
import unittest
from unittest.mock import patch

import torch

from network import Receiver
from round_coordinator import RoundCoordinator
from rsu import RSU
from server import Server, training_done_event


class AodvLivenessTests(unittest.TestCase):
    def test_coordinated_rsu_arms_only_the_round_opened_by_vehicle_cohort(self):
        coordinator = RoundCoordinator(["A"], ["R"])
        try:
            rsu = RSU("R", 0, [60001], 60002,
                      vehicle_names=["A"], security_enabled=False,
                      router=object(), total_rounds=2,
                      round_coordinator=coordinator)
        except TypeError:
            self.fail("RSU must accept the shared round coordinator")
        try:
            with patch("rsu.RSU_ROUND_MAX_WAIT", 10):
                rsu.start()
                self.assertEqual(rsu._round_timers, {})
                coordinator.wait_for_round_start("A", 1)
                self.assertEqual(set(rsu._round_timers), {1})
        finally:
            rsu.shutdown()

    def test_entirely_silent_rsu_reports_each_round_over_backhaul(self):
        received = queue.Queue()
        receiver = Receiver(0, received.put)
        receiver.start()
        rsu = RSU("R", 0, [60001], receiver.sock.getsockname()[1],
                  vehicle_names=["A"], security_enabled=False,
                  router=object(), total_rounds=2)
        try:
            with patch("rsu.RSU_ROUND_MAX_WAIT", 0.03):
                rsu.start()
                first = received.get(timeout=3)
                second = received.get(timeout=3)
            self.assertEqual((first["type"], first["round"]), ("NO_CLUSTER_UPDATE", 1))
            self.assertEqual((second["type"], second["round"]), ("NO_CLUSTER_UPDATE", 2))
            self.assertEqual(rsu.completed_rounds, {1, 2})
            self.assertEqual(rsu._round_timers, {})
        finally:
            rsu.shutdown()
            receiver.shutdown()

    def test_entirely_silent_server_broadcasts_last_known_model_and_finishes(self):
        received = queue.Queue()
        receiver = Receiver(0, received.put)
        receiver.start()
        # Construct only the network/round part; no training or dataset is needed.
        server = Server.__new__(Server)
        server.port = 0
        server.aodv_enabled = True
        server.total_rounds = 2
        server.expected_rsus = 1
        server.security_enabled = False
        server.security_authority = None
        server.signer = None
        server.topology = None
        server.cluster_vehicle_names = {}
        server.model = torch.nn.Linear(1, 1)
        server.round_buffers = {}
        server.round_reported = {}
        server.round_start_times = {}
        server.completed_rounds = set()
        server._round_timers = {}
        server._round_deadlines = {}
        server._lock = threading.Lock()
        server.rsu_directory = {"R": receiver.sock.getsockname()[1]}
        server.receiver = Receiver(0, server.on_receive)
        training_done_event.clear()
        try:
            with patch("server.SERVER_ROUND_MAX_WAIT", 0.03):
                server.start()
                first = received.get(timeout=3)
                second = received.get(timeout=3)
                self.assertTrue(training_done_event.wait(timeout=3))
            self.assertEqual((first["type"], first["round"]), ("GLOBAL_UPDATE", 1))
            self.assertEqual((second["type"], second["round"]), ("GLOBAL_UPDATE", 2))
            self.assertEqual(first["global_weights"], second["global_weights"])
            self.assertEqual(server.completed_rounds, {1, 2})
        finally:
            server.shutdown()
            receiver.shutdown()
            training_done_event.clear()

    def test_failed_server_to_rsu_route_closes_cluster_result_as_unreachable(self):
        coordinator = RoundCoordinator(["A"], ["R"])
        server = Server.__new__(Server)
        server.security_enabled = False
        server.security_authority = None
        server.signer = None
        server.round_coordinator = coordinator
        server.aodv_enabled = True

        with patch("server.send_msg", return_value=False):
            server._broadcast_global(1, {"weight": torch.tensor([1.0])}, {"R": 5000})

        self.assertIs(
            coordinator.wait_for_vehicle_result("A", 1, timeout=0.1),
            False,
        )

    def test_invalid_global_does_not_finalize_trusted_round_result(self):
        coordinator = RoundCoordinator(["A"], ["R"])
        rsu = RSU.__new__(RSU)
        rsu.name = "R"
        rsu.round_coordinator = coordinator
        rsu._decode_server_global = lambda message: None

        rsu.on_receive({
            "type": "GLOBAL_UPDATE",
            "sender": "Server",
            "recipient": "R",
            "round": 1,
        })

        self.assertIsNone(
            coordinator.wait_for_vehicle_result("A", 1, timeout=0.01))

    def test_broadcast_construction_failure_finalizes_each_rsu_and_continues(self):
        coordinator = RoundCoordinator(["A", "B"], ["R1", "R2"])
        server = Server.__new__(Server)
        server.round_coordinator = coordinator
        server._build_global_message = (
            lambda rsu_name, round_num, weights:
            (_ for _ in ()).throw(RuntimeError("signing failed"))
            if rsu_name == "R1" else {"type": "GLOBAL_UPDATE"}
        )

        with patch("server.send_msg", return_value=False):
            server._broadcast_global(
                1, {"weight": torch.tensor([1.0])}, {"R1": 5001, "R2": 5002})

        self.assertIs(
            coordinator.wait_for_vehicle_result("A", 1, timeout=0.1), False)
        self.assertIs(
            coordinator.wait_for_vehicle_result("B", 1, timeout=0.1), False)

    def test_direct_rsu_does_not_start_a_new_watchdog(self):
        rsu = RSU("R", 0, [60001], 60002, security_enabled=False)
        try:
            rsu.start()
            self.assertEqual(rsu._round_timers, {})
        finally:
            rsu.shutdown()


if __name__ == "__main__":
    unittest.main()
