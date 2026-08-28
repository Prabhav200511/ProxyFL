import queue
import threading
import unittest
from unittest.mock import patch

import torch

from network import Receiver
from rsu import RSU
from server import Server, training_done_event


class AodvLivenessTests(unittest.TestCase):
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

    def test_direct_rsu_does_not_start_a_new_watchdog(self):
        rsu = RSU("R", 0, [60001], 60002, security_enabled=False)
        try:
            rsu.start()
            self.assertEqual(rsu._round_timers, {})
        finally:
            rsu.shutdown()


if __name__ == "__main__":
    unittest.main()
