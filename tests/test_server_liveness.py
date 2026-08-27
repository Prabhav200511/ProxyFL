import threading
import time
import unittest
from unittest.mock import MagicMock

import torch

from server import Server, training_done_event


class ServerLivenessTests(unittest.TestCase):
    def test_invalid_only_round_broadcasts_current_model_and_finishes(self):
        training_done_event.clear()
        server = Server.__new__(Server)
        server._lock = threading.Lock()
        server.completed_rounds = set()
        server._round_timers = {}
        server._round_deadlines = {}
        server.round_buffers = {1: [{
            "sender": "RSU_0_Central",
            "raw_msg": {"avg_weights": b"not-a-model"},
            "verified_payload": None,
            "sender_info": None,
        }]}
        server.round_reported = {1: {"RSU_0_Central"}}
        server.round_start_times = {1: time.perf_counter()}
        server.rsu_directory = {"RSU_0_Central": 5000}
        server.security_enabled = False
        server.security_authority = None
        server.verifier = None
        server.batch_verification_enabled = False
        server.topology = None
        server.cluster_vehicle_names = {}
        server.total_rounds = 1
        current = {"weight": torch.tensor([7.0])}
        server.model = MagicMock()
        server.model.state_dict.return_value = current
        server._broadcast_global = MagicMock()

        server.aggregate(1)

        self.assertIn(1, server.completed_rounds)
        server._broadcast_global.assert_called_once()
        args = server._broadcast_global.call_args.args
        self.assertEqual(args[0], 1)
        self.assertTrue(torch.equal(args[1]["weight"], current["weight"]))
        self.assertEqual(args[2], server.rsu_directory)
        self.assertTrue(training_done_event.is_set())


if __name__ == "__main__":
    unittest.main()
