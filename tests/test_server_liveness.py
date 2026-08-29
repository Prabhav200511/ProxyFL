import io
import threading
import time
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

import server as server_module
from config import DEVICE
from logger import TrainingLogger
from metrics import MetricsTracker
from server import Server, training_done_event
from vanet_sim import VanetTopology


class ServerLivenessTests(unittest.TestCase):
    def _make_server(self, total_rounds=3):
        server = Server.__new__(Server)
        server._lock = threading.Lock()
        server.completed_rounds = set()
        server._round_timers = {}
        server._round_deadlines = {}
        server.round_buffers = {}
        server.round_reported = {}
        server.round_start_times = {}
        server.rsu_directory = {"RSU_0_Central": 5000}
        server.security_enabled = False
        server.security_authority = None
        server.verifier = None
        server.batch_verification_enabled = False
        server.total_rounds = total_rounds
        server.aodv_enabled = True

        topology = VanetTopology()
        topology.register_rsu("RSU_0_Central", 0, 0)
        topology.register_vehicle("C0_V1", 100_000, 0, 0, 0)
        server.topology = topology
        server.cluster_vehicle_names = {"RSU_0_Central": ["C0_V1"]}

        model = torch.nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[-1.0], [1.0]]))
        server.model = model.to(DEVICE)
        inputs = torch.tensor([[-1.0], [1.0]])
        targets = torch.tensor([0, 1])
        server.test_loader = DataLoader(
            TensorDataset(inputs, targets), batch_size=2, shuffle=False)
        return server

    def _assert_empty_round_metrics(self, tracker, training_logger, round_num):
        self.assertIn(round_num, training_logger.global_proxy_acc)
        self.assertEqual(training_logger.global_proxy_acc[round_num], 100.0)

        rows = {
            (row["node"], row["round"]): row
            for row in tracker.rows()
        }
        server_row = rows[("Server", round_num)]
        expected_metrics = {
            "global_proxy_accuracy_pct": 100.0,
            "global_proxy_f1": 1.0,
            "global_proxy_recall": 1.0,
            "successful_updates": 0.0,
            "throughput_updates_per_sec": 0.0,
            "throughput_bytes_per_sec": 0.0,
            "model_payload_bytes_rx": 0.0,
            "vehicles_in_range_total": 0.0,
        }
        for metric, expected in expected_metrics.items():
            self.assertIn(metric, server_row)
            self.assertEqual(server_row[metric], expected)

    def test_empty_round_evaluates_current_model_and_continues_training(self):
        training_done_event.clear()
        server = self._make_server(total_rounds=3)
        server.round_buffers = {1: []}
        server.round_reported = {1: set()}
        server.round_start_times = {1: time.perf_counter()}
        broadcasts = []
        armed_rounds = []
        server._broadcast_global = (
            lambda round_num, weights, directory:
            broadcasts.append((round_num, weights, directory)))
        server._arm_silent_round = armed_rounds.append
        tracker = MetricsTracker()
        training_logger = TrainingLogger()
        output = io.StringIO()

        with patch.object(server_module, "metrics_tracker", tracker), \
                patch.object(server_module, "logger", training_logger), \
                redirect_stdout(output):
            server.aggregate(1)

        self._assert_empty_round_metrics(tracker, training_logger, 1)
        self.assertEqual([item[0] for item in broadcasts], [1])
        self.assertEqual(armed_rounds, [2])
        self.assertFalse(training_done_event.is_set())
        rendered = output.getvalue()
        self.assertIn("GLOBAL PROXY METRICS", rendered)
        self.assertIn("Test Accuracy : 100.0%", rendered)
        self.assertIn("F1-Score", rendered)
        self.assertIn("Recall", rendered)

    def test_silent_timeout_evaluates_current_model_and_continues_training(self):
        training_done_event.clear()
        server = self._make_server(total_rounds=3)
        server.round_buffers = {2: []}
        server.round_reported = {2: set()}
        server.round_start_times = {2: time.perf_counter()}
        broadcasts = []
        armed_rounds = []
        server._broadcast_global = (
            lambda round_num, weights, directory:
            broadcasts.append((round_num, weights, directory)))
        server._arm_silent_round = armed_rounds.append
        tracker = MetricsTracker()
        training_logger = TrainingLogger()

        with patch.object(server_module, "metrics_tracker", tracker), \
                patch.object(server_module, "logger", training_logger):
            server._force_aggregate(2)

        self._assert_empty_round_metrics(tracker, training_logger, 2)
        self.assertEqual([item[0] for item in broadcasts], [2])
        self.assertEqual(armed_rounds, [3])
        self.assertFalse(training_done_event.is_set())

    def test_invalid_only_final_round_reports_metrics_and_finishes(self):
        training_done_event.clear()
        server = self._make_server(total_rounds=1)
        server.round_buffers = {1: [{
            "sender": "RSU_0_Central",
            "raw_msg": {"avg_weights": b"not-a-model"},
            "verified_payload": None,
            "sender_info": None,
        }]}
        server.round_reported = {1: {"RSU_0_Central"}}
        server.round_start_times = {1: time.perf_counter()}
        broadcasts = []
        server._broadcast_global = (
            lambda round_num, weights, directory:
            broadcasts.append((round_num, weights, directory)))
        tracker = MetricsTracker()
        training_logger = TrainingLogger()

        with patch.object(server_module, "metrics_tracker", tracker), \
                patch.object(server_module, "logger", training_logger):
            server.aggregate(1)

        self._assert_empty_round_metrics(tracker, training_logger, 1)
        self.assertIn(1, server.completed_rounds)
        self.assertEqual([item[0] for item in broadcasts], [1])
        self.assertTrue(training_done_event.is_set())


if __name__ == "__main__":
    unittest.main()
