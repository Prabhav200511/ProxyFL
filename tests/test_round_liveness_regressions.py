"""Regressions for round-liveness bugs found by running the simulation.

Each test pins one failure mode that silently destroyed a whole FL round:

1. A non-ASCII character in an RSU/Server log line raised UnicodeEncodeError on
   a cp1252 console.  Those prints run inside receiver threads, so the
   exception aborted an in-flight aggregation after the round had already been
   marked complete: no CLUSTER_UPDATE and no NO_CLUSTER_UPDATE ever reached the
   server, and every vehicle stalled until its own failsafe timeout.
2. More generally, any exception inside aggregate() lost the round.  Both the
   RSU and the Server must now always emit exactly one downstream message.
3. A vehicle that was still on round r-1 discarded the global proxy for round r
   and then stalled for the full failsafe timeout once it arrived at round r.
"""

import io
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

import config
from device import Device, ROUND_TRAINING, ROUND_WAITING_GLOBAL
from rsu import RSU
from server import Server, training_done_event
from logger import TrainingLogger

HOT_PATH_MODULES = (
    "main.py", "device.py", "rsu.py", "server.py", "vanet_sim.py",
    "crypto_protocol.py", "metrics.py", "logger.py", "network.py",
    "models.py", "privacy.py", "config.py", "run_grid_experiments.py",
)


class ConsoleEncodingTests(unittest.TestCase):
    def test_no_non_ascii_characters_in_logged_lines(self):
        offenders = []
        for module in HOT_PATH_MODULES:
            with io.open(module, encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    if "print(" not in line:
                        continue
                    if any(ord(char) > 127 for char in line):
                        offenders.append(f"{module}:{number}")
        self.assertEqual(
            offenders, [],
            "non-ASCII log output can raise UnicodeEncodeError inside a "
            "receiver thread and abort an aggregation: " + ", ".join(offenders),
        )


class PersistedRoundOrderTests(unittest.TestCase):
    def test_saved_tables_sort_rounds_even_when_threads_log_out_of_order(self):
        training_logger = TrainingLogger()
        training_logger.log_vehicle(97, "fast", 0.1, 0.9)
        training_logger.log_vehicle(63, "slow", 0.2, 0.8)
        training_logger.log_global(97, 0.9)
        training_logger.log_global(63, 0.8)

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir, "training.log")
            training_logger.save_logs(output)
            rendered = output.read_text(encoding="utf-8")

        vehicle_section, remaining = rendered.split(
            "GLOBAL PROXY MODEL EVALUATION", 1)
        global_section = remaining.split(
            "CLUSTER AGGREGATION DIVERGENCE", 1)[0]

        def rendered_rounds(section):
            rounds = []
            for line in section.splitlines():
                cells = [cell.strip() for cell in line.split("|")]
                if len(cells) > 1 and cells[1].isdigit():
                    rounds.append(int(cells[1]))
            return rounds

        self.assertEqual(rendered_rounds(vehicle_section), [63, 97])
        self.assertEqual(rendered_rounds(global_section), [63, 97])


class RsuAggregationLivenessTests(unittest.TestCase):
    def _rsu(self):
        rsu = RSU.__new__(RSU)
        rsu.name = "RSU_0_Central"
        rsu.port = 5000
        rsu.server_port = 8000
        rsu.cluster_ports = [6000, 6001]
        rsu.vehicle_names = ["C0_V1", "C0_V2"]
        rsu.topology = None
        rsu.global_reference_weights = None
        rsu.security_authority = None
        rsu.signer = None
        rsu.security_enabled = False
        rsu.batch_verification_enabled = False
        rsu.verifier = None
        rsu.round_buffers = {}
        rsu.round_reported = {}
        rsu.completed_rounds = set()
        rsu._round_timers = {}
        rsu._round_deadlines = {}
        rsu._lock = threading.Lock()
        return rsu

    def test_exception_during_aggregation_still_reports_to_server(self):
        rsu = self._rsu()
        rsu.round_buffers[1] = [{
            "sender": "C0_V1",
            "raw_msg": {"weights": b"unused"},
            "verified_payload": None,
            "sender_info": None,
        }]
        rsu.round_reported[1] = {"C0_V1"}
        rsu._aggregate_round = MagicMock(
            side_effect=UnicodeEncodeError(
                "charmap", "\u2202", 0, 1, "unmappable"))
        rsu._send_no_cluster_update = MagicMock(return_value=True)

        rsu.aggregate(1)

        self.assertIn(1, rsu.completed_rounds)
        rsu._send_no_cluster_update.assert_called_once_with(1)

    def test_trust_log_line_is_console_safe(self):
        rsu = self._rsu()
        state = {"fc1.weight": torch.zeros(2, 2)}
        rsu.round_buffers[1] = [
            {"sender": "C0_V1", "raw_msg": {}, "verified_payload": None,
             "sender_info": None},
        ]
        rsu.round_reported[1] = {"C0_V1"}
        sent = []
        with patch("rsu.send_msg", side_effect=lambda *a, **k: sent.append(a)):
            with patch("rsu.deserialize_weights", return_value=state):
                rsu.round_buffers[1][0]["raw_msg"] = {
                    "weights": b"payload"}
                buffer = io.StringIO()
                with patch("sys.stdout", buffer):
                    rsu.aggregate(1)
        printed = buffer.getvalue()
        self.assertIn("[TRUST]", printed)
        printed.encode("cp1252")  # must not raise
        self.assertEqual(len(sent), 1, "cluster update was not forwarded")


class ServerAggregationLivenessTests(unittest.TestCase):
    def test_exception_during_aggregation_still_broadcasts_and_finishes(self):
        training_done_event.clear()
        server = Server.__new__(Server)
        server._lock = threading.Lock()
        server.completed_rounds = set()
        server._round_timers = {}
        server._round_deadlines = {}
        server.round_buffers = {1: [{"sender": "RSU_0_Central"}]}
        server.round_reported = {1: {"RSU_0_Central"}}
        server.round_start_times = {1: time.perf_counter()}
        server.rsu_directory = {"RSU_0_Central": 5000}
        server.total_rounds = 1
        server.model = MagicMock()
        server.model.state_dict.return_value = {"weight": torch.tensor([3.0])}
        server._broadcast_global = MagicMock()
        server._aggregate_round = MagicMock(side_effect=RuntimeError("boom"))

        server.aggregate(1)

        self.assertIn(1, server.completed_rounds)
        server._broadcast_global.assert_called_once()
        self.assertTrue(
            training_done_event.is_set(),
            "a failed final round must still release the simulation barrier")


class DeviceFutureGlobalTests(unittest.TestCase):
    def _device(self):
        device = Device.__new__(Device)
        device.name = "C0_V1"
        device.device_id = 0
        device.current_round = 2
        device.proxy_lock = threading.Lock()
        device.round_event = threading.Event()
        device._pending_global_updates = {}
        device._round_phase = ROUND_TRAINING
        device.proxy_model = MagicMock()
        return device

    def test_global_for_a_future_round_is_stashed_not_discarded(self):
        device = self._device()
        weights = {"fc1.weight": torch.ones(1)}

        device._handle_verified_global(3, weights)

        self.assertIn(3, device._pending_global_updates)
        device.proxy_model.load_state_dict.assert_not_called()

        device.current_round = 3
        device._round_phase = ROUND_WAITING_GLOBAL
        self.assertTrue(device._apply_pending_global(3))
        device.proxy_model.load_state_dict.assert_called_once_with(weights)
        self.assertTrue(device.round_event.is_set())

    def test_global_for_a_past_round_is_still_discarded(self):
        device = self._device()
        device._handle_verified_global(1, {"fc1.weight": torch.ones(1)})
        self.assertEqual(device._pending_global_updates, {})


class TimeoutInvariantTests(unittest.TestCase):
    def test_device_failsafe_covers_the_full_aggregation_cascade(self):
        self.assertGreater(
            config.TIMEOUT,
            config.RSU_ROUND_MAX_WAIT + config.SERVER_ROUND_MAX_WAIT,
            "a vehicle must not give up before the RSU->Server cascade can "
            "reach its own hard caps",
        )
        self.assertLessEqual(config.RSU_ROUND_TIMEOUT, config.RSU_ROUND_MAX_WAIT)
        self.assertLessEqual(
            config.SERVER_ROUND_TIMEOUT, config.SERVER_ROUND_MAX_WAIT)


if __name__ == "__main__":
    unittest.main()
