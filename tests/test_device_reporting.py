"""Regression tests for unavailable-vehicle round reports."""

from __future__ import annotations

import unittest
import threading
from unittest.mock import MagicMock, patch

import torch

from device import Device
from round_coordinator import RoundCoordinator


class DeviceReportingTests(unittest.TestCase):
    def test_inner_round_failure_aborts_coordinator_and_is_not_swallowed(self):
        coordinator = RoundCoordinator(["C0_V1"], ["RSU_0"])
        device = Device.__new__(Device)
        device.name = "C0_V1"
        device.round_coordinator = coordinator

        handler = getattr(device, "_abort_round", None)
        self.assertIsNotNone(handler, "inner round failures must abort the run")
        with self.assertRaisesRegex(RuntimeError, "Round 3 failed"):
            handler(3, ValueError("broken update"))
        with self.assertRaisesRegex(RuntimeError, "broken update"):
            coordinator.raise_if_aborted()

    def test_unresolved_network_round_timeout_aborts_instead_of_advancing(self):
        coordinator = RoundCoordinator(["C0_V1"], ["RSU_0"])
        device = Device.__new__(Device)
        device.name = "C0_V1"
        device.round_coordinator = coordinator
        device.round_event = threading.Event()
        device.current_round = 1
        device._round_phase = "waiting_global"
        device._pending_global_updates = {}
        device._request_sent_at = {}
        device.proxy_lock = threading.Lock()

        with patch("device.TIMEOUT", 0.01), self.assertRaisesRegex(
                RuntimeError, "Network round 1"):
            device._wait_for_round_sync(1)
        with self.assertRaisesRegex(RuntimeError, "Network round 1"):
            coordinator.raise_if_aborted()

    def test_unhandled_worker_failure_aborts_all_round_barriers(self):
        coordinator = RoundCoordinator(["C0_V1"], ["RSU_0"])
        device = Device.__new__(Device)
        device.name = "C0_V1"
        device.round_coordinator = coordinator
        device.send = MagicMock(side_effect=RuntimeError("training crashed"))

        target = getattr(device, "_training_target", None)
        self.assertIsNotNone(target, "device worker failures must abort barriers")
        target()

        with self.assertRaisesRegex(RuntimeError, "training crashed"):
            coordinator.raise_if_aborted()

    def test_unreachable_vehicle_uses_round_result_without_global_wait_timeout(self):
        coordinator = RoundCoordinator(["C0_V1"], ["RSU_0"])
        coordinator.record_rsu_result(1, "RSU_0", set())
        device = Device.__new__(Device)
        device.name = "C0_V1"
        device.round_coordinator = coordinator
        device.round_event = threading.Event()
        device._round_phase = "waiting_global"
        device._pending_global_updates = {}
        device._request_sent_at = {}
        device.proxy_lock = threading.Lock()

        waiter = getattr(device, "_wait_for_round_sync", None)
        self.assertIsNotNone(waiter, "coordinated round waiting is required")
        self.assertFalse(waiter(1))

    def test_delivered_vehicle_applies_global_before_round_barrier(self):
        coordinator = RoundCoordinator(["C0_V1"], ["RSU_0"])
        coordinator.record_rsu_result(1, "RSU_0", {"C0_V1"})
        device = Device.__new__(Device)
        device.name = "C0_V1"
        device.round_coordinator = coordinator
        device.round_event = threading.Event()
        device.current_round = 1
        device._round_phase = "training"
        device._pending_global_updates = {}
        device._request_sent_at = {}
        device.proxy_lock = threading.Lock()
        device.proxy_model = MagicMock()

        device._handle_verified_global(1, {"weight": torch.tensor([2.0])})

        waiter = getattr(device, "_wait_for_round_sync", None)
        self.assertIsNotNone(waiter, "coordinated round waiting is required")
        self.assertTrue(waiter(1))
        device.proxy_model.load_state_dict.assert_called_once()

    def test_early_peer_update_is_not_discarded_before_average(self):
        device = Device.__new__(Device)
        device.name = "C0_V2"
        device.current_round = 1
        device.peer_directory = {"C0_V1": 6000}
        device._peer_lock = threading.Lock()
        device._peer_buffers = {
            1: {"C0_V1": {"w": torch.tensor([9.0])}}}
        device.security_enabled = False
        device.signer = None
        device.security_authority = None
        device.topology = MagicMock()
        device.topology.get_v2v_neighbors.return_value = ["C0_V1"]
        device.topology.get_v2v_distance.return_value = 1.0
        with patch("device.send_msg", return_value=True), patch(
                "device.V2V_COLLECT_TIMEOUT", 0.01):
            result = device._v2v_share_and_aggregate(
                1, {"w": torch.tensor([1.0])})
        self.assertEqual(result["w"].item(), 5.0)

    def test_verified_global_is_queued_until_local_report(self) -> None:
        device = Device.__new__(Device)
        device.current_round = 4
        device._round_phase = "training"
        device._pending_global_updates = {}
        device.proxy_lock = threading.Lock()
        device.proxy_model = MagicMock()
        device.round_event = threading.Event()
        weights = {"weight": torch.tensor([3.0])}

        device._handle_verified_global(4, weights)
        device.proxy_model.load_state_dict.assert_not_called()
        self.assertFalse(device.round_event.is_set())

        device._round_phase = "waiting_global"
        self.assertTrue(device._apply_pending_global(4))
        device.proxy_model.load_state_dict.assert_called_once()
        self.assertTrue(device.round_event.is_set())

    def test_security_mode_rejects_unsigned_peer_update(self) -> None:
        device = Device.__new__(Device)
        device.name = "C2_V4"
        device.current_round = 3
        device.security_enabled = True
        device.security_authority = object()
        device.signer = object()
        device._peer_lock = threading.Lock()
        device._peer_buffers = {}

        device.on_receive({
            "type": "PEER_UPDATE",
            "sender": "C2_V5",
            "recipient": "C2_V4",
            "round": 3,
            "weights": {"weight": 1.0},
        })

        self.assertEqual(device._peer_buffers, {})

    def test_plain_no_update_report_identifies_vehicle_rsu_and_round(self) -> None:
        device = Device.__new__(Device)
        device.name = "C2_V4"
        device.rsu_name = "RSU_2_East"
        device.security_enabled = False
        device.signer = None
        device.security_authority = None

        builder = getattr(device, "_build_no_update_message", None)
        self.assertIsNotNone(builder, "_build_no_update_message is required")
        message = builder(7)

        self.assertEqual(message, {
            "type": "NO_UPDATE",
            "sender": "C2_V4",
            "recipient": "RSU_2_East",
            "round": 7,
        })


if __name__ == "__main__":
    unittest.main(verbosity=2)
