"""Tests for metrics tracking, latency decomposition, and energy calculations."""

from __future__ import annotations

import unittest
import os
import tempfile

import pandas as pd
from metrics import MetricsTracker, energy_joules
from config import OBU_PEAK_POWER_W, X_OP_CRYPTO, X_OP_COMM, X_OP_TRAIN, X_OP_IDLE


class MetricsAndEnergyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = MetricsTracker()

    def test_energy_formula(self) -> None:
        # E = 10.88 * x * (t_ms / 1000)
        e_crypto = energy_joules(100.0, X_OP_CRYPTO)
        expected = 10.88 * X_OP_CRYPTO * 0.1
        self.assertAlmostEqual(e_crypto, expected)

    def test_energy_and_latency_decomposition(self) -> None:
        node = "C1_V1"
        r = 1

        # Simulate durations
        self.tracker.record_duration(node, r, "key_generation", 0.010)       # 10ms
        self.tracker.record_duration(node, r, "signature_generation", 0.005) # 5ms
        self.tracker.record_duration(node, r, "signature_verification", 0.005) # 5ms
        self.tracker.record_duration(node, r, "encryption", 0.002)           # 2ms
        self.tracker.record_duration(node, r, "decryption", 0.002)           # 2ms
        self.tracker.record_duration(node, r, "communication_tx", 0.015)     # 15ms
        self.tracker.record_duration(node, r, "communication_rx", 0.010)     # 10ms
        self.tracker.record_duration(node, r, "training", 0.500)             # 500ms
        self.tracker.record_duration(node, r, "device_round_execution", 0.700) # 700ms

        rows = self.tracker.rows()
        self.assertEqual(len(rows), 1)
        row = rows[0]

        # Verify latency sums
        # Security latency: 10 + 5 + 5 + 2 + 2 = 24ms
        self.assertAlmostEqual(row["security_latency_ms"], 24.0)
        # Communication latency: 15 + 10 = 25ms
        self.assertAlmostEqual(row["communication_latency_ms"], 25.0)

        # Verify End-to-End Time sum
        self.assertAlmostEqual(row["end_to_end_time_ms"], 500.0 + 24.0 + 25.0 + 151.0)

        # Verify Energy decomposition: E_total = E_security + E_comm
        e_sec = row["energy_security_j"]
        e_comm = row["energy_communication_j"]
        e_tot = row["energy_total_j"]
        self.assertAlmostEqual(e_tot, e_sec + e_comm)

        # Verify idle duration: execution (700) - active (500 + 24 + 25) = 151ms
        self.assertAlmostEqual(row["idle_latency_ms"], 151.0)

    def test_export_includes_vanet_capacity_columns(self) -> None:
        self.tracker.record_wireless_delivery(
            "C0_V1", 1, 1000, 8_000_000.0)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "metrics.csv")
            self.tracker.export_csv(path)
            frame = pd.read_csv(path)
        required = {
            "vanet_wireless_bits", "vanet_airtime_s",
            "vanet_link_capacity_bps", "vanet_goodput_bps",
        }
        self.assertTrue(required.issubset(frame.columns))

    def test_export_preserves_global_f1_and_recall(self) -> None:
        self.tracker.record_value("Server", 1, "global_proxy_f1", 0.75)
        self.tracker.record_value("Server", 1, "global_proxy_recall", 0.625)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "metrics.csv")
            self.tracker.export_csv(path)
            frame = pd.read_csv(path)

        self.assertIn("global_proxy_f1", frame.columns)
        self.assertIn("global_proxy_recall", frame.columns)
        self.assertEqual(frame.loc[0, "global_proxy_f1"], 0.75)
        self.assertEqual(frame.loc[0, "global_proxy_recall"], 0.625)


if __name__ == "__main__":
    unittest.main(verbosity=2)
