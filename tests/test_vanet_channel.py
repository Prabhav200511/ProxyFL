import unittest
import threading
from unittest.mock import patch

from metrics import MetricsTracker
from network import Receiver, send_msg
from vanet_channel import (
    VANET_PHY_MAX_RATE_BPS, WirelessLink, link_capacity_bps)


class VanetChannelTests(unittest.TestCase):
    def test_measurement_failure_cannot_change_successful_delivery(self):
        received = []
        event = threading.Event()

        def capture(message):
            received.append(message)
            event.set()

        receiver = Receiver(0, capture)
        receiver.start()
        port = receiver.sock.getsockname()[1]
        try:
            with patch(
                    "network.metrics_tracker.record_wireless_delivery",
                    side_effect=RuntimeError("measurement failed")):
                sent = send_msg(
                    ("127.0.0.1", port),
                    {"type": "PING", "sender": "C0_V1", "round": 1},
                    wireless_link=WirelessLink("v2v", 10.0),
                )
            self.assertTrue(sent)
            self.assertTrue(event.wait(2.0))
            self.assertEqual(received[0]["type"], "PING")
        finally:
            receiver.shutdown()

    def test_capacity_is_capped_and_decreases_with_distance(self):
        near = link_capacity_bps(1.0)
        medium = link_capacity_bps(300.0)
        far = link_capacity_bps(1000.0)
        self.assertLessEqual(near, VANET_PHY_MAX_RATE_BPS)
        self.assertGreater(near, medium)
        self.assertGreater(medium, far)
        self.assertGreater(far, 0.0)

    def test_goodput_uses_delivered_bits_over_modeled_airtime(self):
        tracker = MetricsTracker()
        tracker.record_wireless_delivery("C0_V1", 1, 1000, 8_000_000.0)
        tracker.record_wireless_delivery("C0_V1", 1, 1000, 4_000_000.0)
        row = tracker.rows()[0]
        expected_airtime = 8000 / 8_000_000 + 8000 / 4_000_000
        self.assertAlmostEqual(row["vanet_wireless_bits"], 16000.0)
        self.assertAlmostEqual(row["vanet_airtime_s"], expected_airtime)
        self.assertAlmostEqual(
            row["vanet_goodput_bps"], 16000 / expected_airtime)
        self.assertAlmostEqual(
            row["vanet_link_capacity_bps"], 6_000_000.0)


if __name__ == "__main__":
    unittest.main()
