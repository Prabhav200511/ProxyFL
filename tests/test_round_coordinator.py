"""Regression tests for simulation-wide FL round ordering."""

import threading
import time
import unittest


class RoundCoordinatorTests(unittest.TestCase):
    def _coordinator_class(self):
        try:
            from round_coordinator import RoundCoordinator
        except ModuleNotFoundError:
            self.fail("RoundCoordinator is required to prevent cross-round drift")
        return RoundCoordinator

    def test_fast_vehicle_cannot_enter_next_round_before_slow_vehicle_finishes(self):
        """Removing the end-of-round cohort barrier must break this test."""
        coordinator = self._coordinator_class()(["fast", "slow"], ["RSU"])
        slow_can_finish = threading.Event()
        fast_waiting = threading.Event()
        events = []

        def fast_vehicle():
            coordinator.wait_for_round_start("fast", 1)
            events.append("fast:1")
            fast_waiting.set()
            coordinator.finish_round("fast", 1)
            coordinator.wait_for_round_start("fast", 2)
            events.append("fast:2")

        def slow_vehicle():
            coordinator.wait_for_round_start("slow", 1)
            slow_can_finish.wait(timeout=2)
            events.append("slow:1")
            coordinator.finish_round("slow", 1)
            coordinator.wait_for_round_start("slow", 2)
            events.append("slow:2")

        fast = threading.Thread(target=fast_vehicle)
        slow = threading.Thread(target=slow_vehicle)
        fast.start()
        slow.start()
        self.assertTrue(fast_waiting.wait(timeout=2))
        time.sleep(0.03)
        self.assertNotIn("fast:2", events)

        slow_can_finish.set()
        fast.join(timeout=2)
        slow.join(timeout=2)

        self.assertFalse(fast.is_alive())
        self.assertFalse(slow.is_alive())
        self.assertLess(events.index("slow:1"), events.index("fast:2"))

    def test_round_opens_only_after_every_vehicle_arrives(self):
        """Eagerly arming a future-round watchdog must break this test."""
        coordinator = self._coordinator_class()(["A", "B"], ["RSU"])
        opened = []
        coordinator.add_round_open_callback(opened.append)

        first_arrival = threading.Thread(
            target=coordinator.wait_for_round_start, args=("A", 1))
        first_arrival.start()
        time.sleep(0.03)
        self.assertEqual(opened, [])

        second_arrival = threading.Thread(
            target=coordinator.wait_for_round_start, args=("B", 1))
        second_arrival.start()
        first_arrival.join(timeout=2)
        second_arrival.join(timeout=2)

        self.assertEqual(opened, [1])

    def test_vehicle_result_distinguishes_delivered_and_unreachable_globals(self):
        """A failed AODV route must not be reported as a model delivery."""
        coordinator = self._coordinator_class()(["A", "B"], ["RSU"])

        coordinator.record_rsu_result(1, "RSU", {"A"})

        self.assertIsNone(
            coordinator.wait_for_vehicle_result("A", 1, timeout=0.01),
            "transport handoff alone is not verified model receipt",
        )
        coordinator.record_vehicle_receipt("A", 1)
        self.assertTrue(coordinator.wait_for_vehicle_result("A", 1, timeout=0.1))
        self.assertFalse(coordinator.wait_for_vehicle_result("B", 1, timeout=0.1))

    def test_abort_releases_every_round_barrier_waiter(self):
        coordinator = self._coordinator_class()(["A", "B"], ["RSU"])
        errors = []

        def wait_for_missing_vehicle():
            try:
                coordinator.wait_for_round_start("A", 1)
            except RuntimeError as exc:
                errors.append(str(exc))

        waiter = threading.Thread(target=wait_for_missing_vehicle, daemon=True)
        waiter.start()
        time.sleep(0.03)
        abort = getattr(coordinator, "abort", None)
        self.assertIsNotNone(abort, "worker failures must break round barriers")
        abort("vehicle B failed")
        waiter.join(timeout=2)

        self.assertFalse(waiter.is_alive())
        self.assertIn("vehicle B failed", errors[0])


if __name__ == "__main__":
    unittest.main()
