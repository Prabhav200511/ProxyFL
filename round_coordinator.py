"""Out-of-band simulation control for strictly ordered FL rounds.

The coordinator never carries model parameters or protocol messages.  It only
controls when a simulated round may open and records whether the existing
network path attempted a global delivery to each vehicle.
"""

import threading
import time


class RoundCoordinator:
    """Keep all vehicle workers in one monotonically increasing round."""

    def __init__(self, vehicle_names, rsu_names):
        self._vehicles = frozenset(vehicle_names)
        self._rsus = frozenset(rsu_names)
        if not self._vehicles:
            raise ValueError("at least one vehicle is required")
        if not self._rsus:
            raise ValueError("at least one RSU is required")
        self._condition = threading.Condition()
        self._round_open_callbacks = []
        self._arrivals = {}
        self._finished = {}
        self._opened = set()
        self._rsu_results = {}
        self._delivered_vehicles = {}
        self._verified_receipts = {}
        self._aborted = None

    def add_round_open_callback(self, callback):
        """Run *callback(round_num)* immediately before a round opens."""
        with self._condition:
            if self._opened:
                raise RuntimeError("round callbacks must be registered before training")
            self._round_open_callbacks.append(callback)

    def _validate_vehicle(self, vehicle_name):
        if vehicle_name not in self._vehicles:
            raise ValueError(f"unknown vehicle {vehicle_name!r}")

    @staticmethod
    def _validate_round(round_num):
        if type(round_num) is not int or round_num < 1:
            raise ValueError("round number must be a positive integer")

    def abort(self, reason):
        """Break all barriers and preserve the first worker failure."""
        message = str(reason) or "round coordination aborted"
        with self._condition:
            if self._aborted is None:
                self._aborted = message
            self._condition.notify_all()

    def raise_if_aborted(self):
        with self._condition:
            self._raise_if_aborted_locked()

    def _raise_if_aborted_locked(self):
        if self._aborted is not None:
            raise RuntimeError(f"Round coordination aborted: {self._aborted}")

    def _wait_for_barrier_locked(self, predicate, timeout, description):
        deadline = None if timeout is None else time.monotonic() + timeout
        while not predicate():
            self._raise_if_aborted_locked()
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                self._aborted = f"{description} timed out"
                self._condition.notify_all()
                self._raise_if_aborted_locked()
            self._condition.wait(remaining)
        self._raise_if_aborted_locked()

    def wait_for_round_start(self, vehicle_name, round_num, timeout=None):
        """Block until every configured vehicle is ready to start this round."""
        self._validate_vehicle(vehicle_name)
        self._validate_round(round_num)
        with self._condition:
            self._raise_if_aborted_locked()
            arrivals = self._arrivals.setdefault(round_num, set())
            arrivals.add(vehicle_name)
            if arrivals == self._vehicles and round_num not in self._opened:
                try:
                    for callback in self._round_open_callbacks:
                        callback(round_num)
                except BaseException as exc:
                    self._aborted = (
                        f"opening round {round_num} failed: {exc!r}")
                    self._condition.notify_all()
                    self._raise_if_aborted_locked()
                self._opened.add(round_num)
                self._condition.notify_all()
            else:
                self._wait_for_barrier_locked(
                    lambda: round_num in self._opened,
                    timeout,
                    f"opening round {round_num}",
                )

    def finish_round(self, vehicle_name, round_num, timeout=None):
        """Block at the cohort barrier until every vehicle finishes the round."""
        self._validate_vehicle(vehicle_name)
        self._validate_round(round_num)
        with self._condition:
            self._raise_if_aborted_locked()
            if round_num not in self._opened:
                raise RuntimeError(f"round {round_num} has not opened")
            finished = self._finished.setdefault(round_num, set())
            finished.add(vehicle_name)
            if finished == self._vehicles:
                self._condition.notify_all()
            else:
                self._wait_for_barrier_locked(
                    lambda: self._finished[round_num] == self._vehicles,
                    timeout,
                    f"finishing round {round_num}",
                )

    def record_rsu_result(self, round_num, rsu_name, delivered_vehicle_names):
        """Record the real network delivery outcomes from one RSU broadcast."""
        self._validate_round(round_num)
        if rsu_name not in self._rsus:
            raise ValueError(f"unknown RSU {rsu_name!r}")
        delivered = set(delivered_vehicle_names)
        unknown = delivered - self._vehicles
        if unknown:
            raise ValueError(f"unknown delivered vehicles: {sorted(unknown)!r}")
        with self._condition:
            if self._aborted is not None:
                return
            results = self._rsu_results.setdefault(round_num, {})
            results.setdefault(rsu_name, delivered)
            if set(results) == self._rsus:
                self._delivered_vehicles[round_num] = set().union(*results.values())
                self._condition.notify_all()

    def record_vehicle_receipt(self, vehicle_name, round_num):
        """Acknowledge that a device authenticated and accepted its global."""
        self._validate_vehicle(vehicle_name)
        self._validate_round(round_num)
        with self._condition:
            if self._aborted is not None:
                return
            self._verified_receipts.setdefault(round_num, set()).add(
                vehicle_name)
            self._condition.notify_all()

    def wait_for_vehicle_result(self, vehicle_name, round_num, timeout=None):
        """Return whether AODV/direct transport delivered this round's global.

        ``None`` means the network round itself did not finish before *timeout*.
        """
        self._validate_vehicle(vehicle_name)
        self._validate_round(round_num)
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                self._raise_if_aborted_locked()
                if round_num in self._delivered_vehicles:
                    delivered = self._delivered_vehicles[round_num]
                    if vehicle_name not in delivered:
                        return False
                    if vehicle_name in self._verified_receipts.get(
                            round_num, set()):
                        return True
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)
