"""Thread-safe timing, communication, throughput, and energy instrumentation."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Dict, Iterable, Mapping

import pandas as pd

from config import OBU_PEAK_POWER_W, X_OP_COMM, X_OP_CRYPTO, X_OP_IDLE, X_OP_TRAIN


def energy_joules(duration_ms: float, utilization: float) -> float:
    """``E = 10.88 W * x_op * t_ms / 1000`` for one measured operation."""
    return OBU_PEAK_POWER_W * utilization * max(duration_ms, 0.0) / 1000.0


class MetricsTracker:
    """Collect metrics by ``(node, round)`` without sharing mutable row objects."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.reset()

    def reset(self) -> None:
        with getattr(self, "_lock", threading.RLock()):
            self._rows: Dict[tuple[str, int], Dict[str, float]] = defaultdict(dict)
            self._simulation_started_at: float | None = None
            self._simulation_duration_ms: float | None = None

    def _row(self, node: str, round_num: int) -> Dict[str, float]:
        if not isinstance(round_num, int) or round_num < 0:
            raise ValueError("round number must be a non-negative integer")
        return self._rows[(node, round_num)]

    def record_duration(self, node: str, round_num: int, metric: str, duration_seconds: float) -> None:
        """Accumulate a duration in milliseconds."""
        with self._lock:
            row = self._row(node, round_num)
            key = metric if metric.endswith("_ms") else f"{metric}_ms"
            row[key] = row.get(key, 0.0) + max(duration_seconds, 0.0) * 1000.0

    def record_value(self, node: str, round_num: int, metric: str, value: float) -> None:
        with self._lock:
            self._row(node, round_num)[metric] = float(value)

    def add_value(self, node: str, round_num: int, metric: str, value: float) -> None:
        with self._lock:
            row = self._row(node, round_num)
            row[metric] = row.get(metric, 0.0) + float(value)

    def record_bytes(self, node: str, round_num: int, direction: str, num_bytes: int) -> None:
        if direction not in {"tx", "rx"}:
            raise ValueError("communication direction must be 'tx' or 'rx'")
        self.add_value(node, round_num, f"bytes_{direction}", max(num_bytes, 0))

    def record_wireless_delivery(
        self, node: str, round_num: int, num_wire_bytes: int,
        capacity_bps: float,
    ) -> None:
        """Accumulate one successful wireless hop without changing timing."""
        bits = float(max(num_wire_bytes, 0) * 8)
        if capacity_bps <= 0:
            raise ValueError("wireless capacity must be positive")
        with self._lock:
            row = self._row(node, round_num)
            row["vanet_wireless_bits"] = (
                row.get("vanet_wireless_bits", 0.0) + bits)
            row["vanet_airtime_s"] = (
                row.get("vanet_airtime_s", 0.0) + bits / capacity_bps)
            row["vanet_capacity_sum_bps"] = (
                row.get("vanet_capacity_sum_bps", 0.0) + capacity_bps)
            row["vanet_capacity_samples"] = (
                row.get("vanet_capacity_samples", 0.0) + 1.0)

    def record_batch_duration(
        self, receiver: str, participants: Iterable[str], round_num: int, duration_seconds: float
    ) -> None:
        """Record receiver work and allocate an equal batch-verification share.

        The allocation lets each vehicle's per-round security latency and energy
        sum cleanly while retaining the actual RSU/server wall-clock cost.
        """
        participants = [node for node in participants if node]
        duration_ms = max(duration_seconds, 0.0) * 1000.0
        with self._lock:
            receiver_row = self._row(receiver, round_num)
            receiver_row["batch_verification_receiver_ms"] = (
                receiver_row.get("batch_verification_receiver_ms", 0.0) + duration_ms
            )
            if participants:
                share = duration_ms / len(participants)
                for node in participants:
                    row = self._row(node, round_num)
                    row["batch_verification_ms"] = row.get("batch_verification_ms", 0.0) + share

    def start_simulation(self) -> None:
        self._simulation_started_at = time.perf_counter()

    def finish_simulation(self) -> float:
        if self._simulation_started_at is None:
            return 0.0
        self._simulation_duration_ms = (time.perf_counter() - self._simulation_started_at) * 1000.0
        return self._simulation_duration_ms

    @staticmethod
    def _derived_metrics(row: Mapping[str, float]) -> Dict[str, float]:
        training = row.get("training_ms", 0.0)
        key_generation = row.get("key_generation_ms", 0.0)
        signature_generation = row.get("signature_generation_ms", 0.0)
        signature_verification = row.get("signature_verification_ms", 0.0)
        batch_verification = row.get("batch_verification_ms", 0.0)
        encryption = row.get("encryption_ms", 0.0)
        decryption = row.get("decryption_ms", 0.0)
        comm_tx = row.get("communication_tx_ms", 0.0)
        comm_rx = row.get("communication_rx_ms", 0.0)
        security = (key_generation + signature_generation + signature_verification + batch_verification
                    + encryption + decryption)
        communication = comm_tx + comm_rx
        active = training + security + communication
        execution = max(
            row.get("device_round_execution_ms", 0.0),
            row.get("rsu_round_execution_ms", 0.0),
            row.get("server_round_execution_ms", 0.0),
        )
        idle = max(execution - active, 0.0)
        end_to_end = training + security + communication + idle
        wireless_bits = row.get("vanet_wireless_bits", 0.0)
        wireless_airtime = row.get("vanet_airtime_s", 0.0)
        capacity_samples = row.get("vanet_capacity_samples", 0.0)
        mean_capacity = (
            row.get("vanet_capacity_sum_bps", 0.0) / capacity_samples
            if capacity_samples > 0 else 0.0
        )
        goodput = (
            wireless_bits / wireless_airtime
            if wireless_airtime > 0 else 0.0
        )
        return {
            "security_latency_ms": security,
            "communication_latency_ms": communication,
            "end_to_end_time_ms": end_to_end,
            "energy_training_j": energy_joules(training, X_OP_TRAIN),
            "energy_security_j": energy_joules(security, X_OP_CRYPTO),
            "energy_communication_j": energy_joules(communication, X_OP_COMM),
            "energy_idle_j": energy_joules(idle, X_OP_IDLE),
            # Overhead total covers security + communication components.
            "energy_total_j": energy_joules(security, X_OP_CRYPTO)
            + energy_joules(communication, X_OP_COMM),
            "idle_latency_ms": idle,
            "vanet_link_capacity_bps": mean_capacity,
            "vanet_goodput_bps": goodput,
        }

    def rows(self, quality_metrics: Mapping[tuple[str, int], Mapping[str, float]] | None = None) -> list[dict]:
        quality_metrics = quality_metrics or {}
        with self._lock:
            keys = set(self._rows) | set(quality_metrics)
            result = []
            for node, round_num in sorted(keys, key=lambda key: (key[1], key[0])):
                row = {"node": node, "round": round_num}
                row.update(self._rows.get((node, round_num), {}))
                row.update(quality_metrics.get((node, round_num), {}))
                row.update(self._derived_metrics(row))
                result.append(row)
            return result

    def export_csv(
        self, filename: str, quality_metrics: Mapping[tuple[str, int], Mapping[str, float]] | None = None
    ) -> str:
        data = self.rows(quality_metrics)
        columns = [
            "node", "round", "train_loss", "train_accuracy_pct", "private_test_accuracy_pct",
            "epsilon", "delta", "global_proxy_accuracy_pct", "successful_updates",
            "throughput_updates_per_sec", "throughput_bytes_per_sec",
            "vanet_wireless_bits", "vanet_airtime_s",
            "vanet_link_capacity_bps", "vanet_goodput_bps",
            "vehicles_in_range", "vehicles_assigned",
            "vehicles_in_range_total", "vehicles_assigned_total",
            "bytes_tx", "bytes_rx", "model_payload_bytes_rx", "training_ms",
            "key_generation_ms", "signature_generation_ms", "signature_verification_ms", "batch_verification_ms",
            "batch_verification_receiver_ms", "encryption_ms", "decryption_ms",
            "communication_tx_ms", "communication_rx_ms", "security_latency_ms",
            "communication_latency_ms", "action_to_response_ms",
            "end_to_end_time_ms", "device_round_execution_ms", "rsu_round_execution_ms",
            "server_round_execution_ms", "energy_training_j", "energy_security_j",
            "energy_communication_j", "energy_total_j", "energy_idle_j", "idle_latency_ms",
        ]
        frame = pd.DataFrame(data)
        for column in columns:
            if column not in frame:
                frame[column] = 0.0
        frame = frame[columns].sort_values(["round", "node"])
        frame.to_csv(filename, index=False)
        return filename

    def export_simulation_summary(self, filename: str) -> str:
        total_ms = self._simulation_duration_ms or 0.0
        pd.DataFrame([{
            "simulation_execution_ms": total_ms,
            "simulation_execution_sec": total_ms / 1000.0,
        }]).to_csv(filename, index=False)
        return filename


class Timer:
    """High-resolution context manager that automatically records milliseconds."""

    def __init__(self, node: str, round_num: int, metric: str) -> None:
        self.node = node
        self.round_num = round_num
        self.metric = metric
        self.started_at = 0.0

    def __enter__(self) -> "Timer":
        self.started_at = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is None:
            metrics_tracker.record_duration(
                self.node, self.round_num, self.metric, time.perf_counter() - self.started_at
            )
        return False


class BatchTimer:
    """Measure one batch operation and allocate it across its source updates."""

    def __init__(self, receiver: str, participants: Iterable[str], round_num: int) -> None:
        self.receiver = receiver
        self.participants = list(participants)
        self.round_num = round_num
        self.started_at = 0.0

    def __enter__(self) -> "BatchTimer":
        self.started_at = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is None:
            metrics_tracker.record_batch_duration(
                self.receiver, self.participants, self.round_num,
                time.perf_counter() - self.started_at,
            )
        return False


metrics_tracker = MetricsTracker()
