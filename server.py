"""Central ProxyFL aggregation server with verified RSU updates."""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import DEVICE, RSU_BASE_PORT, SECURITY_ENABLED, SERVER_ROUND_TIMEOUT, TOTAL_ROUNDS
from crypto_protocol import (
    Authority, CertificatelessSigner, CertificatelessVerifier, SecurityError,
    decrypt_payload, message_aad, parse_envelope,
)
from data_utils import VANETDataset, get_vanet_scaler
from metrics import BatchTimer, Timer, metrics_tracker
from model_codec import deserialize_weights, serialize_weights
from models import MNISTProxyModel, ProxyModel, jsd_weighted_average
from network import Receiver, send_msg
from shared_logger import logger


training_done_event = threading.Event()


class Server:
    """Accept only batch-verified cluster models and broadcast the global proxy."""

    def __init__(
        self, port, expected_rsus, dataset_type="mnist", total_rounds=TOTAL_ROUNDS,
        rsu_names=None, security_authority: Authority | None = None,
        security_identity: CertificatelessSigner | None = None,
    ):
        self.port = port
        self.expected_rsus = expected_rsus
        self.dataset_type = dataset_type
        self.total_rounds = total_rounds
        self.expected_rsu_names = set(rsu_names or [])
        self.rsu_ports = [RSU_BASE_PORT + index for index in range(expected_rsus)]
        self.security_authority = security_authority
        self.security_identity = security_identity
        self.security_enabled = bool(
            SECURITY_ENABLED and security_authority is not None and security_identity is not None
        )
        self.verifier = (CertificatelessVerifier(security_authority.P_pub)
                         if self.security_enabled else None)
        self.round_buffers: Dict[int, Dict[str, Dict[str, Any]]] = {}
        self.completed_rounds = set()
        self._round_timers: Dict[int, threading.Timer] = {}
        self._lock = threading.Lock()
        self.receiver = Receiver(self.port, self.on_receive, metric_node="Server")

        torch.manual_seed(42)
        if dataset_type == "mnist":
            self.model = MNISTProxyModel().to(DEVICE)
            print("[SERVER] Loading MNIST test dataset...")
            transform = transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
            ])
            test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
            self.test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        else:
            self.model = ProxyModel().to(DEVICE)
            print("[SERVER] Loading attack test datasets...")
            test_files = [f"attack{index}_test.csv" for index in range(1, 6)]
            frames = [pd.read_csv(path) for path in test_files if os.path.exists(path)]
            if frames:
                test_df = pd.concat(frames, ignore_index=True)
                scaler = get_vanet_scaler()
                test_df[VANETDataset.FEATURE_COLS] = scaler.transform(test_df[VANETDataset.FEATURE_COLS])
                self.test_loader = DataLoader(VANETDataset(test_df), batch_size=1000, shuffle=False)
            else:
                self.test_loader = None

    def _decode_cluster_update(self, msg: Dict[str, Any]) -> Dict[str, Any] | None:
        sender, r = msg.get("sender"), msg.get("round")
        if not isinstance(sender, str) or not isinstance(r, int):
            print("[SERVER] [SECURITY] Rejected cluster update with invalid sender/round")
            return None
        if self.expected_rsu_names and sender not in self.expected_rsu_names:
            print(f"[SERVER] [SECURITY] Rejected update from unknown RSU {sender}")
            return None
        if self.security_enabled:
            try:
                parsed = parse_envelope(self.security_authority, self.security_identity, msg, "CLUSTER_UPDATE")
                with Timer("Server", r, "decryption"):
                    payload = decrypt_payload(
                        self.security_identity.shared_secret_for(parsed.sender_info),
                        msg["ciphertext"], msg["nonce"], msg["tag"], parsed.aad,
                    )
                return {
                    "sender": sender, "payload": payload, "signature": parsed.signature,
                    "public_info": parsed.sender_info,
                }
            except (KeyError, TypeError, SecurityError) as exc:
                print(f"[SERVER] [SECURITY] Rejected {sender} round {r}: {exc}")
                return None
        if msg.get("recipient") != "Server" or not isinstance(msg.get("payload"), bytes):
            print(f"[SERVER] Rejected malformed baseline update from {sender}")
            return None
        return {"sender": sender, "payload": msg["payload"]}

    def on_receive(self, msg: Dict[str, Any]) -> None:
        if msg.get("type") != "CLUSTER_UPDATE":
            return
        pending = self._decode_cluster_update(msg)
        if pending is None:
            return
        r, sender = msg["round"], pending["sender"]
        should_aggregate = False
        with self._lock:
            if r in self.completed_rounds:
                return
            if r not in self.round_buffers:
                self.round_buffers[r] = {}
                timer = threading.Timer(SERVER_ROUND_TIMEOUT, self._force_aggregate, args=[r])
                timer.daemon = True
                timer.start()
                self._round_timers[r] = timer
            if sender in self.round_buffers[r]:
                print(f"[SERVER] [SECURITY] Dropped duplicate cluster update from {sender}, round {r}")
                return
            self.round_buffers[r][sender] = pending
            if len(self.round_buffers[r]) >= self.expected_rsus:
                self._cancel_timer_locked(r)
                should_aggregate = True
        if should_aggregate:
            self.aggregate(r)

    def _verified_records(self, r: int, records: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        if not self.security_enabled:
            return records
        participants = [record["sender"] for record in records]
        batch = [(record["payload"], record["signature"], record["public_info"])
                 for record in records]
        with BatchTimer("Server", participants, r):
            batch_ok = self.verifier.batch_verify(batch)
        if batch_ok:
            return records
        valid_records = []
        for record in records:
            with Timer("Server", r, "signature_verification"):
                valid = self.verifier.verify(
                    record["payload"], record["signature"], record["public_info"])
            if valid:
                valid_records.append(record)
            else:
                print(f"[SERVER] [SECURITY] Dropped invalid signature from "
                      f"{record['sender']} in round {r}")
        return valid_records

    def evaluate_global_model(self, weights):
        if self.test_loader is None:
            return 0.0, 0.0, 0.0
        self.model.load_state_dict(weights)
        self.model.eval()
        predictions, targets = [], []
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data.to(DEVICE))
                predictions.extend(output.argmax(dim=1).cpu().numpy())
                targets.extend(target.cpu().numpy())
        return (
            accuracy_score(targets, predictions),
            f1_score(targets, predictions, average="weighted", zero_division=0),
            recall_score(targets, predictions, average="weighted", zero_division=0),
        )

    def _broadcast_global(self, r: int, weights) -> None:
        outbound = {
            "type": "GLOBAL_UPDATE", "round": r,
            "global_payload": serialize_weights(weights),
        }
        for rsu_port in list(self.rsu_ports):
            send_msg(("127.0.0.1", rsu_port), outbound, metric_node="Server", round_num=r)

    def aggregate(self, r: int) -> None:
        started_at = time.perf_counter()
        try:
            with self._lock:
                if r in self.completed_rounds:
                    return
                self.completed_rounds.add(r)
                self._cancel_timer_locked(r)
                records = list(self.round_buffers.pop(r, {}).values())
            verified = self._verified_records(r, records)
            cluster_weights = []
            for record in verified:
                try:
                    cluster_weights.append(deserialize_weights(record["payload"]))
                except (RuntimeError, ValueError) as exc:
                    print(f"[SERVER] [SECURITY] Dropped undecodable cluster update from "
                          f"{record['sender']}: {exc}")
            if not cluster_weights:
                print(f"[SERVER] [!] No verified cluster updates for round {r}; reusing global proxy")
                self._broadcast_global(r, self.model.state_dict())
                return

            global_weights, divergences = jsd_weighted_average(
                cluster_weights, self.model.state_dict(), alpha=2.0)
            for index, divergence in enumerate(divergences):
                logger.log_jsd(r, f"Cluster_{index + 1}", divergence)
            accuracy, f1, recall = self.evaluate_global_model(global_weights)
            logger.log_global(r, accuracy)
            metrics_tracker.record_value("Server", r, "successful_updates", len(cluster_weights))
            elapsed_seconds = max(time.perf_counter() - started_at, 1e-9)
            metrics_tracker.record_value(
                "Server", r, "throughput_updates_per_sec", len(cluster_weights) / elapsed_seconds)
            print(f"\n[SERVER] --- ROUND {r} GLOBAL PROXY METRICS ---")
            print(f"         Test Accuracy : {accuracy * 100:.2f}%")
            print(f"         F1-Score      : {f1:.4f}")
            print(f"         Recall        : {recall:.4f}\n")
            self._broadcast_global(r, global_weights)
        finally:
            metrics_tracker.record_duration(
                "Server", r, "server_round_execution", time.perf_counter() - started_at)
            if r >= self.total_rounds:
                training_done_event.set()

    def _force_aggregate(self, r: int) -> None:
        with self._lock:
            if r in self.completed_rounds:
                return
            count = len(self.round_buffers.get(r, {}))
        if count:
            print(f"[SERVER] [!] Timeout! Aggregating {count}/{self.expected_rsus} RSUs for round {r}")
            self.aggregate(r)

    def _cancel_timer_locked(self, r: int) -> None:
        timer = self._round_timers.pop(r, None)
        if timer:
            timer.cancel()

    def start(self) -> None:
        print(f"[SERVER] Listening on port {self.port} | device={DEVICE}")
        self.receiver.start()

    def shutdown(self) -> None:
        self.receiver.shutdown()
