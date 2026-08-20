# server.py — Central Server for ProxyFL
#
# Receives cluster-aggregated proxy weights from RSUs, applies
# JSD-weighted global aggregation, evaluates the global proxy model
# on held-out attack test data, and broadcasts the result back.

import threading
import os
import time

import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader

from config import (
    TOTAL_ROUNDS, DEVICE, RSU_BASE_PORT, SERVER_ROUND_TIMEOUT,
    SECURITY_ENABLED, BATCH_VERIFICATION_ENABLED
)
from data_utils import VANETDataset, get_vanet_scaler
from shared_logger import logger
from network import Receiver, send_msg
from torchvision import datasets, transforms
from models import jsd_weighted_average, ProxyModel, MNISTProxyModel
from crypto_protocol import (
    Authority, CertificatelessSigner, CertificatelessVerifier,
    verify_envelope, signature_from_wire
)
from model_codec import serialize_weights, deserialize_weights
from metrics import Timer, BatchTimer, metrics_tracker


training_done_event = threading.Event()


class Server:
    """Central aggregation server.

    Args:
        port:                TCP port to listen on.
        expected_rsus:       Number of RSU cluster updates expected per round.
        dataset_type:        "mnist" or "vanet".
        total_rounds:        Total communication rounds.
        security_authority:  Optional bootstrap Authority (TA/KGC) instance.
        security_identity:   Optional pre-registered CertificatelessSigner.
    """

    def __init__(self, port, expected_rsus, dataset_type="mnist", total_rounds=TOTAL_ROUNDS,
                 security_authority=None, security_identity=None):
        self.port = port
        self.expected_rsus = expected_rsus
        self.dataset_type = dataset_type
        self.total_rounds = total_rounds
        self.security_authority = security_authority
        self.signer = security_identity
        self.verifier = (
            CertificatelessVerifier(security_authority.P_pub)
            if security_authority is not None else None
        )

        if SECURITY_ENABLED and self.signer is None and self.security_authority is not None:
            with Timer("Server", 0, "key_generation"):
                self.signer = self.security_authority.register("Server")

        self.round_buffers = {}
        self.round_start_times = {}
        self.completed_rounds = set()
        self.rsu_ports = [RSU_BASE_PORT + i for i in range(expected_rsus)]
        self._round_timers = {}
        self._lock = threading.Lock()  # protects round_buffers, completed_rounds, rsu_ports, _round_timers
        self.receiver = Receiver(self.port, self.on_receive, node_name="Server")

        torch.manual_seed(42)
        if self.dataset_type == "mnist":
            self.model = MNISTProxyModel().to(DEVICE)
            print("[SERVER] Loading MNIST test dataset...")
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
            self.test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        else:
            self.model = ProxyModel().to(DEVICE)
            print("[SERVER] Loading attack test datasets...")
            test_files = ['attack1_test.csv', 'attack2_test.csv', 'attack3_test.csv', 'attack4_test.csv', 'attack5_test.csv']
            dfs = [pd.read_csv(f) for f in test_files if os.path.exists(f)]
            if dfs:
                test_df = pd.concat(dfs, ignore_index=True)
                scaler = get_vanet_scaler()
                test_df[VANETDataset.FEATURE_COLS] = scaler.transform(test_df[VANETDataset.FEATURE_COLS])
                self.test_dataset = VANETDataset(test_df)
                self.test_loader = DataLoader(self.test_dataset, batch_size=1000, shuffle=False)
            else:
                self.test_loader = None

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------
    def on_receive(self, msg):
        msg_type = msg.get("type") if isinstance(msg, dict) else None
        if msg_type == "CLUSTER_UPDATE":
            r = msg.get("round")
            sender = msg.get("sender") or msg.get("rsu_name")
            if not isinstance(r, int) or r < 0:
                return

            should_aggregate = False
            verified_payload = None
            sender_info = None

            # Security verification
            if SECURITY_ENABLED and self.security_authority is not None and self.signer is not None and "sig" in msg:
                t0 = time.perf_counter()
                result = verify_envelope(self.security_authority, self.signer, msg, "CLUSTER_UPDATE")
                ver_duration = time.perf_counter() - t0
                if sender:
                    metrics_tracker.record_duration(sender, r, "signature_verification", ver_duration)

                if result is None:
                    print(f"[SERVER] [SECURITY] Authentication failed for CLUSTER_UPDATE from {sender} (Round {r}). Dropping.")
                    return
                verified_payload, sender_info = result

            with self._lock:
                if r in self.completed_rounds:
                    return

                if r not in self.round_start_times:
                    self.round_start_times[r] = time.perf_counter()

                rsu_port = msg.get("rsu_port")
                if rsu_port and rsu_port not in self.rsu_ports:
                    self.rsu_ports.append(rsu_port)

                if r not in self.round_buffers:
                    self.round_buffers[r] = []
                    timer = threading.Timer(
                        SERVER_ROUND_TIMEOUT, self._force_aggregate, args=[r])
                    timer.daemon = True
                    timer.start()
                    self._round_timers[r] = timer

                entry = {
                    "sender": sender,
                    "raw_msg": msg,
                    "verified_payload": verified_payload,
                    "sender_info": sender_info,
                }
                self.round_buffers[r].append(entry)

                if len(self.round_buffers[r]) >= self.expected_rsus:
                    self._cancel_timer_locked(r)
                    should_aggregate = True

            if should_aggregate:
                self.aggregate(r)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate_global_model(self, weights):
        if not self.test_loader:
            return 0.0, 0.0, 0.0

        self.model.load_state_dict(weights)
        self.model.eval()

        all_preds, all_targets = [], []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = self.model(data)
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted',
                      zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted',
                              zero_division=0)
        return acc, f1, recall

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def aggregate(self, r):
        server_t0 = time.perf_counter()
        with self._lock:
            if r in self.completed_rounds:
                return
            self.completed_rounds.add(r)
            self._cancel_timer_locked(r)
            data = self.round_buffers.pop(r, [])
            rsu_ports_snapshot = list(self.rsu_ports)
            round_start_t = self.round_start_times.pop(r, server_t0)

        if not data:
            return

        cluster_weights = []
        participants = [d["sender"] for d in data if d.get("sender")]

        # Batch verification & weight extraction
        if SECURITY_ENABLED and self.security_authority is not None and self.verifier is not None:
            batch_items = []
            for d in data:
                raw_msg = d.get("raw_msg", {})
                payload = d.get("verified_payload")
                s_info = d.get("sender_info")
                sig_wire = raw_msg.get("sig")
                if payload is not None and s_info is not None and sig_wire is not None:
                    try:
                        sig = signature_from_wire(sig_wire)
                        batch_items.append((payload, sig, s_info, d))
                    except Exception:
                        pass

            if batch_items:
                if BATCH_VERIFICATION_ENABLED and len(batch_items) > 1:
                    with BatchTimer("Server", participants, r):
                        batch_input = [(p, s, info) for p, s, info, _ in batch_items]
                        batch_ok = self.verifier.batch_verify(batch_input)
                else:
                    batch_ok = True

                if batch_ok:
                    for payload, _, _, _ in batch_items:
                        try:
                            w = deserialize_weights(payload)
                            cluster_weights.append(w)
                        except Exception as e:
                            print(f"[SERVER] Deserialization error: {e}")
                else:
                    print(f"[SERVER] [SECURITY] Batch verification failed for Round {r}. Falling back to single-verify.")
                    for payload, sig, info, d in batch_items:
                        t_single_0 = time.perf_counter()
                        is_valid = self.verifier.verify(payload, sig, info)
                        dur = time.perf_counter() - t_single_0
                        sender = d.get("sender")
                        if sender:
                            metrics_tracker.record_duration(sender, r, "signature_verification", dur)
                        if is_valid:
                            try:
                                cluster_weights.append(deserialize_weights(payload))
                            except Exception:
                                pass
                        else:
                            print(f"[SERVER] [SECURITY] Excluded invalid signature from {sender}")
        else:
            for d in data:
                raw = d.get("raw_msg", {})
                w = raw.get("avg_weights")
                if isinstance(w, bytes):
                    try:
                        w = deserialize_weights(w)
                    except Exception:
                        pass
                if isinstance(w, dict):
                    cluster_weights.append(w)

        if not cluster_weights:
            print(f"[SERVER] [!] 0 valid cluster weights for Round {r}.")
            return

        # JSD-weighted average
        global_weights, divergences = jsd_weighted_average(
            cluster_weights, self.model.state_dict(), alpha=2.0)

        for i, div in enumerate(divergences):
            logger.log_jsd(r, f"Cluster_{i + 1}", div)

        # Evaluate
        acc, f1, recall = self.evaluate_global_model(global_weights)
        logger.log_global(r, acc)

        round_wall_clock = time.perf_counter() - round_start_t
        # Legacy updates/sec (kept for CSV compatibility)
        throughput_ups = len(cluster_weights) / max(round_wall_clock, 0.001)

        # Strict throughput: total bytes delivered to the server this round / wall-clock
        total_bytes = 0.0
        for d in data:
            payload = d.get("verified_payload")
            if isinstance(payload, (bytes, bytearray)):
                total_bytes += len(payload)
                continue
            raw_msg = d.get("raw_msg", {})
            if not isinstance(raw_msg, dict):
                continue
            blob = raw_msg.get("ciphertext")
            if isinstance(blob, (bytes, bytearray)):
                total_bytes += len(blob)
                continue
            weights = raw_msg.get("avg_weights")
            if isinstance(weights, (bytes, bytearray)):
                total_bytes += len(weights)
            elif isinstance(weights, dict):
                try:
                    total_bytes += len(serialize_weights(weights))
                except Exception:
                    pass
        if total_bytes <= 0:
            for w in cluster_weights:
                try:
                    total_bytes += len(serialize_weights(w))
                except Exception:
                    pass
        throughput_bps = total_bytes / max(round_wall_clock, 0.001)

        metrics_tracker.record_value("Server", r, "global_proxy_accuracy_pct", acc * 100.0)
        metrics_tracker.record_value("Server", r, "successful_updates", float(len(cluster_weights)))
        metrics_tracker.record_value("Server", r, "throughput_updates_per_sec", throughput_ups)
        metrics_tracker.record_value("Server", r, "throughput_bytes_per_sec", throughput_bps)
        metrics_tracker.record_value("Server", r, "bytes_rx", total_bytes)
        metrics_tracker.record_duration("Server", r, "server_round_execution", time.perf_counter() - server_t0)

        print(f"\n[SERVER] --- ROUND {r} GLOBAL PROXY METRICS ---")
        print(f"         Test Accuracy : {round(acc * 100, 2)}%")
        print(f"         F1-Score      : {round(f1, 4)}")
        print(f"         Recall        : {round(recall, 4)}")
        print(f"         Throughput    : {round(throughput_bps, 2)} B/s "
              f"({round(throughput_ups, 2)} updates/sec, wall-clock {round(round_wall_clock, 2)}s)\n")

        # Broadcast global proxy to all RSUs
        msg = {
            "type": "GLOBAL_UPDATE",
            "round": r,
            "global_weights": {k: v.cpu() for k, v in global_weights.items()},
        }
        for p in rsu_ports_snapshot:
            send_msg(("127.0.0.1", p), msg, sender_name="Server", round_num=r)

        if r >= self.total_rounds:
            training_done_event.set()

    def _force_aggregate(self, r):
        with self._lock:
            if r in self.completed_rounds:
                return
            has_data = r in self.round_buffers and len(self.round_buffers[r]) > 0
            if has_data:
                n = len(self.round_buffers[r])
                print(f"[SERVER] [!] Timeout! Aggregating {n}/"
                      f"{self.expected_rsus} RSUs for round {r}")
            else:
                print(f"[SERVER] [!] Timeout! 0 RSUs reported for round {r}. Broadcasting current global proxy.")
                self.completed_rounds.add(r)
                self._cancel_timer_locked(r)
                self.round_buffers.pop(r, None)
                rsu_ports_snapshot = list(self.rsu_ports)

        if has_data:
            self.aggregate(r)
        else:
            msg = {
                "type": "GLOBAL_UPDATE",
                "round": r,
                "global_weights": {k: v.cpu() for k, v in self.model.state_dict().items()},
            }
            for p in rsu_ports_snapshot:
                send_msg(("127.0.0.1", p), msg, sender_name="Server", round_num=r)
            if r >= self.total_rounds:
                training_done_event.set()

    def _cancel_timer_locked(self, r):
        """Cancel a round timer. Must be called with self._lock held."""
        timer = self._round_timers.pop(r, None)
        if timer:
            timer.cancel()

    # ------------------------------------------------------------------
    # Startup / Shutdown
    # ------------------------------------------------------------------
    def start(self):
        print(f"[SERVER] Listening on port {self.port} | device={DEVICE}")
        self.receiver.start()

    def shutdown(self):
        """Close the listening socket so the port is freed."""
        try:
            self.receiver.shutdown()
        except Exception:
            pass
