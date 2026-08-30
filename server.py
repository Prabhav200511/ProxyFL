# server.py — Central Server for ProxyFL
#
# Receives cluster-aggregated proxy weights from RSUs, applies unweighted
# FedAvg global aggregation (Eq. 8), evaluates the global proxy model on
# held-out attack test data, and broadcasts the result back.

import threading
import os
import time

import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader

from config import (
    TOTAL_ROUNDS, DEVICE, RSU_BASE_PORT, SERVER_ROUND_TIMEOUT,
    SERVER_ROUND_MAX_WAIT, SIMULATION_SEED,
    SECURITY_ENABLED, BATCH_VERIFICATION_ENABLED
)
from data_utils import VANETDataset, get_vanet_scaler
from shared_logger import logger
from network import Receiver, send_msg
from torchvision import datasets, transforms
from models import average_weights, ProxyModel, MNISTProxyModel
from crypto_protocol import (
    Authority, CertificatelessSigner, CertificatelessVerifier,
    build_envelope, decrypt_envelope, encrypt_payload, message_aad,
    signature_from_wire,
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
        topology:            Spatial topology used for coverage reporting.
        cluster_vehicle_names: Assigned vehicle names keyed by RSU name.
    """

    def __init__(self, port, expected_rsus, dataset_type="mnist", total_rounds=TOTAL_ROUNDS,
                 security_authority=None, security_identity=None, topology=None,
                 cluster_vehicle_names=None, security_enabled=None,
                 batch_verification_enabled=None, rsu_directory=None,
                 vanet_scaler=None, random_seed=SIMULATION_SEED, aodv_enabled=False,
                 round_coordinator=None):
        self.port = port
        self.expected_rsus = expected_rsus
        self.dataset_type = dataset_type
        self.total_rounds = total_rounds
        self.aodv_enabled = aodv_enabled
        self.round_coordinator = round_coordinator
        self._shutdown = False
        self.security_authority = security_authority
        self.signer = security_identity
        self.topology = topology
        self.cluster_vehicle_names = cluster_vehicle_names or {}
        if rsu_directory is None:
            names = list(self.cluster_vehicle_names)
            rsu_directory = {
                name: RSU_BASE_PORT + index
                for index, name in enumerate(names)
            }
        self.rsu_directory = dict(rsu_directory)
        self.security_enabled = (
            SECURITY_ENABLED if security_enabled is None else security_enabled)
        self.batch_verification_enabled = (
            BATCH_VERIFICATION_ENABLED
            if batch_verification_enabled is None
            else batch_verification_enabled
        )
        self.verifier = (
            CertificatelessVerifier(security_authority.P_pub)
            if security_authority is not None else None
        )

        if self.security_enabled and self.signer is None and self.security_authority is not None:
            with Timer("Server", 0, "key_generation"):
                self.signer = self.security_authority.register("Server")

        self.round_buffers = {}
        self.round_reported = {}
        self.round_start_times = {}
        self.completed_rounds = set()
        self.rsu_ports = [RSU_BASE_PORT + i for i in range(expected_rsus)]
        self._round_timers = {}
        self._round_deadlines = {}  # round -> hard cap (perf_counter)
        self._lock = threading.Lock()  # protects round_buffers, completed_rounds, rsu_ports, _round_timers
        self.receiver = Receiver(self.port, self.on_receive, node_name="Server")
        if self.round_coordinator is not None:
            self.round_coordinator.add_round_open_callback(
                self._arm_silent_round)

        torch.manual_seed(random_seed)
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
                scaler = vanet_scaler or get_vanet_scaler()
                test_df[VANETDataset.FEATURE_COLS] = scaler.transform(test_df[VANETDataset.FEATURE_COLS])
                self.test_dataset = VANETDataset(test_df)
                self.test_loader = DataLoader(self.test_dataset, batch_size=1000, shuffle=False)
            else:
                self.test_loader = None

    def _build_global_message(self, rsu_name, round_num, weights):
        """Build one RSU-specific global update using the existing protocol."""
        payload = serialize_weights(weights)
        if (self.security_enabled and self.signer is not None
                and self.security_authority is not None):
            with Timer("Server", round_num, "signature_generation"):
                signature = self.signer.sign(payload)
            with Timer("Server", round_num, "encryption"):
                aad = message_aad(
                    "GLOBAL_UPDATE", "Server", rsu_name, round_num)
                recipient_info = self.security_authority.public_info(rsu_name)
                shared_secret = self.signer.shared_secret_for(recipient_info)
                ciphertext, nonce, tag = encrypt_payload(
                    shared_secret, payload, aad)
            return build_envelope(
                "GLOBAL_UPDATE", self.signer, rsu_name, round_num,
                signature, ciphertext, nonce, tag,
            )
        return {
            "type": "GLOBAL_UPDATE",
            "sender": "Server",
            "recipient": rsu_name,
            "round": round_num,
            "global_weights": payload,
        }

    def _broadcast_global(self, round_num, weights, rsu_directory_snapshot):
        """Send one recipient-bound global message to every known RSU."""
        for rsu_name, port in rsu_directory_snapshot.items():
            coordinator = getattr(self, "round_coordinator", None)
            delivered = False
            try:
                msg = self._build_global_message(
                    rsu_name, round_num, weights)
                delivered = send_msg(
                    ("127.0.0.1", port), msg,
                    sender_name="Server", round_num=round_num,
                )
            except Exception as exc:
                print(f"[SERVER] [ERROR] Could not deliver round {round_num} "
                      f"global to {rsu_name}: {exc!r}")
            if not delivered and coordinator is not None:
                coordinator.record_rsu_result(
                    round_num, rsu_name, set())

    def _print_in_range_vehicle_counts(self, r):
        """Record and print assigned-vehicle coverage for every RSU."""
        if self.topology is None or not self.cluster_vehicle_names:
            return {}
        coverage = self.topology.assigned_rsu_coverage(
            self.cluster_vehicle_names)
        print(f"[SERVER] Round {r} in-range vehicle coverage:")
        for rsu_name in self.cluster_vehicle_names:
            counts = coverage[rsu_name]
            metrics_tracker.record_value(
                rsu_name, r, "vehicles_in_range", counts["in_range"])
            metrics_tracker.record_value(
                rsu_name, r, "vehicles_assigned", counts["assigned"])
            print(f"         {rsu_name}: {counts['in_range']}/"
                  f"{counts['assigned']} assigned vehicles in range")
        totals = coverage["total"]
        metrics_tracker.record_value(
            "Server", r, "vehicles_in_range_total", totals["in_range"])
        metrics_tracker.record_value(
            "Server", r, "vehicles_assigned_total", totals["assigned"])
        print(f"         TOTAL: {totals['in_range']}/"
              f"{totals['assigned']} assigned vehicles in range")
        return coverage

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------
    def on_receive(self, msg):
        msg_type = msg.get("type") if isinstance(msg, dict) else None
        if msg_type in ("CLUSTER_UPDATE", "NO_CLUSTER_UPDATE"):
            r = msg.get("round")
            sender = msg.get("sender") or msg.get("rsu_name")
            if not isinstance(r, int) or r < 0 or not sender:
                return
            if (self.cluster_vehicle_names
                    and sender not in self.cluster_vehicle_names):
                return

            should_aggregate = False
            verified_payload = None
            sender_info = None

            # Security verification
            if self.security_enabled:
                if self.security_authority is None or self.signer is None:
                    return
                if "sig" not in msg:
                    return
                t0 = time.perf_counter()
                result = decrypt_envelope(
                    self.security_authority, self.signer, msg, msg_type)
                ver_duration = time.perf_counter() - t0
                if sender:
                    metrics_tracker.record_duration(sender, r, "decryption", ver_duration)

                if result is None:
                    print(f"[SERVER] [SECURITY] Authentication failed for "
                          f"{msg_type} from {sender} (Round {r}). Dropping.")
                    return
                verified_payload, sender_info, signature = result
                if msg_type == "NO_CLUSTER_UPDATE":
                    verification_started = time.perf_counter()
                    is_valid = self.verifier.verify(
                        verified_payload, signature, sender_info)
                    metrics_tracker.record_duration(
                        sender, r, "signature_verification",
                        time.perf_counter() - verification_started,
                    )
                    if not is_valid:
                        return

            with self._lock:
                if r in self.completed_rounds:
                    return

                if r not in self.round_start_times:
                    self.round_start_times[r] = time.perf_counter()

                rsu_port = msg.get("rsu_port")
                if rsu_port and rsu_port not in self.rsu_ports:
                    self.rsu_ports.append(rsu_port)
                if rsu_port:
                    self.rsu_directory[sender] = rsu_port

                if r not in self.round_buffers:
                    self.round_buffers[r] = []
                    self.round_reported[r] = set()
                    self._round_deadlines[r] = (
                        time.perf_counter() + SERVER_ROUND_MAX_WAIT)

                if sender in self.round_reported[r]:
                    return
                if sender:
                    self.round_reported[r].add(sender)
                # Restart the inactivity window so slow clusters still count.
                self._restart_round_timer_locked(r)

                if msg_type == "CLUSTER_UPDATE":
                    entry = {
                        "sender": sender,
                        "raw_msg": msg,
                        "verified_payload": verified_payload,
                        "sender_info": sender_info,
                    }
                    self.round_buffers[r].append(entry)

                if len(self.round_reported[r]) >= self.expected_rsus:
                    self._cancel_timer_locked(r)
                    should_aggregate = True

            if should_aggregate:
                self.aggregate(r)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate_global_model(self, weights):
        if not getattr(self, "test_loader", None):
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
        """Aggregate one round; always broadcast a global and close the round.

        Aggregation runs inside a receiver thread.  An unhandled exception here
        would leave every RSU (and therefore every vehicle) waiting for a
        global update that is never sent, so the round is finished on every
        path -- including the failure path.
        """
        server_t0 = time.perf_counter()
        with self._lock:
            if r in self.completed_rounds:
                return
            self.completed_rounds.add(r)
            self._cancel_timer_locked(r)
            data = self.round_buffers.pop(r, [])
            self.round_reported.pop(r, None)
            rsu_directory_snapshot = dict(self.rsu_directory)
            round_start_t = self.round_start_times.pop(r, server_t0)

        try:
            self._aggregate_round(
                r, data, rsu_directory_snapshot, round_start_t)
        except Exception as exc:
            print(f"[SERVER] [ERROR] Round {r} aggregation failed: {exc!r}. "
                  "Broadcasting the last known global proxy so the round "
                  "still closes.")
            import traceback
            traceback.print_exc()
            try:
                fallback = {
                    key: value.cpu()
                    for key, value in self.model.state_dict().items()
                }
                self._broadcast_global(r, fallback, rsu_directory_snapshot)
                metrics_tracker.record_value(
                    "Server", r, "successful_updates", 0.0)
            except Exception as fallback_exc:
                print(f"[SERVER] [ERROR] Global broadcast fallback failed for "
                      f"round {r}: {fallback_exc!r}")
        finally:
            metrics_tracker.record_duration(
                "Server", r, "server_round_execution",
                time.perf_counter() - server_t0,
            )
            if r >= self.total_rounds:
                training_done_event.set()
            elif getattr(self, "round_coordinator", None) is None:
                self._arm_silent_round(r + 1)

    def _aggregate_round(self, r, data, rsu_directory_snapshot, round_start_t):
        """Verify, FedAvg, evaluate and broadcast one round's cluster models."""
        if not data:
            coverage = self._print_in_range_vehicle_counts(r)
            if (coverage
                    and coverage.get("total", {}).get("in_range") == 0):
                print(f"[SERVER] Round {r}: all RSUs have 0 directly "
                      "in-range assigned vehicles; training continues with "
                      "the unchanged global proxy.")
            print(f"[SERVER] Round {r}: no RSU supplied a model update; "
                  "broadcasting the current global proxy.")
            weights = {
                k: v.cpu() for k, v in self.model.state_dict().items()}
            self._report_global_metrics(
                r, weights, successful_updates=0, total_bytes=0.0,
                round_start_t=round_start_t,
            )
            self._broadcast_global(r, weights, rsu_directory_snapshot)
            return

        self._print_in_range_vehicle_counts(r)

        cluster_weights = []
        participants = [d["sender"] for d in data if d.get("sender")]

        # Batch verification & weight extraction
        if self.security_enabled and self.security_authority is not None and self.verifier is not None:
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
                if self.batch_verification_enabled and len(batch_items) > 1:
                    with BatchTimer("Server", participants, r):
                        batch_input = [(p, s, info) for p, s, info, _ in batch_items]
                        batch_ok = self.verifier.batch_verify(batch_input)
                else:
                    batch_ok = all(
                        self.verifier.verify(payload, sig, info)
                        for payload, sig, info, _ in batch_items
                    )

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
                if not isinstance(w, bytes):
                    continue
                try:
                    w = deserialize_weights(w)
                except Exception:
                    continue
                if isinstance(w, dict):
                    cluster_weights.append(w)

        if not cluster_weights:
            print(f"[SERVER] [!] 0 valid cluster weights for Round {r}.")
            weights = {
                k: v.cpu() for k, v in self.model.state_dict().items()}
            self._report_global_metrics(
                r, weights, successful_updates=0, total_bytes=0.0,
                round_start_t=round_start_t,
            )
            self._broadcast_global(r, weights, rsu_directory_snapshot)
            return

        # Equation (8): central server forms the arithmetic mean of RSU models.
        global_weights = average_weights(cluster_weights)

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
        self._report_global_metrics(
            r, global_weights,
            successful_updates=len(cluster_weights),
            total_bytes=total_bytes,
            round_start_t=round_start_t,
        )

        # Broadcast global proxy to all RSUs
        self._broadcast_global(r, global_weights, rsu_directory_snapshot)

    def _report_global_metrics(
            self, r, weights, successful_updates, total_bytes,
            round_start_t):
        """Evaluate, print, and persist a complete server row for one round."""
        acc, f1, recall = self.evaluate_global_model(weights)
        logger.log_global(r, acc)

        round_wall_clock = time.perf_counter() - round_start_t
        throughput_ups = (
            float(successful_updates) / max(round_wall_clock, 0.001))
        throughput_bps = float(total_bytes) / max(round_wall_clock, 0.001)

        metrics_tracker.record_value(
            "Server", r, "global_proxy_accuracy_pct", acc * 100.0)
        metrics_tracker.record_value("Server", r, "global_proxy_f1", f1)
        metrics_tracker.record_value(
            "Server", r, "global_proxy_recall", recall)
        metrics_tracker.record_value(
            "Server", r, "successful_updates", float(successful_updates))
        metrics_tracker.record_value(
            "Server", r, "throughput_updates_per_sec", throughput_ups)
        metrics_tracker.record_value(
            "Server", r, "throughput_bytes_per_sec", throughput_bps)
        metrics_tracker.record_value(
            "Server", r, "model_payload_bytes_rx", float(total_bytes))

        print(f"\n[SERVER] --- ROUND {r} GLOBAL PROXY METRICS ---")
        print(f"         Test Accuracy : {round(acc * 100, 2)}%")
        print(f"         F1-Score               : {round(f1, 4)}")
        print(f"         Recall                 : {round(recall, 4)}")
        print(f"         Successful Updates     : {successful_updates}")
        print(f"         Model Payload Received : {round(total_bytes, 2)} bytes")
        print(f"         Server Collection Rate : {round(throughput_bps, 2)} B/s")
        print(f"         Update Processing Rate : {round(throughput_ups, 2)} "
              "updates/sec")
        print(f"         Round Wall-Clock       : "
              f"{round(round_wall_clock, 2)}s\n")

    def _force_aggregate(self, r):
        force_t0 = time.perf_counter()
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
                self.round_reported.pop(r, None)
                round_start_t = self.round_start_times.pop(r, force_t0)
                rsu_directory_snapshot = dict(self.rsu_directory)

        if has_data:
            self.aggregate(r)
            return

        try:
            self._aggregate_round(
                r, [], rsu_directory_snapshot, round_start_t)
        except Exception as exc:
            print(f"[SERVER] [ERROR] Empty-round broadcast failed for round "
                  f"{r}: {exc!r}")
        finally:
            metrics_tracker.record_duration(
                "Server", r, "server_round_execution",
                time.perf_counter() - force_t0,
            )
            if r >= self.total_rounds:
                training_done_event.set()
            elif getattr(self, "round_coordinator", None) is None:
                self._arm_silent_round(r + 1)

    def _cancel_timer_locked(self, r):
        """Cancel a round timer. Must be called with self._lock held."""
        timer = self._round_timers.pop(r, None)
        if timer:
            timer.cancel()
        self._round_deadlines.pop(r, None)

    def _restart_round_timer_locked(self, r):
        """(Re)arm the inactivity window for round *r*, honouring the hard cap.

        Must be called with self._lock held.
        """
        timer = self._round_timers.pop(r, None)
        if timer:
            timer.cancel()
        hard_cap = self._round_deadlines.get(r)
        remaining = SERVER_ROUND_TIMEOUT
        if hard_cap is not None:
            remaining = min(
                remaining, max(hard_cap - time.perf_counter(), 0.0))
        timer = threading.Timer(remaining, self._force_aggregate, args=[r])
        timer.daemon = True
        timer.start()
        self._round_timers[r] = timer

    # ------------------------------------------------------------------
    # Startup / Shutdown
    # ------------------------------------------------------------------
    def start(self):
        print(f"[SERVER] Listening on port {self.port} | device={DEVICE}")
        self.receiver.start()
        if getattr(self, "round_coordinator", None) is None:
            self._arm_silent_round(1)

    def _arm_silent_round(self, r):
        """Start the AODV no-traffic cap without fabricating an RSU report."""
        if not getattr(self, "aodv_enabled", False) or r > self.total_rounds:
            return
        with self._lock:
            if getattr(self, "_shutdown", False) or r in self.completed_rounds or r in self._round_timers:
                return
            self.round_buffers.setdefault(r, [])
            self.round_reported.setdefault(r, set())
            self.round_start_times.setdefault(r, time.perf_counter())
            self._round_deadlines[r] = time.perf_counter() + SERVER_ROUND_MAX_WAIT
            timer = threading.Timer(SERVER_ROUND_MAX_WAIT, self._force_aggregate, args=[r])
            timer.daemon = True
            self._round_timers[r] = timer
            timer.start()

    def shutdown(self):
        """Close the listening socket so the port is freed."""
        with self._lock:
            self._shutdown = True
            for r in list(self._round_timers):
                self._cancel_timer_locked(r)
        try:
            self.receiver.shutdown()
        except Exception:
            pass
