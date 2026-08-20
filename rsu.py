# rsu.py — Road-Side Unit (Cluster Head)
#
# Aggregates proxy model weights from vehicles in its cluster using FedAvg,
# then forwards the cluster average to the central server.
# Broadcasts global updates back to vehicles.
# Uses timeout-based aggregation so out-of-range vehicles don't deadlock.

import time
import threading

from config import (
    RSU_ROUND_TIMEOUT, SECURITY_ENABLED, BATCH_VERIFICATION_ENABLED,
    TRUST_SCORE_ENABLED, TRUST_L2_THRESHOLD, TRUST_MEDIAN_MULTIPLIER,
)
from models import average_weights, filter_trusted_weights
from network import Receiver, send_msg
from crypto_protocol import (
    Authority, CertificatelessSigner, CertificatelessVerifier,
    build_envelope, verify_envelope, encrypt_payload, message_aad,
    signature_from_wire
)
from model_codec import serialize_weights, deserialize_weights
from metrics import Timer, BatchTimer, metrics_tracker


class RSU:
    """Road-Side Unit that aggregates a cluster of vehicles.

    Args:
        name:                RSU identifier (e.g. "Cluster_1").
        port:                TCP port this RSU listens on.
        cluster_ports:       List of vehicle TCP ports in this cluster.
        server_port:         TCP port of the central server.
        topology:            Shared VanetTopology for range checks.
        vehicle_names:       List of vehicle names (parallel to cluster_ports).
        security_authority:  Optional bootstrap Authority (TA/KGC) instance.
        security_identity:   Optional pre-registered CertificatelessSigner.
    """

    def __init__(self, name, port, cluster_ports, server_port,
                 topology=None, vehicle_names=None,
                 security_authority=None, security_identity=None):
        self.name = name
        self.port = port
        self.cluster_ports = cluster_ports
        self.server_port = server_port
        self.topology = topology
        self.vehicle_names = vehicle_names or []
        self.security_authority = security_authority
        self.signer = security_identity
        self.verifier = (
            CertificatelessVerifier(security_authority.P_pub)
            if security_authority is not None else None
        )

        if SECURITY_ENABLED and self.signer is None and self.security_authority is not None:
            with Timer(self.name, 0, "key_generation"):
                self.signer = self.security_authority.register(self.name)

        self.round_buffers = {}
        self.round_reported = {}
        self.completed_rounds = set()
        self._round_timers = {}
        self._lock = threading.Lock()  # protects round_buffers, completed_rounds, _round_timers
        self.receiver = Receiver(self.port, self.on_receive, node_name=self.name)

    def on_receive(self, msg):
        msg_type = msg.get("type") if isinstance(msg, dict) else None
        if msg_type in ("LOCAL_UPDATE", "NO_UPDATE"):
            r = msg.get("round")
            sender = msg.get("sender")
            if not isinstance(r, int) or r < 0:
                return

            should_aggregate = False
            verified_payload = None
            sender_info = None

            # Security verification
            if SECURITY_ENABLED and self.security_authority is not None and self.signer is not None and "sig" in msg:
                t0 = time.perf_counter()
                result = verify_envelope(self.security_authority, self.signer, msg, msg_type)
                ver_duration = time.perf_counter() - t0
                if sender:
                    metrics_tracker.record_duration(sender, r, "signature_verification", ver_duration)

                if result is None:
                    print(f"[{self.name}] [SECURITY] Authentication failed for {msg_type} from {sender} (Round {r}). Dropping.")
                    return
                verified_payload, sender_info = result

            with self._lock:
                if r in self.completed_rounds:
                    return

                if r not in self.round_buffers:
                    self.round_buffers[r] = []
                    self.round_reported[r] = set()
                    timer = threading.Timer(
                        RSU_ROUND_TIMEOUT, self._force_aggregate, args=[r])
                    timer.daemon = True
                    timer.start()
                    self._round_timers[r] = timer

                if sender:
                    self.round_reported[r].add(sender)

                if msg_type == "LOCAL_UPDATE":
                    entry = {
                        "sender": sender,
                        "raw_msg": msg,
                        "verified_payload": verified_payload,
                        "sender_info": sender_info,
                    }
                    self.round_buffers[r].append(entry)

                # If all vehicles reported (either with LOCAL_UPDATE or NO_UPDATE)
                expected_count = max(len(self.cluster_ports), len(self.vehicle_names))
                if len(self.round_reported[r]) >= expected_count:
                    self._cancel_timer_locked(r)
                    should_aggregate = True

            if should_aggregate:
                self.aggregate(r)

        elif msg_type == "GLOBAL_UPDATE":
            # Broadcast global proxy back to cluster vehicles
            for port in self.cluster_ports:
                send_msg(("127.0.0.1", port), msg, sender_name=self.name, round_num=msg.get("round"))

    def aggregate(self, r):
        """FedAvg the received proxy weights and forward to the server."""
        rsu_t0 = time.perf_counter()
        with self._lock:
            if r in self.completed_rounds:
                return
            self.completed_rounds.add(r)
            self._cancel_timer_locked(r)
            data = self.round_buffers.pop(r, [])
            self.round_reported.pop(r, None)

        valid_entries = []  # (sender, weights)
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
                    with BatchTimer(self.name, participants, r):
                        batch_input = [(p, s, info) for p, s, info, _ in batch_items]
                        batch_ok = self.verifier.batch_verify(batch_input)
                else:
                    batch_ok = True

                if batch_ok:
                    for payload, _, _, d in batch_items:
                        try:
                            w = deserialize_weights(payload)
                            valid_entries.append((d.get("sender", "?"), w))
                        except Exception as e:
                            print(f"[{self.name}] Deserialization error: {e}")
                else:
                    # Fallback to individual verification to drop any invalid item
                    print(f"[{self.name}] [SECURITY] Batch verification failed for Round {r}. Falling back to single-verify.")
                    for payload, sig, info, d in batch_items:
                        t_single_0 = time.perf_counter()
                        is_valid = self.verifier.verify(payload, sig, info)
                        dur = time.perf_counter() - t_single_0
                        sender = d.get("sender")
                        if sender:
                            metrics_tracker.record_duration(sender, r, "signature_verification", dur)
                        if is_valid:
                            try:
                                valid_entries.append((sender or "?", deserialize_weights(payload)))
                            except Exception:
                                pass
                        else:
                            print(f"[{self.name}] [SECURITY] Excluded invalid signature from {sender}")
        else:
            # Plaintext fallback
            for d in data:
                raw = d.get("raw_msg", {})
                w = raw.get("weights")
                if isinstance(w, bytes):
                    try:
                        w = deserialize_weights(w)
                    except Exception:
                        pass
                if isinstance(w, dict):
                    valid_entries.append((d.get("sender", "?"), w))

        n_received = len(valid_entries)
        n_expected = max(len(self.cluster_ports), len(self.vehicle_names))

        # Trust score filter (Eq. 9–10)
        if TRUST_SCORE_ENABLED and valid_entries:
            valid_weights, trust_log = filter_trusted_weights(
                valid_entries,
                threshold=TRUST_L2_THRESHOLD,
                median_multiplier=TRUST_MEDIAN_MULTIPLIER,
            )
            for name, deviation, accepted in trust_log:
                tau = 1 if accepted else 0
                print(f"[{self.name}] [TRUST] {name}: ∂={deviation:.4f} → τ={tau}")
                if not accepted:
                    print(f"[{self.name}] [TRUST] Declined communication from {name} (malicious/outlier)")
        else:
            valid_weights = [w for _, w in valid_entries]

        if valid_weights:
            avg_weights = average_weights(valid_weights)
            print(f"[{self.name}] Aggregated {len(valid_weights)}/{n_expected} trusted "
                  f"vehicles (received {n_received}, Round {r}) -> forwarding to Server")

            if SECURITY_ENABLED and self.signer is not None and self.security_authority is not None:
                raw_avg = serialize_weights(avg_weights)
                with Timer(self.name, r, "signature_generation"):
                    sig = self.signer.sign(raw_avg)
                with Timer(self.name, r, "encryption"):
                    aad = message_aad("CLUSTER_UPDATE", self.name, "Server", r)
                    server_info = self.security_authority.public_info("Server")
                    shared_secret = self.signer.shared_secret_for(server_info)
                    ciphertext, nonce, tag = encrypt_payload(shared_secret, raw_avg, aad)
                msg = build_envelope(
                    "CLUSTER_UPDATE", self.signer, "Server", r,
                    sig, ciphertext, nonce, tag
                )
                msg["rsu_port"] = self.port
            else:
                msg = {
                    "type": "CLUSTER_UPDATE",
                    "rsu_port": self.port,
                    "sender": self.name,
                    "round": r,
                    "avg_weights": avg_weights,
                }
            send_msg(("127.0.0.1", self.server_port), msg, sender_name=self.name, round_num=r)
        else:
            print(f"[{self.name}] [!] 0 valid vehicle weights for Round {r}.")

        metrics_tracker.record_duration(
            self.name, r, "rsu_round_execution", time.perf_counter() - rsu_t0
        )

    def _force_aggregate(self, r):
        """Timeout handler: aggregate whatever we have so far."""
        with self._lock:
            if r in self.completed_rounds:
                return
            has_data = r in self.round_buffers and len(self.round_buffers[r]) > 0
            if has_data:
                n = len(self.round_buffers[r])
                print(f"[{self.name}] [!] Timeout! Aggregating {n}/"
                      f"{len(self.cluster_ports)} vehicles for round {r}")

        self.aggregate(r)

    def _cancel_timer_locked(self, r):
        """Cancel a round timer. Must be called with self._lock held."""
        timer = self._round_timers.pop(r, None)
        if timer:
            timer.cancel()

    def start(self):
        self.receiver.start()

    def shutdown(self):
        self.receiver.shutdown()
