# rsu.py — Road-Side Unit (Cluster Head)
#
# Aggregates proxy model weights from vehicles in its cluster using FedAvg,
# then forwards the cluster average to the central server.
# Broadcasts global updates back to vehicles.
# Uses timeout-based aggregation so out-of-range vehicles don't deadlock.

import time
import threading

from config import (
    RSU_ROUND_TIMEOUT, RSU_ROUND_MAX_WAIT,
    SECURITY_ENABLED, BATCH_VERIFICATION_ENABLED,
    TRUST_SCORE_ENABLED, TRUST_L2_THRESHOLD, TRUST_MEDIAN_MULTIPLIER,
)
from models import average_weights, filter_trusted_weights
from network import Receiver, send_msg
from crypto_protocol import (
    Authority, CertificatelessSigner, CertificatelessVerifier,
    build_envelope, decrypt_envelope, verify_envelope, encrypt_payload, message_aad,
    signature_from_wire
)
from model_codec import serialize_weights, deserialize_weights
from metrics import Timer, BatchTimer, metrics_tracker
from vanet_channel import WirelessLink


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
                 security_authority=None, security_identity=None,
                 security_enabled=None, batch_verification_enabled=None,
                 initial_global_weights=None):
        self.name = name
        self.port = port
        self.cluster_ports = cluster_ports
        self.server_port = server_port
        self.topology = topology
        self.vehicle_names = vehicle_names or []
        self.global_reference_weights = (
            {
                key: value.detach().cpu().clone()
                for key, value in initial_global_weights.items()
            }
            if initial_global_weights is not None else None
        )
        self.security_authority = security_authority
        self.signer = security_identity
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
            with Timer(self.name, 0, "key_generation"):
                self.signer = self.security_authority.register(self.name)

        self.round_buffers = {}
        self.round_reported = {}
        self.completed_rounds = set()
        self._round_timers = {}
        self._round_deadlines = {}  # round -> hard cap (perf_counter) for collection
        self._lock = threading.Lock()  # protects round_buffers, completed_rounds, _round_timers
        self.receiver = Receiver(self.port, self.on_receive, node_name=self.name)

    def _decode_server_global(self, msg):
        """Authenticate and decode a Server-to-RSU global update."""
        if msg.get("sender") != "Server" or msg.get("recipient") != self.name:
            return None
        if self.security_enabled:
            if self.security_authority is None or self.signer is None:
                return None
            round_num = msg.get("round", 0)
            with Timer(self.name, round_num, "decryption"):
                result = decrypt_envelope(
                    self.security_authority, self.signer, msg,
                    "GLOBAL_UPDATE")
            if result is None:
                return None
            payload, sender_info, signature = result
            verifier = CertificatelessVerifier(
                self.security_authority.P_pub)
            with Timer(self.name, round_num, "signature_verification"):
                if not verifier.verify(payload, signature, sender_info):
                    return None
        else:
            payload = msg.get("global_weights")
            if not isinstance(payload, bytes):
                return None
        try:
            return deserialize_weights(payload)
        except Exception:
            return None

    def _build_vehicle_global(self, vehicle_name, round_num, weights):
        """Build one vehicle-specific RSU-to-vehicle global update."""
        payload = serialize_weights(weights)
        if (self.security_enabled and self.signer is not None
                and self.security_authority is not None):
            with Timer(self.name, round_num, "signature_generation"):
                signature = self.signer.sign(payload)
            with Timer(self.name, round_num, "encryption"):
                aad = message_aad(
                    "GLOBAL_UPDATE", self.name, vehicle_name, round_num)
                recipient_info = self.security_authority.public_info(
                    vehicle_name)
                shared_secret = self.signer.shared_secret_for(recipient_info)
                ciphertext, nonce, tag = encrypt_payload(
                    shared_secret, payload, aad)
            return build_envelope(
                "GLOBAL_UPDATE", self.signer, vehicle_name, round_num,
                signature, ciphertext, nonce, tag,
            )
        return {
            "type": "GLOBAL_UPDATE",
            "sender": self.name,
            "recipient": vehicle_name,
            "round": round_num,
            "global_weights": payload,
        }

    def on_receive(self, msg):
        msg_type = msg.get("type") if isinstance(msg, dict) else None
        if msg_type in ("LOCAL_UPDATE", "NO_UPDATE"):
            r = msg.get("round")
            sender = msg.get("sender")
            if not isinstance(r, int) or r < 0 or not sender:
                return
            if self.vehicle_names and sender not in self.vehicle_names:
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
                result = decrypt_envelope(self.security_authority, self.signer, msg, msg_type)
                ver_duration = time.perf_counter() - t0
                if sender:
                    metrics_tracker.record_duration(sender, r, "decryption", ver_duration)

                if result is None:
                    print(f"[{self.name}] [SECURITY] Authentication failed for {msg_type} from {sender} (Round {r}). Dropping.")
                    return
                verified_payload, sender_info, signature = result
                # NO_UPDATE changes the round barrier immediately, so verify
                # this control message before counting it as a report.
                if msg_type == "NO_UPDATE":
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

                if r not in self.round_buffers:
                    self.round_buffers[r] = []
                    self.round_reported[r] = set()
                    self._round_deadlines[r] = (
                        time.perf_counter() + RSU_ROUND_MAX_WAIT)

                if sender in self.round_reported[r]:
                    return
                self.round_reported[r].add(sender)
                # Restart the inactivity window: stragglers queued behind
                # TRAINING_SEMAPHORE must not be silently excluded.
                self._restart_round_timer_locked(r)

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
            weights = self._decode_server_global(msg)
            if weights is None:
                print(f"[{self.name}] [SECURITY] Dropped invalid GLOBAL_UPDATE")
                return
            round_num = msg.get("round")
            self.global_reference_weights = {
                key: value.detach().cpu().clone()
                for key, value in weights.items()
            }
            # Re-authenticate the global proxy separately for each vehicle.
            # One vehicle's failure must not deprive the rest of the cluster.
            for index, port in enumerate(self.cluster_ports):
                if index >= len(self.vehicle_names):
                    continue
                vehicle_name = self.vehicle_names[index]
                try:
                    wireless_link = None
                    if self.topology is not None:
                        distance = self.topology.get_distance_to_rsu(
                            vehicle_name, self.name)
                        if distance != float("inf"):
                            wireless_link = WirelessLink("rsu2v", distance)
                    vehicle_msg = self._build_vehicle_global(
                        vehicle_name, round_num, weights)
                    send_msg(
                        ("127.0.0.1", port), vehicle_msg,
                        sender_name=self.name, round_num=round_num,
                        wireless_link=wireless_link,
                    )
                except Exception as exc:
                    print(f"[{self.name}] [ERROR] Could not deliver round "
                          f"{round_num} global to {vehicle_name}: {exc!r}")

    def aggregate(self, r):
        """FedAvg the received proxy weights and forward to the server.

        Once the round is claimed it MUST produce exactly one downstream
        message.  Aggregation runs inside a receiver thread, so an unhandled
        exception here used to abort the round silently: the server never
        learned the cluster had reported and every vehicle in it stalled until
        its own failsafe timeout.
        """
        rsu_t0 = time.perf_counter()
        with self._lock:
            if r in self.completed_rounds:
                return
            self.completed_rounds.add(r)
            self._cancel_timer_locked(r)
            data = self.round_buffers.pop(r, [])
            self.round_reported.pop(r, None)

        try:
            self._aggregate_round(r, data)
        except Exception as exc:
            print(f"[{self.name}] [ERROR] Round {r} aggregation failed: "
                  f"{exc!r}. Reporting NO_CLUSTER_UPDATE so the server round "
                  "can still close.")
            import traceback
            traceback.print_exc()
            try:
                self._send_no_cluster_update(r)
            except Exception as fallback_exc:
                print(f"[{self.name}] [ERROR] NO_CLUSTER_UPDATE fallback "
                      f"failed for round {r}: {fallback_exc!r}")
        finally:
            metrics_tracker.record_duration(
                self.name, r, "rsu_round_execution",
                time.perf_counter() - rsu_t0,
            )

    def _aggregate_round(self, r, data):
        """Verify, trust-filter, FedAvg and forward one round's cluster data."""
        valid_entries = []  # (sender, weights)
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
                    with BatchTimer(self.name, participants, r):
                        batch_input = [(p, s, info) for p, s, info, _ in batch_items]
                        batch_ok = self.verifier.batch_verify(batch_input)
                else:
                    batch_ok = all(
                        self.verifier.verify(payload, sig, info)
                        for payload, sig, info, _ in batch_items
                    )

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
                if not isinstance(w, bytes):
                    continue
                try:
                    w = deserialize_weights(w)
                except Exception:
                    continue
                if isinstance(w, dict):
                    valid_entries.append((d.get("sender", "?"), w))

        n_received = len(valid_entries)
        n_expected = max(len(self.cluster_ports), len(self.vehicle_names))

        # Trust score filter (Eq. 9–10)
        if TRUST_SCORE_ENABLED and valid_entries:
            valid_weights, trust_log = filter_trusted_weights(
                valid_entries,
                reference_weights=self.global_reference_weights,
                threshold=TRUST_L2_THRESHOLD,
                median_multiplier=TRUST_MEDIAN_MULTIPLIER,
            )
            for name, deviation, accepted in trust_log:
                tau = 1 if accepted else 0
                print(f"[{self.name}] [TRUST] {name}: deviation={deviation:.4f} -> tau={tau}")
                if not accepted:
                    print(f"[{self.name}] [TRUST] Declined communication from {name} (malicious/outlier)")
        else:
            valid_weights = [w for _, w in valid_entries]

        if valid_weights:
            avg_weights = average_weights(valid_weights)
            print(f"[{self.name}] Aggregated {len(valid_weights)}/{n_expected} trusted "
                  f"vehicles (received {n_received}, Round {r}) -> forwarding to Server")

            if self.security_enabled and self.signer is not None and self.security_authority is not None:
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
                    "avg_weights": serialize_weights(avg_weights),
                }
            send_msg(("127.0.0.1", self.server_port), msg, sender_name=self.name, round_num=r)
        else:
            print(f"[{self.name}] [!] 0 valid vehicle weights for Round {r}.")
            self._send_no_cluster_update(r)

    def _send_no_cluster_update(self, r):
        """Authenticated control-plane report: this cluster has no model."""
        if (self.security_enabled and self.signer is not None
                and self.security_authority is not None):
            payload = b""
            with Timer(self.name, r, "signature_generation"):
                signature = self.signer.sign(payload)
            with Timer(self.name, r, "encryption"):
                aad = message_aad(
                    "NO_CLUSTER_UPDATE", self.name, "Server", r)
                server_info = self.security_authority.public_info("Server")
                shared_secret = self.signer.shared_secret_for(server_info)
                ciphertext, nonce, tag = encrypt_payload(
                    shared_secret, payload, aad)
            msg = build_envelope(
                "NO_CLUSTER_UPDATE", self.signer, "Server", r,
                signature, ciphertext, nonce, tag,
            )
            msg["rsu_port"] = self.port
        else:
            msg = {
                "type": "NO_CLUSTER_UPDATE",
                "rsu_port": self.port,
                "sender": self.name,
                "recipient": "Server",
                "round": r,
            }
        return send_msg(
            ("127.0.0.1", self.server_port), msg,
            sender_name=self.name, round_num=r,
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
        self._round_deadlines.pop(r, None)

    def _restart_round_timer_locked(self, r):
        """(Re)arm the inactivity window for round *r*, honouring the hard cap.

        Must be called with self._lock held.
        """
        timer = self._round_timers.pop(r, None)
        if timer:
            timer.cancel()
        hard_cap = self._round_deadlines.get(r)
        remaining = RSU_ROUND_TIMEOUT
        if hard_cap is not None:
            remaining = min(
                remaining, max(hard_cap - time.perf_counter(), 0.0))
        timer = threading.Timer(remaining, self._force_aggregate, args=[r])
        timer.daemon = True
        timer.start()
        self._round_timers[r] = timer

    def start(self):
        self.receiver.start()

    def shutdown(self):
        self.receiver.shutdown()
