"""Road-side unit aggregation with authenticated local-update handling."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict

from config import RSU_ROUND_TIMEOUT, SECURITY_ENABLED
from crypto_protocol import (
    Authority, CertificatelessSigner, CertificatelessVerifier, SecurityError,
    build_envelope, decrypt_payload, encrypt_payload, message_aad, parse_envelope,
)
from metrics import BatchTimer, Timer, metrics_tracker
from model_codec import deserialize_weights, serialize_weights
from models import average_weights
from network import Receiver, send_msg


class RSU:
    """Aggregate only decrypted and certificate-less verified device updates."""

    def __init__(
        self, name, port, cluster_ports, server_port, topology=None, vehicle_names=None,
        security_authority: Authority | None = None,
        security_identity: CertificatelessSigner | None = None,
    ):
        self.name = name
        self.port = port
        self.cluster_ports = list(cluster_ports)
        self.server_port = server_port
        self.topology = topology
        self.vehicle_names = list(vehicle_names or [])
        self.expected_senders = set(self.vehicle_names)
        self.security_authority = security_authority
        self.security_identity = security_identity
        self.security_enabled = bool(
            SECURITY_ENABLED and security_authority is not None and security_identity is not None
        )
        self.verifier = (CertificatelessVerifier(security_authority.P_pub)
                         if self.security_enabled else None)
        # round -> {sender -> pending authenticated (but not yet batch-verified) update}
        self.round_buffers: Dict[int, Dict[str, Dict[str, Any]]] = {}
        self.no_update_senders: Dict[int, set[str]] = {}
        self.completed_rounds = set()
        self._round_timers: Dict[int, threading.Timer] = {}
        self._lock = threading.Lock()
        self.receiver = Receiver(self.port, self.on_receive, metric_node=self.name)

    def _decode_local_update(self, msg: Dict[str, Any]) -> Dict[str, Any] | None:
        """Perform cheap link checks + AEAD decryption; defer signature to batch."""
        sender = msg.get("sender")
        r = msg.get("round")
        if not isinstance(sender, str) or not isinstance(r, int) or sender not in self.expected_senders:
            print(f"[{self.name}] [SECURITY] Rejected local update with invalid sender/round")
            return None
        if self.security_enabled:
            try:
                parsed = parse_envelope(self.security_authority, self.security_identity, msg, "LOCAL_UPDATE")
                with Timer(self.name, r, "decryption"):
                    payload = decrypt_payload(
                        self.security_identity.shared_secret_for(parsed.sender_info),
                        msg["ciphertext"], msg["nonce"], msg["tag"], parsed.aad,
                    )
                return {
                    "sender": sender, "payload": payload, "signature": parsed.signature,
                    "public_info": parsed.sender_info,
                }
            except (KeyError, TypeError, SecurityError) as exc:
                print(f"[{self.name}] [SECURITY] Rejected {sender} round {r}: {exc}")
                return None
        if msg.get("recipient") != self.name or not isinstance(msg.get("payload"), bytes):
            print(f"[{self.name}] Rejected malformed baseline update from {sender}")
            return None
        return {"sender": sender, "payload": msg["payload"]}

    def _accept_no_update(self, msg: Dict[str, Any]) -> bool:
        """Verify an explicit no-upload signal without counting it as a model."""
        sender, r = msg.get("sender"), msg.get("round")
        if not isinstance(sender, str) or not isinstance(r, int) or sender not in self.expected_senders:
            return False
        if self.security_enabled:
            try:
                parsed = parse_envelope(self.security_authority, self.security_identity, msg, "NO_UPDATE")
                with Timer(self.name, r, "decryption"):
                    payload = decrypt_payload(
                        self.security_identity.shared_secret_for(parsed.sender_info),
                        msg["ciphertext"], msg["nonce"], msg["tag"], parsed.aad,
                    )
                with Timer(self.name, r, "signature_verification"):
                    valid = self.verifier.verify(payload, parsed.signature, parsed.sender_info)
                if valid and payload == b"NO_UPDATE":
                    return True
            except (KeyError, TypeError, SecurityError):
                pass
        elif msg.get("recipient") == self.name and msg.get("payload") == b"NO_UPDATE":
            return True
        print(f"[{self.name}] [SECURITY] Rejected no-update signal from {sender}, round {r}")
        return False

    def _ensure_timer_locked(self, r: int) -> None:
        if r not in self.round_buffers:
            self.round_buffers[r] = {}
        if r not in self._round_timers:
            timer = threading.Timer(RSU_ROUND_TIMEOUT, self._force_aggregate, args=[r])
            timer.daemon = True
            timer.start()
            self._round_timers[r] = timer

    def on_receive(self, msg: Dict[str, Any]) -> None:
        message_type = msg.get("type")
        if message_type == "LOCAL_UPDATE":
            pending = self._decode_local_update(msg)
            if pending is None:
                return
            r, sender = msg["round"], pending["sender"]
            should_aggregate = False
            with self._lock:
                if r in self.completed_rounds:
                    return
                self._ensure_timer_locked(r)
                if sender in self.round_buffers[r]:
                    print(f"[{self.name}] [SECURITY] Dropped duplicate update from {sender} for round {r}")
                    return
                self.round_buffers[r][sender] = pending
                if (len(self.round_buffers[r])
                        + len(self.no_update_senders.get(r, set())) >= len(self.cluster_ports)):
                    self._cancel_timer_locked(r)
                    should_aggregate = True
            if should_aggregate:
                self.aggregate(r)

        elif message_type == "NO_UPDATE":
            if not self._accept_no_update(msg):
                return
            r, sender = msg["round"], msg["sender"]
            should_aggregate = False
            with self._lock:
                if r in self.completed_rounds:
                    return
                self._ensure_timer_locked(r)
                if sender in self.round_buffers[r] or sender in self.no_update_senders.setdefault(r, set()):
                    print(f"[{self.name}] [SECURITY] Dropped duplicate response from {sender} for round {r}")
                    return
                self.no_update_senders[r].add(sender)
                if (len(self.round_buffers[r]) + len(self.no_update_senders[r])
                        >= len(self.cluster_ports)):
                    self._cancel_timer_locked(r)
                    should_aggregate = True
            if should_aggregate:
                self.aggregate(r)

        elif message_type == "GLOBAL_UPDATE":
            # Server-provided global model is an opaque tensor-only payload. It
            # is broadcast to every device, including one that missed its upload.
            for port in self.cluster_ports:
                send_msg(("127.0.0.1", port), msg, metric_node=self.name,
                         round_num=msg.get("round"))

    def _verified_records(self, r: int, records: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        if not self.security_enabled:
            return records
        participants = [record["sender"] for record in records]
        batch = [(record["payload"], record["signature"], record["public_info"])
                 for record in records]
        with BatchTimer(self.name, participants, r):
            batch_ok = self.verifier.batch_verify(batch)
        if batch_ok:
            return records

        # A failed batch proves at least one invalid signature. Verify only then
        # to localize and exclude individual bad senders rather than lose a round.
        valid_records = []
        for record in records:
            with Timer(self.name, r, "signature_verification"):
                valid = self.verifier.verify(
                    record["payload"], record["signature"], record["public_info"])
            if valid:
                valid_records.append(record)
            else:
                print(f"[{self.name}] [SECURITY] Dropped invalid signature from "
                      f"{record['sender']} in round {r}")
        return valid_records

    def aggregate(self, r: int) -> None:
        """Batch-verify, FedAvg verified models, then authenticate to the server."""
        started_at = time.perf_counter()
        try:
            with self._lock:
                if r in self.completed_rounds:
                    return
                self.completed_rounds.add(r)
                self._cancel_timer_locked(r)
                records = list(self.round_buffers.pop(r, {}).values())
                self.no_update_senders.pop(r, None)
            verified = self._verified_records(r, records)
            weights = []
            accepted_senders = []
            for record in verified:
                try:
                    weights.append(deserialize_weights(record["payload"]))
                    accepted_senders.append(record["sender"])
                except (RuntimeError, ValueError) as exc:
                    print(f"[{self.name}] [SECURITY] Dropped undecodable update from "
                          f"{record['sender']}: {exc}")
            if not weights:
                print(f"[{self.name}] [!] No verified updates to aggregate for round {r}")
                self._forward_cluster_payload(r, b"")
                return

            avg_weights = average_weights(weights)
            metrics_tracker.record_value(self.name, r, "successful_updates", len(weights))
            print(f"[{self.name}] Aggregated {len(weights)}/{len(self.cluster_ports)} verified vehicles "
                  f"(Round {r}) -> forwarding to Server")
            self._forward_cluster_payload(r, serialize_weights(avg_weights))
        finally:
            metrics_tracker.record_duration(
                self.name, r, "rsu_round_execution", time.perf_counter() - started_at)

    def _forward_cluster_payload(self, r: int, payload: bytes) -> None:
        """Send an authenticated model or an authenticated empty-cluster status."""
        if self.security_enabled:
            server_public = self.security_authority.public_info("Server")
            aad = message_aad("CLUSTER_UPDATE", self.name, "Server", r)
            with Timer(self.name, r, "signature_generation"):
                signature = self.security_identity.sign(payload)
            with Timer(self.name, r, "encryption"):
                ciphertext, nonce, tag = encrypt_payload(
                    self.security_identity.shared_secret_for(server_public), payload, aad)
            outbound = build_envelope(
                "CLUSTER_UPDATE", self.security_identity, "Server", r,
                signature, ciphertext, nonce, tag,
            )
        else:
            outbound = {
                "type": "CLUSTER_UPDATE", "sender": self.name, "recipient": "Server",
                "round": r, "payload": payload,
            }
        send_msg(("127.0.0.1", self.server_port), outbound,
                 metric_node=self.name, round_num=r)

    def _force_aggregate(self, r: int) -> None:
        with self._lock:
            if r in self.completed_rounds:
                return
            count = len(self.round_buffers.get(r, {}))
            no_update_count = len(self.no_update_senders.get(r, set()))
        if count or no_update_count:
            print(f"[{self.name}] [!] Timeout! Aggregating {count}/{len(self.cluster_ports)} "
                  f"vehicles for round {r}")
            self.aggregate(r)

    def _cancel_timer_locked(self, r: int) -> None:
        timer = self._round_timers.pop(r, None)
        if timer:
            timer.cancel()

    def start(self) -> None:
        self.receiver.start()

    def shutdown(self) -> None:
        self.receiver.shutdown()
