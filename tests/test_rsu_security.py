"""Ensure failed authentication cannot enter the RSU aggregation buffer."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from crypto_protocol import Authority, build_envelope, encrypt_payload, message_aad
from rsu import RSU


class RSUSecurityTests(unittest.TestCase):
    def test_corrupt_update_never_reaches_average_weights(self) -> None:
        authority = Authority()
        device = authority.register("C1_D1")
        rsu_identity = authority.register("Cluster_1")
        authority.register("Server")
        rsu = RSU(
            "Cluster_1", 0, [6000], 8000, vehicle_names=["C1_D1"],
            security_authority=authority, security_identity=rsu_identity,
        )
        try:
            payload = b"tensor-only state would be here"
            aad = message_aad("LOCAL_UPDATE", "C1_D1", "Cluster_1", 1)
            signature = device.sign(payload)
            ciphertext, nonce, tag = encrypt_payload(
                device.shared_secret_for(rsu_identity.get_public_info()), payload, aad)
            corrupt = build_envelope(
                "LOCAL_UPDATE", device, "Cluster_1", 1, signature,
                bytes([ciphertext[0] ^ 1]) + ciphertext[1:], nonce, tag,
            )
            with patch("rsu.average_weights") as average_weights:
                rsu.on_receive(corrupt)
                average_weights.assert_not_called()
            self.assertNotIn(1, rsu.round_buffers)
        finally:
            rsu.shutdown()

    def test_authenticated_no_update_completes_an_empty_cluster(self) -> None:
        authority = Authority()
        device = authority.register("C1_D1")
        rsu_identity = authority.register("Cluster_1")
        authority.register("Server")
        rsu = RSU(
            "Cluster_1", 0, [6000], 1, vehicle_names=["C1_D1"],
            security_authority=authority, security_identity=rsu_identity,
        )
        try:
            payload = b"NO_UPDATE"
            aad = message_aad("NO_UPDATE", "C1_D1", "Cluster_1", 1)
            signature = device.sign(payload)
            ciphertext, nonce, tag = encrypt_payload(
                device.shared_secret_for(rsu_identity.get_public_info()), payload, aad)
            rsu.on_receive(build_envelope(
                "NO_UPDATE", device, "Cluster_1", 1, signature, ciphertext, nonce, tag))
            self.assertIn(1, rsu.completed_rounds)
        finally:
            rsu.shutdown()


if __name__ == "__main__":
    unittest.main(verbosity=2)
