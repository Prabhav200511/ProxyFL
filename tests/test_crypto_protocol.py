"""Focused regression tests for the certificateless authentication layer."""

from __future__ import annotations

import copy
import time
import unittest

from crypto_protocol import (
    Authority, CertificatelessVerifier, build_envelope, encrypt_payload,
    message_aad, point_to_bytes, q, verify_envelope,
)


class CertificatelessProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.authority = Authority()
        self.device = self.authority.register("C1_D1")
        self.rsu = self.authority.register("Cluster_1")
        self.verifier = CertificatelessVerifier(self.authority.P_pub)

    def _envelope(self, payload: bytes = b"proxy-state") -> dict:
        aad = message_aad("LOCAL_UPDATE", self.device.name, self.rsu.name, 1)
        signature = self.device.sign(payload)
        ciphertext, nonce, tag = encrypt_payload(
            self.device.shared_secret_for(self.rsu.get_public_info()), payload, aad)
        return build_envelope(
            "LOCAL_UPDATE", self.device, self.rsu.name, 1,
            signature, ciphertext, nonce, tag,
        )

    def test_pairwise_secret_is_symmetric(self) -> None:
        self.assertEqual(
            self.device.shared_secret_for(self.rsu.get_public_info()),
            self.rsu.shared_secret_for(self.device.get_public_info()),
        )

    def test_valid_envelope_decrypts_and_verifies(self) -> None:
        result = verify_envelope(self.authority, self.rsu, self._envelope(), "LOCAL_UPDATE")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], b"proxy-state")
        self.assertEqual(result[1]["name"], self.device.name)

    def test_ciphertext_signature_and_claimed_key_tampering_are_rejected(self) -> None:
        ciphertext_tampered = self._envelope()
        ciphertext = bytearray(ciphertext_tampered["ciphertext"])
        ciphertext[0] ^= 1
        ciphertext_tampered["ciphertext"] = bytes(ciphertext)
        self.assertIsNone(verify_envelope(
            self.authority, self.rsu, ciphertext_tampered, "LOCAL_UPDATE"))

        signature_tampered = self._envelope()
        signature_tampered["sig"]["eta"] = (signature_tampered["sig"]["eta"] % (q - 1)) + 1
        self.assertIsNone(verify_envelope(
            self.authority, self.rsu, signature_tampered, "LOCAL_UPDATE"))

        impostor = self.authority.register("C1_D2")
        key_tampered = self._envelope()
        key_tampered["pk"] = copy.deepcopy(key_tampered["pk"])
        key_tampered["pk"]["Q"] = point_to_bytes(impostor.get_public_info()["Q"])
        self.assertIsNone(verify_envelope(
            self.authority, self.rsu, key_tampered, "LOCAL_UPDATE"))

    def test_batch_verification_and_benchmark(self) -> None:
        signers = [self.authority.register(f"C2_D{index}") for index in range(12)]
        batch = []
        for index, signer in enumerate(signers):
            payload = f"state-{index}".encode()
            batch.append((payload, signer.sign(payload), signer.get_public_info()))
        self.assertTrue(self.verifier.batch_verify(batch))

        tampered = list(batch)
        tampered[5] = (b"altered", tampered[5][1], tampered[5][2])
        self.assertFalse(self.verifier.batch_verify(tampered))

        started_at = time.perf_counter()
        for payload, signature, public_info in batch:
            self.assertTrue(self.verifier.verify(payload, signature, public_info))
        single_seconds = time.perf_counter() - started_at
        started_at = time.perf_counter()
        self.assertTrue(self.verifier.batch_verify(batch))
        batch_seconds = time.perf_counter() - started_at
        self.assertGreater(single_seconds, 0.0)
        self.assertGreater(batch_seconds, 0.0)
        print(
            f"verification benchmark: single={single_seconds * 1000:.2f}ms, "
            f"batch={batch_seconds * 1000:.2f}ms, "
            f"speedup={single_seconds / batch_seconds:.2f}x"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
