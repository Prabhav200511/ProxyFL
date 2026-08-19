"""Certificateless authentication for ProxyFL's simulated VANET links.

The implementation uses raw P-256 point arithmetic from PyCryptodome rather
than an ECDSA wrapper. MIRACL Core was the requested first choice, but its
interactive generated Python package is not portable in this project; this is
the permitted raw-EC PyCryptodome fallback. Only wire-safe bytes and scalar
values leave this module -- no ECC objects are deserialized from the network.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Dict, Iterable, Mapping, Tuple

from Crypto.Cipher import AES
from Crypto.PublicKey import ECC
from Crypto.PublicKey.ECC import EccPoint


CURVE_NAME = "P-256"
_curve_key = ECC.generate(curve=CURVE_NAME)
P = _curve_key._curve.G
q = int(_curve_key._curve.order)
POINT_BYTES = 32


class SecurityError(ValueError):
    """Raised when a security envelope is malformed or fails authentication."""


def _nonzero_scalar() -> int:
    return secrets.randbelow(q - 1) + 1


def point_to_bytes(point: EccPoint) -> bytes:
    """Encode an affine P-256 point without relying on pickle or ``str``."""
    return (int(point.x).to_bytes(POINT_BYTES, "big")
            + int(point.y).to_bytes(POINT_BYTES, "big"))


def point_from_bytes(data: bytes) -> EccPoint:
    if not isinstance(data, bytes) or len(data) != 2 * POINT_BYTES:
        raise SecurityError("invalid P-256 point encoding")
    try:
        return EccPoint(
            int.from_bytes(data[:POINT_BYTES], "big"),
            int.from_bytes(data[POINT_BYTES:], "big"),
            curve=CURVE_NAME,
        )
    except (TypeError, ValueError) as exc:
        raise SecurityError("point is not on P-256") from exc


def _hash_part(value: Any) -> bytes:
    if isinstance(value, bytes):
        raw = value
    elif isinstance(value, str):
        raw = value.encode("utf-8")
    elif isinstance(value, int):
        raw = value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big")
    elif isinstance(value, EccPoint):
        raw = point_to_bytes(value)
    elif isinstance(value, tuple):
        raw = b"".join(_hash_part(item) for item in value)
    else:
        raise TypeError(f"cannot hash protocol value of type {type(value)!r}")
    return len(raw).to_bytes(4, "big") + raw


def hash_to_scalar(domain: bytes, *args: Any) -> int:
    """Domain-separated SHA-256 hash-to-scalar for H0/H1/H2/H3."""
    digest = sha256(domain + b"".join(_hash_part(arg) for arg in args)).digest()
    return int.from_bytes(digest, "big") % q


def _aid_key(aid: Tuple[EccPoint, bytes]) -> bytes:
    return point_to_bytes(aid[0]) + aid[1]


def aid_to_wire(aid: Tuple[EccPoint, bytes]) -> Dict[str, bytes]:
    return {"point": point_to_bytes(aid[0]), "token": aid[1]}


def aid_from_wire(data: Mapping[str, Any]) -> Tuple[EccPoint, bytes]:
    try:
        token = data["token"]
        if not isinstance(token, bytes) or len(token) != 32:
            raise SecurityError("invalid pseudo-identity token")
        return point_from_bytes(data["point"]), token
    except (KeyError, TypeError) as exc:
        raise SecurityError("malformed pseudo-identity") from exc


def public_info_to_wire(info: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "aid": aid_to_wire(info["aid"]),
        "Q": point_to_bytes(info["Q"]),
        "U": point_to_bytes(info["U"]),
        "X": point_to_bytes(info["X"]),
    }


def public_info_from_wire(data: Mapping[str, Any], name: str = "") -> Dict[str, Any]:
    try:
        return {
            "name": name,
            "aid": aid_from_wire(data["aid"]),
            "Q": point_from_bytes(data["Q"]),
            "U": point_from_bytes(data["U"]),
            "X": point_from_bytes(data["X"]),
        }
    except (KeyError, TypeError) as exc:
        raise SecurityError("malformed public key") from exc


def signature_to_wire(signature: Tuple[int, EccPoint]) -> Dict[str, Any]:
    eta, point = signature
    return {"eta": eta, "R": point_to_bytes(point)}


def signature_from_wire(data: Mapping[str, Any]) -> Tuple[int, EccPoint]:
    try:
        eta = data["eta"]
        if not isinstance(eta, int) or not 0 < eta < q:
            raise SecurityError("signature scalar is outside the curve order")
        return eta, point_from_bytes(data["R"])
    except (KeyError, TypeError) as exc:
        raise SecurityError("malformed signature") from exc


def message_aad(message_type: str, sender: str, recipient: str, round_num: int) -> bytes:
    """Bind routing metadata to AES-GCM so it cannot be relabelled in transit."""
    if not isinstance(round_num, int) or round_num < 0:
        raise SecurityError("invalid message round")
    return json.dumps(
        {"recipient": recipient, "round": round_num, "sender": sender, "type": message_type},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def derive_shared_secret(
    private_key: Tuple[int, int], peer_public: Mapping[str, Any], p_pub: EccPoint
) -> bytes:
    """Derive the corrected, symmetric ECDH-style pairwise secret.

    ``x_i X_j + w_i (U_j + H1(AID_j,U_j,P_pub) P_pub)`` equals the same
    expression with i and j swapped. The source image's direct use of a
    public-key pair was not well typed.
    """
    w_i, x_i = private_key
    aid = peer_public["aid"]
    alpha = hash_to_scalar(b"H1", aid, peer_public["U"], p_pub)
    kgc_component = peer_public["U"] + alpha * p_pub
    secret_point = x_i * peer_public["X"] + w_i * kgc_component
    return sha256(b"ProxyFL/psi/v1" + point_to_bytes(secret_point)).digest()


def encrypt_payload(shared_secret: bytes, payload: bytes, aad: bytes = b"") -> Tuple[bytes, bytes, bytes]:
    """AES-256-GCM encryption; the returned tag authenticates the payload."""
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    key = sha256(b"ProxyFL/AES-GCM/v1" + shared_secret).digest()
    cipher = AES.new(key, AES.MODE_GCM)
    cipher.update(aad)
    ciphertext, tag = cipher.encrypt_and_digest(payload)
    return ciphertext, cipher.nonce, tag


def decrypt_payload(
    shared_secret: bytes, ciphertext: bytes, nonce: bytes, tag: bytes, aad: bytes = b""
) -> bytes:
    key = sha256(b"ProxyFL/AES-GCM/v1" + shared_secret).digest()
    try:
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        cipher.update(aad)
        return cipher.decrypt_and_verify(ciphertext, tag)
    except (TypeError, ValueError) as exc:
        raise SecurityError("AES-GCM authentication failed") from exc


class Authority:
    """Combined TA/KGC provisioned once, before simulation threads start."""

    def __init__(self) -> None:
        self.t = _nonzero_scalar()
        self.T_pub = self.t * P
        self.s = _nonzero_scalar()
        self.P_pub = self.s * P
        self._identities: Dict[str, CertificatelessSigner] = {}
        self._public_by_aid: Dict[bytes, Dict[str, Any]] = {}

    def generate_pseudo_identity(self, real_id: str) -> Tuple[EccPoint, bytes]:
        k_i = _nonzero_scalar()
        aid_point = k_i * P
        mask = sha256(
            b"H0" + point_to_bytes(self.t * aid_point) + point_to_bytes(self.T_pub)
        ).digest()
        real_id_digest = sha256(real_id.encode("utf-8")).digest()
        return aid_point, bytes(left ^ right for left, right in zip(real_id_digest, mask))

    def extract_partial_private_key(self, aid: Tuple[EccPoint, bytes]) -> Tuple[int, EccPoint]:
        u_i = _nonzero_scalar()
        U_i = u_i * P
        alpha = hash_to_scalar(b"H1", aid, U_i, self.P_pub)
        return (u_i + alpha * self.s) % q, U_i

    def register(self, name: str) -> "CertificatelessSigner":
        if name in self._identities:
            raise ValueError(f"identity {name!r} is already registered")
        signer = CertificatelessSigner(self, name)
        public_info = signer.get_public_info()
        self._identities[name] = signer
        self._public_by_aid[_aid_key(public_info["aid"])] = public_info
        return signer

    def identity(self, name: str) -> "CertificatelessSigner":
        return self._identities[name]

    def public_info(self, name: str) -> Dict[str, Any]:
        return self._identities[name].get_public_info()

    def resolve_public_info(self, sender: str, aid_wire: Mapping[str, Any]) -> Dict[str, Any]:
        aid = aid_from_wire(aid_wire)
        info = self._public_by_aid.get(_aid_key(aid))
        if info is None or info["name"] != sender:
            raise SecurityError("sender does not own the claimed pseudo-identity")
        return info


class CertificatelessSigner:
    """Vehicle/RSU/server identity with full certificateless key material."""

    def __init__(self, authority: Authority, name: str) -> None:
        self.authority = authority
        self.name = name
        self.aid = authority.generate_pseudo_identity(name)
        self.w_i, self.U_i = authority.extract_partial_private_key(self.aid)
        self.x_i = _nonzero_scalar()
        self.X_i = self.x_i * P
        self.beta_i = hash_to_scalar(b"H2", self.aid, self.X_i)
        self.Q_i = self.U_i + self.beta_i * self.X_i

        alpha_i = hash_to_scalar(b"H1", self.aid, self.U_i, authority.P_pub)
        if self.w_i * P != self.U_i + alpha_i * authority.P_pub:
            raise SecurityError("KGC partial private-key validation failed")

    @property
    def private_key(self) -> Tuple[int, int]:
        return self.w_i, self.x_i

    def get_public_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "aid": self.aid,
            "Q": self.Q_i,
            "U": self.U_i,
            "X": self.X_i,
        }

    def shared_secret_for(self, peer_public: Mapping[str, Any]) -> bytes:
        return derive_shared_secret(self.private_key, peer_public, self.authority.P_pub)

    def sign(self, message: bytes) -> Tuple[int, EccPoint]:
        r_i = _nonzero_scalar()
        R_i = r_i * P
        gamma = hash_to_scalar(b"H3", self.aid, message, self.Q_i, self.U_i, R_i)
        eta = (r_i + gamma * (self.w_i + self.beta_i * self.x_i)) % q
        return eta, R_i


class CertificatelessVerifier:
    """Verifier used by RSUs and the central server."""

    def __init__(self, p_pub: EccPoint) -> None:
        self.P_pub = p_pub

    def verify(
        self, message: bytes, signature: Tuple[int, EccPoint], public_info: Mapping[str, Any]
    ) -> bool:
        try:
            eta, R_i = signature
            if not isinstance(eta, int) or not 0 < eta < q:
                return False
            aid = public_info["aid"]
            Q_i, U_i = public_info["Q"], public_info["U"]
            alpha = hash_to_scalar(b"H1", aid, U_i, self.P_pub)
            gamma = hash_to_scalar(b"H3", aid, message, Q_i, U_i, R_i)
            return eta * P == R_i + gamma * (Q_i + alpha * self.P_pub)
        except (KeyError, TypeError, ValueError):
            return False

    def batch_verify(
        self, batch: Iterable[Tuple[bytes, Tuple[int, EccPoint], Mapping[str, Any]]]
    ) -> bool:
        """Random-coefficient batch verification from Equation (13)."""
        sum_eta = 0
        right = None
        count = 0
        try:
            for message, signature, public_info in batch:
                eta, R_i = signature
                if not isinstance(eta, int) or not 0 < eta < q:
                    return False
                y_i = _nonzero_scalar()
                aid, Q_i, U_i = public_info["aid"], public_info["Q"], public_info["U"]
                alpha = hash_to_scalar(b"H1", aid, U_i, self.P_pub)
                gamma = hash_to_scalar(b"H3", aid, message, Q_i, U_i, R_i)
                sum_eta = (sum_eta + y_i * eta) % q
                term = y_i * R_i + ((y_i * gamma) % q) * (Q_i + alpha * self.P_pub)
                right = term if right is None else right + term
                count += 1
        except (KeyError, TypeError, ValueError):
            return False
        return count == 0 or sum_eta * P == right


@dataclass(frozen=True)
class ParsedEnvelope:
    sender_info: Dict[str, Any]
    signature: Tuple[int, EccPoint]
    aad: bytes


def build_envelope(
    message_type: str,
    sender: CertificatelessSigner,
    recipient: str,
    round_num: int,
    signature: Tuple[int, EccPoint],
    ciphertext: bytes,
    nonce: bytes,
    tag: bytes,
) -> Dict[str, Any]:
    """Construct the JSON-safe part of a signed and encrypted update."""
    return {
        "type": message_type,
        "sender": sender.name,
        "recipient": recipient,
        "round": round_num,
        "aid": aid_to_wire(sender.aid),
        # The receiver resolves the authoritative copy from AID and compares
        # this claimed public material before accepting it for verification.
        "pk": public_info_to_wire(sender.get_public_info()),
        "sig": signature_to_wire(signature),
        "ciphertext": ciphertext,
        "nonce": nonce,
        "tag": tag,
    }


def parse_envelope(
    authority: Authority, receiver: CertificatelessSigner, msg: Mapping[str, Any], expected_type: str
) -> ParsedEnvelope:
    try:
        if msg["type"] != expected_type or msg["recipient"] != receiver.name:
            raise SecurityError("message type or recipient does not match this link")
        sender = msg["sender"]
        round_num = msg["round"]
        if not isinstance(sender, str):
            raise SecurityError("invalid sender")
        sender_info = authority.resolve_public_info(sender, msg["aid"])
        claimed_public = public_info_from_wire(msg["pk"], name=sender)
        if public_info_to_wire(claimed_public) != public_info_to_wire(sender_info):
            raise SecurityError("claimed public key does not match TA/KGC registry")
        signature = signature_from_wire(msg["sig"])
        aad = message_aad(expected_type, sender, receiver.name, round_num)
        return ParsedEnvelope(sender_info=sender_info, signature=signature, aad=aad)
    except (KeyError, TypeError) as exc:
        raise SecurityError("malformed security envelope") from exc


def verify_envelope(
    authority: Authority, receiver: CertificatelessSigner, msg: Mapping[str, Any], expected_type: str
) -> Tuple[bytes, Dict[str, Any]] | None:
    """Decrypt and verify one envelope; return ``None`` for any forgery/corruption."""
    try:
        parsed = parse_envelope(authority, receiver, msg, expected_type)
        payload = decrypt_payload(
            receiver.shared_secret_for(parsed.sender_info),
            msg["ciphertext"], msg["nonce"], msg["tag"], parsed.aad,
        )
        verified = CertificatelessVerifier(authority.P_pub).verify(
            payload, parsed.signature, parsed.sender_info
        )
        return (payload, parsed.sender_info) if verified else None
    except (KeyError, TypeError, SecurityError):
        return None
