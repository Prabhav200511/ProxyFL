"""Certificateless authentication for ProxyFL's simulated VANET links.

All cryptographic operations use the vendored MIRACL Core NIST256 source:
its generated Python modules supply elliptic-curve arithmetic, while a narrow
MIRACL C bridge supplies SHA-256 and AES-256-GCM.
"""

from __future__ import annotations

import ctypes
import json
import secrets
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# MIRACL Core Python (configured NIST256 package under miracl_python/)
_MIRACL_ROOT = Path(__file__).resolve().parent / "miracl_python"
_miracl_path = str(_MIRACL_ROOT)
_miracl_path_added = _miracl_path not in sys.path
if _miracl_path_added:
    sys.path.insert(0, str(_MIRACL_ROOT))
try:
    from nist256 import big as miracl_big  # noqa: E402
    from nist256.ecp import ECp, generator  # noqa: E402
    from nist256 import curve as _curve  # noqa: E402
finally:
    if _miracl_path_added:
        sys.path.remove(_miracl_path)

P: ECp = generator()
q: int = int(_curve.r)
POINT_BYTES = int(_curve.EFS)  # 32 for NIST256

_MIRACL_BRIDGE = Path(__file__).resolve().parent / "crypto_protocol" / "miracl_core.dll"
_bridge = ctypes.CDLL(str(_MIRACL_BRIDGE)) if _MIRACL_BRIDGE.is_file() else None

# True only when crypto_protocol/miracl_core.dll has been built (see
# crypto_protocol/build_miracl_bridge.bat).  Elliptic-curve arithmetic always
# comes from the vendored MIRACL Core Python modules; SHA-256 and AES-256-GCM
# come from MIRACL only when this flag is True, and otherwise fall back to
# hashlib / the `cryptography` package.  Reported crypto timings differ between
# the two backends, so the active backend is part of every measurement.
MIRACL_BRIDGE_AVAILABLE = _bridge is not None
MIRACL_SYMMETRIC_BACKEND = (
    "MIRACL Core C (hash.c/aes.c/gcm.c)" if MIRACL_BRIDGE_AVAILABLE
    else "hashlib SHA-256 + cryptography AES-256-GCM (MIRACL bridge not built)"
)
_UBytePointer = ctypes.POINTER(ctypes.c_ubyte)
if _bridge is not None:
    _bridge.proxyfl_miracl_sha256.argtypes = [
        _UBytePointer, ctypes.c_int, _UBytePointer]
    _bridge.proxyfl_miracl_sha256.restype = None
    _bridge.proxyfl_miracl_gcm_encrypt.argtypes = [
        _UBytePointer, ctypes.c_int, _UBytePointer, ctypes.c_int,
        _UBytePointer, ctypes.c_int, _UBytePointer, ctypes.c_int,
        _UBytePointer, _UBytePointer,
    ]
    _bridge.proxyfl_miracl_gcm_encrypt.restype = ctypes.c_int
    _bridge.proxyfl_miracl_gcm_decrypt.argtypes = [
        _UBytePointer, ctypes.c_int, _UBytePointer, ctypes.c_int,
        _UBytePointer, ctypes.c_int, _UBytePointer, ctypes.c_int,
        _UBytePointer, _UBytePointer,
    ]
    _bridge.proxyfl_miracl_gcm_decrypt.restype = ctypes.c_int


class SecurityError(ValueError):
    """Raised when a security envelope is malformed or fails authentication."""


def _nonzero_scalar() -> int:
    return miracl_big.rand(q + 1) - 1


def _batch_coefficient() -> int:
    """Uniform non-zero element of Z_q^*, as required by Equation (13)."""
    return _nonzero_scalar()


def _buffer(data: bytes) -> Any:
    if not isinstance(data, bytes):
        raise TypeError("MIRACL bridge inputs must be bytes")
    return (ctypes.c_ubyte * max(1, len(data))).from_buffer_copy(data or b"\x00")


def _sha256(data: bytes) -> bytes:
    if _bridge is None:
        return sha256(data).digest()
    source = _buffer(data)
    digest = (ctypes.c_ubyte * 32)()
    _bridge.proxyfl_miracl_sha256(source, len(data), digest)
    return bytes(digest)


def point_to_bytes(point: ECp) -> bytes:
    """Encode an affine P-256 point as uncompressed x||y."""
    x, y = point.get()
    if x == 0 and y == 0 and point.isinf():
        raise SecurityError("cannot encode point at infinity")
    return int(x).to_bytes(POINT_BYTES, "big") + int(y).to_bytes(POINT_BYTES, "big")


def point_from_bytes(data: bytes) -> ECp:
    if not isinstance(data, bytes) or len(data) != 2 * POINT_BYTES:
        raise SecurityError("invalid P-256 point encoding")
    x = int.from_bytes(data[:POINT_BYTES], "big")
    y = int.from_bytes(data[POINT_BYTES:], "big")
    point = ECp()
    if not point.setxy(x, y):
        raise SecurityError("point is not on P-256")
    return point


def points_equal(a: ECp, b: ECp) -> bool:
    return a.get() == b.get()


def point_add(a: ECp, b: ECp) -> ECp:
    """Return a+b (MIRACL ECp has .add() but no __add__)."""
    result = a.copy()
    result.add(b)
    return result


def _hash_part(value: Any) -> bytes:
    if isinstance(value, bytes):
        raw = value
    elif isinstance(value, str):
        raw = value.encode("utf-8")
    elif isinstance(value, int):
        raw = value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big")
    elif isinstance(value, ECp):
        raw = point_to_bytes(value)
    elif isinstance(value, tuple):
        raw = b"".join(_hash_part(item) for item in value)
    else:
        raise TypeError(f"cannot hash protocol value of type {type(value)!r}")
    return len(raw).to_bytes(4, "big") + raw


def hash_to_scalar(domain: bytes, *args: Any) -> int:
    """Domain-separated SHA-256 hash-to-scalar for H0/H1/H2/H3."""
    digest = _sha256(domain + b"".join(_hash_part(arg) for arg in args))
    return (int.from_bytes(digest, "big") % (q - 1)) + 1


def _h0_mask(t_times_aid1: ECp, t_pub: ECp) -> bytes:
    """H0(t·AID_{i,1} || T_pub) → 32-byte mask for AID XOR (paper Phase ii)."""
    return _sha256(
        b"H0" + point_to_bytes(t_times_aid1) + point_to_bytes(t_pub)
    )


def _real_id_token(real_id: str) -> bytes:
    """Fixed-width encoding of ID_i for AID_{i,2} = ID ⊕ H0(...)."""
    raw = real_id.encode("utf-8")
    if len(raw) > 32:
        return _sha256(raw)
    return raw.ljust(32, b"\x00")


def _aid_key(aid: Tuple[ECp, bytes]) -> bytes:
    return point_to_bytes(aid[0]) + aid[1]


def aid_to_wire(aid: Tuple[ECp, bytes]) -> Dict[str, bytes]:
    return {"point": point_to_bytes(aid[0]), "token": aid[1]}


def aid_from_wire(data: Mapping[str, Any]) -> Tuple[ECp, bytes]:
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


def reconstruct_public_key(aid: Tuple[ECp, bytes], U: ECp, X: ECp) -> ECp:
    """Paper public-key reconstruction: Q' = U + H2(AID, X)·X."""
    beta = hash_to_scalar(b"H2", aid, X)
    return point_add(U, beta * X)


def signature_to_wire(signature: Tuple[int, ECp]) -> Dict[str, Any]:
    eta, point = signature
    return {"eta": eta, "R": point_to_bytes(point)}


def signature_from_wire(data: Mapping[str, Any]) -> Tuple[int, ECp]:
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
    private_key: Tuple[int, int], peer_public: Mapping[str, Any], p_pub: ECp
) -> bytes:
    """Symmetric pairwise secret ψ_{i,j}.

    Paper writes x_i·pk_j + w_i·pk_j (ill-typed). Correct ECDH form used here:
    x_i X_j + w_i (U_j + H1(AID_j,U_j,P_pub) P_pub).
    """
    w_i, x_i = private_key
    aid = peer_public["aid"]
    alpha = hash_to_scalar(b"H1", aid, peer_public["U"], p_pub)
    kgc_component = point_add(peer_public["U"], alpha * p_pub)
    secret_point = point_add(x_i * peer_public["X"], w_i * kgc_component)
    return _sha256(b"ProxyFL/psi/v1" + point_to_bytes(secret_point))


def encrypt_payload(shared_secret: bytes, payload: bytes, aad: bytes = b"") -> Tuple[bytes, bytes, bytes]:
    """AES-256-GCM encryption (paper Phase ii channel protection)."""
    if not isinstance(shared_secret, bytes) or not isinstance(payload, bytes) or not isinstance(aad, bytes):
        raise TypeError("shared secret, payload, and AAD must be bytes")
    key = _sha256(b"ProxyFL/AES-GCM/v1" + shared_secret)
    if _bridge is None:
        nonce = secrets.token_bytes(12)
        sealed = AESGCM(key).encrypt(nonce, payload, aad)
        return sealed[:-16], nonce, sealed[-16:]
    nonce = secrets.token_bytes(12)
    ciphertext = (ctypes.c_ubyte * max(1, len(payload)))()
    tag = (ctypes.c_ubyte * 16)()
    if not _bridge.proxyfl_miracl_gcm_encrypt(
        _buffer(key), len(key), _buffer(nonce), len(nonce), _buffer(aad), len(aad),
        _buffer(payload), len(payload), ciphertext, tag,
    ):
        raise SecurityError("MIRACL AES-GCM encryption failed")
    return bytes(ciphertext[:len(payload)]), nonce, bytes(tag)


def decrypt_payload(
    shared_secret: bytes, ciphertext: bytes, nonce: bytes, tag: bytes, aad: bytes = b""
) -> bytes:
    if (
        not isinstance(shared_secret, bytes) or not isinstance(ciphertext, bytes)
        or not isinstance(nonce, bytes) or not isinstance(tag, bytes) or not isinstance(aad, bytes)
        or not nonce or len(tag) != 16
    ):
        raise SecurityError("invalid AES-GCM inputs")
    key = _sha256(b"ProxyFL/AES-GCM/v1" + shared_secret)
    if _bridge is None:
        try:
            return AESGCM(key).decrypt(nonce, ciphertext + tag, aad)
        except Exception as exc:
            raise SecurityError("AES-GCM authentication failed") from exc
    plaintext = (ctypes.c_ubyte * max(1, len(ciphertext)))()
    if not _bridge.proxyfl_miracl_gcm_decrypt(
        _buffer(key), len(key), _buffer(nonce), len(nonce), _buffer(aad), len(aad),
        _buffer(ciphertext), len(ciphertext), _buffer(tag), plaintext,
    ):
        raise SecurityError("AES-GCM authentication failed")
    return bytes(plaintext[:len(ciphertext)])


class Authority:
    """Combined TA / KGC with Motor Vehicle Department (MVD) enrollment."""

    def __init__(self) -> None:
        self.t = _nonzero_scalar()
        self.T_pub = self.t * P
        self.s = _nonzero_scalar()
        self.P_pub = self.s * P
        self._mvd: Dict[str, bytes] = {}  # real_id → ID token
        self._token_to_id: Dict[bytes, str] = {}
        self._identities: Dict[str, CertificatelessSigner] = {}
        self._public_by_aid: Dict[bytes, Dict[str, Any]] = {}
        self._aid_to_real_id: Dict[bytes, str] = {}

    def enroll_mvd(self, real_id: str) -> None:
        """Enroll a real identity with the simulated MVD before registration."""
        if not real_id or not isinstance(real_id, str):
            raise SecurityError("invalid MVD identity")
        token = _real_id_token(real_id)
        existing = self._token_to_id.get(token)
        if existing is not None and existing != real_id:
            raise SecurityError("MVD identity token collision")
        self._mvd[real_id] = token
        self._token_to_id[token] = real_id

    def is_enrolled(self, real_id: str) -> bool:
        return real_id in self._mvd

    def generate_pseudo_identity(self, real_id: str) -> Tuple[ECp, bytes]:
        """AID_{i,1}=k_i P, AID_{i,2}=ID_i ⊕ H0(t·AID_{i,1} || T_pub)."""
        if real_id not in self._mvd:
            raise SecurityError(f"identity {real_id!r} is not enrolled with MVD")
        k_i = _nonzero_scalar()
        aid_point = k_i * P
        mask = _h0_mask(self.t * aid_point, self.T_pub)
        token = self._mvd[real_id]
        aid_token = bytes(left ^ right for left, right in zip(token, mask))
        return aid_point, aid_token

    def recover_identity(self, aid: Tuple[ECp, bytes]) -> str:
        """TA recovers real ID_i from AID (paper registration check)."""
        aid_point, aid_token = aid
        mask = _h0_mask(self.t * aid_point, self.T_pub)
        token = bytes(left ^ right for left, right in zip(aid_token, mask))
        real_id = self._token_to_id.get(token)
        if real_id is None:
            raise SecurityError("AID does not map to an enrolled MVD identity")
        return real_id

    def extract_partial_private_key(self, aid: Tuple[ECp, bytes]) -> Tuple[int, ECp]:
        u_i = _nonzero_scalar()
        U_i = u_i * P
        alpha = hash_to_scalar(b"H1", aid, U_i, self.P_pub)
        return (u_i + alpha * self.s) % q, U_i

    def register(self, name: str, real_id: Optional[str] = None) -> "CertificatelessSigner":
        if name in self._identities:
            raise ValueError(f"identity {name!r} is already registered")
        rid = real_id if real_id is not None else name
        if rid not in self._mvd:
            raise SecurityError(f"identity {rid!r} is not enrolled with MVD")
        signer = CertificatelessSigner(self, name, rid)
        # Verify AID recovers to the enrolled identity
        recovered = self.recover_identity(signer.aid)
        if recovered != rid:
            raise SecurityError("AID recovery failed MVD cross-check")
        public_info = signer.get_public_info()
        self._identities[name] = signer
        self._public_by_aid[_aid_key(public_info["aid"])] = public_info
        self._aid_to_real_id[_aid_key(public_info["aid"])] = rid
        return signer

    def identity(self, name: str) -> "CertificatelessSigner":
        return self._identities[name]

    def public_info(self, name: str) -> Dict[str, Any]:
        return self._identities[name].get_public_info()

    def resolve_public_info(self, sender: str, aid_wire: Mapping[str, Any]) -> Dict[str, Any]:
        aid = aid_from_wire(aid_wire)
        # MVD / TA ownership check via AID recovery
        try:
            real_id = self.recover_identity(aid)
        except SecurityError as exc:
            raise SecurityError("AID failed MVD recovery") from exc
        info = self._public_by_aid.get(_aid_key(aid))
        if info is None or info["name"] != sender:
            raise SecurityError("sender does not own the claimed pseudo-identity")
        if self._aid_to_real_id.get(_aid_key(aid)) != real_id:
            raise SecurityError("AID / MVD binding mismatch")
        return info


class CertificatelessSigner:
    """Vehicle/RSU/server identity with full certificateless key material."""

    def __init__(self, authority: Authority, name: str, real_id: str) -> None:
        self.authority = authority
        self.name = name
        self.real_id = real_id
        self.aid = authority.generate_pseudo_identity(real_id)
        self.w_i, self.U_i = authority.extract_partial_private_key(self.aid)
        self.x_i = _nonzero_scalar()
        self.X_i = self.x_i * P
        self.beta_i = hash_to_scalar(b"H2", self.aid, self.X_i)
        self.Q_i = point_add(self.U_i, self.beta_i * self.X_i)

        alpha_i = hash_to_scalar(b"H1", self.aid, self.U_i, authority.P_pub)
        if not points_equal(self.w_i * P, point_add(self.U_i, alpha_i * authority.P_pub)):
            raise SecurityError("KGC partial private-key validation failed (Eq. 11)")

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
            "real_id": self.real_id,
        }

    def shared_secret_for(self, peer_public: Mapping[str, Any]) -> bytes:
        return derive_shared_secret(self.private_key, peer_public, self.authority.P_pub)

    def sign(self, message: bytes) -> Tuple[int, ECp]:
        """Sign plaintext d_i: σ=η,R with γ=H3(AID,d,pk,R) (paper §Encryption)."""
        while True:
            r_i = _nonzero_scalar()
            R_i = r_i * P
            # pk_i = (Q_i, U_i) as in the paper
            gamma = hash_to_scalar(
                b"H3", self.aid, message, self.Q_i, self.U_i, R_i)
            eta = (
                r_i + gamma * (self.w_i + self.beta_i * self.x_i)
            ) % q
            if eta != 0:
                return eta, R_i


class CertificatelessVerifier:
    """Verifier used by RSUs and the central server."""

    def __init__(self, p_pub: ECp) -> None:
        self.P_pub = p_pub

    def _reconstruct_ok(self, public_info: Mapping[str, Any]) -> bool:
        try:
            aid, Q, U, X = (
                public_info["aid"], public_info["Q"],
                public_info["U"], public_info["X"],
            )
            return points_equal(reconstruct_public_key(aid, U, X), Q)
        except (KeyError, TypeError, SecurityError):
            return False

    def verify(
        self, message: bytes, signature: Tuple[int, ECp], public_info: Mapping[str, Any]
    ) -> bool:
        """Single verification (Eq. 12) after public-key reconstruction."""
        try:
            if not self._reconstruct_ok(public_info):
                return False
            eta, R_i = signature
            if not isinstance(eta, int) or not 0 < eta < q:
                return False
            aid = public_info["aid"]
            Q_i, U_i = public_info["Q"], public_info["U"]
            alpha = hash_to_scalar(b"H1", aid, U_i, self.P_pub)
            gamma = hash_to_scalar(b"H3", aid, message, Q_i, U_i, R_i)
            left = eta * P
            right = point_add(R_i, gamma * point_add(Q_i, alpha * self.P_pub))
            return points_equal(left, right)
        except (KeyError, TypeError, ValueError, SecurityError):
            return False

    def batch_verify(
        self, batch: Iterable[Tuple[bytes, Tuple[int, ECp], Mapping[str, Any]]]
    ) -> bool:
        """Batch verification with random coefficients (Eq. 13)."""
        sum_eta = 0
        sum_y_r: Optional[ECp] = None
        sum_yg_q: Optional[ECp] = None
        p_pub_scalar = 0
        count = 0
        try:
            for message, signature, public_info in batch:
                if not self._reconstruct_ok(public_info):
                    return False
                eta, R_i = signature
                if not isinstance(eta, int) or not 0 < eta < q:
                    return False
                y_i = _batch_coefficient()
                aid, Q_i, U_i = public_info["aid"], public_info["Q"], public_info["U"]
                alpha = hash_to_scalar(b"H1", aid, U_i, self.P_pub)
                gamma = hash_to_scalar(b"H3", aid, message, Q_i, U_i, R_i)
                y_gamma = (y_i * gamma) % q
                sum_eta = (sum_eta + y_i * eta) % q
                y_r = y_i * R_i
                y_gamma_q = y_gamma * Q_i
                sum_y_r = y_r if sum_y_r is None else point_add(sum_y_r, y_r)
                sum_yg_q = y_gamma_q if sum_yg_q is None else point_add(sum_yg_q, y_gamma_q)
                p_pub_scalar = (p_pub_scalar + y_gamma * alpha) % q
                count += 1
        except (KeyError, TypeError, ValueError, SecurityError):
            return False
        if count == 0:
            return True
        right = point_add(point_add(sum_y_r, sum_yg_q), p_pub_scalar * self.P_pub)
        return points_equal(sum_eta * P, right)


@dataclass(frozen=True)
class ParsedEnvelope:
    sender_info: Dict[str, Any]
    signature: Tuple[int, ECp]
    aad: bytes


def build_envelope(
    message_type: str,
    sender: CertificatelessSigner,
    recipient: str,
    round_num: int,
    signature: Tuple[int, ECp],
    ciphertext: bytes,
    nonce: bytes,
    tag: bytes,
) -> Dict[str, Any]:
    """Construct req_i = (σ_i, c_i, AID_i) plus routing metadata."""
    return {
        "type": message_type,
        "sender": sender.name,
        "recipient": recipient,
        "round": round_num,
        "aid": aid_to_wire(sender.aid),
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
        # Explicit reconstruction check (paper verification step)
        if not points_equal(
            reconstruct_public_key(
                claimed_public["aid"], claimed_public["U"], claimed_public["X"]
            ),
            claimed_public["Q"],
        ):
            raise SecurityError("public key reconstruction failed")
        signature = signature_from_wire(msg["sig"])
        aad = message_aad(expected_type, sender, receiver.name, round_num)
        return ParsedEnvelope(sender_info=sender_info, signature=signature, aad=aad)
    except (KeyError, TypeError) as exc:
        raise SecurityError("malformed security envelope") from exc


def decrypt_envelope(
    authority: Authority, receiver: CertificatelessSigner, msg: Mapping[str, Any], expected_type: str
) -> Tuple[bytes, Dict[str, Any], Tuple[int, ECp]] | None:
    """Parse and AEAD-decrypt one envelope without individual signature verification.

    RSU/server aggregation uses the returned signature in Equation (13)'s
    simultaneous batch verification step.  Call ``verify_envelope`` when an
    individual check is required.
    """
    try:
        parsed = parse_envelope(authority, receiver, msg, expected_type)
        payload = decrypt_payload(
            receiver.shared_secret_for(parsed.sender_info),
            msg["ciphertext"], msg["nonce"], msg["tag"], parsed.aad,
        )
        return payload, parsed.sender_info, parsed.signature
    except (KeyError, TypeError, SecurityError):
        return None


def verify_envelope(
    authority: Authority, receiver: CertificatelessSigner, msg: Mapping[str, Any], expected_type: str
) -> Tuple[bytes, Dict[str, Any]] | None:
    """Individually decrypt and verify one envelope."""
    result = decrypt_envelope(authority, receiver, msg, expected_type)
    if result is None:
        return None
    payload, sender_info, signature = result
    verified = CertificatelessVerifier(authority.P_pub).verify(payload, signature, sender_info)
    return (payload, sender_info) if verified else None
