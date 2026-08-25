# ProxyFL Correctness and VANET Capacity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve ProxyFL's existing FL and certificateless-security concepts while fixing the validated transport, authentication, synchronization, V2V, liveness, trust-reference, data-isolation, reproducibility, and reporting defects and replacing the throughput graph with a measurement-only VANET link-capacity/goodput model.

**Architecture:** Introduce two focused boundary modules: `wire_codec.py` for bounded safe message serialization and `vanet_channel.py` for observational PHY-capacity calculations. Existing `Device`, `RSU`, and `Server` classes retain their learning and cryptographic algorithms but gain authenticated per-hop global delivery, explicit round state, correct V2V buffering, and liveness fallbacks. Metrics collect wireless bits/airtime at successful send boundaries, while data preparation and plotting consume the corrected measurements without changing participation or network timing.

**Tech Stack:** Python 3.11, PyTorch, MIRACL NIST P-256, AES-256-GCM, standard-library JSON/base64/socket/threading, pandas, NumPy, matplotlib, and `unittest`.

**Spec:** `docs/superpowers/specs/2026-08-25-proxyfl-correctness-and-vanet-capacity-design.md`

## Global Constraints

- Do not replace DML, per-sample DP-SGD, Equation 6 V2V averaging, Equation 7 RSU averaging, or Equation 8 server averaging.
- Do not replace the Equation 9-10 trust decision or configured threshold policy; only correct its reference model.
- Do not replace the TA/KGC, pseudo-identity, shared-secret, signature, AES-GCM, single-verification, or batch-verification construction.
- Do not change the five-RSU layout, configured mobility, privacy parameters, training epochs, optimizer settings, or aggregation weights.
- The VANET channel model is measurement-only: it must not sleep, drop messages, alter timeouts, or affect participation.
- Keep legacy CSV columns for compatibility; new `vanet_link_capacity_bps` and `vanet_goodput_bps` fields must be explicit and drive the throughput plot.
- The current working tree contains user-owned modifications in every major target file. Do not reset, restore, stage, or commit those files. Use `git diff --check` and test results as task checkpoints; commit only newly created documentation or wholly new files when their history can remain isolated.
- Use this test runtime on Windows: `$env:KMP_DUPLICATE_LIB_OK='TRUE'; & 'C:\Users\manik\anaconda3\python.exe'`.

---

### Task 1: Bounded Safe Wire Codec and Tensor Payload Normalization

**Files:**
- Create: `wire_codec.py`
- Modify: `config.py`
- Modify: `network.py`
- Modify: `device.py`
- Modify: `rsu.py`
- Modify: `server.py`
- Create: `tests/test_wire_codec.py`
- Modify: `tests/test_rsu_security.py`

**Interfaces:**
- Produces: `encode_message(message: Mapping[str, Any]) -> bytes` and `decode_message(data: bytes) -> dict[str, Any]`.
- Produces: `MAX_NETWORK_MESSAGE_BYTES: int = 16 * 1024 * 1024` in `config.py`.
- Preserves: `send_msg(addr, msg, sender_name=None, round_num=None) -> bool` until Task 2 adds an optional link argument.
- Requires every model field crossing the outer message boundary to be `bytes` from `serialize_weights`, never a raw tensor dictionary.

- [ ] **Step 1: Write codec and receiver failure tests**

Create `tests/test_wire_codec.py` with concrete cases:

```python
import base64
import binascii
import json
import math
import socket
import struct
import threading
import unittest

from network import Receiver
from wire_codec import WireCodecError, decode_message, encode_message


class WireCodecTests(unittest.TestCase):
    def test_round_trip_preserves_nested_bytes(self):
        message = {
            "type": "LOCAL_UPDATE",
            "sender": "C0_V1",
            "round": 3,
            "ciphertext": b"\x00\x01payload",
            "pk": {"aid": {"token": b"token"}},
        }
        self.assertEqual(decode_message(encode_message(message)), message)

    def test_rejects_unsupported_tensor_like_value(self):
        with self.assertRaises(WireCodecError):
            encode_message({"type": "BAD", "weights": object()})

    def test_rejects_malformed_bytes_tag(self):
        malformed = json.dumps({
            "type": "BAD",
            "payload": {"__proxyfl_type__": "bytes", "base64": "%%%"},
        }).encode("utf-8")
        with self.assertRaises(WireCodecError):
            decode_message(malformed)

    def test_receiver_rejects_oversized_frame_before_callback(self):
        received = []
        receiver = Receiver(0, received.append, max_message_bytes=32)
        port = receiver.sock.getsockname()[1]
        receiver.start()
        with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
            client.sendall(struct.pack(">I", 33))
        threading.Event().wait(0.1)
        receiver.shutdown()
        self.assertEqual(received, [])
```

- [ ] **Step 2: Run the focused tests and confirm the missing-module failure**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& 'C:\Users\manik\anaconda3\python.exe' -m unittest tests.test_wire_codec -v
```

Expected: import failure for `wire_codec` or missing `Receiver(port, callback, node_name=None, max_message_bytes=MAX_NETWORK_MESSAGE_BYTES)` support.

- [ ] **Step 3: Implement the minimal safe codec**

Create `wire_codec.py` around these exact rules:

```python
import base64
import json
from collections.abc import Mapping
from typing import Any


class WireCodecError(ValueError):
    pass


def _to_wire(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {
            "__proxyfl_type__": "bytes",
            "base64": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, (list, tuple)):
        return [_to_wire(item) for item in value]
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return {key: _to_wire(item) for key, item in value.items()}
    raise WireCodecError(f"unsupported wire value: {type(value).__name__}")


def _from_wire(value: Any) -> Any:
    if isinstance(value, list):
        return [_from_wire(item) for item in value]
    if isinstance(value, dict):
        if value.get("__proxyfl_type__") == "bytes":
            if set(value) != {"__proxyfl_type__", "base64"}:
                raise WireCodecError("malformed byte-string tag")
            try:
                return base64.b64decode(value["base64"], validate=True)
            except (binascii.Error, TypeError, ValueError) as exc:
                raise WireCodecError("invalid base64 byte string") from exc
        return {key: _from_wire(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        raise WireCodecError("non-finite numbers are not allowed")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise WireCodecError("decoded unsupported JSON value")


def encode_message(message: Mapping[str, Any]) -> bytes:
    if not isinstance(message, Mapping):
        raise WireCodecError("message must be a mapping")
    return json.dumps(_to_wire(message), separators=(",", ":"), allow_nan=False).encode("utf-8")


def decode_message(data: bytes) -> dict[str, Any]:
    try:
        decoded = _from_wire(json.loads(data.decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WireCodecError("malformed JSON message") from exc
    if not isinstance(decoded, dict):
        raise WireCodecError("top-level message must be an object")
    return decoded
```

In `network.py`, replace `pickle.dumps/loads` with the codec, accept `max_message_bytes=MAX_NETWORK_MESSAGE_BYTES`, reject `msglen <= 0` or `msglen > max_message_bytes` before `_recvall`, and preserve byte/duration accounting.

Normalize plaintext model messages at every sending boundary:

```python
# Device plaintext LOCAL_UPDATE
"weights": serialize_weights(proxy_weights)

# RSU plaintext CLUSTER_UPDATE
"avg_weights": serialize_weights(avg_weights)

# Server/RSU plaintext GLOBAL_UPDATE
"global_weights": serialize_weights(global_weights)
```

Receiving paths already using `deserialize_weights` must reject malformed bytes instead of accepting raw tensor dictionaries. Update plaintext unit fixtures to serialize actual tensor dictionaries.

- [ ] **Step 4: Run codec, security, and device-reporting tests**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& 'C:\Users\manik\anaconda3\python.exe' -m unittest tests.test_wire_codec tests.test_rsu_security tests.test_device_reporting -v
```

Expected: all tests pass; malformed and oversized frames never invoke callbacks.

- [ ] **Step 5: Check the task diff without staging user files**

Run:

```powershell
git diff --check -- wire_codec.py config.py network.py device.py rsu.py server.py tests/test_wire_codec.py tests/test_rsu_security.py
git status --short
```

Expected: no whitespace errors; only intended files differ in addition to the pre-existing dirty set.

---

### Task 2: Measurement-Only VANET Capacity and Goodput

**Files:**
- Create: `vanet_channel.py`
- Modify: `config.py`
- Modify: `metrics.py`
- Modify: `network.py`
- Modify: `vanet_sim.py`
- Modify: `device.py`
- Modify: `rsu.py`
- Create: `tests/test_vanet_channel.py`
- Modify: `tests/test_metrics_and_energy.py`

**Interfaces:**
- Produces: immutable `WirelessLink(kind: str, distance_m: float)`.
- Produces: `link_capacity_bps(distance_m: float) -> float` and `modeled_airtime_seconds(num_bits: int, capacity_bps: float) -> float`.
- Extends: `send_msg(addr, msg, sender_name=None, round_num=None, wireless_link: WirelessLink | None = None) -> bool`.
- Extends: `MetricsTracker.record_wireless_delivery(node, round_num, num_wire_bytes, capacity_bps)`.
- Produces derived row fields `vanet_wireless_bits`, `vanet_airtime_s`, `vanet_link_capacity_bps`, and `vanet_goodput_bps`.

- [ ] **Step 1: Write analytical channel and metric tests**

Create tests asserting the formula and measurement-only behavior:

```python
import math
import unittest

from metrics import MetricsTracker
from vanet_channel import VANET_PHY_MAX_RATE_BPS, link_capacity_bps


class VanetChannelTests(unittest.TestCase):
    def test_capacity_is_capped_and_decreases_with_distance(self):
        near = link_capacity_bps(1.0)
        medium = link_capacity_bps(300.0)
        far = link_capacity_bps(1000.0)
        self.assertLessEqual(near, VANET_PHY_MAX_RATE_BPS)
        self.assertGreater(near, medium)
        self.assertGreater(medium, far)
        self.assertGreater(far, 0.0)

    def test_goodput_uses_delivered_bits_over_modeled_airtime(self):
        tracker = MetricsTracker()
        tracker.record_wireless_delivery("C0_V1", 1, 1000, 8_000_000.0)
        tracker.record_wireless_delivery("C0_V1", 1, 1000, 4_000_000.0)
        row = tracker.rows()[0]
        expected_airtime = 8000 / 8_000_000 + 8000 / 4_000_000
        self.assertAlmostEqual(row["vanet_wireless_bits"], 16000.0)
        self.assertAlmostEqual(row["vanet_airtime_s"], expected_airtime)
        self.assertAlmostEqual(row["vanet_goodput_bps"], 16000 / expected_airtime)
        self.assertAlmostEqual(row["vanet_link_capacity_bps"], 6_000_000.0)
```

- [ ] **Step 2: Run the tests and confirm missing capacity interfaces**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& 'C:\Users\manik\anaconda3\python.exe' -m unittest tests.test_vanet_channel tests.test_metrics_and_energy -v
```

Expected: missing `vanet_channel` or `record_wireless_delivery` failure.

- [ ] **Step 3: Implement channel equations and metric accumulation**

Add the approved constants to `config.py` and implement:

```python
from dataclasses import dataclass
import math

from config import (
    VANET_BANDWIDTH_HZ, VANET_TX_POWER_DBM, VANET_PATH_LOSS_1M_DB,
    VANET_PATH_LOSS_EXPONENT, VANET_NOISE_FIGURE_DB, VANET_PHY_MAX_RATE_BPS,
)


@dataclass(frozen=True)
class WirelessLink:
    kind: str
    distance_m: float

    def __post_init__(self):
        if self.kind not in {"v2v", "v2rsu", "rsu2v"}:
            raise ValueError("invalid VANET wireless link kind")
        if not math.isfinite(self.distance_m) or self.distance_m < 0:
            raise ValueError("wireless distance must be finite and non-negative")


def link_capacity_bps(distance_m: float) -> float:
    distance = max(float(distance_m), 1.0)
    path_loss = VANET_PATH_LOSS_1M_DB + 10.0 * VANET_PATH_LOSS_EXPONENT * math.log10(distance)
    received_dbm = VANET_TX_POWER_DBM - path_loss
    noise_dbm = -174.0 + 10.0 * math.log10(VANET_BANDWIDTH_HZ) + VANET_NOISE_FIGURE_DB
    snr_linear = 10.0 ** ((received_dbm - noise_dbm) / 10.0)
    shannon = VANET_BANDWIDTH_HZ * math.log2(1.0 + snr_linear)
    return min(VANET_PHY_MAX_RATE_BPS, max(shannon, 1.0))
```

`record_wireless_delivery` must add bits, airtime, capacity sum, and sample count atomically. `_derived_metrics` computes arithmetic mean capacity and bits/airtime goodput without including backhaul.

After a successful socket send, `send_msg` computes capacity and records the encoded frame length plus four-byte prefix. It must not sleep or alter the return value.

Add `VanetTopology.get_v2v_distance(a, b)` and use existing RSU distance for call sites. Supply `WirelessLink` only for Device-to-peer, Device-to-RSU, and RSU-to-vehicle sends. Server-to-RSU and RSU-to-server remain `None`.

- [ ] **Step 4: Run focused tests and verify no timing behavior changed**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& 'C:\Users\manik\anaconda3\python.exe' -m unittest tests.test_vanet_channel tests.test_metrics_and_energy tests.test_topology_reporting -v
```

Expected: all pass; capacity decreases with distance and goodput uses modeled airtime.

- [ ] **Step 5: Check the task diff**

Run:

```powershell
git diff --check -- vanet_channel.py config.py metrics.py network.py vanet_sim.py device.py rsu.py tests/test_vanet_channel.py tests/test_metrics_and_energy.py
```

Expected: no whitespace errors.

---

### Task 3: Exact Nonzero Scalar Domains

**Files:**
- Modify: `crypto_protocol.py`
- Modify: `tests/test_crypto_protocol.py`

**Interfaces:**
- Preserves all public protocol interfaces and equations.
- Corrects `_nonzero_scalar() -> int` and `hash_to_scalar(domain, *args) -> int` to return `1 <= value < q`.
- Keeps wire-format validation `0 < eta < q` by resampling a signing nonce if `eta == 0`.

- [ ] **Step 1: Add deterministic scalar-domain regression tests**

Add:

```python
from unittest.mock import patch


def test_hash_to_scalar_never_returns_zero(self):
    with patch("crypto_protocol._sha256", return_value=b"\x00" * 32):
        self.assertEqual(hash_to_scalar(b"H1", b"value"), 1)


def test_random_scalar_includes_one_and_excludes_zero(self):
    with patch("crypto_protocol.miracl_big.rand", return_value=2):
        self.assertEqual(_nonzero_scalar(), 1)
```

Extend imports with `hash_to_scalar` and `_nonzero_scalar`.

- [ ] **Step 2: Run the two tests and verify the old zero/excluded-one behavior fails**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& 'C:\Users\manik\anaconda3\python.exe' -m unittest tests.test_crypto_protocol.CertificatelessProtocolTests.test_hash_to_scalar_never_returns_zero tests.test_crypto_protocol.CertificatelessProtocolTests.test_random_scalar_includes_one_and_excludes_zero -v
```

Expected: failures showing `0` and `2` under the old helpers.

- [ ] **Step 3: Correct the domain without changing primitives**

Implement:

```python
def _nonzero_scalar() -> int:
    return miracl_big.rand(q + 1) - 1


def hash_to_scalar(domain: bytes, *args: Any) -> int:
    digest = _sha256(domain + b"".join(_hash_part(arg) for arg in args))
    return (int.from_bytes(digest, "big") % (q - 1)) + 1
```

Wrap `CertificatelessSigner.sign` nonce generation in a loop that returns only when `eta != 0`; all signature equations remain unchanged.

- [ ] **Step 4: Run the complete crypto suite**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& 'C:\Users\manik\anaconda3\python.exe' -m unittest tests.test_crypto_protocol -v
```

Expected: all protocol, tampering, symmetry, and batch tests pass.

- [ ] **Step 5: Check the crypto diff**

Run:

```powershell
git diff --check -- crypto_protocol.py tests/test_crypto_protocol.py
```

Expected: no whitespace errors.

---

### Task 4: Authenticated Per-Hop Global Model Delivery

**Files:**
- Modify: `main.py`
- Modify: `server.py`
- Modify: `rsu.py`
- Modify: `device.py`
- Modify: `tests/test_rsu_security.py`
- Modify: `tests/test_device_reporting.py`
- Create: `tests/test_global_update_security.py`

**Interfaces:**
- Adds `Server.rsu_directory: dict[str, int]` from RSU name to port.
- Adds `Server._build_global_message(rsu_name: str, round_num: int, weights: Mapping[str, Tensor]) -> dict`.
- Adds `RSU._decode_server_global(msg: Mapping[str, Any]) -> dict[str, Tensor] | None`.
- Adds `RSU._build_vehicle_global(vehicle_name: str, round_num: int, weights: Mapping[str, Tensor]) -> dict`.
- Adds `Device._decode_rsu_global(msg: Mapping[str, Any]) -> dict[str, Tensor] | None`.
- Uses the existing `build_envelope`, `verify_envelope`, shared secrets, signatures, AES-GCM, AAD, and tensor codec.

- [ ] **Step 1: Write end-to-end envelope tests**

Create authority/server/RSU/device identities and assert:

```python
class GlobalUpdateSecurityTests(unittest.TestCase):
    def setUp(self):
        self.authority = Authority()
        for name in ["Server", "RSU_0_Central", "C0_V1"]:
            self.authority.enroll_mvd(name)
        self.server_id = self.authority.register("Server")
        self.rsu_id = self.authority.register("RSU_0_Central")
        self.vehicle_id = self.authority.register("C0_V1")
        self.weights = {"weight": torch.tensor([1.0, 2.0])}

    def test_server_to_rsu_and_rsu_to_vehicle_are_authenticated(self):
        server = Server.__new__(Server)
        server.security_enabled = True
        server.security_authority = self.authority
        server.signer = self.server_id
        server_msg = server._build_global_message("RSU_0_Central", 2, self.weights)

        rsu = RSU.__new__(RSU)
        rsu.name = "RSU_0_Central"
        rsu.security_enabled = True
        rsu.security_authority = self.authority
        rsu.signer = self.rsu_id
        decoded = rsu._decode_server_global(server_msg)
        self.assertTrue(torch.equal(decoded["weight"], self.weights["weight"]))

        vehicle_msg = rsu._build_vehicle_global("C0_V1", 2, decoded)
        device = Device.__new__(Device)
        device.name = "C0_V1"
        device.rsu_name = "RSU_0_Central"
        device.security_enabled = True
        device.security_authority = self.authority
        device.signer = self.vehicle_id
        final = device._decode_rsu_global(vehicle_msg)
        self.assertTrue(torch.equal(final["weight"], self.weights["weight"]))

    def test_unsigned_or_tampered_global_update_is_rejected(self):
        device = Device.__new__(Device)
        device.name = "C0_V1"
        device.rsu_name = "RSU_0_Central"
        device.security_enabled = True
        device.security_authority = self.authority
        device.signer = self.vehicle_id
        unsigned = {"type": "GLOBAL_UPDATE", "sender": "RSU_0_Central", "round": 2}
        self.assertIsNone(device._decode_rsu_global(unsigned))
```

- [ ] **Step 2: Run the focused test and confirm helper methods are missing**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& 'C:\Users\manik\anaconda3\python.exe' -m unittest tests.test_global_update_security -v
```

Expected: missing helper failures.

- [ ] **Step 3: Implement existing-protocol global delivery**

Use payload `serialize_weights(weights)` and message type `GLOBAL_UPDATE` at each hop. Server envelope recipient is the RSU name; RSU envelope recipient is the vehicle name. Time signature generation, encryption, decryption, and verification with existing metrics.

In no-security mode, build:

```python
{
    "type": "GLOBAL_UPDATE",
    "sender": sender_name,
    "recipient": recipient_name,
    "round": round_num,
    "global_weights": serialize_weights(weights),
}
```

In `main.py`, pass:

```python
rsu_directory = {
    rsu_name: RSU_BASE_PORT + index
    for index, (rsu_name, _, _, _) in enumerate(cluster_specs)
}
```

The server sends a distinct message for each RSU. The RSU verifies/decrypts before updating its reference or forwarding and sends a distinct message to each known vehicle, adding Task 2's `WirelessLink("rsu2v", distance)` context. The device accepts only its assigned RSU as sender.

- [ ] **Step 4: Run global, crypto, RSU, and device tests**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& 'C:\Users\manik\anaconda3\python.exe' -m unittest tests.test_global_update_security tests.test_crypto_protocol tests.test_rsu_security tests.test_device_reporting -v
```

Expected: all pass; unsigned/tampered global updates are rejected.

- [ ] **Step 5: Check the cross-hop diff**

Run:

```powershell
git diff --check -- main.py server.py rsu.py device.py tests/test_global_update_security.py tests/test_rsu_security.py tests/test_device_reporting.py
```

Expected: no whitespace errors.

---

### Task 5: Device Round State and Server Zero-Valid Liveness

**Files:**
- Modify: `device.py`
- Modify: `server.py`
- Modify: `tests/test_device_reporting.py`
- Create: `tests/test_server_liveness.py`

**Interfaces:**
- Adds device phases `ROUND_TRAINING`, `ROUND_REPORTING`, `ROUND_WAITING_GLOBAL`, and `ROUND_IDLE`.
- Adds `Device._pending_global_updates: dict[int, dict[str, Tensor]]`.
- Adds `Device._handle_verified_global(round_num, weights)` and `_apply_pending_global(round_num) -> bool`.
- Adds `Server._broadcast_global(round_num, weights, rsu_directory_snapshot)` used by every aggregation exit.

- [ ] **Step 1: Write state and liveness regression tests**

Add a device test that proves a global update cannot overwrite training:

```python
def test_verified_global_is_queued_until_local_report(self):
    device = Device.__new__(Device)
    device.current_round = 4
    device._round_phase = "training"
    device._pending_global_updates = {}
    device.proxy_lock = threading.Lock()
    device.proxy_model = MagicMock()
    device.round_event = threading.Event()
    weights = {"weight": torch.tensor([3.0])}

    device._handle_verified_global(4, weights)
    device.proxy_model.load_state_dict.assert_not_called()
    self.assertFalse(device.round_event.is_set())

    device._round_phase = "waiting_global"
    self.assertTrue(device._apply_pending_global(4))
    device.proxy_model.load_state_dict.assert_called_once()
    self.assertTrue(device.round_event.is_set())
```

Create `tests/test_server_liveness.py` using the previously reproduced invalid payload and patch `Server._broadcast_global`. Assert the round is completed, broadcast happens once with current `model.state_dict()`, and `training_done_event` is set when `round_num == total_rounds`.

- [ ] **Step 2: Run focused tests and confirm current eager-load/no-broadcast failures**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& 'C:\Users\manik\anaconda3\python.exe' -m unittest tests.test_device_reporting tests.test_server_liveness -v
```

Expected: missing phase helpers and zero broadcast on invalid aggregation.

- [ ] **Step 3: Implement phase-gated application and one liveness helper**

Set phase and initialize the round buffer before training. `on_receive` verifies/decodes first, then calls `_handle_verified_global`. Only `ROUND_WAITING_GLOBAL` loads immediately; earlier phases queue by exact round.

After a successful `LOCAL_UPDATE` or `NO_UPDATE` send:

```python
self._round_phase = ROUND_WAITING_GLOBAL
if not self._apply_pending_global(r):
    self.round_event.wait(timeout=TIMEOUT)
```

Set `_request_sent_at[r]` only when the relevant `send_msg` call returns `True`. Clear stale pending entries and set idle after each round.

Refactor server broadcast construction through `_broadcast_global`. In `if not cluster_weights`, broadcast `self.model.state_dict()`, record zero successful updates, record server execution, and set `training_done_event` on the final round instead of returning silently.

- [ ] **Step 4: Run state, liveness, and global-security tests**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& 'C:\Users\manik\anaconda3\python.exe' -m unittest tests.test_device_reporting tests.test_server_liveness tests.test_global_update_security -v
```

Expected: all pass; training-phase updates queue and every terminal server path broadcasts.

- [ ] **Step 5: Check the task diff**

Run:

```powershell
git diff --check -- device.py server.py tests/test_device_reporting.py tests/test_server_liveness.py
```

Expected: no whitespace errors.

---

### Task 6: V2V Readiness and Early-Message Preservation

**Files:**
- Modify: `config.py`
- Modify: `vanet_sim.py`
- Modify: `device.py`
- Modify: `tests/test_topology_reporting.py`
- Modify: `tests/test_device_reporting.py`

**Interfaces:**
- Adds `V2V_READY_TIMEOUT` while preserving `V2V_COLLECT_TIMEOUT` for message collection.
- Adds `VanetTopology.mark_v2v_ready(name, round_num)`, `wait_for_v2v_ready(name, round_num, peer_names, timeout) -> bool`, and `clear_v2v_ready(name, round_num)`.
- Initializes `Device._peer_buffers[r]` at round entry and never clears it before sharing.

- [ ] **Step 1: Add topology barrier and early-buffer tests**

Add:

```python
def test_v2v_readiness_releases_when_all_peers_mark_ready(self):
    topology = VanetTopology()
    topology.mark_v2v_ready("C0_V1", 2)
    topology.mark_v2v_ready("C0_V2", 2)
    self.assertTrue(topology.wait_for_v2v_ready(
        "C0_V1", 2, ["C0_V2"], timeout=0.01))
    topology.clear_v2v_ready("C0_V1", 2)
    topology.clear_v2v_ready("C0_V2", 2)


def test_early_peer_update_is_not_discarded_before_average(self):
    device = Device.__new__(Device)
    device.name = "C0_V2"
    device.current_round = 1
    device.peer_directory = {"C0_V1": 6000}
    device._peer_lock = threading.Lock()
    device._peer_buffers = {1: {"C0_V1": {"w": torch.tensor([9.0])}}}
    device.security_enabled = False
    device.signer = None
    device.security_authority = None
    device.topology = MagicMock()
    device.topology.get_v2v_neighbors.return_value = ["C0_V1"]
    with patch("device.send_msg", return_value=True), patch("device.V2V_COLLECT_TIMEOUT", 0.01):
        result = device._v2v_share_and_aggregate(1, {"w": torch.tensor([1.0])})
    self.assertEqual(result["w"].item(), 5.0)
```

- [ ] **Step 2: Run tests and reproduce the existing discarded update**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& 'C:\Users\manik\anaconda3\python.exe' -m unittest tests.test_topology_reporting tests.test_device_reporting -v
```

Expected: missing readiness API and early average remains `1.0` instead of `5.0`.

- [ ] **Step 3: Implement condition-based readiness and buffer lifecycle**

Use `threading.Condition(self._lock)` with a set of `(round_num, vehicle_name)` markers. `wait_for_v2v_ready` waits in a deadline loop until every peer marker exists or timeout expires, and notifies on mark/clear.

At device round entry:

```python
with self._peer_lock:
    self._peer_buffers[r] = {}
```

After training and before sends, mark ready and wait on only currently in-range peers. Remove the entry-clearing `pop` at the start of `_v2v_share_and_aggregate`. `on_receive` accepts a peer message only when `r == current_round`, sender exists in `peer_directory`, and the topology still reports it as a neighbor. At round exit, pop the round buffer and clear the readiness marker.

- [ ] **Step 4: Run topology/device/global-state tests**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& 'C:\Users\manik\anaconda3\python.exe' -m unittest tests.test_topology_reporting tests.test_device_reporting tests.test_global_update_security -v
```

Expected: all pass; early peer update averages to `5.0` and readiness cleans up.

- [ ] **Step 5: Check the V2V diff**

Run:

```powershell
git diff --check -- config.py vanet_sim.py device.py tests/test_topology_reporting.py tests/test_device_reporting.py
```

Expected: no whitespace errors.

---

### Task 7: Equation 9 Global Trust Reference

**Files:**
- Modify: `models.py`
- Modify: `main.py`
- Modify: `rsu.py`
- Create: `tests/test_trust_filter.py`
- Modify: `tests/test_rsu_security.py`

**Interfaces:**
- Extends `filter_trusted_weights(weight_entries, reference_weights=None, threshold=None, median_multiplier=3.0)`.
- Adds `RSU.global_reference_weights`, initialized from the server's initial proxy state and updated only after an authenticated global message.
- Preserves the current cutoff policy when `TRUST_L2_THRESHOLD is None`.

- [ ] **Step 1: Write a reference-selection test**

Create:

```python
import torch
import unittest

from models import filter_trusted_weights


class TrustFilterTests(unittest.TestCase):
    def test_deviation_uses_supplied_global_reference(self):
        reference = {"w": torch.tensor([0.0])}
        entries = [
            ("honest", {"w": torch.tensor([0.1])}),
            ("outlier", {"w": torch.tensor([10.0])}),
        ]
        trusted, log = filter_trusted_weights(
            entries, reference_weights=reference, threshold=1.0)
        outcomes = {name: accepted for name, _, accepted in log}
        self.assertEqual(outcomes, {"honest": True, "outlier": False})
        self.assertEqual(len(trusted), 1)
```

- [ ] **Step 2: Run the focused test and confirm the signature is unsupported**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& 'C:\Users\manik\anaconda3\python.exe' -m unittest tests.test_trust_filter -v
```

Expected: unexpected keyword `reference_weights`.

- [ ] **Step 3: Implement the paper-aligned reference without changing cutoff logic**

Use:

```python
weights_only = [weights for _, weights in weight_entries]
reference = reference_weights if reference_weights is not None else average_weights(weights_only)
deviations = [
    (name, model_l2_deviation(reference, weights), weights)
    for name, weights in weight_entries
]
```

Keep the existing median-multiplier or absolute cutoff code unchanged. Pass `initial_global_weights` from `server.model.state_dict()` into each RSU at construction. Update the RSU reference only after successful Server-to-RSU verification and decoding; pass it into `filter_trusted_weights`.

- [ ] **Step 4: Run trust, RSU, and global authentication tests**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& 'C:\Users\manik\anaconda3\python.exe' -m unittest tests.test_trust_filter tests.test_rsu_security tests.test_global_update_security -v
```

Expected: all pass; tampered globals cannot replace the trust reference.

- [ ] **Step 5: Check the trust diff**

Run:

```powershell
git diff --check -- models.py main.py rsu.py tests/test_trust_filter.py tests/test_rsu_security.py
```

Expected: no whitespace errors.

---

### Task 8: DP Objective Consistency and Honest Epsilon Warning

**Files:**
- Modify: `config.py`
- Modify: `device.py`
- Create: `tests/test_dp_objective.py`

**Interfaces:**
- Produces pure helper `proxy_training_objective(logits, targets, soft_targets, class_weights=None) -> Tensor` in `device.py`.
- Adds `DP_EPSILON_WARNING_THRESHOLD = 10.0` as reporting only.
- Uses the same helper in vectorized per-sample DP gradients and non-DP proxy training.
- Does not change `DP_NOISE_MULTIPLIER`, `DP_CLIP_NORM`, `DP_DELTA`, or `DP_MAX_EPSILON`.

- [ ] **Step 1: Write pure objective and warning tests**

Create:

```python
import torch
import unittest

from device import proxy_training_objective


class DPObjectiveTests(unittest.TestCase):
    def test_class_weights_change_dp_and_batch_objective_consistently(self):
        logits = torch.tensor([[3.0, 0.5], [0.1, 2.0]])
        targets = torch.tensor([0, 1])
        soft = torch.softmax(torch.tensor([[2.5, 0.2], [0.3, 1.7]]) / 3.0, dim=1)
        weights = torch.tensor([1.0, 4.0])
        weighted = proxy_training_objective(logits, targets, soft, weights)
        unweighted = proxy_training_objective(logits, targets, soft, None)
        self.assertFalse(torch.isclose(weighted, unweighted))
        per_sample = torch.stack([
            proxy_training_objective(
                logits[index:index + 1], targets[index:index + 1],
                soft[index:index + 1], weights)
            for index in range(2)
        ]).mean()
        self.assertTrue(torch.isfinite(per_sample))
```

Add a log-capture test that an epsilon above the threshold includes `PRIVACY WARNING` and does not set `budget_exhausted` when `DP_MAX_EPSILON is None`.

- [ ] **Step 2: Run the focused tests and confirm helper absence**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& 'C:\Users\manik\anaconda3\python.exe' -m unittest tests.test_dp_objective -v
```

Expected: missing helper.

- [ ] **Step 3: Centralize the existing proxy objective**

Implement the helper with existing constants:

```python
def proxy_training_objective(logits, targets, soft_targets, class_weights=None):
    ce = F.cross_entropy(logits, targets, weight=class_weights)
    kl = dml_loss(logits, soft_targets, DML_TEMPERATURE)
    return (1 - DML_BETA) * ce + DML_BETA * kl
```

Store VANET class weights separately from `self.criterion`, pass them into the helper in both DP single-sample and non-DP batch paths, and keep MNIST weights as `None`. Print a warning after epsilon logging when `eps > DP_EPSILON_WARNING_THRESHOLD`; do not alter training or sharing.

- [ ] **Step 4: Run objective, metrics, and device tests**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& 'C:\Users\manik\anaconda3\python.exe' -m unittest tests.test_dp_objective tests.test_metrics_and_energy tests.test_device_reporting -v
```

Expected: all pass and no privacy configuration value changes.

- [ ] **Step 5: Check the DP diff**

Run:

```powershell
git diff --check -- config.py device.py tests/test_dp_objective.py
```

Expected: no whitespace errors.

---

### Task 9: Leakage-Free VANET Partitions and Reproducible Seeds

**Files:**
- Modify: `config.py`
- Modify: `data_utils.py`
- Modify: `main.py`
- Modify: `device.py`
- Modify: `vanet_sim.py`
- Create: `tests/test_data_partitioning.py`
- Modify: `tests/test_topology_reporting.py`

**Interfaces:**
- Adds `SIMULATION_SEED = 42` and CLI `--seed`.
- Produces `prepare_vanet_partitions(train_path: str, total_vehicles: int) -> tuple[StandardScaler, list[tuple[pd.DataFrame, pd.DataFrame]]]`.
- Adds optional `vanet_partition` and `random_seed` Device constructor inputs.
- Adds optional `random_seed` to `VanetTopology`/`spawn_vehicle` and uses a per-vehicle steering RNG.

- [ ] **Step 1: Write leakage and determinism tests**

Build a temporary six-row CSV with four training-like values and two extreme held-out values. Call `prepare_vanet_partitions(path, total_vehicles=2)` and assert `scaler.mean_` equals the union of each partition's training rows, not all six rows. Call twice and assert train/test indices and transformed tensors match.

Add topology test:

```python
def test_seed_reproduces_vehicle_spawn_and_motion(self):
    first = VanetTopology(random_seed=7)
    second = VanetTopology(random_seed=7)
    for topology in [first, second]:
        topology.register_rsu("RSU", 0, 0)
        spawn_vehicle(topology, "C0_V1", "RSU")
        topology.move_vehicle("C0_V1")
    self.assertEqual(
        first.get_vehicle_position("C0_V1"),
        second.get_vehicle_position("C0_V1"),
    )
```

- [ ] **Step 2: Run tests and confirm old global-scaler/nondeterministic behavior**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& 'C:\Users\manik\anaconda3\python.exe' -m unittest tests.test_data_partitioning tests.test_topology_reporting -v
```

Expected: missing partition helper/seed interfaces or scaler mean includes held-out rows.

- [ ] **Step 3: Implement shared training-only scaling and seed plumbing**

Preserve current contiguous per-device slicing and local 80/20 semantics. First calculate every device slice and split, concatenate only its training portions to fit one `StandardScaler`, and return copied/scaled train/test frames per device.

In `run_single_simulation`, seed before any random operation:

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
```

Pass precomputed VANET partitions and `seed + device_id` into devices. Use a seeded `torch.Generator` for each DataLoader. Replace `torch.seed()` with explicit per-device initialization. Give `VanetTopology` a master RNG for spawning and a deterministic `random.Random(master_seed + stable_vehicle_index)` for each vehicle's steering jitter so thread order cannot change the steering sequence.

Update CLI subprocess propagation for `--dataset both` to pass the seed.

- [ ] **Step 4: Run data, topology, and device tests**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& 'C:\Users\manik\anaconda3\python.exe' -m unittest tests.test_data_partitioning tests.test_topology_reporting tests.test_device_reporting -v
```

Expected: all pass; held-out extrema do not affect scaler statistics and identical seeds reproduce topology.

- [ ] **Step 5: Check the data/reproducibility diff**

Run:

```powershell
git diff --check -- config.py data_utils.py main.py device.py vanet_sim.py tests/test_data_partitioning.py tests/test_topology_reporting.py
```

Expected: no whitespace errors.

---

### Task 10: Correct CSV Semantics, Throughput Plot, and Explanations

**Files:**
- Modify: `metrics.py`
- Modify: `server.py`
- Modify: `plot_metrics.py`
- Modify: `tests/test_metrics_and_energy.py`
- Modify: `tests/test_plot_reporting.py`

**Interfaces:**
- CSV adds raw/derived VANET fields while retaining `throughput_updates_per_sec` and `throughput_bytes_per_sec` as legacy server collection measurements.
- Throughput plot aggregates wireless rows per round as `sum(vanet_wireless_bits) / sum(vanet_airtime_s)` and labels the y-axis `Modeled VANET Goodput (Mbps)`.
- Explanation generator detects constant versus changing coverage from observed values.

- [ ] **Step 1: Add export/plot/explanation regression tests**

Extend CSV test to assert columns:

```python
required = {
    "vanet_wireless_bits", "vanet_airtime_s",
    "vanet_link_capacity_bps", "vanet_goodput_bps",
}
self.assertTrue(required.issubset(frame.columns))
```

Build two rounds with wireless bits/airtime across multiple vehicle and RSU rows, patch `matplotlib.pyplot.plot`, call `plot_system_metrics`, and assert plotted Mbps equals the round aggregates rather than the Server `throughput_bytes_per_sec` values.

For constant coverage rows, assert the explanation contains `remained constant` and does not contain `moving vehicles enter`. For varying coverage, assert it describes observed entries/exits without claiming causation from the graph alone.

- [ ] **Step 2: Run plot and metric tests and confirm legacy series is still selected**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& 'C:\Users\manik\anaconda3\python.exe' -m unittest tests.test_metrics_and_energy tests.test_plot_reporting -v
```

Expected: missing columns or plot call uses server wall-clock throughput.

- [ ] **Step 3: Implement round aggregation and honest explanations**

Add new columns to `MetricsTracker.export_csv`. Do not overwrite Receiver-recorded `bytes_rx` in `server.py`; store application model payload bytes in a separate `model_payload_bytes_rx` field if compatibility requires it.

In plotting:

```python
wireless = df.groupby("round", as_index=True).agg(
    bits=("vanet_wireless_bits", "sum"),
    airtime=("vanet_airtime_s", "sum"),
)
goodput_mbps = (
    wireless["bits"] / wireless["airtime"].replace(0.0, np.nan)
).fillna(0.0) / 1_000_000.0
```

Use this as the primary throughput graph. Keep a documented legacy fallback only when new fields are absent. Generate coverage prose from `min == max` versus observed variation and mention stationary configuration when coverage is constant.

- [ ] **Step 4: Run plotting, metrics, and compatibility tests**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& 'C:\Users\manik\anaconda3\python.exe' -m unittest tests.test_metrics_and_energy tests.test_plot_reporting -v
```

Expected: all pass; y-axis is Mbps and explanations match data.

- [ ] **Step 5: Check the reporting diff**

Run:

```powershell
git diff --check -- metrics.py server.py plot_metrics.py tests/test_metrics_and_energy.py tests/test_plot_reporting.py
```

Expected: no whitespace errors.

---

### Task 11: Integrated Verification and Regression Audit

**Files:**
- Verify: all modified source and test files
- Update only if a failure exposes a requirement gap: the test file closest to that requirement and its corresponding source file

**Interfaces:**
- Confirms all tasks compose under the existing CLI and security defaults.
- Produces no new algorithm or configuration changes.

- [ ] **Step 1: Compile every project Python module**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& 'C:\Users\manik\anaconda3\python.exe' -m compileall -q . -x 'core-master|crypto_protocol[\\/]miracl_native'
```

Expected: exit code 0 and no syntax errors.

- [ ] **Step 2: Run the complete unittest suite**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& 'C:\Users\manik\anaconda3\python.exe' -m unittest discover -s tests -v
```

Expected: all existing and new tests pass with zero errors and zero failures.

- [ ] **Step 3: Run focused adversarial reproductions**

Run the new tests individually so their outcomes are visible:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& 'C:\Users\manik\anaconda3\python.exe' -m unittest tests.test_wire_codec tests.test_global_update_security tests.test_server_liveness tests.test_trust_filter tests.test_vanet_channel -v
```

Expected: unsigned global, malformed wire message, zero-valid liveness, wrong trust reference, and nonphysical throughput regressions are all rejected by passing tests.

- [ ] **Step 4: Audit preserved concepts against the approved spec**

Run:

```powershell
git diff -- config.py models.py privacy.py crypto_protocol.py device.py rsu.py server.py
```

Verify manually that:

- `average_weights` remains equal-weight arithmetic averaging.
- DML alpha, beta, and temperature are unchanged.
- DP clipping/noise/accounting equations and parameter values are unchanged.
- Signature, verification, batch-verification, and shared-secret equations are unchanged except nonzero-domain correction and global-message reuse.
- VANET measurement never calls `sleep` and never changes `send_msg` success.

Expected: every invariant holds; if not, revert only the unintended new hunk with `apply_patch` while preserving pre-existing user edits.

- [ ] **Step 5: Perform final diff and workspace safety checks**

Run:

```powershell
git diff --check
git status --short
git diff --stat
```

Expected: no whitespace errors; no generated metrics, plots, model binaries, or user-owned files were deleted or reset. Do not stage or commit overlapping dirty source files without explicit user authorization.
