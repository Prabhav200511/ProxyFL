# ProxyFL VANET — Architecture Teardown

**What this repository actually builds, how a communication round really flows through it, where it
departs from the ProxyFL paper and from the certificateless-authentication protocol figures — and
the defects that were silently destroying every federated round.**

| | |
|---|---|
| Repo | `Prabhav200511/ProxyFL` |
| Entry point | `main.py` |
| Topology | 5 RSUs, 2–10 vehicles each (seeded random) |
| Source of truth | the code in this repository, not the papers |
| Verified by | 54 unit tests + a 3-round live VANET run (seed 42, 21 vehicles) |

> **Read this first.** Before the fixes in this branch, the system trained locally but never learned
> collectively. Every RSU aggregation crashed on a single non-ASCII character in a log line, so no
> cluster model ever reached the server, no global model was ever computed, and all 21 vehicles timed
> out every round. **The plots and CSVs currently committed at the repo root were produced by a
> pipeline whose top two aggregation tiers were dead** and should be regenerated. Section 8 has the
> evidence; sections 1–7 describe the architecture as the code now stands.

### At a glance

| Metric | Value |
|---|---|
| Round-fatal bugs found | 5 |
| Total defects fixed | 12 |
| Open limitations reported (not changed) | 11 |
| Vehicles now aggregated per round | 21 / 21 |
| Global proxy accuracy, rounds 1–3 | 60.0% → 61.2% → 62.0% |

### Contents

1. [System components](#1-system-components)
2. [Network hierarchy](#2-network-hierarchy)
3. [Complete communication-round flow](#3-complete-communication-round-flow)
4. [Security architecture](#4-security-architecture)
5. [Aggregation hierarchy](#5-aggregation-hierarchy)
6. [Metrics](#6-metrics)
7. [File-wise implementation mapping](#7-file-wise-implementation-mapping)
8. [Worked example, diagrams and defects](#8-worked-example-diagrams-and-defects)

---

## 1. System components

Everything runs inside **one Python process**. Each node is a thread (or a pair of threads) with a
real TCP listener bound to `127.0.0.1`. "Distributed" here means distributed *message passing* over
loopback sockets, not distributed hosts.

### Terminology

- **VANET** — Vehicular Ad-hoc NETwork. Vehicles carry an On-Board Unit (OBU) that talks to fixed
  roadside radios and to other vehicles. Two link types matter here: **V2V** (vehicle↔vehicle, short
  range) and **V2I / V2RSU** (vehicle↔infrastructure, longer range).
- **RSU** — Road-Side Unit. A fixed radio at a known position. In this design it is also the
  *cluster head*: it owns the vehicles assigned to it and aggregates on their behalf.
- **Proxy model** — ProxyFL's core idea. Each vehicle keeps a **private model** it never shares, plus
  a small **proxy model** whose architecture every participant agrees on. Only the proxy is
  transmitted, so private architectures can differ and only the proxy needs differential-privacy
  protection.

### 1.1 Trusted Authority and Key Generation Center

One class, `crypto_protocol.Authority`, plays three roles from the protocol figures at once:

- **MVD** (Motor Vehicle Department) — `enroll_mvd(real_id)` records a fixed-width 32-byte token for
  a real identity. Registration is refused for anything not enrolled.
- **TA** — holds master secret `t` and public `T_pub = t·P`. `generate_pseudo_identity` issues
  `AID = (AID1, AID2)` with `AID1 = k·P` and `AID2 = ID XOR H0(t·AID1 || T_pub)`.
  `recover_identity` reverses it — that is how the TA de-anonymises a misbehaving vehicle.
- **KGC** — holds master secret `s` and public `P_pub = s·P`. `extract_partial_private_key` returns
  `(w, U)` with `w = u + alpha·s mod q`, `alpha = H1(AID, U, P_pub)`.

`main.py` builds exactly one `Authority`, enrols and registers `"Server"`, all five RSU names and
every vehicle name *before any thread starts*, then hands the same object to every node. There is no
network-facing registration protocol — the TLS/SSL-secured enrolment channel in the figure is not
modelled.

### 1.2 Central FL server

`server.Server` on port **8000**. It holds the reference global proxy model (`ProxyModel` for VANET,
`MNISTProxyModel` for MNIST), collects one `CLUSTER_UPDATE` or `NO_CLUSTER_UPDATE` per RSU per round,
batch-verifies them, averages with plain FedAvg, evaluates on a held-out attack test set, and
broadcasts the result. It also owns the simulation barrier: `training_done_event` is set once the
final round closes.

### 1.3 RSUs as intermediate aggregators

Five `rsu.RSU` instances on ports **5000–5004**, positions fixed by `config.RSU_LAYOUT`:

| Index | Name | Direction | Position (m) | Port | Vehicle IDs |
|---|---|---|---|---|---|
| 0 | `RSU_0_Central` | Central | (0, 0) | 5000 | `C0_V1` … |
| 1 | `RSU_1_North` | North | (0, 1800) | 5001 | `C1_V1` … |
| 2 | `RSU_2_East` | East | (1800, 0) | 5002 | `C2_V1` … |
| 3 | `RSU_3_South` | South | (0, −1800) | 5003 | `C3_V1` … |
| 4 | `RSU_4_West` | West | (−1800, 0) | 5004 | `C4_V1` … |

Each RSU gets `random.randint(2, 10)` vehicles for the run (`VEHICLES_PER_CLUSTER_RANGE`), drawn from
the seeded RNG — so the count differs per RSU but is reproducible per seed. Vehicle names come from
`vanet_sim.format_vehicle_id(cluster_index, j+1)` → `C{cluster}_V{n}`, cluster index 0-based, vehicle
index 1-based.

### 1.4 Vehicles as FL clients

`device.Device` on port `6000 + cluster_index*100 + j`. Each vehicle owns:

- a **private model** — `VanetIDSSmall` / `VanetIDS` / `VanetIDSLarge` (2,790 / 11,622 / 45,670
  parameters), assigned round-robin by `device_id % 3` when `--heterogeneous` (the default);
- a **proxy model** — `ProxyModel`, 4→32→6, **358 parameters**, identical architecture on every
  vehicle and seeded identically so all proxies start from the same weights;
- a contiguous slice of `Main_data_shuffled.csv` (105,320 rows → ~5,015 rows per vehicle at 21
  vehicles), split 80/20 into a local train and local test set;
- an `RDPAccountant` tracking cumulative epsilon at delta = 1e-5.

Two threads per vehicle: the socket listener and the round loop (`Device.send`). A process-wide
`TRAINING_SEMAPHORE` caps concurrent training at 8 on CPU — **the single most important non-obvious
fact about this system's timing**, and the root of one of the round-fatal bugs in section 8.

### 1.5 VANET mobility and communication layer

Three files with a strict division of labour:

- `vanet_sim.VanetTopology` — a lock-protected position store. Answers `get_v2v_neighbors`,
  `can_reach_rsu`, `get_distance_to_rsu` and `assigned_rsu_coverage`; advances vehicles once per
  round via `move_vehicle`; hosts the V2V rendezvous barrier (`mark_v2v_ready` /
  `wait_for_v2v_ready`) on a `threading.Condition`.
- `vanet_channel` — an IEEE 802.11p-style link budget (23 dBm, 10 MHz, path-loss exponent 2.7, capped
  at 27 Mb/s) that computes Shannon capacity from distance. Its own docstring says it is
  *measurement-only*: it never adds delay, never drops a packet, never blocks participation.
- `network.py` — the actual transport. Length-prefixed frames (4-byte big-endian) over one TCP
  connection per message, capped at 16 MiB, encoded by `wire_codec` as JSON with base64-tagged byte
  strings.

> **Read this before interpreting any latency number.** Range gates *whether a vehicle is allowed to
> upload*. It does not gate delivery: once a node calls `send_msg`, the frame goes over loopback TCP
> at host speed. So `communication_tx_ms` and `communication_rx_ms` measure Python socket and
> scheduler cost, **not radio time**. The radio numbers live in separate `vanet_airtime_s` /
> `vanet_goodput_bps` columns and are derived, not observed.

### 1.6 Metrics and plotting subsystem

- `metrics.MetricsTracker` — a global, lock-protected dict keyed by `(node, round)`. `Timer` and
  `BatchTimer` accumulate milliseconds; `record_bytes`, `record_value` and
  `record_wireless_delivery` accumulate counters. `_derived_metrics` computes latency roll-ups and
  the OBU energy model at export time.
- `logger.TrainingLogger` — PrettyTable text tables plus round-keyed dicts for accuracy, loss,
  private accuracy and privacy spend. Writes `*_training_logs.txt`.
- `plot_metrics.plot_all` — parses the log text and the CSV, emits 13 PNGs into `plots/` plus a
  data-backed `vanet_plot_explanations.md`.

---

## 2. Network hierarchy

Three tiers plus one lateral link. Six message types move between them.

| Message | Direction | Payload | Built by |
|---|---|---|---|
| `PEER_UPDATE` | vehicle → in-range vehicle | **model parameters** — proxy `state_dict` | `Device._v2v_share_and_aggregate` |
| `LOCAL_UPDATE` | vehicle → its RSU | **model parameters** — V2V-averaged proxy `state_dict` | `Device.send` |
| `NO_UPDATE` | vehicle → its RSU | *control only* — empty `b""`, barrier signal | `Device._build_no_update_message` |
| `CLUSTER_UPDATE` | RSU → server | **model parameters** — cluster-averaged proxy `state_dict` | `RSU._aggregate_round` |
| `NO_CLUSTER_UPDATE` | RSU → server | *control only* — empty, "cluster produced nothing" | `RSU._send_no_cluster_update` |
| `GLOBAL_UPDATE` | server → RSU, then RSU → vehicle | **model parameters** — global proxy `state_dict` | `Server._build_global_message` / `RSU._build_vehicle_global` |

Every one of these is wrapped in a signed + AES-GCM-encrypted envelope when `--security` is on
(the default).

### 2.1 Server → RSUs

`Server._broadcast_global` iterates `rsu_directory` (name → port, seeded from `RSU_BASE_PORT + i` and
refreshed from the `rsu_port` field of each incoming cluster update) and builds a **separate,
recipient-bound** `GLOBAL_UPDATE` per RSU. Separate because the envelope is encrypted under the
pairwise secret psi(Server, RSU) and the AAD names that specific recipient — one broadcast blob would
not authenticate.

### 2.2 RSU → vehicles

On receiving the server's global, the RSU decrypts and verifies it, caches it as
`global_reference_weights` (next round's trust-score reference), then **re-signs and re-encrypts it
once per vehicle** and sends to every port in `cluster_ports`. Note it sends to *all* assigned
vehicles regardless of range or contribution — the RSU has no notion of "this vehicle is
unreachable", so even an out-of-range vehicle receives its global over loopback.

### 2.3 Vehicle → RSU

Exactly one report per vehicle per round, chosen by two predicates in `Device.send`:

| Condition | Sent |
|---|---|
| `can_reach_rsu` and privacy budget intact | `LOCAL_UPDATE` (proxy weights) |
| in range but `budget_exhausted` | `NO_UPDATE` |
| out of range | `NO_UPDATE` |

Because a vehicle *always* reports something, the RSU's "all assigned vehicles reported" condition is
reachable in the normal case and the collection timer is a failsafe rather than the primary
mechanism. `NO_UPDATE` deliberately carries no parameters: an out-of-range vehicle contributes to the
barrier, not to the model.

### 2.4 Vehicle ↔ vehicle: the V2V proxy exchange that is actually implemented

This is real, not decorative, and it implements Eq. (6) of the protocol. In
`Device._v2v_share_and_aggregate`:

1. Ask the topology for all vehicles within `V2V_RANGE = 350 m`. The peer directory holds **every
   vehicle in the simulation**, not just cluster-mates, so a cross-cluster neighbour is geometrically
   possible — clusters are 1800 m apart and the spawn radius is 800 m, so two vehicles from adjacent
   clusters can be as close as 200 m.
2. Send a signed, encrypted `PEER_UPDATE` to each neighbour.
3. Collect for up to `V2V_COLLECT_TIMEOUT`, or until every neighbour has answered.
4. Average own proxy with each received proxy at equal weight — the `b = 1/(received+1)` reading of
   Eq. (6).

Two constraints worth knowing: V2V runs **only for vehicles that can also reach their RSU** (it sits
inside that branch), and the averaged result is sent upward but is *not* loaded back into the local
proxy model — the incoming `GLOBAL_UPDATE` overwrites it anyway. A receiving vehicle re-checks the
sender against its current V2V neighbour set before accepting, so a distant vehicle cannot inject a
peer proxy.

> **Was broken, now fixed (defect B4).** The rendezvous barrier that lines peers up before the
> exchange was set to **2 seconds**, while `TRAINING_SEMAPHORE` spreads a cluster's training over tens
> of seconds. The barrier therefore always timed out, its return value was ignored, and every vehicle
> logged `V2V: no peer proxies received`. Eq. (6) never executed.

### 2.5 RSU → server

One `CLUSTER_UPDATE` (with the cluster mean) or `NO_CLUSTER_UPDATE` (empty) per round. Both carry a
plaintext `rsu_port` field so the server can learn or correct the RSU's return address. The server
refuses any sender not present in `cluster_vehicle_names`, i.e. not a known RSU name.

### 2.6 How range and movement affect participation

Range enters the system in exactly three places: `can_reach_rsu` (gates `LOCAL_UPDATE`),
`get_v2v_neighbors` (gates `PEER_UPDATE` in both directions), and `assigned_rsu_coverage` (the
reported coverage metric). Mobility is one call to `move_vehicle(name, dt=10.0)` at the end of each
round: `x += speed*cos(dir)*dt`, then the heading is jittered by ±0.3 rad.

> **Open limitation, not changed (L2).** `config.SPEED_RANGE = (0, 0)` while the comment beside it
> still reads "7–28 km/h". Every vehicle has speed 0, so `move_vehicle` jitters the heading but the
> position never changes. Combined with `spawn_vehicle` placing vehicles inside
> `0.8 * V2RSU_RANGE` = 800 m of a 1000 m radio, **no vehicle is ever out of range**: the
> out-of-range branch and the range-driven `NO_UPDATE` path are dead code, and coverage reads a
> constant 21/21 in every round. A parameter choice rather than a code fault, so it was left alone —
> but nothing in the current results exercises mobility.

---

## 3. Complete communication-round flow

The sequence below is `Device.send` for one round `r`, with the RSU and server reactions interleaved
where they occur. Everything is event-driven: RSUs and the server act inside socket-receiver threads,
not on a clock.

### Step 0 — global model distribution

There is no explicit "round 0 push". Instead every vehicle's proxy is **seeded identically**:
`Device.__init__` calls `torch.manual_seed(effective_seed - device_id)` where
`effective_seed = seed + device_id`, so the seed collapses to the same value everywhere and all
proxies start from bit-identical weights. Each RSU is separately handed `server.model.state_dict()`
as its initial `global_reference_weights` so its trust filter has a reference from round 1. From
round 2 onward, distribution is simply the tail of the previous round: server → RSU → vehicle.

### Step 1 — local vehicle training (deep mutual learning)

`LOCAL_EPOCHS = 3` passes over the local loader, throttled by `TRAINING_SEMAPHORE`. Per mini-batch
both models teach each other:

- Detached soft targets are taken from both models at temperature `T = 3.0`.
- **Private model** (Eq. 4): `L = (1-alpha)*CE + alpha*KL(private || proxy_soft)`, alpha = 0.5, plain
  Adam, **no DP**. This is the model the vehicle actually uses for inference, and keeping DP off it
  is the whole point of ProxyFL.
- **Proxy model** (Eq. 5): `L = (1-beta)*CE + beta*KL(proxy || private_soft)`, beta = 0.5, trained
  with DP-SGD.

For VANET the cross-entropy is class-weighted `[1, 2, 2, 2, 2, 2]` — class 0 is benign, classes 1–5
are attacks. Both models then step an `ExponentialLR(gamma=0.95)` scheduler.

### Step 2 — proxy-model generation and DP masking

The proxy is the masked artefact. `Device._dp_sgd_step` implements Eq. (6)–(7) of the ProxyFL paper
properly, and vectorised:

1. `torch.func.vmap(torch.func.grad(...))` produces one gradient *per sample* in the batch.
2. Flatten across all parameters, take per-sample L2 norms, compute the clip factor
   `min(1, C/||g_i||_2)` with `C = DP_CLIP_NORM = 1.0`.
3. Sum the clipped gradients, divide by batch size, add `N(0, (sigma*C/B)^2)` Gaussian noise per
   parameter.
4. Step the proxy optimiser; charge one step to the `RDPAccountant`.

The protocol figure describes an *adaptive dimension-wise masking vector* (its Eq. 3). The code does
**not** implement that; it implements per-sample-clipped Gaussian DP-SGD from the ProxyFL paper
instead. The privacy protection is real and it is accounted for — it is simply a different mechanism
from the one drawn in the figure.

> **Open limitation — parameter, not code (L1).** `DP_NOISE_MULTIPLIER = 0.05` against the paper's
> sigma = 1.0. The accountant reports it honestly: measured epsilon = **1.48e5** after round 1 and
> **2.95e5** after round 2 at delta = 1e-5. The mechanism is right; the noise level makes the
> (epsilon, delta) claim vacuous. Raise sigma to >= 1.0 before quoting any privacy guarantee.

### Step 3 — proxy exchange and vehicle-level aggregation

The vehicle marks itself V2V-ready, waits at the rendezvous barrier for its in-range neighbours, then
exchanges `PEER_UPDATE`s and averages own + received proxies. That average — not the raw local
proxy — is what goes upward.

### Step 4 — upload

Sign the serialised weights, derive psi(vehicle, RSU), AES-256-GCM-encrypt with the routing metadata
as AAD, wrap as an envelope, send. The send timestamp is stored so the round's *action-to-response*
latency can be measured when the global comes back.

### Step 5 — RSU-level aggregation

1. Reject senders not in `vehicle_names`; reject envelopes with no `sig`.
2. `decrypt_envelope` — parse, resolve the AID through the TA, cross-check the claimed public key
   against the registry, reconstruct the public key, then AEAD-open. A `NO_UPDATE` is
   **signature-verified immediately** because it moves the barrier; a `LOCAL_UPDATE`'s signature is
   deferred to the batch check.
3. Duplicate senders for the round are dropped (`round_reported` set).
4. When all assigned vehicles have reported — or the inactivity window expires — aggregate:
   `batch_verify` every signature in one curve equation (Eq. 13), falling back to per-signature
   verification if the aggregate fails, so one bad signature cannot discard honest updates.
5. Trust filter (Eq. 9–10): L2 deviation of each proxy from `global_reference_weights`; accept if
   `deviation <= median(deviations) * 3.0`. Rejected vehicles are logged with `tau=0` and dropped.
6. `average_weights` over the survivors, then sign / encrypt / send as `CLUSTER_UPDATE`.

### Step 6 — server-level global aggregation

Same shape one tier up: reject unknown RSUs and unsigned envelopes, decrypt, dedupe, batch-verify,
then `average_weights` over the cluster models (Eq. 8) — a plain arithmetic mean, with **no trust
filtering at this tier**. Evaluate the result on the merged `attack1..5_test.csv` for accuracy,
weighted F1 and weighted recall; record coverage, throughput and payload bytes.

### Step 7 — distribution of the updated global model

`_broadcast_global` → each RSU → each vehicle. The vehicle verifies, then calls
`proxy_model.load_state_dict(weights)` and sets its round event. The private model is never
overwritten — it only ever absorbs the global through the DML KL term on the following round.

### Step 8 — degenerate cases

| Situation | Handling |
|---|---|
| Vehicle out of range | Sends `NO_UPDATE`, then waits on the round barrier for the global like everyone else. No parameters transmitted. |
| Vehicle privacy budget exhausted | `budget_exhausted` stops proxy training and sharing; the vehicle sends `NO_UPDATE` and keeps training its private model only. Inactive by default (`DP_MAX_EPSILON = None`). |
| Vehicle arrives after its RSU closed the round | The RSU drops it (round in `completed_rounds`). The vehicle is not stranded — it picks the round's global out of the pending-global stash. Its update for that round is lost. |
| RSU has 0 valid vehicle weights | Sends `NO_CLUSTER_UPDATE` so the server's round can still close. |
| RSU aggregation throws | Guarded: logs the traceback and still emits `NO_CLUSTER_UPDATE`. *(New — defect B2.)* |
| No RSU supplied a model | Server re-broadcasts the current global unchanged and records `successful_updates = 0`. |
| Server aggregation throws | Guarded: broadcasts the last known global and always releases `training_done_event` on the final round. *(New — defect B2.)* |
| Vehicle never receives a global | Gives up after `TIMEOUT` and starts the next round on its own proxy. Logged as a timeout. |

---

## 4. Security architecture

### Terminology

- **Certificateless public-key cryptography (CL-PKC)** — a middle ground between PKI and
  identity-based crypto. There are no certificates to distribute, and — unlike identity-based
  schemes — the key generation centre *cannot* forge your signature. Your private key has two halves:
  a **partial private key** issued by the KGC, and a **secret value** you generate yourself and never
  reveal. Verifiers *reconstruct* your public key from public material instead of checking a signed
  certificate.
- **Pseudonymous identity (AID)** — a per-vehicle alias used on the air interface so observers cannot
  link messages to a real vehicle, while the TA retains the ability to recover the real identity for
  accountability.
- **AEAD / AAD** — Authenticated Encryption with Associated Data. AES-GCM both encrypts the payload
  and authenticates it together with unencrypted "associated data" — here the routing header. Change
  the header and decryption fails.

Curve: **NIST P-256**, via the vendored MIRACL Core `nist256` package. Generator `P`, order `q`
(256-bit), points encoded uncompressed as 32-byte x || 32-byte y. Four domain-separated
hash-to-scalar functions `H0`…`H3`, each SHA-256 over a length-prefixed encoding of every argument,
reduced into `[1, q-1]`.

### 4.1 Vehicle registration and pseudonymous identities

`enroll_mvd(real_id)` → `register(name, real_id)`. Registration builds a `CertificatelessSigner`,
then the Authority immediately re-derives the real identity from the fresh AID and refuses the
registration if it does not match. `main.py` asserts the same round-trip for every vehicle
(`assert authority.recover_identity(signer.aid) == dev_name`), so identity recoverability is checked
at startup rather than assumed.

### 4.2 Certificateless public/private keys

| Quantity | Definition | Held by |
|---|---|---|
| `t`, `T_pub = t·P` | TA master secret / public | TA |
| `s`, `P_pub = s·P` | KGC master secret / public | KGC |
| `AID1 = k·P` | random pseudonym point | node (public) |
| `AID2 = ID XOR H0(t·AID1 \|\| T_pub)` | masked real identity | node (public) |
| `alpha = H1(AID, U, P_pub)` | KGC binding scalar | public |
| `w = u + alpha·s mod q`, `U = u·P` | partial private key from the KGC | node (`w` secret, `U` public) |
| `x`, `X = x·P` | self-chosen secret value | node (`x` secret, `X` public) |
| `beta = H2(AID, X)` | reconstruction scalar | public |
| `Q = U + beta·X` | full public key component | public |
| `pk = (Q, U)`, `sk = (w, x)` | the node's key pair | node |

The signer verifies its own partial key at construction — Eq. (11), `w·P == U + alpha·P_pub` — and
raises `SecurityError` if the KGC misbehaved.

### 4.3 Pairwise shared-secret establishment

The figure writes `psi_ij = x_i·pk_j + w_i·pk_j`, which is ill-typed: it adds points without saying
which component of `pk_j` is meant. `derive_shared_secret` substitutes the well-typed ECDH form and
documents the substitution in the source:

```
psi(i -> j) = x_i * X_j  +  w_i * (U_j + H1(AID_j, U_j, P_pub) * P_pub)
            = x_i x_j P  +  w_i w_j P

psi(j -> i) = x_j * X_i  +  w_j * (U_i + H1(AID_i, U_i, P_pub) * P_pub)
            = x_j x_i P  +  w_j w_i P            <-- identical

key = SHA-256( "ProxyFL/AES-GCM/v1" || SHA-256("ProxyFL/psi/v1" || psi) )
```

The inner bracket reconstructs `w_j·P` from public values only, which is why the peer's secret `w_j`
is never needed. Both halves of the private key contribute, so compromising the KGC alone does not
yield the channel key.

### 4.4 Encryption and authenticated decryption

AES-256-GCM with a fresh 12-byte random nonce per message. The AAD is a canonical, key-sorted JSON
object `{recipient, round, sender, type}` — so a captured envelope cannot be relabelled as a
different message type, re-addressed to another node, or replayed into a different round without the
tag failing. Ciphertext, nonce and 16-byte tag ride the envelope as separate base64-tagged byte
fields.

### 4.5 Signature generation

```
r     <- random in Z_q*
R     =  r * P
gamma =  H3(AID, message, Q, U, R)          # message = plaintext proxy bytes
eta   = (r + gamma * (w + beta * x)) mod q  # retried if eta == 0
sig   = (eta, R)
```

### 4.6 Public-key reconstruction and individual signature verification

Before any signature check, `_reconstruct_ok` recomputes `Q' = U + H2(AID, X)·X` and compares it to
the advertised `Q`. Only then:

```
alpha =  H1(AID, U, P_pub)
gamma =  H3(AID, message, Q, U, R)
accept  iff   eta * P  ==  R + gamma * (Q + alpha * P_pub)

why it holds:  eta*P = r*P + gamma*(w + beta*x)*P
                     = R   + gamma*(w*P + beta*X)
                     = R   + gamma*(U + alpha*P_pub + beta*X)
                     = R   + gamma*(Q + alpha*P_pub)
```

### 4.7 Batch signature verification

Eq. (13), enabled by default and used whenever an aggregator holds more than one update. Each
signature gets a fresh random coefficient `y_i` from Z_q*, and one curve equation covers the whole
batch:

```
(SUM y_i eta_i) * P  ==  SUM y_i R_i  +  SUM y_i gamma_i Q_i
                                      +  (SUM y_i gamma_i alpha_i) * P_pub
```

The random coefficients matter: without them an attacker could submit two signatures whose errors
cancel. The aggregate check is all-or-nothing, so on failure both the RSU and the server **fall back
to per-signature verification**, exclude only the offending senders, and log each exclusion — one
forged update cannot poison a whole batch of honest work.

### 4.8 Which cryptographic operations use MIRACL

| Operation | Implementation | MIRACL? |
|---|---|---|
| P-256 point multiply / add, on-curve checks, field arithmetic, random scalars | `miracl_python/nist256` (`ecp`, `fp`, `big`) | **Yes** — MIRACL Core, pure Python |
| SHA-256 for H0–H3, key derivation, AID masking | `crypto_protocol._sha256` | Only if `miracl_core.dll` is built |
| AES-256-GCM seal / open | `encrypt_payload` / `decrypt_payload` | Only if `miracl_core.dll` is built |

> **Open limitation — measurement caveat (L5).** `crypto_protocol/miracl_core.dll` **is not present in
> the repo** — only the C source and `build_miracl_bridge.bat`. So `_bridge is None` and the symmetric
> primitives fall back to Python's `hashlib` and the `cryptography` package's AES-GCM. That fallback
> was invisible; the run banner now states the active backend, because every reported crypto timing
> depends on which one is live. Elliptic-curve arithmetic is *always* MIRACL and it is pure Python,
> which is why it dominates the crypto cost.

### 4.9 How invalid, duplicate, unsigned or unauthorized updates are rejected

| Attack / fault | Rejected by | Where |
|---|---|---|
| Sender is not an assigned cluster member / known RSU | name allow-list | `RSU.on_receive`, `Server.on_receive` |
| Unsigned update | `"sig" not in msg` → drop | both aggregators |
| Wrong message type or wrong recipient | envelope field check + AAD mismatch | `parse_envelope` |
| Unregistered or forged pseudonym | TA recovers the AID; fails if it maps to no enrolled MVD identity | `Authority.resolve_public_info` |
| Sender claiming someone else's AID | AID→owner lookup must match the claimed sender name | `Authority.resolve_public_info` |
| Substituted public key | wire `pk` compared byte-for-byte with the TA/KGC registry, *and* `Q' = U + beta·X` recomputed | `parse_envelope` |
| Malformed signature (`eta` outside `(0, q)`, off-curve `R`) | range and on-curve checks | `signature_from_wire`, `verify` |
| Tampered ciphertext or relabelled header | AES-GCM tag over payload + AAD | `decrypt_payload` |
| Forged signature | Eq. 13 batch check, then per-signature Eq. 12 fallback | `CertificatelessVerifier` |
| Duplicate submission in one round | `round_reported` set — the second report is ignored | both aggregators |
| Replay from an earlier round | round bound into the AAD; old rounds sit in `completed_rounds` | `message_aad` + round bookkeeping |
| Model-poisoning / outlier update | Eq. 9–10 trust score: L2 deviation vs. previous global, cutoff = median x 3 | `models.filter_trusted_weights` |
| Non-tensor or pickled-object payload | `torch.load(weights_only=True)` + tensor-only dict check | `model_codec.deserialize_weights` |
| Oversized or non-JSON frame | 16 MiB frame cap; strict wire codec | `network.Receiver`, `wire_codec` |

> **Two residual weaknesses in the protocol as coded.**
> **(L7) Sign-then-encrypt with the signature outside the envelope.** The signature is computed over
> the *plaintext* and travels in the clear next to the ciphertext, so anyone who can guess a candidate
> payload can confirm it by verifying the signature. Moving `sig` inside the AEAD would close that.
> **(L8) No nonce cache.** Replay is stopped by AAD round binding plus per-round duplicate-sender
> dedup, not by remembering nonces — adequate here, but not a general anti-replay defence.

---

## 5. Aggregation hierarchy

```
TIER 0   private model  f_phi   (VanetIDSSmall / VanetIDS / VanetIDSLarge)
         |  never transmitted. never aggregated. learns only via the DML KL term.
         |  Adam, no DP.
         v
TIER 1   proxy model  h_theta   (ProxyModel, 4->32->6, 358 params, common arch)
         |  trained by DP-SGD: per-sample clip C=1.0 then Gaussian noise
         |  = "masked" proxy for this vehicle, this round
         v
TIER 1b  V2V-averaged proxy                                        Eq. (6)
         |  mean( own proxy , each in-range peer's proxy )  equal weights
         |  transmitted upward; NOT loaded back into the local proxy
         v
TIER 2   RSU cluster proxy                                         Eq. (7)
         |  batch-verify -> trust filter Eq.(9-10) -> average_weights()
         |  arithmetic mean over the surviving vehicles of this cluster
         v
TIER 3   server global proxy                                       Eq. (8)
         |  batch-verify -> average_weights()        (no trust filter here)
         |  arithmetic mean over the reporting clusters
         v
         broadcast down: server -> RSU -> every assigned vehicle
         each vehicle does proxy_model.load_state_dict(global)
```

### What kind of value each thing is

| Value | Kind | Representation on the wire |
|---|---|---|
| Private model weights | model parameters | never leaves the vehicle |
| Proxy model weights | model parameters | `torch.save` of a tensor-only `state_dict` → opaque bytes |
| Per-sample gradients, clip factors, noise | gradients | never transmitted — consumed inside `_dp_sgd_step` |
| V2V / cluster / global averages | masked proxy parameters | same serialisation as above |
| Envelope | encrypted container | `{type, sender, recipient, round, aid, pk, sig, ciphertext, nonce, tag}` |
| Signature | signature | `{eta: int, R: 64 bytes}`, cleartext in the envelope |
| AID and `pk` | public key material | cleartext — the verifier needs them before it can open anything |
| `NO_UPDATE` / `NO_CLUSTER_UPDATE` | control message | signed and encrypted `b""` — carries only its header |
| `rsu_port` | routing metadata | plaintext, outside the AEAD, used only to learn a return address |
| Trust score tau, deviation | control decision | computed at the RSU, never transmitted |

> **Open limitation (L3).** `average_weights` is an **unweighted** arithmetic mean at both tiers. A
> 2-vehicle cluster therefore counts as much as a 6-vehicle cluster in the global model, and a vehicle
> with more local data counts no more than one with less. This matches Eq. (7)/(8) exactly as drawn in
> the protocol figures, but it is not sample-size-weighted FedAvg. Note also that
> `models.jsd_weighted_average` and `calculate_jsd` exist and are *never called* — the server does
> plain FedAvg. The stale docstring claiming JSD-weighted aggregation has been corrected.

---

## 6. Metrics

Every metric is a row in `metrics.csv` keyed by `(node, round)`. Nodes include all vehicles, all five
RSUs and `Server`; round 0 holds key-generation cost only. Durations accumulate in milliseconds via
`Timer`; roll-ups and energy are computed at export time by `_derived_metrics`.

### 6.1 Accuracy and loss

| Column | Measured on | By |
|---|---|---|
| `train_accuracy_pct` / `train_loss` | private model, local training batches, averaged over 3 epochs | `Device.train_epoch` |
| `private_test_accuracy_pct` | private model, the vehicle's local 20% held-out split | `evaluate_private_model` |
| `global_proxy_accuracy_pct` | global proxy, merged `attack1..5_test.csv` | `Server.evaluate_global_model` |
| `epsilon` / `delta` | cumulative RDP→(epsilon, delta) for the proxy | `privacy.RDPAccountant` |

The server also prints weighted F1 and weighted recall, which are *not* persisted to the CSV. Proxy
test accuracy is printed per vehicle but not logged either, and **proxy training loss is never
recorded at all** — `train_epoch` returns a hard-coded `0.0` as its third value and the caller
discards it (L4). If you need proxy-side learning curves, that is the gap to close.

### 6.2 Training and non-training energy

One linear OBU power model: `E = 10.88 W * x_op * t_ms / 1000`, with utilisation factors from
`config`.

| Column | `x_op` | Time source |
|---|---|---|
| `energy_training_j` | 1.0 | `training_ms` — the 3-epoch DML + DP-SGD block |
| `energy_security_j` | 0.4 | keygen + sign + verify + batch-verify + encrypt + decrypt |
| `energy_communication_j` | 0.6 | `communication_tx_ms + communication_rx_ms` |
| `energy_idle_j` | 0.2 | `idle_latency_ms` = round execution − active time |
| `energy_total_j` | — | **security + communication only.** Despite the name it excludes training and idle — it is the non-training overhead figure. Do not read it as a total. |

### 6.3 Communication latency

`communication_tx_ms` is the wall-clock of encode + connect + `sendall`, charged to the sender.
`communication_rx_ms` is frame-receive wall-clock, charged to the receiving node. Both are loopback
TCP and Python scheduling, *not* radio time. The modelled radio side is separate:
`vanet_wireless_bits`, `vanet_airtime_s` and `vanet_link_capacity_bps`, accumulated per successful
hop from the Shannon capacity at that hop's distance.

### 6.4 Action-to-response latency

The one strict request→response measurement in the system: from the moment a vehicle finishes sending
its `LOCAL_UPDATE` (or `NO_UPDATE`) to the moment a *verified* `GLOBAL_UPDATE` for that round is
accepted. It therefore contains the RSU's remaining collection wait, RSU aggregation, the server's
collection wait, server aggregation and evaluation, and both downward hops. If the round times out,
no value is recorded.

### 6.5 Throughput

- `throughput_bytes_per_sec` — total cluster-model payload bytes the server accepted, divided by the
  round's server-side wall-clock (first `CLUSTER_UPDATE` arrival → aggregation). A server-collection
  throughput, not a link rate.
- `throughput_updates_per_sec` — the same denominator with a count of cluster models in the numerator.
  Kept for CSV compatibility.
- `vanet_goodput_bps` — modelled bits divided by modelled airtime. Because airtime is itself
  `bits / capacity`, this is a capacity-weighted harmonic mean of the hops used that round,
  hard-capped at 27 Mb/s. Derived, never observed.

### 6.6 Cryptographic operation and signature-verification time

| Column | What it actually times |
|---|---|
| `key_generation_ms` | one-off certificateless registration. Round 0 only. |
| `signature_generation_ms` | one `sign()`: random scalar, `r·P`, H3, one modular multiply. |
| `signature_verification_ms` | per-signature Eq. 12 checks — `NO_UPDATE` control messages, and the single-verify fallback after a failed batch. |
| `batch_verification_ms` | the receiver's Eq. 13 wall-clock **divided equally among the participants of that batch**, so per-vehicle security latency sums cleanly. |
| `batch_verification_receiver_ms` | the same batch cost, undivided, charged to the RSU or server that paid it. |
| `encryption_ms` | AAD build + peer-key lookup + ECDH shared secret + AES-GCM seal. |
| `decryption_ms` | the whole `decrypt_envelope`: parse, AID recovery, registry cross-check, public-key reconstruction, ECDH, AES-GCM open. Charged to the *sender's* node name, not the receiver's. Much more than "AES time". |

### 6.7 Vehicles remaining in range, per RSU and total

`Server._print_in_range_vehicle_counts` runs once per aggregated round, asks the topology for
`assigned_rsu_coverage`, and records `vehicles_in_range` / `vehicles_assigned` per RSU plus
`vehicles_in_range_total` / `vehicles_assigned_total` on `Server`. "In range" means Euclidean distance
to the *assigned* RSU <= 1000 m — a vehicle that has drifted inside another RSU's radius still counts
as out of range for its own.

### 6.8 Why these values rise and fall between rounds

- **Everything timed is real wall-clock on a contended host.** 21 vehicle threads, 5 RSU threads and
  the server share 8 training slots, one GIL and a single-threaded BLAS. `training_ms` for an
  identical workload varies with how many peers were training at the same moment.
- **Action-to-response depends on queue position, not on the network.** The first vehicle in a cluster
  to report waits for its whole cluster, then for the slowest cluster, then for the server. The last
  vehicle waits almost nothing. This is the dominant source of spread in the latency plots.
- **Security latency scales with envelope count, and per-vehicle batch cost scales *inversely*.** A
  round where six vehicles reported has a larger receiver-side batch cost but a *smaller*
  `batch_verification_ms` per vehicle, because the cost is divided by the number of participants.
- **Idle absorbs every barrier.** `idle_latency_ms = round_execution - (training + security +
  communication)`, so V2V rendezvous waits and RSU/server collection waits land here. It moves
  inversely to how well the round's participants were synchronised.
- **Throughput is denominator-driven.** The proxy is 358 parameters (~2 KB serialised), so the
  numerator is nearly constant. A tightly-clustered round reports high throughput; a straggler-spread
  round reports low. In the verification run this alone moved the figure between 522 B/s and
  1616 B/s.
- **Vehicles in range is currently flat at 21/21** because speed is zero and the spawn radius sits
  inside the RSU radius. With a non-zero `SPEED_RANGE` it would fall as vehicles disperse and rise
  when a heading jitter turns one back.
- **Global accuracy can dip as well as climb.** Proxies are DP-noised before averaging, participation
  varies, and a trust rejection removes a vehicle from that round's mean — so a monotone curve is not
  expected.

---

## 7. File-wise implementation mapping

| File | Responsibility | Owns |
|---|---|---|
| `main.py` | Orchestration and bootstrap. Parses the CLI, seeds RNGs, draws 2–10 vehicles per RSU, prepares the VANET data partitions, builds the single `Authority` and enrols/registers every identity, places the five RSUs, spawns vehicles, constructs Server → RSUs → Devices in that order, starts the threads, waits on `training_done_event`, then shuts sockets down and writes logs, CSVs and plots. | process lifecycle |
| `device.py` | The vehicle. Two models, data partitioning with an 80/20 local split, DML training, vectorised DP-SGD on the proxy, RDP accounting, local evaluation, V2V rendezvous and gossip, envelope construction, the round loop, and the pending-global barrier. | all local learning |
| `rsu.py` | Cluster head. Verifies and decrypts vehicle envelopes, dedupes per round, runs the inactivity-window collection timer, batch-verifies, applies the Eq. 9–10 trust filter, FedAvgs the cluster, forwards upward, and re-authenticates the global downward once per vehicle. | tier-2 aggregation |
| `server.py` | Central aggregator. Collects one report per RSU, batch-verifies, FedAvgs the clusters, evaluates the global proxy on the attack test set, records coverage / throughput / payload metrics, broadcasts recipient-bound globals, and releases the simulation barrier. | tier-3 aggregation |
| `vanet_sim.py` | Spatial ground truth. Thread-safe vehicle and RSU positions, V2V and V2RSU range queries, per-round mobility, the V2V readiness barrier, per-RSU coverage counting, and the `C{c}_V{n}` identifier format. | geometry and mobility |
| `crypto_protocol.py` | The whole certificateless stack: MIRACL P-256 binding, H0–H3, the TA/KGC/MVD `Authority`, `CertificatelessSigner`, `CertificatelessVerifier` (single Eq. 12 and batch Eq. 13), pairwise ECDH, AES-256-GCM with AAD, and envelope build / parse / decrypt / verify. | security plane |
| `metrics.py` | Instrumentation. The global `MetricsTracker`, `Timer` / `BatchTimer`, byte and wireless-hop counters, derived latency roll-ups, the OBU energy model, and CSV export with a fixed column order. | measurement |
| `logger.py` | Human-readable training record. PrettyTable tables plus round-keyed dicts for train loss / accuracy, private accuracy, privacy spend and global accuracy; writes `*_training_logs.txt` and delegates plotting. | learning record |
| `plot_metrics.py` | Reporting. Parses the log text and the metrics CSV and renders 13 PNGs (accuracy, loss, energy x3, latency x3, action-to-response, crypto x2, throughput, coverage) with legends placed outside the axes, plus a data-backed `vanet_plot_explanations.md`. | figures |
| `config.py` | Single source of tunables: rounds, batch size, local epochs, collection deadlines, security and trust flags, V2V windows, the OBU energy profile, ranges and speeds, the fixed five-RSU layout, DP parameters, DML alpha/beta/T, ports — plus the thread, BLAS and console safety setup and the training semaphore. | configuration |

**Supporting files:** `models.py` (architectures, `dml_loss`, `average_weights`,
`filter_trusted_weights`), `privacy.py` (RDP accountant), `data_utils.py` (dataset + leakage-free
partition scaler), `network.py` (framed TCP), `wire_codec.py` (safe JSON + base64 outer encoding),
`model_codec.py` (tensor-only `torch.load`), `vanet_channel.py` (measurement-only link budget),
`shared_logger.py` (the logger singleton), `run_grid_experiments.py` (parameter sweeps).

---

## 8. Worked example, diagrams and defects

### 8.1 One vehicle update, end to end

`C2_V3`, assigned to `RSU_2_East` at (1800, 0), sitting 583 m away, round 3:

```
 1  TRAIN      3 epochs of DML on ~4,012 local rows (semaphore slot acquired).
               private VanetIDS : (1-0.5)*CE + 0.5*KL(private || proxy_soft), Adam, no DP
               proxy ProxyModel : per-sample grads via vmap -> clip to C=1.0
                                  -> mean -> + N(0, (0.05*1.0/32)^2) -> Adam step
                                  -> RDPAccountant.step(1)  once per batch

 2  EVALUATE   private acc on the local 20% split; proxy acc on the same split (printed).
               log train_loss, train_accuracy_pct, private_test_accuracy_pct, epsilon.

 3  RENDEZVOUS topology.mark_v2v_ready("C2_V3", 3)
               wait_for_v2v_ready(peers within 350 m = [C2_V1, C2_V2, C2_V5])

 4  RANGE      get_distance_to_rsu = 583 m  <=  V2RSU_RANGE 1000 m   -> may upload

 5  V2V        snapshot proxy state_dict (358 params)
               for each neighbour:  sign(bytes) -> psi(C2_V3, peer) -> AES-GCM(aad)
                                    -> PEER_UPDATE -> peer port
               collect <= 10 s  ->  received 3 peer proxies
               proxy_out = mean(own, C2_V1, C2_V2, C2_V5)              <-- Eq. (6)

 6  UPLOAD     raw = torch.save(proxy_out)                             ~2 KB
               sig = (eta, R) over raw                                 <- Timer signature_generation
               aad = {"recipient":"RSU_2_East","round":3,
                      "sender":"C2_V3","type":"LOCAL_UPDATE"}
               ct, nonce, tag = AES-256-GCM(SHA256(psi), raw, aad)      <- Timer encryption
               envelope = {type, sender, recipient, round, aid, pk, sig, ct, nonce, tag}
               send_msg -> 127.0.0.1:5002        _request_sent_at[3] = now

 7  RSU IN     sender in vehicle_names? yes.  "sig" present? yes.
               decrypt_envelope: AID -> TA recovery -> registry pk match
                                 -> Q' = U + H2(AID,X)*X == Q -> AES-GCM open
               round_reported[3].add("C2_V3");  inactivity timer restarted
               ... all 6 assigned vehicles reported -> cancel timer, aggregate

 8  RSU AGG    batch_verify 6 signatures in one curve equation          <- Eq. (13)
               trust: ||prev_global - w||_2 per vehicle; cutoff = median * 3
                      C2_V3 deviation within cutoff -> tau = 1 -> kept
               cluster = average_weights(6 survivors)                   <- Eq. (7)
               sign + encrypt under psi(RSU_2_East, Server) -> CLUSTER_UPDATE -> :8000

 9  SERVER     5 of 5 RSUs reported -> batch_verify -> global = mean(5 clusters)  <- Eq. (8)
               evaluate on attack1..5_test.csv -> acc 62.0%, F1 0.5012, recall 0.620
               record coverage 21/21, throughput, payload bytes

10  DOWN       server -> 5 recipient-bound GLOBAL_UPDATEs
               RSU_2_East decrypts, caches as global_reference_weights,
               re-signs + re-encrypts once per vehicle -> all 6 cluster ports

11  APPLY      C2_V3 verifies -> action_to_response_ms recorded
               proxy_model.load_state_dict(global);  round_event.set()
               private model untouched -- it absorbs the global next round via KL

12  MOVE       topology.move_vehicle("C2_V3")   speed 0 -> position unchanged
               record device_round_execution_ms;  proceed to round 4
```

### 8.2 Architecture diagram

```
                     +---------------------------------------------------+
                     |  Authority  (TA + KGC + MVD)    crypto_protocol.py|
                     |  t, T_pub  |  s, P_pub  |  MVD registry           |
                     |  AID issue / recover . partial keys . pk registry  |
                     +---------------------------------------------------+
                        ^ shared in-process reference (no network hop)
                        | used by every node for sign / verify / psi
   .....................|...............................................
                        |
                +---------------------+
                |  CENTRAL FL SERVER  |  server.py            port 8000
                |  global proxy model |  FedAvg over clusters  Eq. (8)
                |  eval: attack1..5   |  coverage / throughput
                +----------+----------+
                           |  GLOBAL_UPDATE (one per RSU, recipient-bound)
      +--------------+-----+------+--------------+--------------+
      |              |            |              |              |
 +---------+    +---------+  +---------+    +---------+    +---------+
 | RSU_0   |    | RSU_1   |  | RSU_2   |    | RSU_3   |    | RSU_4   |
 | Central |    | North   |  | East    |    | South   |    | West    |
 | (0,0)   |    |(0,1800) |  |(1800,0) |    |(0,-1800)|    |(-1800,0)|
 |  :5000  |    |  :5001  |  |  :5002  |    |  :5003  |    |  :5004  |
 +----+----+    +----+----+  +----+----+    +----+----+    +----+----+
      |              |            |              |              |
      | CLUSTER_UPDATE upward   /   GLOBAL_UPDATE downward, re-signed per vehicle
      | verify -> trust filter Eq.(9-10) -> FedAvg Eq.(7)
      |              |            |              |              |
 +----+----+    +----+----+  +----+----+    +----+----+    +----+----+
 | C0_V1.. |    | C1_V1.. |  | C2_V1.. |    | C3_V1.. |    | C4_V1.. |
 |  2-10   |    |  2-10   |  |  2-10   |    |  2-10   |    |  2-10   |
 | vehicles|    | vehicles|  | vehicles|    | vehicles|    | vehicles|
 +---------+    +---------+  +---------+    +---------+    +---------+
    :6000+         :6100+       :6200+         :6300+         :6400+

 inside one vehicle (device.py):
    +-----------------------------------------------------------+
    |  private model f_phi   <--- DML KL --->   proxy h_theta    |
    |  heterogeneous, no DP                     common arch,     |
    |  never transmitted                        DP-SGD, shared   |
    |  local data: 80% train / 20% test    RDPAccountant         |
    +-----------------------------------------------------------+

 lateral V2V (<= 350 m, any cluster):    C2_V1 <--PEER_UPDATE--> C2_V3
                                               \                /
                                                --- C2_V5 ------

 range gates PARTICIPATION only.  transport is always loopback TCP.
 vanet_channel.py computes capacity for METRICS -- never delay, never loss.
```

### 8.3 Message-sequence diagram for one FL round

```
 Vehicle A     Vehicle B       RSU_k         SERVER      A and B are in one cluster
 (Ck_V1)       (Ck_V2)                                   and within 350 m of each other
    |              |              |              |
    |-- DML + DP-SGD, 3 epochs ---|              |       semaphore-limited
    |              |-- DML + DP-SGD ...          |       finishes later than A
    |              |              |              |
    |-- mark_v2v_ready(r) --------|              |
    |<=== barrier: wait for B (V2V_READY_TIMEOUT) |
    |              |-- mark_v2v_ready(r)         |
    |<-- PEER_UPDATE (signed+enc) -|              |      both directions
    |--- PEER_UPDATE (signed+enc) ->              |
    | proxy_A' = mean(A,B)        | proxy_B' = mean(B,A)      <-- Eq. (6)
    |              |              |              |
    |-- LOCAL_UPDATE(proxy_A') -->|              |       t0 for action-to-response
    |              |              | verify+decrypt, dedupe
    |              |              | restart inactivity timer
    |              |-- LOCAL_UPDATE(proxy_B') -->|
    |              |              | all assigned reported
    |              |              | batch_verify   Eq.(13)
    |              |              | trust filter   Eq.(9-10)
    |              |              | FedAvg         Eq.(7)
    |              |              |-- CLUSTER_UPDATE -->|
    |              |              |              | verify+decrypt, dedupe
    |              |              |              | restart inactivity timer
    |              |              |         (other 4 RSUs report)
    |              |              |              | batch_verify Eq.(13)
    |              |              |              | FedAvg       Eq.(8)
    |              |              |              | evaluate on attack test set
    |              |              |              | record coverage / throughput
    |              |              |<-- GLOBAL_UPDATE (bound to RSU_k)
    |              |              | verify, cache as trust reference
    |<-- GLOBAL_UPDATE (bound to A)|              |
    |              |<-- GLOBAL_UPDATE (bound to B)|
    | verify -> load_state_dict   |              |       t1: action-to-response
    | round_event.set()           |              |
    |-- move_vehicle() ---------- round r+1 -----|

 Degenerate branches:
   A out of range        -> A sends NO_UPDATE (empty, signed) and waits on the barrier
   A budget exhausted    -> A sends NO_UPDATE, keeps training its private model only
   RSU has 0 valid       -> RSU sends NO_CLUSTER_UPDATE
   RSU aggregate raises  -> guarded: still sends NO_CLUSTER_UPDATE
   no RSU reported       -> SERVER re-broadcasts the unchanged global
   A gets no global      -> A gives up after TIMEOUT, starts r+1 on its own proxy
```

### 8.4 Protocol phase to implementing file/functions

| Protocol phase | File | Functions |
|---|---|---|
| (i) Initialization / Setup — TA `t,T_pub`, KGC `s,P_pub`, H1–H3 | `crypto_protocol.py` | `Authority.__init__`, `hash_to_scalar`, `_h0_mask` |
| (ii) Registration — MVD enrolment, `AID1=k·P`, `AID2=ID XOR H0` | `crypto_protocol.py`, `main.py` | `enroll_mvd`, `generate_pseudo_identity`, `recover_identity`, `register` |
| (ii) Key Generation — secret value, partial key, Eq. 11 check | `crypto_protocol.py` | `extract_partial_private_key`, `CertificatelessSigner.__init__` |
| Shared secret `psi_ij` | `crypto_protocol.py` | `derive_shared_secret`, `CertificatelessSigner.shared_secret_for` |
| (iii) Training initiation — local loss and gradient (Eq. 1–2) | `device.py`, `models.py` | `Device.train_epoch`, `dml_loss`, `proxy_training_objective` |
| Proxy generation / masking (Eq. 3–5) | `device.py`, `privacy.py` | `Device._dp_sgd_step`, `RDPAccountant.step` |
| (iii) Encryption — `E_psi(d_i) = c_i` | `crypto_protocol.py` | `encrypt_payload`, `message_aad`, `build_envelope` |
| (iii) Signing — `gamma=H3`, `eta=r+gamma(w+beta·x)`, `sigma=(eta,R)` | `crypto_protocol.py` | `CertificatelessSigner.sign` |
| Request assembly — `req_i = (sigma_i, c_i, AID_i)` | `crypto_protocol.py`, `device.py` | `build_envelope`, `Device.send` |
| (iv) Authenticated decryption — `D_psi(c_i) = d_i` | `crypto_protocol.py` | `decrypt_payload`, `parse_envelope`, `decrypt_envelope` |
| (iv) Public-key reconstruction — `beta=H2(AID,X)`, `Q'=U+beta·X`, compare | `crypto_protocol.py` | `reconstruct_public_key`, `CertificatelessVerifier._reconstruct_ok` |
| (iv) Single signature verification (Eq. 12 / 14) | `crypto_protocol.py` | `CertificatelessVerifier.verify`, `verify_envelope` |
| (iv) Batch verification (Eq. 13) | `crypto_protocol.py` | `CertificatelessVerifier.batch_verify`, `_batch_coefficient` |
| Trust score / malicious detection (Eq. 9–10) | `models.py`, `rsu.py` | `model_l2_deviation`, `filter_trusted_weights`, `RSU._aggregate_round` |
| V2V proxy aggregation (Eq. 6) | `device.py`, `vanet_sim.py` | `_v2v_share_and_aggregate`, `mark_v2v_ready`, `wait_for_v2v_ready`, `get_v2v_neighbors` |
| Low-level RSU aggregation (Eq. 7) | `rsu.py`, `models.py` | `RSU._aggregate_round`, `average_weights` |
| Global aggregation (Eq. 8) | `server.py`, `models.py` | `Server._aggregate_round`, `average_weights` |
| Global distribution | `server.py`, `rsu.py`, `device.py` | `_broadcast_global`, `_build_vehicle_global`, `_decode_rsu_global`, `_handle_verified_global` |
| Mobility and range constraints | `vanet_sim.py` | `move_vehicle`, `can_reach_rsu`, `get_v2v_neighbors`, `assigned_rsu_coverage` |
| Privacy accounting | `privacy.py` | `RDPAccountant._rdp_one_order`, `_rdp_to_epsilon` |

### 8.5 Implementation differences from the ProxyFL paper

Reference: Kalra, Wen, Cresswell, Volkovs and Tizhoosh, *Decentralized federated learning through
proxy model sharing*, Nature Communications 14:2899 (2023).

| Paper | This implementation |
|---|---|
| **Fully decentralized, no server.** PushSum gossip over a directed exponential graph, `Theta^(t+1) = P^(t) Theta^(t)`. | **Three-tier hierarchy with a central server** plus a lateral V2V average. A deliberate extension, not a defect — but it reintroduces the single central party ProxyFL set out to remove. |
| De-biasing weights `w^(t+1) = P^(t) w^(t)`, then de-bias `theta/w`. | Not present. Plain arithmetic means at every tier; no PushSum, no column-stochastic `P`, no de-biasing weights. |
| Exponential protocol: each client sends exactly one proxy per round, `O(1)` per round regardless of client count. | Each vehicle sends one proxy to its RSU *plus* one to every in-range V2V neighbour; each RSU sends one to the server. Communication grows with neighbour density. |
| OpenMPI across 8 V100 GPUs, one client per GPU. | One Python process, threads, loopback TCP, an 8-way training semaphore on CPU. |
| sigma = 1.0, C = 1.0, batch 250, Poisson sub-sampling with replacement for correct DP accounting. | sigma = 0.05, C = 1.0, batch 32, a standard shuffled `DataLoader`. The accountant assumes Poisson sub-sampling at rate `B/N`, so the reported epsilon is optimistic for the sampler actually used — on top of sigma being 20x too small. |
| Datasets: MNIST, Fashion-MNIST, CIFAR-10, Kvasir, Camelyon-17. | MNIST plus a VANET intrusion-detection set (VeReMi-style, 4 features, 6 classes). Fashion-MNIST, CIFAR-10 and the histopathology tracks are absent. |
| Baselines: FedAvg, AvgPush, CWT, FML, Regular, Joint. | No baselines. ProxyFL-VANET only. |
| alpha, beta = 0.5 (0.3 for histopathology); DML alternates gradient steps between the two models. | alpha, beta = 0.5, T = 3.0. Same alternation, but soft targets for both models are snapshotted *before* either update, so within a batch each learns from the other's pre-update state. |
| Per-client epsilon tracked; a client drops out when its budget is spent. | Implemented (`budget_exhausted`) but disabled: `DP_MAX_EPSILON = None`. |
| No trust filtering — not part of ProxyFL. | Adds the Eq. 9–10 L2 trust score at the RSU tier, from the VANET protocol. |
| No cryptographic layer beyond DP. | Adds a full certificateless authentication and AEAD layer — this project's contribution, not the paper's. |
| Proxy is "generally smaller than the private model". | Holds: a 358-parameter proxy against 2,790–45,670-parameter private models. |

### 8.6 Implementation differences from the certificateless-protocol figures

- **Shared secret.** Figure: `psi_ij = x_i·pk_j + w_i·pk_j`. Code:
  `x_i·X_j + w_i·(U_j + H1(AID_j, U_j, P_pub)·P_pub)`, then SHA-256. The figure's expression is
  ill-typed; the substitution is symmetric and documented in the source. *Deliberate correction.*
- **Adaptive masking (Eq. 3).** The figure's dimension-wise mask is not implemented. The code uses
  per-sample-clipped Gaussian DP-SGD from the ProxyFL paper instead — a different mechanism, with
  real protection.
- **AES-256 → AES-256-GCM with AAD.** The figure specifies AES-256 for the KGC channel and
  `E_psi(d)` for the data channel. The code uses authenticated encryption throughout and binds the
  routing header. Stronger than specified.
- **KGC delivery channel.** The figure sends `ppk_i` over TLS/SSL or AES-256. In code the partial key
  is returned by a direct method call on a shared in-process object — no channel is modelled.
- **Hierarchical aggregation weight (Eq. 6).** The figure leaves the per-neighbour weight general;
  the code fixes it at `1/(n+1)`, i.e. a plain mean.
- **Trust filtering is RSU-only.** Eq. 9–10 is applied to vehicle proxies at the RSU. The server
  accepts every cluster model unconditionally, so a compromised RSU is unchecked.

### 8.7 Confirmed bugs and the fixes applied

Every item is backed by run output or by a test that fails without the fix. The suite is now 54
tests; `tests/test_round_liveness_regressions.py` pins the round-fatal ones.

#### B1 — A single non-ASCII log character destroyed every RSU aggregation (fixed)

**Evidence.** `[NET] Receive error on port 5003: 'charmap' codec can't encode character '∂'` on
RSU ports 5000, 5001 and 5003; **zero** "Aggregated ... forwarding to Server" lines; **zero**
"GLOBAL PROXY METRICS" blocks; all 21 vehicles logged "Timed out waiting for global update" in every
round.

**Cause.** The trust log printed the characters `∂`, `→` and `τ`. `RSU.aggregate` runs on a
socket-receiver thread whose stdout on Windows is cp1252, so the print raised `UnicodeEncodeError`.
`Receiver._handle` swallowed it — but only *after* `completed_rounds.add(r)` and the buffer pop. The
round was consumed and neither `CLUSTER_UPDATE` nor `NO_CLUSTER_UPDATE` was ever sent. **The top two
aggregation tiers were dead in every run.**

**Fix.** ASCII trust log (`deviation=... -> tau=...`);
`sys.stdout/stderr.reconfigure(encoding="utf-8", errors="replace")` in `config.py` so logging can
never affect protocol behaviour; the same glyph class removed from `run_grid_experiments.py`; and a
test asserts that no `print(` line in any hot-path module contains a non-ASCII character.

#### B2 — No exception isolation in either aggregator; any throw silently lost the round (fixed)

**Cause.** Both `RSU.aggregate` and `Server.aggregate` claimed the round
(`completed_rounds.add(r)`, buffer pop) and *then* did all the fallible work: verification,
deserialisation, trust filtering, evaluation, serialisation. Any exception left the round claimed but
unanswered. At the server that also meant `training_done_event` was never set, so a failure on the
final round would hang the whole simulation.

**Fix.** Each is split into a guarded shell plus `_aggregate_round`. The RSU always emits
`NO_CLUSTER_UPDATE` on failure; the server always broadcasts the last known global and always
releases the barrier in a `finally`. The RSU's per-vehicle global relay is now individually guarded so
one bad recipient cannot starve the rest of the cluster.

#### B3 — Fixed collection deadlines discarded most of every cluster (fixed)

**Evidence.** `[RSU_4_West] Timeout! Aggregating 1/5 vehicles for round 1`;
`[RSU_2_East] Timeout! Aggregating 3/6 vehicles for round 1`. Up to 80% of a cluster's work discarded
per round.

**Cause.** `RSU_ROUND_TIMEOUT = 25 s` was measured from the *first* report. `TRAINING_SEMAPHORE`
allows 8 concurrent trainers for 21 vehicles, so a cluster's last vehicle finishes far more than 25 s
after its first. The deadline systematically excluded the slow half of every cluster.

**Fix.** The deadline is now an **inactivity window** that restarts on every new report, bounded by a
hard cap from the round's first report (`RSU_ROUND_MAX_WAIT = 150 s`,
`SERVER_ROUND_MAX_WAIT = 180 s`), with the device failsafe `TIMEOUT` derived from those caps.
Verification run: **3/3, 2/2, 6/6, 5/5, 5/5 in every round, zero timeouts.**

#### B4 — V2V proxy sharing (Eq. 6) never actually executed (fixed)

**Evidence.** Pre-fix run: 4x `V2V: no peer proxies received`, **0x**
`V2V: averaged local + N peer proxies`.

**Cause.** `V2V_READY_TIMEOUT = 2.0 s` for a barrier that must span the semaphore-induced straggler
spread, and `wait_for_v2v_ready`'s return value was discarded, so the failure was invisible.

**Fix.** `V2V_READY_TIMEOUT = 90 s`, `V2V_COLLECT_TIMEOUT = 10 s`, and the barrier now logs when it
gives up. Verification run: **21 successful V2V aggregations across 3 rounds**, including `C2_V3`
averaging its own proxy with 3 peers.

#### B5 — A lagging vehicle threw away the global model it was waiting for (fixed)

**Cause.** `_handle_verified_global` required `round_num == current_round`. A vehicle still finishing
round r-1 when round r's global arrived discarded it, then stalled on round r for the full failsafe.
The round-start cleanup also unconditionally popped round r from the pending stash, discarding
anything legitimately queued for it.

**Fix.** Globals for future rounds are stashed and consumed at that round's barrier; only strictly
older entries are dropped at round start. Two regression tests cover stash-and-apply and
still-discard-the-past.

#### B6 — Round-phase state written without the lock its readers hold (fixed)

All three `_round_phase` transitions in `Device.send` mutated state that `_handle_verified_global`
reads under `proxy_lock`, without taking that lock. All transitions now hold it.

#### B7 — Server's empty-round path lost its metric and leaked round state (fixed)

`Server._force_aggregate`'s zero-RSU branch never recorded `server_round_execution_ms` and left
`round_start_times[r]` in place. Both corrected, and the branch is now exception-guarded too.

#### B8 — Docstring claimed JSD-weighted aggregation; a dead loop pretended to log it (fixed)

`server.py`'s header said "applies JSD-weighted global aggregation" while the code called plain
`average_weights`, and `for i, div in enumerate(divergences)` iterated a list that was always empty.
Docstring corrected to Eq. 8 FedAvg; dead loop removed. `jsd_weighted_average` and `calculate_jsd`
remain in `models.py`, unused.

#### B9 — Round banner printed five times per round (fixed)

`self.name.endswith("_V1")` matched `C0_V1` through `C4_V1`, so each round header appeared once per
cluster. Now keyed on `device_id == 0`.

#### B10 — A clean checkout could not import the project (fixed)

`logger.py` imports `prettytable` at module scope and no requirements file declared it — `main.py`,
`server.py`, `device.py` and the entire test suite failed with `ModuleNotFoundError` on a fresh
environment (5 of 13 test modules could not even be collected). Added `requirements.txt` covering
every runtime and test dependency.

#### B11 — The MIRACL symmetric fallback was silent (fixed)

`crypto_protocol/miracl_core.dll` is not in the repo, so SHA-256 and AES-GCM silently fell back to
`hashlib` and `cryptography` while the module docstring claimed MIRACL supplied both. Every crypto
timing in the results depended on an invisible switch. The run banner now names the active backend,
and `MIRACL_BRIDGE_AVAILABLE` is exported.

#### B12 — Non-ASCII prints in the grid-experiment driver (fixed)

The `x` and check-mark characters in `run_grid_experiments.py` — the same crash class as B1, latent
only because the sweep driver was not being run. Replaced, and now covered by the B1 regression test.

### 8.8 Open limitations and architectural constraints

Each of these alters experiment semantics rather than correctness, so it is reported for a decision
rather than changed silently.

| # | Limitation | Consequence | Recommended change |
|---|---|---|---|
| L1 | `DP_NOISE_MULTIPLIER = 0.05` vs. the paper's 1.0 | epsilon ~= 1.5e5 per round. The (epsilon, delta) guarantee is vacuous. | sigma >= 1.0, then re-baseline accuracy |
| L2 | `SPEED_RANGE = (0, 0)`; spawn radius 800 m inside a 1000 m radio | No vehicle is ever out of range. The out-of-range and coverage-loss paths are dead code; coverage is a flat 21/21. | restore a non-zero speed and widen the spawn radius past `V2RSU_RANGE` |
| L3 | Unweighted means at both aggregation tiers | A 2-vehicle cluster equals a 6-vehicle cluster in the global model. | weight by `n_k` for true FedAvg; matches Eq. 7/8 as-is |
| L4 | Proxy training loss never recorded (`train_epoch` returns `0.0`) | No proxy-side learning curve; proxy accuracy printed but not logged. | return and log the real proxy objective |
| L5 | MIRACL C bridge not built | Crypto timings mix pure-Python EC with library symmetric primitives. | run `build_miracl_bridge.bat` before any timing run |
| L6 | TA/KGC is a shared in-process object with no lock | Safe today (registration is single-threaded in `main`), but no registration protocol is modelled. | acceptable for simulation; state it in the paper |
| L7 | Signature over plaintext, carried outside the AEAD | An eavesdropper who guesses a payload can confirm it. | move `sig` inside the encrypted envelope |
| L8 | No nonce cache; replay defence is round-binding + dedup only | Adequate here, not a general anti-replay mechanism. | add a per-sender seen-nonce window if claiming replay resistance |
| L9 | `vanet_channel` is observational; transport is loopback TCP | `communication_*_ms` measures host scheduling, not radio time. | state this explicitly wherever latency is reported |
| L10 | No trust filtering at the server tier | A compromised RSU's cluster model is accepted unconditionally. | apply `filter_trusted_weights` to cluster models too |
| L11 | `Device`'s VANET fallback fits the scaler on the full CSV | Test-set leakage — unreachable from `main.py`, live if `Device` is constructed directly. | delete the fallback or route it through `prepare_vanet_partitions` |

### 8.9 Verification evidence

3-round VANET run, seed 42, 21 vehicles across 5 RSUs:

| Check | Before | After |
|---|---|---|
| RSU aggregations completed | 0 | 15 of 15 (5 RSUs x 3 rounds) |
| Cluster participation | 1/5, 3/6, then nothing | 3/3, 2/2, 6/6, 5/5, 5/5 every round |
| Server global aggregations | 0 | 3 of 3 |
| Global proxy accuracy | never computed | 60.0% -> 61.2% -> 62.0% |
| Vehicle round timeouts | every vehicle, every round | 0 |
| V2V aggregations | 0 | 21 |
| Console / receiver errors | one per RSU aggregation | 0 |
| Unit tests | 47 (5 modules failed to import) | 54, all passing |

---

## Reproducing

```bash
pip install -r requirements.txt
python main.py --dataset vanet --rounds 3          # VANET IDS
python main.py --dataset mnist --rounds 3          # MNIST
python main.py --dataset both  --rounds 3          # both, sequentially
python -m pytest tests/ -q                         # 54 tests
```

Useful flags: `--no-security` (disable the certificateless layer), `--no-batch` (per-signature
verification instead of Eq. 13), `--homogeneous` (identical private architectures), `--seed N`.
