# ProxyFL Correctness and VANET Capacity Design

Date: 2026-08-25

## Objective

Correct the validated implementation, synchronization, transport, privacy-reporting, and measurement defects without replacing or modifying ProxyFL's base federated-learning or certificateless-security concepts.

The result must preserve the paper-aligned model flow while making the implementation safe, deterministic, live under failure, and honest about what its throughput graph measures.

## Preserved Concepts

The following are invariants and must not be replaced:

- Private and proxy models trained by the existing Deep Mutual Learning objectives.
- Per-sample clipping and Gaussian-noise DP-SGD on the proxy model.
- V2V proxy averaging followed by equal-weight RSU and server aggregation from Equations 6-8.
- The existing trust decision and configured threshold policy from Equations 9-10.
- The TA/KGC pseudo-identity, certificateless key, shared-secret, signature, single-verification, and batch-verification construction from Equations 11-14.
- MIRACL NIST P-256 operations and the existing AES-256-GCM payload protection.
- The five-RSU spatial layout, configured mobility, and current training hyperparameters.

The work may correct code that fails to implement these concepts consistently, but it must not substitute a different FL algorithm, aggregation weighting, privacy mechanism, cryptographic protocol, or mobility policy.

## VANET Link-Capacity Measurement

The throughput graph will stop using localhost wall-clock collection time. Wireless V2V and vehicle-to-RSU transmissions will instead be instrumented with a measurement-only 5.9 GHz, 10 MHz VANET channel model.

For link distance `d`, received signal strength and capacity are:

```text
PL(d) = PL(1 m) + 10 n log10(max(d, 1 m))
Pr_dBm = Pt_dBm - PL(d)
Noise_dBm = -174 + 10 log10(B_Hz) + noise_figure_dB
SNR_linear = 10 ^ ((Pr_dBm - Noise_dBm) / 10)
C_bps = min(27 Mbps, B_Hz log2(1 + SNR_linear))
airtime_s = delivered_wire_bits / C_bps
```

Constants will be centralized in `config.py`: 10 MHz bandwidth, 23 dBm transmit power, 46.4 dB one-metre path loss at 5.9 GHz, path-loss exponent 2.7, 9 dB receiver noise figure, and 27 Mbps PHY cap.

For every successful wireless send, metrics will record delivered wire bits, modeled airtime, and link capacity. Per-round VANET goodput will be `sum(delivered bits) / sum(modeled airtime)`, representing a shared-channel application goodput. Mean link capacity will also be exported separately. RSU-to-server backhaul is not a VANET wireless hop and is excluded.

This model is observational only: it must not sleep, drop packets, alter timeouts, or change which FL updates participate. Existing wall-clock communication timings remain available under their current latency columns. Legacy throughput columns remain readable for compatibility, but new unambiguous `vanet_link_capacity_bps` and `vanet_goodput_bps` columns drive the throughput plot, whose axis is Mbps.

## Safe Transport and Existing Security Protocol

`pickle` will be removed from the network boundary. A standard-library wire codec will encode the existing message dictionaries as bounded JSON, using an explicit tagged base64 representation for byte strings. It will reject unsupported types, malformed tags, non-dictionary top-level messages, and frames larger than a configured maximum before allocating or decoding their payload.

Model state dictionaries will continue to use the existing tensor-only `torch.save`/`torch.load(weights_only=True)` codec inside authenticated payload bytes.

Global model delivery will use the existing protocol rather than a new security concept:

1. The server signs the serialized global proxy and encrypts it with its existing pairwise RSU shared secret.
2. The RSU verifies and decrypts that envelope using the existing single-verification path.
3. The RSU signs the same serialized global proxy and creates a recipient-specific AES-GCM envelope for each vehicle.
4. A vehicle verifies and decrypts the RSU envelope before accepting the global model.

The no-security mode retains plaintext semantics but uses serialized tensor bytes and the safe outer wire codec.

Hash-to-scalar and random-scalar helpers will be corrected to return the specified nonzero `Z_q*` domain. Signature generation will resample its nonce in the negligible case that modular arithmetic produces a zero signature scalar. No equation or primitive changes.

## Round Synchronization and Liveness

Each device will maintain explicit round state: training, sharing/reporting, and waiting-for-global. A valid global update received during training or sharing is stored by round and applied only after the device has sent its `LOCAL_UPDATE` or authenticated `NO_UPDATE`. This prevents a server timeout from overwriting a proxy that is still being locally trained.

Global envelopes from the wrong sender, recipient, or round are rejected by the existing AAD and authority bindings. Stale completed rounds are discarded; a valid current-round update may be queued once.

RSU and server timeout aggregation remains as the existing straggler/liveness policy. Late reports for an already completed round remain excluded, but they cannot corrupt a device's in-progress training state.

If server aggregation receives reports but extracts zero valid cluster models, it broadcasts its unchanged current global model and signals completion on the final round, exactly like the existing no-data path. Every terminal aggregation path must either broadcast a model or produce an explicit terminal signal.

## V2V Buffer Correctness

Peer buffers are initialized when a device enters a round and are never cleared at the start of the later sharing call. An update is accepted only for the device's active round, from an in-range known peer, and once per sender.

The topology manager will expose an in-process, round-scoped V2V readiness condition. After local training, a vehicle marks itself ready and waits until its currently in-range peers are ready or the configured readiness timeout expires. It then performs the existing Equation 6 exchange and equal-weight average. This preserves the algorithm while preventing training-time skew from making valid peer updates one-sided or deleting early arrivals.

Readiness bookkeeping is removed after a round so late messages cannot leak memory or contaminate later rounds.

## Trust Reference

RSUs will retain the last authenticated global proxy they broadcast. Vehicle deviation will be calculated against that established global proxy, matching Equation 9, rather than against the average of the same updates being judged. The existing absolute threshold or median-multiplier fallback remains unchanged.

The initial reference is the server's deterministic initial proxy state supplied when RSUs are constructed. Trust filtering still occurs before the existing equal-weight RSU average.

## DP Consistency and Privacy Reporting

Privacy parameters and the DP-SGD mechanism remain unchanged. The per-sample proxy loss will use the same configured class weights as the current non-DP proxy loss so enabling DP does not silently change the classification objective.

The simulation will emit a clear warning when reported epsilon exceeds a configurable reporting threshold, while continuing with the user's configured noise multiplier and budget policy. It will not silently tune the noise multiplier or stop training unless the existing `DP_MAX_EPSILON` setting requests that behavior.

## Data Isolation and Reproducibility

The VANET file will be partitioned deterministically per vehicle. Each vehicle keeps its current 80/20 local train/test meaning. The shared feature scaler is fitted only on the union of local training rows and is then applied to both train and held-out rows, preventing held-out feature leakage without introducing per-client feature spaces.

A simulation seed will be accepted by the CLI and applied to Python, NumPy, PyTorch, vehicle counts, placement, headings, data-loader shuffling, and per-device model initialization. The same seed and parameters must reproduce topology and partitions. Different private architectures remain selected by the existing heterogeneous architecture mapping.

## Reporting

CSV export will distinguish:

- Localhost wall-clock communication latency.
- Actual serialized bytes sent and received.
- Modeled VANET wireless bits and airtime.
- Mean theoretical VANET link capacity.
- Modeled shared-channel VANET goodput.
- Successful vehicle and RSU update counts.

The throughput plot and explanation will use modeled VANET goodput in Mbps. Explanations will detect constant coverage and state that vehicles remained stationary/in range instead of attributing a flat series to mobility. Partial RSU participation will be described as timeout or validation behavior, not a coverage change.

## Error Handling

- Oversized or malformed frames are closed without invoking application callbacks.
- Failed global authentication never loads model weights or releases the round barrier.
- Failed sends do not create an action-to-response timer.
- Invalid model payloads are excluded, and zero-valid aggregation follows the unchanged-global fallback.
- V2V readiness timeouts degrade to the existing available-neighbor average rather than blocking the simulation indefinitely.

## Verification

Regression tests will cover:

- Safe wire-codec round trips, malformed data, unsupported values, and maximum frame size.
- Signed/encrypted Server-to-RSU and RSU-to-vehicle global delivery, including tampering and unsigned-message rejection.
- A global update arriving during training being queued rather than loaded immediately.
- Preservation of an early peer update and V2V readiness cleanup.
- Zero-valid server aggregation broadcasting the current model and completing the final round.
- Trust deviation using the previous global reference.
- Nonzero scalar domains and existing signature/batch-verification behavior.
- Equal class weighting between DP and non-DP proxy objectives.
- Scaler fitting on training rows only and deterministic partitions/topology.
- Analytical capacity values, distance monotonicity, wireless-only aggregation, and Mbps plotting.
- The complete existing test suite to ensure the preserved FL and security behavior does not regress.

## Compatibility and Rollout

No existing user-owned result files will be deleted. Existing public constructors will keep defaults where possible; new seed, initial-global, topology, and link-context inputs will be optional or supplied by `main.py`. Old CSV columns remain present for downstream readers, with corrected metrics added alongside them. The implementation will be delivered as focused changes and tests in the current working tree without overwriting unrelated user modifications.
