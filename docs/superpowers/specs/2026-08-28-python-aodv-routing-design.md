# Native Python AODV routing and communication metrics

Status: proposed design for user review; implementation has not started.

## Goal and branch

Implement an event-driven Python AODV routing model that governs wireless
message delivery in ProxyFL, and produce routing-overhead, communication-byte
breakdown, and normalized-routing-load graphs from recorded events.

Work on `codex/python-aodv-routing`, branched from the cleaned repository at
`2270f2b`. Do not modify or force-push `main` or `codex/repository-cleanup`.
Preserve existing uncommitted configuration, logger, and experiment results.
Execute inline, without introducing ns-3 or another simulator dependency.

## Selected approach and alternatives

Selected: a shared in-process, discrete-event routing engine with per-node
route tables and an idealized wireless model. The existing TCP connections
remain host-side delivery plumbing, not the source of simulated radio timing
or routing-byte measurements. A wireless message is allowed through only
after the model finds and traverses a valid route.

A trace-only routing calculator is insufficient: it could draw graphs while
the FL application continued bypassing unreachable paths. An ns-3 integration
is outside the user's selected native-Python scope.

The result will be labeled an **ideal-link AODV simulation**, not a complete
IEEE 802.11p implementation or a physical-network measurement.

## Invariants

- Keep local training, DML, DP accounting, model architectures, trust filtering,
  averaging, and RSU/global aggregation algorithms unchanged.
- Keep the existing encryption, signatures, identity system, and authenticated
  message formats. Relay nodes forward opaque bytes; they neither decrypt nor
  re-sign a model update.
- Keep assigned RSU identities and the existing immediate-neighbor V2V gossip
  participant selection. Routing must not make every reachable vehicle an FL
  gossip peer or change which RSU aggregates a vehicle's update.
- Multi-hop delivery to an assigned RSU is permitted even when direct V2RSU
  range fails. Changed delivery/participation can change training results;
  preserving the FL algorithms does not promise identical results.
- AODV control packets are not automatically authenticated by FL-message
  security. Do not claim secure routing or malicious-relay resistance.
- Preserve a `direct` mode for baseline comparisons. Add `--routing aodv` to
  enable the new model; default remains `direct`. Propagate the option through
  the `both` dataset subprocesses and experiment entry points.

## Routing engine

Use the core behavior in [RFC 3561](https://www.rfc-editor.org/rfc/rfc3561.html):

1. Each vehicle and RSU has a route table containing destination, next hop,
   hop count, destination sequence number and validity, expiry, and precursors.
   Compare 32-bit sequence numbers with wraparound-aware freshness rules.
2. A missing or expired route causes an actual RREQ broadcast, identified by
   originator and request ID. Each node suppresses duplicate RREQs and builds
   a reverse route from the accepted request.
3. Use destination-only replies (RREQ D flag). The destination generates a
   RREP, which follows reverse routes and installs forward routes. Reject stale
   replies. Do not substitute a global shortest-path query for discovery.
4. Reuse active routes and refresh their lifetimes on use. Retain invalid-route
   sequence state long enough to reject stale control packets.
5. On link-layer feedback for a broken next hop, invalidate affected routes,
   update sequence state, and send RERR notifications through precursors.
   Subsequent traffic can rediscover a route, or fail after bounded attempts.
6. Enforce finite hop limits, request retries, and duplicate-cache lifetimes.
   No disconnected request may hang indefinitely or be delivered by a TCP
   fallback. No-route failure must remain distinguishable from host IPC failure.

Version-one choices: full-network bounded flooding rather than expanding-ring
optimization; no intermediate-node replies, local repair, multicast, or
unidirectional links. HELLO and RREP-ACK are disabled because the ideal-link
model supplies bidirectional connectivity and link-break feedback. Their
absence is stated in metadata, not presented as measured savings.

Initial configurable routing defaults: active-route timeout 3 simulated
seconds, node-traversal time 0.04 seconds, network diameter 64 hops, and two
RREQ retries after the first attempt. Derive discovery/duplicate lifetimes
from these constants consistently. Record actual configured values with each
run. Reject non-positive durations, invalid hop limits, and negative retries.

## Topology and simulated time

The existing position/range model supplies an immutable adjacency snapshot
for each submitted wireless message. Edges are V2V or vehicle-to-RSU links
within their existing range limits; there are no invented wireless RSU-to-RSU
edges. RSU/server backhaul remains separate from wireless AODV accounting.

A topology snapshot stays fixed while that message's route discovery and data
forwarding are simulated. This is an explicit snapshot-mobility approximation;
version one does not claim to model link changes during a single packet.
Between messages, changed links generate link-layer feedback to their endpoints.
Existing vehicle movement remains the source of topology updates.
Serialize submitted message simulations through the shared engine. This first
version does not claim realistic competition between simultaneous FL flows;
recorded submission order is part of the experiment definition.

Use one shared monotonic event clock and a stable event-queue tie breaker.
Advance it using modeled serialization/propagation/processing delays and
explicit traffic arrival times, never `sleep()` or ML execution durations.
For live FL integration, a round r request is submitted no earlier than
`(r - 1) * 10` simulated seconds or the current event-clock time, whichever is
later. Late requests never rewind the clock. Record submission ordering so
the exact network trace can be replayed; host-thread ordering can otherwise
vary between full ML runs.

Use the existing distance-based capacity function for link serialization.
Serialize transmissions from a given transmitter; separate transmitters do
not contend in this ideal-radio version. Broadcast sends once at a rate that
reaches all neighbors in that snapshot. A broadcast with no neighbors still
consumes one transmitted control packet. Exclude interference, fading, MAC
contention, MAC ACKs, and radio retransmissions, and say so in run metadata.
AODV discovery retries are still modeled and counted.

Never add simulated milliseconds to existing wall-clock training, security,
or socket timing columns. Keep simulated network latency and host execution
time as separately named quantities. Stationary vehicles do not force a
declining routing-overhead curve: inactive routes may expire between rounds.

## FL transport integration and failures

- Register all vehicle and RSU endpoints with the router when a simulation
  starts. Reset routing state for each run and dataset.
- Route PEER_UPDATE, LOCAL_UPDATE, NO_UPDATE, and RSU-to-vehicle GLOBAL_UPDATE.
  Missing topology or unregistered wireless endpoints must not silently bypass
  AODV. Only explicitly designated backhaul messages remain direct TCP.
- In AODV mode, remove the direct-range-only upload decision and attempt routing
  to the same assigned RSU. Keep privacy-budget and trust decisions unchanged.
- A disconnected vehicle's NO_UPDATE is also subject to routing; it cannot
  magically inform the RSU over localhost. Existing bounded round-liveness
  handling must still close rounds with missing participants.
- The current RSU/server timers start only after a first report. In AODV mode,
  initialize a no-traffic watchdog for round one at node startup and for later
  rounds after the previous round closes, using the configured maximum-wait
  budgets. All-silent clusters must emit their existing NO_CLUSTER_UPDATE over
  backhaul, and an all-silent server round must finish with its existing
  last-known-model fallback. These are wall-clock execution safeguards, not
  modeled network delays or synthetic reports from disconnected vehicles.
- Preserve endpoint verification of the original sender and recipient. Deliver
  each envelope at most once; forwarded packets do not enter FL aggregation.
- Never invoke application callbacks while holding the routing-state lock.
  Keep communication simulation failures separate from metrics-export failures.
  A metrics failure must not undo successful delivery.
- Record modeled packet arrival separately from successful application/security
  acceptance and from TCP host handoff. A host send failure must not be counted
  as a successfully delivered FL message.

## Packet and byte accounting

Model datagrams with 1,200 bytes maximum application content plus 20 IPv4 and
8 UDP header bytes per datagram. This is an explicit virtual packetization
choice, not a claim that the localhost TCP stream is UDP. Reassemble the full
encoded envelope for endpoint handoff; use stable message and packet IDs.

Control message bodies: RREQ 24 bytes, RREP 20 bytes, RERR `4 + 8*n` bytes for
n unreachable destinations. Split RERR destination lists if needed to fit
the configured packet limit. Count headers for control packets too.

The ledger records each control and data transmission at each hop, including
rebroadcasts and unsuccessful discovery traffic. A broadcast counts once at
its sender, not once per receiver. RX counters are diagnostic and are not
added to TX counters to estimate transmitted volume.

Expose disjoint byte components, at the modeled IP-packet boundary:

`total_wireless_bytes_tx = fl_application_bytes_tx + security_bytes_tx + routing_control_bytes_tx + ip_udp_header_bytes_tx`

Here FL/application bytes are the encoded unsecured counterpart of the actual
message, including its application framing and metadata. Security bytes are
the increase in encoded size caused by the existing secure envelope. Construct
the unsecured counterpart at the endpoint from the same serialized payload;
do not infer security overhead from mismatched global TX and model-RX totals.
When security is disabled, its byte increment is zero. Check conservation of
the byte partition. The ideal snapshot model completes all fragments on a
valid route; no-route failures produce routing traffic without data delivery.

The sum includes useful FL data, so label it **communication volume**, not
overhead above useful payload. Transport headers remain visible rather than
being silently omitted from the user's three-term conceptual formula. Keep
wireless and wired-backhaul totals separate, with an explicitly labeled
combined application-level total only if their boundaries match.

NRL is `routing_control_packet_transmissions / data_packets_delivered` over
the same observation window. Each control forwarding hop contributes to the
numerator. Count data packets once at their final network destination, not
once per relay or per accepted model update. Authentication rejection does
not retroactively erase a network arrival. Zero data deliveries yield N/A,
not zero. Aggregate sums before division; do not average per-node ratios.

## Outputs and usability

Store routing event traces and per-round routing summaries alongside each
run's outputs, with routing mode, timer settings, packetization, radio
assumptions, simulation seed, and accounting boundary.

Generate these plots through the existing shared layout/output pipeline:

- `plots/<dataset>_aodv_routing_overhead_vs_rounds.png`: RREQ/RREP/RERR transmitted
  bytes, with headers attributed explicitly and total routing volume.
- `plots/<dataset>_communication_volume_vs_rounds.png`: disjoint FL/application,
  security, routing-control, and IP/UDP-header byte components.
- `plots/<dataset>_normalized_routing_load_vs_rounds.png`: packet ratio, with
  undefined rounds shown as gaps and annotated.
- `plots/<dataset>_aodv_network_latency_vs_rounds.png`: modeled network latency,
  explicitly distinct from FL wall-clock round latency.

Direct-mode and legacy CSVs must say routing is not modeled; they must not
produce an apparently measured zero-valued AODV curve. Do not overwrite old
throughput definitions silently or force favorable trends.

Provide a deterministic small routing experiment with connected, stationary,
link-break, alternate-path, and disconnected scenarios so users can inspect
these graphs without running a costly training job. Label its synthetic
traffic explicitly; also verify integration using actual serialized messages.

## Components

- `aodv.py`: protocol state, control packets, route discovery/maintenance.
- `routing_sim.py`: event clock, topology snapshots, packet forwarding and traces.
- `routing_metrics.py`: conservation-checked byte accounting and NRL summaries.
- `network.py`: optional routing adapter around existing endpoint delivery.
- `vanet_sim.py`: safe topology snapshots; existing mobility remains intact.
- `device.py`, `rsu.py`, `server.py`, `main.py`: endpoint registration, routing-mode
  selection, all-silent round watchdogs, and removal of wireless delivery
  bypasses only in AODV mode.
- `plot_metrics.py` and focused plot helpers: new graphs with existing layout.
- `run_routing_experiment.py`: deterministic, explicitly synthetic scenarios.
- Dedicated routing/transport/metrics/plot tests and updated run documentation.

## Acceptance tests

1. A three-node line requires RREQ/RREP events before A-to-C delivery; repeated
   traffic within route lifetime reuses the route without another discovery.
2. A diamond suppresses duplicate requests while still counting all actual
   broadcasts. TTL and retry limits terminate an unreachable discovery.
3. Expiry triggers rediscovery. A broken active next hop causes RERR propagation,
   followed by alternate-route discovery or a bounded failure.
4. Stale sequence numbers cannot replace fresh routes; sequence wraparound is
   covered. Forwarding cannot create an unbounded loop.
5. Counts match hand-calculated small topologies, include every forwarding hop,
   partition bytes exactly, and handle fragmentation and zero-delivery NRL.
6. A successful multi-hop secure envelope verifies unchanged at its intended
   endpoint. Tampering and unsigned messages remain rejected as before.
7. A disconnected route never falls through to TCP; NO_UPDATE and GLOBAL_UPDATE
   obey the same wireless routing constraint. Backhaul is not charged as AODV.
8. Simulated time does not depend on wall-clock sleeps; the same ordered traffic
   and topology trace reproduces the same routing events and counters.
9. Missing participants, including an entirely silent cluster or server round,
   still allow bounded FL round completion; routing and receiver callbacks do
   not deadlock one another.
10. Direct mode and the existing 55-test baseline remain valid. New graphs have
    tested data series, readable labels, correct units, and no fabricated NRL.

Implementation begins after review of this specification. No implementation
or AODV measurement is claimed by this design document itself.
