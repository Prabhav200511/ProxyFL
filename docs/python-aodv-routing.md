# Python AODV routing

This branch adds an **ideal-link, destination-only AODV simulation** in Python.
It governs wireless delivery; it is not ns-3, an IEEE 802.11p MAC simulation, or
a physical-network benchmark. FL training, averaging, trust decisions and
endpoint encryption/signatures are unchanged. Relays do not decrypt or re-sign
updates. AODV control messages themselves are **not authenticated**.

## Run

From the repository, using the installed ML environment:

```powershell
& '.\.venv_gpu\Scripts\python.exe' -B main.py --dataset vanet --rounds 5 --routing aodv
& '.\.venv_gpu\Scripts\python.exe' -B main.py --dataset both --rounds 5 --routing aodv
& '.\.venv_gpu\Scripts\python.exe' -B run_routing_experiment.py
```

The last command runs fast **synthetic network traffic**, without training,
and writes two scenarios and eight graphs to `routing_results/`. It includes
stationary route reuse, a broken link, an alternate route, complete disconnection,
and recovery. Its default inter-arrival interval is 0.5 simulated seconds.
Use `--interval 10 --output-dir routing_results/interval10` to compare against
the live FL round-arrival floor. `--active-route-timeout` changes the experiment's
route lifetime; it does not change FL or security settings.

`--routing direct` is the default and preserves the prior delivery behavior.
Direct mode does not model routing overhead/NRL and does not draw fake zero
curves. AODV runs export `<dataset>_routing_rounds.csv`, `_routing_events.jsonl`
and `_routing_metadata.json`. The CSV and metadata must be kept together to
replot them. Per-run traces contain topology and endpoint names/sizes, not model
plaintext, ciphertext, private keys or real vehicle identities.

The grid entry point accepts `--routing aodv` and keeps direct/AODV results in
separate subdirectories. Each grid configuration retains its routing traces.
Its previously unsupported `--clusters` and `--vehicles` arguments now work in
`main.py`: default topology is unchanged; explicitly requested extra RSUs
extend the existing layout on a deterministic 1,800m lattice. Counts are bounded
to 20 clusters and 99 vehicles per cluster to preserve unique existing TCP port
allocation. These are explicit experimental topology choices, not FL changes.

```powershell
& '.\.venv_gpu\Scripts\python.exe' -B run_grid_experiments.py --datasets vanet --clusters 2 --vehicles 2 --rounds 2 --routing aodv
```

## What is measured

Each RREQ broadcast, RREP forwarding transmission and RERR transmission is
counted at its transmitter. A broadcast is one transmission, not one per
receiver. Data fragments are counted at every forwarding hop; final data
arrivals are counted once. No route means no FL data transmission, but discovery
traffic still counts. Host TCP failures remain distinct from network arrival;
successful handoff is not a claim that endpoint security/trust accepted a model.

At the modeled IPv4/UDP boundary:

```
total wireless TX bytes = FL/application bytes
                        + security increment bytes
                        + AODV control body bytes
                        + IP/UDP header bytes
NRL = total control packet transmissions / final data packet arrivals
```

NRL is dimensionless and remains separate from communication volume. Zero
arrivals yield NaN/N/A, never zero. Ratios use sums across the network.
The total includes useful data, so the graph calls it **communication volume**,
not overhead above payload. Routing overhead includes RREQ/RREP/RERR bodies
and their headers; the four-component graph instead places **all** headers in
one disjoint component. The two views must not be added together.

Virtual datagrams carry at most 1,200 encoded application bytes plus 20-byte
IPv4 and 8-byte UDP headers. RREQ bodies are 24 bytes, RREP bodies 20 bytes,
and RERR bodies `4 + 8 * unreachable_destinations`; long RERR lists are split.
The envelope includes the existing four-byte application length prefix. This
packetization models sizes; the original encoded bytes are handed to TCP once
all virtual fragments arrive, with no relay modification. It does not claim
the localhost TCP connection actually uses UDP or these segment boundaries.
The endpoint constructs the unsecured counterpart from the same serialized
payload; security overhead is the encoded envelope-size increment. No-security
messages have zero security increment. Allocation across fragments is accounting
only (baseline bytes first), not a claim about the location of encrypted fields.

RSU/server backhaul is excluded from wireless routing totals. Existing `bytes_tx`
and socket-duration metrics remain host-side application-frame measurements;
do not add them to wireless per-hop IP totals (different boundaries).

## Graphs

The shared plot layout writes these files under `plots/` for live FL runs:

- `<dataset>_aodv_routing_overhead_vs_rounds.png`: each control type plus headers.
- `<dataset>_communication_volume_vs_rounds.png`: four disjoint components.
- `<dataset>_normalized_routing_load_vs_rounds.png`: NRL with undefined gaps.
- `<dataset>_aodv_network_latency_vs_rounds.png`: simulated end-to-end envelope
  latency, separately showing successful envelopes and all discovery attempts.

The existing throughput graph remains a delivered-bit/serialization-airtime
link-capacity proxy. In AODV mode its inputs come from actual simulated **data
hops**, including relays, instead of an out-of-range source-to-RSU distance.
It excludes discovery waits and is not application goodput over elapsed time.
No simulated milliseconds are added to wall-clock FL/security/socket durations.
The host cost of executing the simulator is still real wall-clock work.

## Stationary vehicles and limitations

### Why the synthetic mobility graph rises and falls

The example is a deliberately scripted network scenario, not a measured FL
training trajectory. Its routing-control volume (including headers) is:

| Round | Routing bytes transmitted | Cause |
| --- | ---: | --- |
| 1 | 200 | Initial route discovery: requests and replies |
| 2–3 | 0 | Reuse the established route |
| 4 | 392 | Break a link, report the error, discover a longer alternate route |
| 5 | 0 | Reuse the alternate route |
| 6 | 312 | Disconnected: three failed discovery attempts; no data arrives |
| 7 | 352 | Connectivity returns; discover the route again |
| 8 | 0 | Reuse the recovered route |

Zero routing-control bytes do **not** mean zero communication: data packets
continue over a cached route. Conversely, the fall in total communication volume
at round 6 means failed data delivery, not an efficiency gain. Normalized Routing
Load is undefined there because its delivered-packet denominator is zero.
The latency spike at round 6 is the modeled discovery timeout/retry wait.

Graph labels spell out **Ad hoc On-Demand Distance Vector (AODV)**, **Route
Request (RREQ)**, **Route Reply (RREP)**, **Route Error (RERR)**, **Normalized
Routing Load (NRL)**, **Federated Learning (FL)**, **Internet Protocol (IP)**,
and **User Datagram Protocol (UDP)**. KiB means kibibytes (1,024 bytes).

### What to expect without mobility

Stationarity prevents mobility-induced breaks but does not increase the radio's
capacity by itself. Successful route reuse can reduce discovery overhead and
network latency, followed by a plateau. With the default 3-second active-route
timeout and live round arrival floors 10 seconds apart, routes may expire even
when vehicles never move. Payload sizes, path lengths, failures and offered
traffic also matter. No monotonic trend is forced into any metric.

The implementation follows the core mechanisms described in
[RFC 3561](https://www.rfc-editor.org/rfc/rfc3561.html): sequence-aware route
tables, RREQ duplicate suppression/reverse routes, destination RREPs, precursor
RERR propagation, route expiry and bounded discovery retries. It deliberately
omits intermediate replies, expanding rings, local repair and unidirectional
links. HELLO and RREP-ACK are disabled under explicit ideal bidirectional-link
feedback assumptions, not counted as experimentally measured savings.

Messages use immutable snapshots of the existing V2V and V2RSU range graph.
Only the assigned RSU receives a vehicle upload; multihop does not reassign it.
V2V gossip participants remain the existing immediate neighbors, not all
reachable vehicles. No wireless RSU-to-RSU edges are invented.

One deterministic event clock serializes message submissions, with stable event
ordering and per-transmitter serialization. Fragments are store-and-forward
without pipelining. Separate transmitters do not contend. There is no fading,
interference, MAC acknowledgment/retransmission or mid-message mobility.
Residual control floods finish under the same snapshot; their time is not added
to an envelope already delivered. Discovery retries use bounded exponential
timeout backoff. Each transmission is attributed to its triggering message's
round, even if its simulated clock time extends beyond the round-arrival floor.

The same **ordered** topology/traffic trace is reproducible; live ML thread
submission order is not promised identical between runs. Each run gets a fresh
router. Metadata includes protocol/radio configuration, seed, units and modeling
assumptions. All-silent RSU/server watchdogs use existing maximum-wait budgets
from startup/previous round closure in AODV mode. Very slow training can therefore
miss a wall-clock collection deadline even without network loss. Those fallback
rounds are not modeled radio control packets. Receivers remain alive until
vehicle loops finish so final traces are not exported prematurely.

## Verification

```powershell
& '.\.venv_gpu\Scripts\python.exe' -B -m unittest discover -s tests -q
```

Tests cover hand-calculated multihop bytes, duplicate suppression, sequence
wrap/staleness, route reuse/expiry, RERR/alternate paths, unreachable/late replies,
deterministic traces, unchanged secure envelopes over real TCP, no routing
bypass, host-vs-network failures, all-silent rounds, CLI propagation and plot
series/undefined NRL. Generated outputs and local training results are not
committed to Git.
