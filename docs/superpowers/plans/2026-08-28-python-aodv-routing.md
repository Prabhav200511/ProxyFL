# Python AODV implementation plan

> Execute inline, as requested. Approved specification:
> `docs/superpowers/specs/2026-08-28-python-aodv-routing-design.md`.

## Constraints and interfaces

Work on `codex/python-aodv-routing`; preserve dirty configuration, logger and
results. No FL mathematics, trust criteria, crypto formats or RSU assignments
change. Default routing is direct. Use unittest, test first, then implement.

The protocol core is standard-library Python. `TopologySnapshot` owns immutable
bidirectional distance maps. `RoutingSimulator.submit(source, destination,
wire_bytes, application_bytes, round_num, snapshot, arrival_time=None)` returns
a `Delivery` with a message ID, success flag, path and modeled duration.
`AodvProtocol` owns per-node route state and uses the simulator's event queue
for actual control transmissions, not a shortest-path oracle. `RoutingLedger`
records TX events and separately records delivered packets and host handoffs.
The optional `WirelessRouter` adapter resolves registered endpoint addresses,
captures topology, and gates the existing TCP transport. No application callbacks run under
the simulator lock. Endpoint callers supply the unsecured counterpart for
security overhead accounting; no metadata is added to authenticated messages.

## 1. Protocol, event simulation and metrics

- [x] Add `tests/test_aodv_routing.py`; observe missing-module failure.
- [x] Test a three-node line: 2 RREQ + 2 RREP transmissions, two fragments of at most 1,200 bytes
  for 1,250 bytes, four data-hop transmissions, two final arrivals.
  At 28 IP/UDP bytes per transmission: control 88, headers 224, application
  2,500, total 2,812 bytes; NRL 2.0.
- [x] Implement `aodv.py`, `routing_sim.py`, `routing_metrics.py`: bounded
  destination-only flooding, duplicate suppression, sequence freshness,
  reverse/forward routes, timers, precursors/RERR, packetization, stable event
  ordering, transmitter serialization and conservation-checked round summaries.
- [x] Add reuse, expiry, diamond, sequence-wrap, break/alternate, unreachable,
  replay, invalid-input and security-byte partition regressions.
- [x] Run `python -B -m unittest discover -s tests -p test_aodv_routing.py -v`.

## 2. Transport integration and round liveness

- [x] Add integration tests before modifying production callers: real encoded
  messages over a routed TCP receiver; disconnected NO_UPDATE/GLOBAL_UPDATE
  must not reach it; secure recipient verification unchanged; backhaul separate.
- [x] Add topology snapshots to `vanet_sim.py` and `WirelessRouter` to
  `network.py`. Extend `send_msg` with optional `router` and `unsecured_msg`.
- [x] Pass router into Device/RSU, including all four wireless message types;
  allow assigned-RSU multihop uploads without expanding V2V participants.
- [x] Add AODV-only round-start watchdogs to RSU/server; exercise entirely
  silent rounds using short test budgets and real timeout/aggregate paths.
- [x] Wire `--routing direct|aodv` through `main.py`, both-dataset subprocesses
  and `run_grid_experiments.py`; create a fresh simulator per run and export
  routing metadata, events JSONL and round CSV beside dataset outputs.

## 3. Graphs and reproducible experiment

- [x] Test plotted series against hand-checked routing CSV including N/A NRL.
- [x] Add `routing_plots.py`, using `plot_metrics.py` layout helpers, and an
  optional routing CSV argument to the shared plotting entry point.
- [x] Add `run_routing_experiment.py` with stationary/alternate-path/link-break and
  disconnected traffic, emitting explicitly synthetic dataset-prefixed plots.
- [x] Document CLI examples, accounting boundaries, timer implications for
  stationary vehicles, radio simplifications and unauthenticated routing.
- [x] Run experiment to a separate ignored results directory; inspect graphs.

## 4. Verify and publish

- [x] Run focused tests plus full `.venv_gpu` unittest discovery (baseline 55).
- [x] Run CLI help checks and actual secure routed transport smoke tests.
- [x] Self-review the diff for FL/security changes, routing bypasses, timer
  races, wrong ratios, measurement failure behavior and dirty-file preservation.
Release action after the verified implementation commit: stage only feature
files, commit and push `codex/python-aodv-routing`, and verify remote HEAD
without merging or force-pushing main.
