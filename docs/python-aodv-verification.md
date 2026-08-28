# AODV verification record — 2026-08-28

Branch: `codex/python-aodv-routing`, based on the existing repository-cleanup
branch. Work and review were performed inline. No main-branch merge, protocol
crypto rewrite, FL mathematics change, or user configuration change.

## Automated checks

Final full command:

```powershell
& '.\.venv_gpu\Scripts\python.exe' -B -m unittest discover -s tests -q
```

Result: **89 tests passed in 42.310 seconds**, including all 55 baseline tests.
Expected exception traces from existing aggregation-fallback tests are deliberate
fixtures, not failed tests. The native crypto bridge build regression also ran.
CLI help checks passed for `main.py` and `run_grid_experiments.py`.
`git diff --check` reported no whitespace errors.

Review regressions include unrelated-flood latency inflation, stale source
routes with expired relays, discovery deadlines, expired-route RERR suppression,
multi-precursor RERR broadcast accounting, RERR splitting, phantom unregistered
relays, real-hop link-capacity accounting, and clipped trailing N/A plot rounds.
Real socket tests also cover secure multihop upload and the actual RSU secure
GLOBAL_UPDATE forwarding path to the assigned vehicle's verifier.

## Actual FL smoke run

Command: `main.py --dataset vanet --rounds 1 --clusters 1 --vehicles 2 --routing aodv`.
Default DML/DP training, heterogeneous private models, security and batch
verification remained enabled. Dataset files were copied into the isolated,
ignored `routing_results/fl_smoke/` directory so existing outputs were preserved.

- Both vehicles completed training; both updates passed existing trust filtering.
- RSU and server aggregation completed; global proxy reached both vehicles.
- Four wireless envelopes arrived and all four host handoffs succeeded.
- 24 final virtual data-packet arrivals; two RREQ and two RREP transmissions.
- 19,420 FL/application + 5,025 security + 88 routing-body + 784 IP/UDP-header
  bytes = **25,317 wireless transmitted bytes**.
- NRL = 4 / 24 = **0.166667**.
- Simulation wall time: 111.187 seconds. Mean modeled network latency:
  0.284668 seconds per envelope. These are different measurements.

This one-round smoke run is not a convergence study, a mobility benchmark, or a
full MNIST/grid evaluation. Later edge-case fixes were verified in the final
suite. The existing privacy configuration printed epsilon about 1,523,653 at
delta 1e-5; this is a weak privacy regime, not strong DP protection. Its settings
were deliberately left unchanged under the user's constraint.

## Graph checks

`run_routing_experiment.py` completed both eight-round synthetic scenarios,
produced eight graphs, JSONL event traces, round CSVs and explicit metadata.
The final graphs were visually inspected; routing-series and N/A behavior are
also tested numerically. Stationary traffic at 0.5-second arrival intervals
shows discovery on first use then route reuse/latency plateau. The separate
mobility scenario shows link-break errors, alternate routing, disconnection
(undefined NRL) and recovery. These graphs are labeled synthetic, not FL runs.

Scope limitations and exact measurement boundaries are in
[Python AODV routing](python-aodv-routing.md).
