"""Deterministic synthetic network-only experiment; no FL accuracy claims."""
import argparse
from pathlib import Path

from aodv import AodvSettings
from routing_sim import RoutingSimulator, TopologySnapshot


def run_experiment(output_dir="routing_results", rounds=8, interval=0.5, seed=42,
                   active_route_timeout=3.0):
    if rounds < 1 or interval < 0:
        raise ValueError("rounds must be positive and interval non-negative")
    output_dir = Path(output_dir)
    outputs = []
    for scenario in ("stationary", "mobility"):
        simulator = RoutingSimulator(AodvSettings(active_route_timeout=active_route_timeout), seed=seed)
        phases = []
        for round_num in range(1, rounds + 1):
            edges = [("A", "B", 300), ("B", "RSU", 900)]
            phase = "connected stationary"
            if scenario == "mobility" and round_num >= 4:
                edges = [("A", "B", 300), ("A", "D", 300), ("D", "E", 300), ("E", "RSU", 900)]
                phase = "broken B-RSU link, alternate A-D-E-RSU"
            if scenario == "mobility" and round_num == 6:
                edges = [("A", "B", 300)]
                phase = "disconnected"
            snapshot = TopologySnapshot.from_edges(["A", "B", "D", "E", "RSU"], edges)
            simulator.submit("A", "RSU", 2600, 2400, round_num, snapshot,
                             arrival_time=(round_num - 1) * interval)
            phases.append({"round": round_num, "scenario": phase})
        prefix = "synthetic_" + scenario
        metadata = simulator.metadata(traffic="synthetic fixed-size envelopes, not FL training")
        metadata.update(scenario=scenario, phases=phases, traffic_interval_s=interval)
        simulator.ledger.export(output_dir / prefix, metadata)
        from routing_plots import plot_routing_metrics
        outputs.extend(plot_routing_metrics(output_dir / (prefix + "_routing_rounds.csv"),
                                            prefix=prefix, output_dir=output_dir / "plots"))
        print(f"[ROUTING] {scenario}: {len(simulator.ledger.events)} recorded events; {output_dir / prefix}")
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="routing_results")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Simulated inter-arrival seconds; use 10 to match live FL round floors")
    parser.add_argument("--active-route-timeout", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_experiment(args.output_dir, args.rounds, args.interval, args.seed, args.active_route_timeout)


if __name__ == "__main__":
    main()
