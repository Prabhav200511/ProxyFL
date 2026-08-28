"""Graphs from measured ideal-link routing events, never legacy zero estimates."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import plot_metrics as shared


def build_routing_figures(csv_path, prefix=""):
    csv_path = Path(csv_path)
    suffix = "_routing_rounds.csv"
    metadata_path = csv_path.with_name(csv_path.name.removesuffix(suffix) + "_routing_metadata.json")
    if not csv_path.exists() or not metadata_path.exists():
        print("[ROUTING] Routing is not modeled in this input; no AODV graphs generated.")
        return {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("routing_mode") != "aodv" or csv_path.stat().st_size == 0:
        print("[ROUTING] No AODV observations available; no routing graphs generated.")
        return {}
    frame = pd.read_csv(csv_path).sort_values("round")
    rounds = frame["round"]
    title_suffix = f" ({prefix.replace('_', ' ').upper()})" if prefix else ""
    if "synthetic" in metadata.get("traffic", "").lower():
        title_suffix += " - synthetic traffic"
    figures = {}

    def figure(name, title, ylabel):
        fig, ax = plt.subplots(figsize=(10, 6))
        figures[name] = fig
        ax.set_title(title + ("\n" + title_suffix.strip() if title_suffix else ""), fontsize=12)
        ax.set_xlabel("Communication round")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.4)
        shared._set_round_ticks(rounds)
        # Matplotlib otherwise shrinks to finite points and can clip trailing
        # N/A rounds and their annotations entirely out of the chart.
        margin = max(0.1, (rounds.max() - rounds.min()) * 0.03)
        ax.set_xlim(rounds.min() - margin, rounds.max() + margin)
        return ax

    ax = figure("aodv_routing_overhead_vs_rounds",
                "Ad hoc On-Demand Distance Vector (AODV)\n"
                "Routing transmissions, including Internet Protocol / User Datagram Protocol headers",
                "Transmitted volume (kibibytes, KiB)")
    total = frame["rreq_bytes_tx"] + frame["rrep_bytes_tx"] + frame["rerr_bytes_tx"]
    for kind, full_name in (("rreq", "Route Request (RREQ)"),
                            ("rrep", "Route Reply (RREP)"),
                            ("rerr", "Route Error (RERR)")):
        ax.plot(rounds, frame[kind + "_bytes_tx"] / 1024, marker="o",
                label=full_name + "\n+ headers")
    ax.plot(rounds, total / 1024, marker="s", linewidth=2, label="Total routing")
    shared._legend_above()

    ax = figure("communication_volume_vs_rounds", "Wireless communication volume (Internet Protocol boundary)",
                "Transmitted volume (kibibytes, KiB)")
    for column, label in (("fl_application_bytes_tx", "Federated Learning (FL) /\napplication"),
                          ("security_bytes_tx", "Security increment"),
                          ("routing_control_bytes_tx", "Routing bodies"),
                          ("ip_udp_header_bytes_tx", "Internet Protocol (IP) /\nUser Datagram Protocol (UDP) headers")):
        ax.plot(rounds, frame[column] / 1024, marker="o", label=label)
    ax.plot(rounds, frame["total_wireless_bytes_tx"] / 1024, color="black", linestyle="--", label="Total")
    shared._legend_above()

    ax = figure("normalized_routing_load_vs_rounds", "Normalized Routing Load (NRL)",
                "Control packet transmissions /\nfinal data packet arrivals")
    ax.plot(rounds, frame["normalized_routing_load"], marker="o", label="Normalized Routing Load (NRL)")
    for round_num in frame.loc[frame["normalized_routing_load"].isna(), "round"]:
        ax.annotate("Undefined\n(no data arrivals)", (round_num, 0.03),
                    xycoords=("data", "axes fraction"), ha="center", fontsize=8)
    ax.set_ylim(bottom=0)
    shared._legend_above()

    ax = figure("aodv_network_latency_vs_rounds", "Modeled network latency\n(not Federated Learning wall-clock time)",
                "Mean simulated latency (seconds)")
    ax.plot(rounds, frame["successful_network_latency_mean_s"], marker="o", label="Successful envelopes")
    ax.plot(rounds, frame["network_latency_mean_s"], marker="s", linestyle="--",
            label="All attempts, including failed route discovery")
    shared._legend_above()
    return figures


def plot_routing_metrics(csv_path, prefix="", output_dir=None):
    figures = build_routing_figures(csv_path, prefix)
    old_directory = shared.PLOT_OUTPUT_DIR
    if output_dir is not None:
        shared.PLOT_OUTPUT_DIR = str(output_dir)
    paths = []
    try:
        for name, figure in figures.items():
            plt.figure(figure.number)
            filename = f"{prefix}_{name}.png" if prefix else f"{name}.png"
            shared._save_fig(filename, f"{prefix}_" if prefix else "", also_unprefixed=False)
            paths.append(str(Path(shared.PLOT_OUTPUT_DIR, filename).resolve()))
    finally:
        shared.PLOT_OUTPUT_DIR = old_directory
        for figure in figures.values():
            plt.close(figure)
    return paths
