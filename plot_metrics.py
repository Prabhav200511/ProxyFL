# plot_metrics.py — Accuracy, Loss, Energy, End-to-End Time, Latency,
# Cryptographic Operations, and modeled VANET goodput graphs.
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PLOT_OUTPUT_DIR = "plots"


def _canonical_vehicle_id(name):
    """Convert legacy C{id}_D{id} output labels to C{id}_V{id}."""
    return re.sub(r"^(C\d+)_D(\d+)$", r"\1_V\2", str(name))


def parse_logs(filepath="training_logs.txt"):
    if not os.path.exists(filepath):
        print(f"[!] Error: '{filepath}' not found.")
        return None, None, None

    vehicle_train_loss = {}
    vehicle_private_acc = {}
    global_proxy_acc = {}
    current_section = None

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if "VEHICLE TRAINING UPDATES" in line:
                current_section = "VEHICLE_TRAIN"
                continue
            elif "GLOBAL PROXY MODEL EVALUATION" in line:
                current_section = "GLOBAL_PROXY"
                continue
            elif "PRIVATE MODEL TEST ACCURACY" in line:
                current_section = "PRIVATE_TEST"
                continue
            elif "CLUSTER AGGREGATION DIVERGENCE" in line or "DIFFERENTIAL PRIVACY" in line:
                current_section = "OTHER"
                continue

            if line.startswith('+') or "Round" in line:
                continue

            parts = [p.strip() for p in line.split('|') if p.strip()]

            if current_section == "VEHICLE_TRAIN" and len(parts) == 4:
                try:
                    r_num, vehicle, loss, acc = (
                        int(parts[0]), parts[1], float(parts[2]),
                        float(parts[3].replace('%', '')),
                    )
                    vehicle = _canonical_vehicle_id(vehicle)
                    vehicle_train_loss.setdefault(vehicle, {})[r_num] = loss
                except ValueError:
                    pass

            elif current_section == "GLOBAL_PROXY" and len(parts) == 2:
                try:
                    r_num, acc = int(parts[0]), float(parts[1].replace('%', ''))
                    global_proxy_acc[r_num] = acc
                except ValueError:
                    pass

            elif current_section == "PRIVATE_TEST" and len(parts) == 3:
                try:
                    r_num, vehicle, acc = (
                        int(parts[0]), parts[1], float(parts[2].replace('%', '')),
                    )
                    vehicle = _canonical_vehicle_id(vehicle)
                    vehicle_private_acc.setdefault(vehicle, {})[r_num] = acc
                except ValueError:
                    pass

    return global_proxy_acc, vehicle_private_acc, vehicle_train_loss


def _save_fig(path, prefix_str, also_unprefixed=True):
    # Reserve a band above the axes for the legend.  Legends are deliberately
    # kept outside the plotting area so they cannot cover plotted data.
    _apply_plot_layout()

    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

    def save_png(output_path):
        """Avoid leaving a partial image if a synced output is momentarily busy."""
        target = os.path.abspath(os.path.join(PLOT_OUTPUT_DIR, os.path.basename(output_path)))
        root, ext = os.path.splitext(target)
        temporary = f"{root}.rendering{ext}"
        try:
            plt.savefig(temporary, dpi=300, format="png")
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    save_png(path)
    if also_unprefixed and prefix_str:
        bare = path[len(prefix_str):] if path.startswith(prefix_str) else path
        save_png(bare)
        print(f"[PLOT] Generated: '{PLOT_OUTPUT_DIR}/{path}' and '{PLOT_OUTPUT_DIR}/{bare}'")
    else:
        print(f"[PLOT] Generated: '{PLOT_OUTPUT_DIR}/{path}'")
    plt.close()


def _legend_above():
    """Place the current axes' legend above, rather than over, its data."""
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=min(3, len(handles)),
        frameon=True,
        borderaxespad=0.0,
    )


def _apply_plot_layout():
    """Lay out the figure and enforce separation between title and legend."""
    figure = plt.gcf()
    figure.tight_layout(rect=(0, 0, 1, 0.72))
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for axes in figure.axes:
        legend = axes.get_legend()
        if legend is None:
            continue
        title_box = axes.title.get_window_extent(renderer)
        legend_box = legend.get_window_extent(renderer)
        required_bottom = title_box.y1 + 8.0
        if legend_box.y0 < required_bottom:
            axes_box = axes.get_window_extent(renderer)
            if axes_box.height <= 0:
                continue
            anchor = legend.get_bbox_to_anchor().transformed(
                axes.transAxes.inverted())
            shift = (required_bottom - legend_box.y0) / axes_box.height
            legend.set_bbox_to_anchor(
                (anchor.x0, anchor.y0 + shift), transform=axes.transAxes)
    figure.canvas.draw()


def _set_round_ticks(rounds, max_ticks=12):
    """Show readable integer round ticks while preserving both endpoints."""
    round_values = sorted({int(round_num) for round_num in rounds})
    if not round_values:
        return
    if len(round_values) <= max_ticks:
        ticks = round_values
    else:
        positions = np.linspace(
            0, len(round_values) - 1, num=max_ticks, dtype=int)
        ticks = [round_values[position] for position in positions]
    plt.xticks(ticks)


def _vehicle_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Accept canonical C{id}_V{id} identifiers and legacy C{id}_D{id}
    # metrics so older runs can still be replotted without including RSU rows.
    vehicle_df = df[df["node"].str.contains(r"^C\d+_[VD]\d+$", regex=True, na=False)]
    if vehicle_df.empty:
        vehicle_df = df[
            ~df["node"].isin(["Server"])
            & ~df["node"].str.startswith(("Cluster", "RSU_"), na=False)
        ]
    return vehicle_df


def _series(group, col, fill=0.0):
    if col not in group.columns:
        return pd.Series(fill, index=group.index, dtype=float)
    return group[col].fillna(fill)


def _extrema_summary(series, unit=""):
    """Describe finite minimum and maximum values with their rounds."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return "No values were recorded for this run."
    minimum_round = numeric.idxmin()
    maximum_round = numeric.idxmax()
    suffix = f" {unit}" if unit else ""
    return (
        f"Minimum: {numeric.loc[minimum_round]:.3f}{suffix} in round "
        f"{minimum_round}; maximum: {numeric.loc[maximum_round]:.3f}{suffix} "
        f"in round {maximum_round}."
    )


def _wireless_goodput_by_round(df):
    """Aggregate delivered wireless bits over modeled airtime per round."""
    required = {"round", "vanet_wireless_bits", "vanet_airtime_s"}
    if not required.issubset(df.columns):
        return pd.Series(dtype=float)
    wireless = df.groupby("round", as_index=True).agg(
        bits=("vanet_wireless_bits", "sum"),
        airtime=("vanet_airtime_s", "sum"),
    )
    wireless = wireless[wireless.index > 0]
    return (
        wireless["bits"]
        / wireless["airtime"].replace(0.0, np.nan)
    ).fillna(0.0)


def _write_vanet_plot_explanations(df, vehicle_group, coverage_total=None):
    """Write data-backed interpretation notes for every VANET plot."""
    server_rows = df[df["node"] == "Server"].groupby("round").mean(
        numeric_only=True)

    def vehicle_metric(column):
        return _series(vehicle_group, column) if column in vehicle_group else pd.Series(dtype=float)

    def server_metric(column):
        return _series(server_rows, column) if column in server_rows else pd.Series(dtype=float)

    security = vehicle_metric("security_latency_ms")
    communication = vehicle_metric("communication_latency_ms")
    training = vehicle_metric("training_ms")
    idle = vehicle_metric("idle_latency_ms")
    end_to_end = vehicle_metric("end_to_end_time_ms")
    action_response = vehicle_metric("action_to_response_ms")
    signature_generation = vehicle_metric("signature_generation_ms")
    signature_verification = (
        vehicle_metric("signature_verification_ms")
        .add(vehicle_metric("batch_verification_ms"), fill_value=0.0)
    )
    encryption = vehicle_metric("encryption_ms")
    wireless_goodput_mbps = _wireless_goodput_by_round(df) / 1_000_000.0
    coverage_numeric = (
        pd.to_numeric(coverage_total, errors="coerce").dropna()
        if coverage_total is not None else pd.Series(dtype=float)
    )
    if coverage_numeric.empty:
        coverage_reason = "No assigned-RSU coverage values were recorded."
    elif coverage_numeric.min() == coverage_numeric.max():
        coverage_reason = (
            f"Coverage remained constant at {coverage_numeric.iloc[0]:.0f} "
            "vehicles. This is consistent with the stationary configuration; "
            "the observed graph contains no assigned-RSU entries or exits."
        )
    else:
        coverage_reason = (
            f"Observed assigned-RSU coverage varied from "
            f"{coverage_numeric.min():.0f} to {coverage_numeric.max():.0f} "
            "vehicles, recording entries or exits relative to the 1000 m "
            "radius. The graph alone does not establish the cause of each "
            "change."
        )

    entries = [
        (
            "vanet_accuracy_vs_rounds.png",
            _extrema_summary(vehicle_metric("private_test_accuracy_pct"), "%"),
            "Accuracy can rise as local and global proxy models learn, but it can dip "
            "because vehicle data are non-IID, the participating set changes with "
            "mobility, and DP-SGD adds noise to shared proxy updates.",
        ),
        (
            "vanet_loss_vs_rounds.png",
            _extrema_summary(vehicle_metric("train_loss")),
            "Loss generally falls while models fit their local samples. Short rises are "
            "expected when DML transfers changing predictions between private and proxy "
            "models, batches differ, and DP noise perturbs proxy gradients.",
        ),
        (
            "vanet_energy_training_vs_rounds.png",
            _extrema_summary(vehicle_metric("energy_training_j"), "J"),
            "Training energy is computed from measured training time, so its rises and "
            "falls follow per-round GPU/CPU scheduling, batch execution time, and the "
            "number of active local samples rather than model accuracy alone.",
        ),
        (
            "vanet_energy_other_vs_rounds.png",
            _extrema_summary(vehicle_metric("energy_total_j"), "J"),
            "Non-training energy follows cryptographic and communication durations. It "
            "increases when more peers are in range, more messages are verified, or a "
            "round spends longer transmitting updates.",
        ),
        (
            "vanet_energy_breakdown.png",
            _extrema_summary(vehicle_metric("energy_total_j"), "J"),
            "The component lines move differently because training, security, and TCP "
            "work are timed separately. The total follows whichever measured component "
            "dominates that round.",
        ),
        (
            "vanet_end_to_end_time_vs_rounds.png",
            _extrema_summary(end_to_end, "ms"),
            "End-to-end time includes training, cryptography, communication, and idle "
            "synchronization. Peaks usually indicate stragglers, RSU/server collection "
            "waits, V2V collection, or operating-system scheduling contention.",
        ),
        (
            "vanet_latency_breakdown.png",
            _extrema_summary(end_to_end, "ms"),
            "This is the legacy alias of the end-to-end component graph. Its total rises "
            "when training or synchronization wait rises and falls when all participants "
            "complete the hierarchy promptly.",
        ),
        (
            "vanet_end_to_end_security_comm_detail.png",
            f"Security: {_extrema_summary(security, 'ms')} Communication: "
            f"{_extrema_summary(communication, 'ms')}",
            "Security varies with batch size and verification work; communication varies "
            "with message arrival order, model payload transfer time, V2V neighbours, "
            "and local TCP thread scheduling.",
        ),
        (
            "vanet_action_to_response_latency.png",
            _extrema_summary(action_response, "ms"),
            "This timer starts when a vehicle sends LOCAL_UPDATE and stops when its "
            "GLOBAL_UPDATE returns. It rises when that vehicle waits for cluster/server "
            "stragglers and falls when the complete aggregation path finishes quickly.",
        ),
        (
            "vanet_cryptographic_operations.png",
            f"Generation: {_extrema_summary(signature_generation, 'ms')} Verification: "
            f"{_extrema_summary(signature_verification, 'ms')} Encryption: "
            f"{_extrema_summary(encryption, 'ms')}",
            "The curves change with the number of signed messages, the number of items "
            "in each batch, individual fallback verification, and runtime contention. "
            "They are operation-time measurements, not convergence indicators.",
        ),
        (
            "vanet_security_overhead.png",
            _extrema_summary(signature_verification, "ms"),
            "This legacy alias contains the same cryptographic series. Verification "
            "increases with more received signatures or fallback work and decreases "
            "when batches contain fewer items or execute faster.",
        ),
        (
            "vanet_throughput_vs_rounds.png",
            _extrema_summary(wireless_goodput_mbps, "Mbps"),
            "Modeled VANET goodput is the sum of successfully delivered wireless "
            "bits divided by their modeled PHY airtime. It reflects V2V, V2RSU, "
            "and RSU-to-vehicle link capacities, excluding wired backhaul timing.",
        ),
        (
            "vanet_vehicles_in_range_vs_rounds.png",
            _extrema_summary(
                coverage_total if coverage_total is not None else pd.Series(dtype=float),
                "vehicles",
            ),
            coverage_reason,
        ),
    ]

    lines = [
        "# VANET plot explanations",
        "",
        "These notes use the measured CSV values from this run. The listed causes "
        "explain the implementation mechanisms that can produce the observed changes; "
        "a graph alone does not prove which mechanism caused a particular point.",
        "",
    ]
    for filename, observation, reason in entries:
        lines.extend([
            f"## `{filename}`",
            "",
            f"![{filename}]({filename})",
            "",
            f"Observed range: {observation}",
            "",
            f"Why the line rises and falls: {reason}",
            "",
        ])

    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(PLOT_OUTPUT_DIR, "vanet_plot_explanations.md")
    temporary_path = f"{output_path}.rendering"
    try:
        with open(temporary_path, "w", encoding="utf-8") as explanation_file:
            explanation_file.write("\n".join(lines))
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    print(f"[PLOT] Generated: '{output_path}'")


def plot_all(log_file="training_logs.txt", csv_file=None, prefix="", routing_csv=None):
    prefix_str = f"{prefix}_" if prefix else ""
    g_acc_dict, v_priv_acc, v_train_loss = parse_logs(log_file)
    dataset_title = f" ({prefix.upper()})" if prefix else ""

    if g_acc_dict is not None and (g_acc_dict or v_priv_acc or v_train_loss):
        markers = ['o', 's', '^', 'v', 'd', 'x', '*', '+']
        all_rounds = set(g_acc_dict.keys())
        for v_dict in v_priv_acc.values():
            all_rounds.update(v_dict.keys())
        for v_dict in v_train_loss.values():
            all_rounds.update(v_dict.keys())
        sorted_rounds = sorted(list(all_rounds))

        if sorted_rounds:
            plt.figure(figsize=(10, 6))
            for i, (v, r_dict) in enumerate(sorted(v_priv_acc.items())):
                v_rounds = sorted(r_dict.keys())
                plt.plot(
                    v_rounds, [r_dict[r] for r in v_rounds],
                    label=f"Private: {v}", linestyle='--', alpha=0.5,
                    marker=markers[i % len(markers)],
                )
            mean_priv_rounds, mean_priv_accs = [], []
            for r in sorted_rounds:
                accs = [v_priv_acc[v][r] for v in v_priv_acc if r in v_priv_acc[v]]
                if accs:
                    mean_priv_rounds.append(r)
                    mean_priv_accs.append(sum(accs) / len(accs))
            if mean_priv_rounds:
                plt.plot(
                    mean_priv_rounds, mean_priv_accs,
                    label="Mean Private Model Accuracy", color="darkgreen",
                    linewidth=3.0, marker="D",
                )
            if g_acc_dict:
                gp_rounds = sorted(g_acc_dict.keys())
                plt.plot(
                    gp_rounds, [g_acc_dict[r] for r in gp_rounds],
                    label="Global Proxy Model Accuracy", color="royalblue",
                    linewidth=3.0, marker="o",
                )
            plt.title(f"Accuracy vs. Communication Rounds{dataset_title}", fontsize=14, fontweight='bold')
            plt.xlabel("Communication Round", fontsize=12)
            plt.ylabel("Accuracy (%)", fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.7)
            _legend_above()
            _set_round_ticks(sorted_rounds)
            _save_fig(f"{prefix_str}accuracy_vs_rounds.png", prefix_str)

            plt.figure(figsize=(10, 6))
            for i, (v, r_dict) in enumerate(sorted(v_train_loss.items())):
                v_rounds = sorted(r_dict.keys())
                plt.plot(
                    v_rounds, [r_dict[r] for r in v_rounds],
                    label=f"{v} Training Loss", linestyle='--', alpha=0.5,
                    marker=markers[i % len(markers)],
                )
            mean_loss_rounds, mean_loss_vals = [], []
            for r in sorted_rounds:
                losses = [v_train_loss[v][r] for v in v_train_loss if r in v_train_loss[v]]
                if losses:
                    mean_loss_rounds.append(r)
                    mean_loss_vals.append(sum(losses) / len(losses))
            if mean_loss_rounds:
                plt.plot(
                    mean_loss_rounds, mean_loss_vals,
                    label="Mean Training Loss", color="crimson",
                    linewidth=3.0, marker="s",
                )
            plt.title(f"Training Loss vs. Communication Rounds{dataset_title}", fontsize=14, fontweight='bold')
            plt.xlabel("Communication Round", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.7)
            _legend_above()
            _set_round_ticks(sorted_rounds)
            _save_fig(f"{prefix_str}loss_vs_rounds.png", prefix_str)

    target_csv = csv_file or (
        f"{prefix_str}metrics.csv" if os.path.exists(f"{prefix_str}metrics.csv") else "metrics.csv"
    )
    if os.path.exists(target_csv):
        plot_metrics_from_csv(target_csv, prefix=prefix)
    # Explicit opt-in prevents stale AODV files from labeling a direct run.
    if routing_csv is not None:
        from routing_plots import plot_routing_metrics
        plot_routing_metrics(routing_csv, prefix=prefix)


def plot_metrics_from_csv(csv_path, prefix=""):
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    prefix_str = f"{prefix}_" if prefix else ""
    dataset_title = f" ({prefix.upper()})" if prefix else ""

    vehicle_df = _vehicle_frame(df)
    if vehicle_df.empty:
        return

    r_group = vehicle_df.groupby("round", as_index=True).mean(numeric_only=True)
    r_group = r_group[r_group.index > 0]
    if r_group.empty:
        return
    r_indices = np.array(r_group.index)

    # ------------------------------------------------------------------
    # 1A. Energy — training only
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    train_energy = _series(r_group, "energy_training_j").values
    plt.plot(
        r_indices, train_energy, marker='d', linewidth=2.5,
        label="E_training (DML+DP Compute)", color="#FF6F91",
    )
    plt.title(f"Training Energy vs. Communication Rounds{dataset_title}", fontsize=13, fontweight='bold')
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("Energy (Joules)", fontsize=11)
    _set_round_ticks(r_indices)
    _legend_above()
    plt.grid(True, linestyle="--", alpha=0.5)
    _save_fig(f"{prefix_str}energy_training_vs_rounds.png", prefix_str)

    # ------------------------------------------------------------------
    # 1B. Energy — all other components
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sec_energy = _series(r_group, "energy_security_j").values
    comm_energy = _series(r_group, "energy_communication_j").values
    idle_energy = _series(r_group, "energy_idle_j").values
    tot_energy = _series(r_group, "energy_total_j").values
    plt.plot(r_indices, sec_energy, marker='o', linewidth=2.5,
             label="E_security (Keygen + Sig + Verify + Encrypt)", color="#845EC2")
    plt.plot(r_indices, comm_energy, marker='s', linewidth=2.5,
             label="E_communication (TX + RX)", color="#00C9A7")
    #plt.plot(r_indices, idle_energy, marker='v', linewidth=2.0, linestyle='--',
           #  label="E_idle (Sync / Wait)", color="#C34A36", alpha=0.85)
    plt.plot(r_indices, tot_energy, marker='^', linewidth=3.0,
             label="E_total = E_security + E_comm", color="#D65DB1")
    plt.title(f"Non-Training Energy vs. Communication Rounds{dataset_title}", fontsize=13, fontweight='bold')
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("Energy (Joules)", fontsize=11)
    _set_round_ticks(r_indices)
    _legend_above()
    plt.grid(True, linestyle="--", alpha=0.5)
    _save_fig(f"{prefix_str}energy_other_vs_rounds.png", prefix_str)
    # Keep legacy filename as alias of the non-training plot for older docs
    plt.figure(figsize=(10, 6))
    plt.plot(r_indices, sec_energy, marker='o', linewidth=2.5, label="E_security", color="#845EC2")
    plt.plot(r_indices, comm_energy, marker='s', linewidth=2.5, label="E_communication", color="#00C9A7")
    plt.plot(r_indices, tot_energy, marker='^', linewidth=3.0, label="E_total", color="#D65DB1")
    plt.plot(r_indices, train_energy, marker='d', linewidth=2.0, linestyle='--',
             label="E_training", color="#FF6F91", alpha=0.7)
    plt.title(f"Per-Vehicle Energy Consumption vs. Communication Rounds{dataset_title}",
              fontsize=13, fontweight='bold')
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("Energy (Joules)", fontsize=11)
    _set_round_ticks(r_indices)
    _legend_above()
    plt.grid(True, linestyle="--", alpha=0.5)
    _save_fig(f"{prefix_str}energy_breakdown.png", prefix_str)

    # ------------------------------------------------------------------
    # 2. End-to-End Time (line graph — formerly "Latency" stacked bars)
    #    Components are separate lines so security/comm are visible
    #    against large training / idle magnitudes.
    # ------------------------------------------------------------------
    sec_e2e = _series(r_group, "security_latency_ms").values
    comm_e2e = _series(r_group, "communication_latency_ms").values
    train_e2e = _series(r_group, "training_ms").values
    idle_e2e = _series(r_group, "idle_latency_ms").values
    if "end_to_end_time_ms" in r_group.columns:
        total_e2e = _series(r_group, "end_to_end_time_ms").values
    else:
        total_e2e = sec_e2e + comm_e2e + train_e2e + idle_e2e

    plt.figure(figsize=(11, 6))
    plt.plot(r_indices, total_e2e, marker='o', linewidth=3.0,
             label="End-to-End Time (total)", color="#2C73D2")
    plt.plot(r_indices, train_e2e, marker='d', linewidth=2.0,
             label="Training", color="#FF9671")
    plt.plot(r_indices, idle_e2e, marker='v', linewidth=2.0, linestyle='--',
             label="Idle / Sync Wait", color="#C4A484")
    plt.plot(r_indices, sec_e2e, marker='s', linewidth=2.0,
             label="Security (Crypto)", color="#845EC2")
    plt.plot(r_indices, comm_e2e, marker='^', linewidth=2.0,
             label="Communication (TX/RX)", color="#00C9A7")
    plt.title(f"End-to-End Time vs. Communication Rounds{dataset_title}", fontsize=13, fontweight='bold')
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("End-to-End Time (ms)", fontsize=11)
    _set_round_ticks(r_indices)
    _legend_above()
    plt.grid(True, linestyle="--", alpha=0.5)
    _save_fig(f"{prefix_str}end_to_end_time_vs_rounds.png", prefix_str)
    # Legacy filename alias
    plt.figure(figsize=(11, 6))
    plt.plot(r_indices, total_e2e, marker='o', linewidth=3.0, label="End-to-End Time (total)", color="#2C73D2")
    plt.plot(r_indices, train_e2e, marker='d', linewidth=2.0, label="Training", color="#FF9671")
    plt.plot(r_indices, idle_e2e, marker='v', linewidth=2.0, linestyle='--', label="Idle / Sync Wait", color="#C4A484")
    plt.plot(r_indices, sec_e2e, marker='s', linewidth=2.0, label="Security (Crypto)", color="#845EC2")
    plt.plot(r_indices, comm_e2e, marker='^', linewidth=2.0, label="Communication (TX/RX)", color="#00C9A7")
    plt.title(f"End-to-End Time vs. Communication Rounds{dataset_title}", fontsize=13, fontweight='bold')
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("End-to-End Time (ms)", fontsize=11)
    _set_round_ticks(r_indices)
    _legend_above()
    plt.grid(True, linestyle="--", alpha=0.5)
    _save_fig(f"{prefix_str}latency_breakdown.png", prefix_str)

    # Zoomed inset-style companion: security + communication only (visibility)
    plt.figure(figsize=(10, 5))
    plt.plot(r_indices, sec_e2e, marker='s', linewidth=2.5, label="Security (Crypto)", color="#845EC2")
    plt.plot(r_indices, comm_e2e, marker='^', linewidth=2.5, label="Communication (TX/RX)", color="#00C9A7")
    plt.title(f"End-to-End Time — Security & Communication Detail{dataset_title}",
              fontsize=13, fontweight='bold')
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("End-to-End Time (ms)", fontsize=11)
    _set_round_ticks(r_indices)
    _legend_above()
    plt.grid(True, linestyle="--", alpha=0.5)
    _save_fig(f"{prefix_str}end_to_end_security_comm_detail.png", prefix_str)

    # ------------------------------------------------------------------
    # 3. Action-to-Response Latency (strict request→response delay)
    # ------------------------------------------------------------------
    if "action_to_response_ms" in vehicle_df.columns and vehicle_df["action_to_response_ms"].notna().any():
        a2r = _series(r_group, "action_to_response_ms").values
        a2r_label = "Action-to-Response Latency"
    else:
        # Fallback for older CSVs: round time after local training ends
        a2r = np.maximum(
            _series(r_group, "device_round_execution_ms").values - train_e2e, 0.0
        )
        a2r_label = "Action-to-Response Latency (approx: round − training)"

    plt.figure(figsize=(10, 5))
    plt.plot(r_indices, a2r, marker='o', linewidth=2.5, color="#B83227", label=a2r_label)
    plt.title(f"Latency (Action-to-Response) vs. Communication Rounds{dataset_title}",
              fontsize=13, fontweight='bold')
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("Latency (ms)", fontsize=11)
    _set_round_ticks(r_indices)
    _legend_above()
    plt.grid(True, linestyle="--", alpha=0.5)
    _save_fig(f"{prefix_str}action_to_response_latency.png", prefix_str)

    # ------------------------------------------------------------------
    # 4. Cryptographic Operations  (exact heading; no "overhead")
    # ------------------------------------------------------------------
    sig_gen = _series(r_group, "signature_generation_ms").values
    sig_ver = _series(r_group, "signature_verification_ms").values
    batch_ver = _series(r_group, "batch_verification_ms").values
    verification_total = sig_ver + batch_ver
    enc_ms = _series(r_group, "encryption_ms").values

    plt.figure(figsize=(10, 5))
    plt.plot(r_indices, sig_gen, marker='o', label="Signature Generation", color="#4D8076", linewidth=2)
    plt.plot(r_indices, batch_ver, marker='s', label="Batch Verification", color="#845EC2", linewidth=2)
    plt.plot(r_indices, verification_total, marker='^', label="Signature Verification (total)",
             color="#C34A36", linestyle=":", linewidth=1.5)
    plt.plot(r_indices, enc_ms, marker='x', label="AES-GCM Encryption", color="#0081CF", linewidth=1.5)
    plt.title(f"Cryptographic Operations{dataset_title}", fontsize=13, fontweight='bold')
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("Time (ms)", fontsize=11)
    _set_round_ticks(r_indices)
    _legend_above()
    plt.grid(True, linestyle="--", alpha=0.5)
    _save_fig(f"{prefix_str}cryptographic_operations.png", prefix_str)
    # Legacy filename for older docs / main.py prints
    plt.figure(figsize=(10, 5))
    plt.plot(r_indices, sig_gen, marker='o', label="Signature Generation", color="#4D8076", linewidth=2)
    plt.plot(r_indices, batch_ver, marker='s', label="Batch Verification", color="#845EC2", linewidth=2)
    plt.plot(r_indices, verification_total, marker='^', label="Signature Verification (total)",
             color="#C34A36", linestyle=":", linewidth=1.5)
    plt.plot(r_indices, enc_ms, marker='x', label="AES-GCM Encryption", color="#0081CF", linewidth=1.5)
    plt.title(f"Cryptographic Operations{dataset_title}", fontsize=13, fontweight='bold')
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("Time (ms)", fontsize=11)
    _set_round_ticks(r_indices)
    _legend_above()
    plt.grid(True, linestyle="--", alpha=0.5)
    _save_fig(f"{prefix_str}security_overhead.png", prefix_str)

    # ------------------------------------------------------------------
    # 5. Measurement-only VANET goodput — delivered bits / modeled airtime
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    goodput = _wireless_goodput_by_round(df)
    if not goodput.empty:
        rounds = np.array(goodput.index)
        plt.plot(
            rounds, goodput.values / 1_000_000.0, marker="o",
            color="#2C73D2", linewidth=2.5,
            label="Modeled VANET Goodput",
        )
        plt.ylabel("Modeled VANET Goodput (Mbps)", fontsize=11)
        _set_round_ticks(rounds)
    else:
        # Documented compatibility fallback for CSVs written before VANET
        # wireless-bit and modeled-airtime fields were introduced.
        server_df = df[df["node"] == "Server"]
        if not server_df.empty:
            s_group = server_df.groupby("round").mean(numeric_only=True)
            s_group = s_group[s_group.index > 0]
            s_rounds = np.array(s_group.index)
            if ("throughput_bytes_per_sec" in s_group.columns
                    and s_group["throughput_bytes_per_sec"].fillna(0).sum() > 0):
                bps = s_group["throughput_bytes_per_sec"].fillna(0).values
            else:
                # Reconstruct B/s from older CSVs.
                bps = []
                for r in s_rounds:
                    ups = float(s_group.loc[r].get(
                        "throughput_updates_per_sec", 0) or 0)
                    n_upd = float(s_group.loc[r].get(
                        "successful_updates", 0) or 0)
                    wall = (n_upd / ups) if ups > 0 else float("nan")
                    srv_rx = float(s_group.loc[r].get("bytes_rx", 0) or 0)
                    if srv_rx <= 0 and "bytes_tx" in df.columns:
                        round_rows = df[df["round"] == r]
                        srv_rx = float(
                            round_rows["bytes_tx"].fillna(0).sum())
                    bps.append(
                        (srv_rx / wall) if wall and wall > 0 else 0.0)
                bps = np.array(bps)
            plt.plot(s_rounds, bps, marker='o', color="#2C73D2", linewidth=2.5,
                     label="Legacy Server Collection Throughput")
            plt.ylabel("Legacy Throughput (bytes / sec)", fontsize=11)
            _set_round_ticks(s_rounds)
        else:
            bytes_total = (
                _series(r_group, "bytes_tx").values
                + _series(r_group, "bytes_rx").values
            )
            round_s = np.maximum(
                _series(r_group, "device_round_execution_ms").values,
                1.0,
            ) / 1000.0
            plt.plot(
                r_indices, bytes_total / round_s, marker='o',
                color="#2C73D2", linewidth=2.5,
                label="Legacy Device Throughput",
            )
            plt.ylabel("Legacy Throughput (bytes / sec)", fontsize=11)
            _set_round_ticks(r_indices)

    plt.title(f"VANET Link Goodput vs. Rounds{dataset_title}", fontsize=13, fontweight='bold')
    plt.xlabel("Communication Round", fontsize=11)
    _legend_above()
    plt.grid(True, linestyle="--", alpha=0.5)
    _save_fig(f"{prefix_str}throughput_vs_rounds.png", prefix_str)

    # ------------------------------------------------------------------
    # 6. Assigned vehicles remaining in RSU range (individual + total)
    # ------------------------------------------------------------------
    coverage_total = None
    coverage_rows = df[
        df["node"].str.startswith("RSU_", na=False)
        & df.get("vehicles_in_range", pd.Series(index=df.index, dtype=float)).notna()
    ]
    if not coverage_rows.empty:
        coverage = coverage_rows.pivot_table(
            index="round", columns="node", values="vehicles_in_range",
            aggfunc="max",
        ).sort_index()
        plt.figure(figsize=(11, 6))
        for rsu_name in coverage.columns:
            plt.plot(
                coverage.index, coverage[rsu_name], marker="o", linewidth=1.8,
                label=rsu_name.replace("_", " "),
            )
        server_coverage = df[
            (df["node"] == "Server")
            & df.get(
                "vehicles_in_range_total",
                pd.Series(index=df.index, dtype=float),
            ).notna()
        ].set_index("round")
        if not server_coverage.empty:
            coverage_total = server_coverage[
                "vehicles_in_range_total"].sort_index()
        else:
            coverage_total = coverage.sum(axis=1)
        plt.plot(
            coverage_total.index, coverage_total.values,
            marker="D", linewidth=3.2, color="black", label="Total in range",
        )
        plt.title(
            f"Vehicles Remaining in Assigned RSU Range vs. Rounds{dataset_title}",
            fontsize=13, fontweight="bold",
        )
        plt.xlabel("Communication Round", fontsize=11)
        plt.ylabel("Number of vehicles in range", fontsize=11)
        _set_round_ticks(coverage.index)
        max_count = int(max(coverage_total.max(), coverage.max().max()))
        plt.yticks(range(0, max_count + 2))
        _legend_above()
        plt.grid(True, linestyle="--", alpha=0.5)
        _save_fig(
            f"{prefix_str}vehicles_in_range_vs_rounds.png", prefix_str)

    if prefix.lower() == "vanet":
        _write_vanet_plot_explanations(df, r_group, coverage_total)


if __name__ == "__main__":
    plot_all()
