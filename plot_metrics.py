# plot_metrics.py — Accuracy, Loss, Energy, End-to-End Time, Latency,
# Cryptographic Operations, and Throughput (bytes/sec) graphs.
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
                    vehicle_private_acc.setdefault(vehicle, {})[r_num] = acc
                except ValueError:
                    pass

    return global_proxy_acc, vehicle_private_acc, vehicle_train_loss


def _save_fig(path, prefix_str, also_unprefixed=True):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    if also_unprefixed and prefix_str:
        bare = path[len(prefix_str):] if path.startswith(prefix_str) else path
        plt.savefig(bare, dpi=300)
        print(f"[PLOT] Generated: '{path}' and '{bare}'")
    else:
        print(f"[PLOT] Generated: '{path}'")
    plt.close()


def _vehicle_frame(df: pd.DataFrame) -> pd.DataFrame:
    vehicle_df = df[df["node"].str.contains(r"^C\d+_D\d+$", regex=True, na=False)]
    if vehicle_df.empty:
        vehicle_df = df[~df["node"].isin(["Server"]) & ~df["node"].str.startswith("Cluster", na=False)]
    return vehicle_df


def _series(group, col, fill=0.0):
    if col not in group.columns:
        return pd.Series(fill, index=group.index, dtype=float)
    return group[col].fillna(fill)


def plot_all(log_file="training_logs.txt", csv_file=None, prefix=""):
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
            plt.legend(loc="lower right")
            plt.xticks(sorted_rounds)
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
            plt.legend(loc="upper right")
            plt.xticks(sorted_rounds)
            _save_fig(f"{prefix_str}loss_vs_rounds.png", prefix_str)

    target_csv = csv_file or (
        f"{prefix_str}metrics.csv" if os.path.exists(f"{prefix_str}metrics.csv") else "metrics.csv"
    )
    if os.path.exists(target_csv):
        plot_metrics_from_csv(target_csv, prefix=prefix)


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
    plt.xticks(r_indices)
    plt.legend(loc="best")
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
    plt.plot(r_indices, idle_energy, marker='v', linewidth=2.0, linestyle='--',
             label="E_idle (Sync / Wait)", color="#C34A36", alpha=0.85)
    plt.plot(r_indices, tot_energy, marker='^', linewidth=3.0,
             label="E_total = E_security + E_comm", color="#D65DB1")
    plt.title(f"Non-Training Energy vs. Communication Rounds{dataset_title}", fontsize=13, fontweight='bold')
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("Energy (Joules)", fontsize=11)
    plt.xticks(r_indices)
    plt.legend(loc="best")
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
    plt.xticks(r_indices)
    plt.legend(loc="best")
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
    plt.xticks(r_indices)
    plt.legend(loc="best")
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
    plt.xticks(r_indices)
    plt.legend(loc="best")
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
    plt.xticks(r_indices)
    plt.legend(loc="best")
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
    plt.xticks(r_indices)
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.5)
    _save_fig(f"{prefix_str}action_to_response_latency.png", prefix_str)

    # ------------------------------------------------------------------
    # 4. Cryptographic Operations  (exact heading; no "overhead")
    # ------------------------------------------------------------------
    sig_gen = _series(r_group, "signature_generation_ms").values
    sig_ver = _series(r_group, "signature_verification_ms").values
    batch_ver = _series(r_group, "batch_verification_ms").values
    enc_ms = _series(r_group, "encryption_ms").values

    plt.figure(figsize=(10, 5))
    plt.plot(r_indices, sig_gen, marker='o', label="Signature Generation", color="#4D8076", linewidth=2)
    plt.plot(r_indices, batch_ver, marker='s', label="Batch Verification", color="#845EC2", linewidth=2)
    if np.nanmax(sig_ver) > 0:
        plt.plot(r_indices, sig_ver, marker='^', label="Single Verification",
                 color="#C34A36", linestyle=":", linewidth=1.5)
    plt.plot(r_indices, enc_ms, marker='x', label="AES-GCM Encryption", color="#0081CF", linewidth=1.5)
    plt.title(f"Cryptographic Operations{dataset_title}", fontsize=13, fontweight='bold')
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("Time (ms)", fontsize=11)
    plt.xticks(r_indices)
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.5)
    _save_fig(f"{prefix_str}cryptographic_operations.png", prefix_str)
    # Legacy filename for older docs / main.py prints
    plt.figure(figsize=(10, 5))
    plt.plot(r_indices, sig_gen, marker='o', label="Signature Generation", color="#4D8076", linewidth=2)
    plt.plot(r_indices, batch_ver, marker='s', label="Batch Verification", color="#845EC2", linewidth=2)
    if np.nanmax(sig_ver) > 0:
        plt.plot(r_indices, sig_ver, marker='^', label="Single Verification",
                 color="#C34A36", linestyle=":", linewidth=1.5)
    plt.plot(r_indices, enc_ms, marker='x', label="AES-GCM Encryption", color="#0081CF", linewidth=1.5)
    plt.title(f"Cryptographic Operations{dataset_title}", fontsize=13, fontweight='bold')
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("Time (ms)", fontsize=11)
    plt.xticks(r_indices)
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.5)
    _save_fig(f"{prefix_str}security_overhead.png", prefix_str)

    # ------------------------------------------------------------------
    # 5. System Throughput — bytes / second
    # ------------------------------------------------------------------
    server_df = df[df["node"] == "Server"]
    plt.figure(figsize=(10, 5))
    if not server_df.empty:
        s_group = server_df.groupby("round").mean(numeric_only=True)
        s_group = s_group[s_group.index > 0]
        s_rounds = np.array(s_group.index)

        if "throughput_bytes_per_sec" in s_group.columns and s_group["throughput_bytes_per_sec"].fillna(0).sum() > 0:
            bps = s_group["throughput_bytes_per_sec"].fillna(0).values
        else:
            # Reconstruct B/s from older CSVs:
            # wall_clock ≈ successful_updates / updates_per_sec
            # bytes ≈ Server bytes_rx or sum of vehicle+RSU bytes_tx that round
            bps = []
            for r in s_rounds:
                ups = float(s_group.loc[r].get("throughput_updates_per_sec", 0) or 0)
                n_upd = float(s_group.loc[r].get("successful_updates", 0) or 0)
                wall = (n_upd / ups) if ups > 0 else float("nan")
                srv_rx = float(s_group.loc[r].get("bytes_rx", 0) or 0)
                if srv_rx <= 0:
                    round_rows = df[df["round"] == r]
                    srv_rx = float(round_rows["bytes_tx"].fillna(0).sum())
                bps.append((srv_rx / wall) if wall and wall > 0 else 0.0)
            bps = np.array(bps)

        plt.plot(s_rounds, bps, marker='o', color="#2C73D2", linewidth=2.5,
                 label="System Throughput")
        plt.ylabel("Throughput (bytes / sec)", fontsize=11)
        plt.xticks(s_rounds)
    else:
        # Fallback: vehicle-level bytes / device round time
        bytes_total = (
            _series(r_group, "bytes_tx").values + _series(r_group, "bytes_rx").values
        )
        round_s = np.maximum(_series(r_group, "device_round_execution_ms").values, 1.0) / 1000.0
        plt.plot(r_indices, bytes_total / round_s, marker='o', color="#2C73D2",
                 linewidth=2.5, label="System Throughput")
        plt.ylabel("Throughput (bytes / sec)", fontsize=11)
        plt.xticks(r_indices)

    plt.title(f"System Throughput vs. Rounds{dataset_title}", fontsize=13, fontweight='bold')
    plt.xlabel("Communication Round", fontsize=11)
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.5)
    _save_fig(f"{prefix_str}throughput_vs_rounds.png", prefix_str)


if __name__ == "__main__":
    plot_all()
