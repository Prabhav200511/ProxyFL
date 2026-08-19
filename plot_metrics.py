# plot_metrics.py — Generates Accuracy vs Rounds and Loss vs Rounds from logs
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_logs(filepath="training_logs.txt"):
    if not os.path.exists(filepath):
        print(f"[!] Error: '{filepath}' not found.")
        return None, None, None, None

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

            # 1. Vehicle Train Updates: [Round, Vehicle, Train Loss, Train Accuracy]
            if current_section == "VEHICLE_TRAIN" and len(parts) == 4:
                r_num, vehicle, loss, acc = int(parts[0]), parts[1], float(parts[2]), float(parts[3].replace('%', ''))
                if vehicle not in vehicle_train_loss:
                    vehicle_train_loss[vehicle] = {}
                vehicle_train_loss[vehicle][r_num] = loss

            # 2. Global Proxy Evaluation: [Round, Global Proxy Test Accuracy]
            elif current_section == "GLOBAL_PROXY" and len(parts) == 2:
                r_num, acc = int(parts[0]), float(parts[1].replace('%', ''))
                global_proxy_acc[r_num] = acc

            # 3. Private Model Test Accuracy: [Round, Vehicle, Private Test Accuracy]
            elif current_section == "PRIVATE_TEST" and len(parts) == 3:
                r_num, vehicle, acc = int(parts[0]), parts[1], float(parts[2].replace('%', ''))
                if vehicle not in vehicle_private_acc:
                    vehicle_private_acc[vehicle] = {}
                vehicle_private_acc[vehicle][r_num] = acc

    return global_proxy_acc, vehicle_private_acc, vehicle_train_loss


def plot_all(log_file="training_logs.txt", prefix=""):
    prefix_str = f"{prefix}_" if prefix else ""
    g_acc_dict, v_priv_acc, v_train_loss = parse_logs(log_file)
    if g_acc_dict is None:
        return

    markers = ['o', 's', '^', 'v', 'd', 'x', '*', '+']
    all_rounds = set(g_acc_dict.keys())
    for v_dict in v_priv_acc.values():
        all_rounds.update(v_dict.keys())
    for v_dict in v_train_loss.values():
        all_rounds.update(v_dict.keys())
    sorted_rounds = sorted(list(all_rounds))

    if not sorted_rounds:
        return

    # 1. Accuracy vs. Rounds
    plt.figure(figsize=(10, 6))

    for i, (v, r_dict) in enumerate(sorted(v_priv_acc.items())):
        v_rounds = sorted(r_dict.keys())
        v_accs = [r_dict[r] for r in v_rounds]
        plt.plot(v_rounds, v_accs, label=f"Private: {v}", linestyle='--', alpha=0.5, marker=markers[i % len(markers)])

    mean_priv_rounds = []
    mean_priv_accs = []
    for r in sorted_rounds:
        accs_at_r = [v_priv_acc[v][r] for v in v_priv_acc if r in v_priv_acc[v]]
        if accs_at_r:
            mean_priv_rounds.append(r)
            mean_priv_accs.append(sum(accs_at_r) / len(accs_at_r))

    if mean_priv_rounds:
        plt.plot(mean_priv_rounds, mean_priv_accs, label="Mean Private Model Accuracy", color="darkgreen", linewidth=3.0, marker="D")

    if g_acc_dict:
        gp_rounds = sorted(g_acc_dict.keys())
        gp_accs = [g_acc_dict[r] for r in gp_rounds]
        plt.plot(gp_rounds, gp_accs, label="Global Proxy Model Accuracy", color="royalblue", linewidth=3.0, marker="o")

    dataset_title = f" ({prefix.upper()})" if prefix else ""
    plt.title(f"Accuracy vs. Rounds{dataset_title}", fontsize=14, fontweight='bold')
    plt.xlabel("Communication Round", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="lower right")
    plt.xticks(sorted_rounds)

    acc_path = f"{prefix_str}accuracy_vs_rounds.png"
    plt.tight_layout()
    plt.savefig(acc_path, dpi=300)
    if prefix_str:
        plt.savefig("accuracy_vs_rounds.png", dpi=300)
    plt.close()
    print(f"[PLOT] Generated: '{acc_path}' and 'accuracy_vs_rounds.png'")

    # 2. Loss vs. Rounds
    plt.figure(figsize=(10, 6))

    for i, (v, r_dict) in enumerate(sorted(v_train_loss.items())):
        v_rounds = sorted(r_dict.keys())
        v_losses = [r_dict[r] for r in v_rounds]
        plt.plot(v_rounds, v_losses, label=f"{v} Training Loss", linestyle='--', alpha=0.5, marker=markers[i % len(markers)])

    mean_loss_rounds = []
    mean_loss_vals = []
    for r in sorted_rounds:
        losses_at_r = [v_train_loss[v][r] for v in v_train_loss if r in v_train_loss[v]]
        if losses_at_r:
            mean_loss_rounds.append(r)
            mean_loss_vals.append(sum(losses_at_r) / len(losses_at_r))

    if mean_loss_rounds:
        plt.plot(mean_loss_rounds, mean_loss_vals, label="Mean Training Loss", color="crimson", linewidth=3.0, marker="s")

    plt.title(f"Loss vs. Rounds{dataset_title}", fontsize=14, fontweight='bold')
    plt.xlabel("Communication Round", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="upper right")
    plt.xticks(sorted_rounds)

    loss_path = f"{prefix_str}loss_vs_rounds.png"
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300)
    if prefix_str:
        plt.savefig("loss_vs_rounds.png", dpi=300)
    plt.close()
    print(f"[PLOT] Generated: '{loss_path}' and 'loss_vs_rounds.png'")


if __name__ == "__main__":
    plot_all()