# logger.py — Metrics logging and automatic plot generation for ProxyFL
import os
import matplotlib.pyplot as plt
from prettytable import PrettyTable


class TrainingLogger:
    def __init__(self):
        self.reset()

    def reset(self):
        self.vehicle_table = PrettyTable()
        self.vehicle_table.field_names = ["Round", "Vehicle", "Train Loss", "Train Accuracy"]

        self.global_table = PrettyTable()
        self.global_table.field_names = ["Round", "Global Proxy Test Accuracy"]

        self.jsd_table = PrettyTable()
        self.jsd_table.field_names = ["Round", "Cluster", "JS Divergence"]

        self.private_accuracy_table = PrettyTable()
        self.private_accuracy_table.field_names = ["Round", "Vehicle", "Private Test Accuracy"]

        self.privacy_table = PrettyTable()
        self.privacy_table.field_names = ["Round", "Vehicle", "Epsilon (eps)", "Delta (delta)"]

        # Structured round-keyed metrics (deduplicated by round)
        self.global_proxy_acc = {}         # round -> accuracy (%)
        self.vehicle_train_loss = {}       # vehicle -> {round: loss}
        self.vehicle_train_acc = {}        # vehicle -> {round: acc (%)}
        self.vehicle_private_acc = {}      # vehicle -> {round: acc (%)}
        self.vehicle_privacy = {}          # vehicle -> {round: (eps, delta)}
        self.cluster_jsd = {}              # cluster -> {round: jsd}

    def log_vehicle(self, round_num, vehicle, loss, accuracy):
        self.vehicle_table.add_row([
            round_num, vehicle, round(loss, 4), f"{round(accuracy * 100, 2)}%"])
        if vehicle not in self.vehicle_train_loss:
            self.vehicle_train_loss[vehicle] = {}
            self.vehicle_train_acc[vehicle] = {}
        self.vehicle_train_loss[vehicle][round_num] = loss
        self.vehicle_train_acc[vehicle][round_num] = accuracy * 100.0

    def log_global(self, round_num, accuracy):
        self.global_table.add_row([round_num, f"{round(accuracy * 100, 2)}%"])
        self.global_proxy_acc[round_num] = accuracy * 100.0

    def log_jsd(self, round_num, cluster, jsd):
        self.jsd_table.add_row([round_num, cluster, round(jsd, 4)])
        if cluster not in self.cluster_jsd:
            self.cluster_jsd[cluster] = {}
        self.cluster_jsd[cluster][round_num] = jsd

    def log_private_accuracy(self, round_num, vehicle, accuracy):
        self.private_accuracy_table.add_row([
            round_num, vehicle, f"{round(accuracy * 100, 2)}%"])
        if vehicle not in self.vehicle_private_acc:
            self.vehicle_private_acc[vehicle] = {}
        self.vehicle_private_acc[vehicle][round_num] = accuracy * 100.0

    def log_privacy(self, round_num, vehicle, epsilon, delta):
        self.privacy_table.add_row([
            round_num, vehicle, round(epsilon, 4), f"{delta:.1e}"])
        if vehicle not in self.vehicle_privacy:
            self.vehicle_privacy[vehicle] = {}
        self.vehicle_privacy[vehicle][round_num] = (epsilon, delta)

    def save_logs(self, filename="training_logs.txt"):
        with open(filename, "w", encoding="utf-8") as f:
            f.write("VEHICLE TRAINING UPDATES\n")
            f.write(self.vehicle_table.get_string())
            f.write("\n\nGLOBAL PROXY MODEL EVALUATION\n")
            f.write(self.global_table.get_string())
            f.write("\n\nCLUSTER AGGREGATION DIVERGENCE\n")
            f.write(self.jsd_table.get_string())
            f.write("\n\nPRIVATE MODEL TEST ACCURACY\n")
            f.write(self.private_accuracy_table.get_string())
            f.write("\n\nDIFFERENTIAL PRIVACY ACCOUNTING\n")
            f.write(self.privacy_table.get_string())

    def generate_plots(self, prefix=""):
        """Generate clean, monotonically ordered Accuracy vs Rounds and Loss vs Rounds plots."""
        prefix_str = f"{prefix}_" if prefix else ""
        markers = ['o', 's', '^', 'v', 'd', 'x', '*', '+']

        # Determine all unique sorted rounds
        all_rounds = set(self.global_proxy_acc.keys())
        for v_dict in self.vehicle_private_acc.values():
            all_rounds.update(v_dict.keys())
        for v_dict in self.vehicle_train_loss.values():
            all_rounds.update(v_dict.keys())
        sorted_rounds = sorted(list(all_rounds))

        if not sorted_rounds:
            print("[PLOT] No metrics recorded to plot.")
            return

        dataset_title = f" ({prefix.upper()})" if prefix else ""

        # ----------------------------------------------------
        # 1. Accuracy vs. Rounds
        # ----------------------------------------------------
        plt.figure(figsize=(10, 6))

        # Plot individual vehicle private test accuracies
        for i, (v, r_dict) in enumerate(sorted(self.vehicle_private_acc.items())):
            v_rounds = sorted(r_dict.keys())
            v_accs = [r_dict[r] for r in v_rounds]
            plt.plot(v_rounds, v_accs,
                     label=f"Private: {v}",
                     linestyle='--',
                     alpha=0.5,
                     marker=markers[i % len(markers)])

        # Plot Mean Private Model Accuracy across vehicles per round
        mean_priv_rounds = []
        mean_priv_accs = []
        for r in sorted_rounds:
            accs_at_r = [self.vehicle_private_acc[v][r] for v in self.vehicle_private_acc if r in self.vehicle_private_acc[v]]
            if accs_at_r:
                mean_priv_rounds.append(r)
                mean_priv_accs.append(sum(accs_at_r) / len(accs_at_r))

        if mean_priv_rounds:
            plt.plot(mean_priv_rounds, mean_priv_accs,
                     label="Mean Private Model Accuracy (Local Validation)",
                     color="darkgreen",
                     linewidth=3.0,
                     marker="D")

        # Plot Global Proxy Model Accuracy
        if self.global_proxy_acc:
            gp_rounds = sorted(self.global_proxy_acc.keys())
            gp_accs = [self.global_proxy_acc[r] for r in gp_rounds]
            plt.plot(gp_rounds, gp_accs,
                     label="Global Proxy Model Accuracy (Attack Test)",
                     color="royalblue",
                     linewidth=3.0,
                     marker="o")

        plt.title(f"Accuracy vs. Communication Rounds{dataset_title}", fontsize=14, fontweight='bold')
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
        print(f"[PLOT] Saved Accuracy Plot: '{acc_path}'")

        # ----------------------------------------------------
        # 2. Loss vs. Rounds
        # ----------------------------------------------------
        plt.figure(figsize=(10, 6))

        # Individual vehicle training losses
        for i, (v, r_dict) in enumerate(sorted(self.vehicle_train_loss.items())):
            v_rounds = sorted(r_dict.keys())
            v_losses = [r_dict[r] for r in v_rounds]
            plt.plot(v_rounds, v_losses,
                     label=f"{v} Train Loss",
                     linestyle='--',
                     alpha=0.5,
                     marker=markers[i % len(markers)])

        # Mean Training Loss across vehicles
        mean_loss_rounds = []
        mean_loss_vals = []
        for r in sorted_rounds:
            losses_at_r = [self.vehicle_train_loss[v][r] for v in self.vehicle_train_loss if r in self.vehicle_train_loss[v]]
            if losses_at_r:
                mean_loss_rounds.append(r)
                mean_loss_vals.append(sum(losses_at_r) / len(losses_at_r))

        if mean_loss_rounds:
            plt.plot(mean_loss_rounds, mean_loss_vals,
                     label="Mean Training Loss",
                     color="crimson",
                     linewidth=3.0,
                     marker="s")

        plt.title(f"Training Loss vs. Communication Rounds{dataset_title}", fontsize=14, fontweight='bold')
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
        print(f"[PLOT] Saved Loss Plot: '{loss_path}'")