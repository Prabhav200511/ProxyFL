# logger.py — Metrics logging and automatic plot generation for ProxyFL
import os
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
            f.write(self.vehicle_table.get_string(sortby="Round"))
            f.write("\n\nGLOBAL PROXY MODEL EVALUATION\n")
            f.write(self.global_table.get_string(sortby="Round"))
            f.write("\n\nCLUSTER AGGREGATION DIVERGENCE\n")
            f.write(self.jsd_table.get_string(sortby="Round"))
            f.write("\n\nPRIVATE MODEL TEST ACCURACY\n")
            f.write(self.private_accuracy_table.get_string(sortby="Round"))
            f.write("\n\nDIFFERENTIAL PRIVACY ACCOUNTING\n")
            f.write(self.privacy_table.get_string(sortby="Round"))

    def generate_plots(self, prefix=""):
        """Generate every plot through the shared non-overlapping layout."""
        from plot_metrics import plot_all

        prefix_str = f"{prefix}_" if prefix else ""
        log_candidate = f"{prefix_str}training_logs.txt"
        if not os.path.exists(log_candidate):
            log_candidate = "training_logs.txt"
        csv_candidate = f"{prefix_str}metrics.csv"
        if not os.path.exists(csv_candidate):
            csv_candidate = (
                "metrics.csv" if os.path.exists("metrics.csv") else None)
        plot_all(
            log_file=log_candidate,
            csv_file=csv_candidate,
            prefix=prefix,
        )
