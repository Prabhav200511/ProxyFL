from prettytable import PrettyTable

class TrainingLogger:
    def __init__(self):
        self.vehicle_table = PrettyTable()
        self.vehicle_table.field_names = ["Round", "Vehicle", "Train Loss", "Train Accuracy"]
        
        self.global_table = PrettyTable()
        self.global_table.field_names = ["Round", "Global Test Accuracy"]

        # --- NEW: DML Drift Table ---
        self.jsd_table = PrettyTable()
        self.jsd_table.field_names = ["Round", "Cluster", "JS Divergence"]

    def log_vehicle(self, round_num, vehicle, loss, accuracy):
        self.vehicle_table.add_row([round_num, vehicle, round(loss, 4), f"{round(accuracy * 100, 2)}%"])

    def log_global(self, round_num, accuracy):
        self.global_table.add_row([round_num, f"{round(accuracy * 100, 2)}%"])

    # --- NEW: Logging function for DML ---
    def log_jsd(self, round_num, cluster, jsd):
        self.jsd_table.add_row([round_num, cluster, round(jsd, 4)])

    def save_logs(self, filename="training_logs.txt"):
        with open(filename, "w") as f:
            f.write("VEHICLE UPDATES\n")
            f.write(self.vehicle_table.get_string())
            f.write("\n\nGLOBAL MODEL EVALUATION\n")
            f.write(self.global_table.get_string())
            # --- NEW: Save DML JSD data to file ---
            f.write("\n\nCLUSTER JS DIVERGENCE (DML DRIFT)\n")
            f.write(self.jsd_table.get_string())