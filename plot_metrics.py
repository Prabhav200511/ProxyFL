import matplotlib.pyplot as plt
import os

def parse_logs(filepath="training_logs.txt"):
    if not os.path.exists(filepath):
        print(f"❌ Error: '{filepath}' not found.")
        return None, None, None, None

    vehicle_data = {}
    jsd_data = {}
    global_rounds = []
    global_acc = []
    current_section = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue

            # Detect section
            if "VEHICLE UPDATES" in line:
                current_section = "VEHICLES"
                continue
            elif "GLOBAL MODEL EVALUATION" in line:
                current_section = "GLOBAL"
                continue
            elif "CLUSTER JS DIVERGENCE" in line:
                current_section = "JSD"
                continue

            if line.startswith('+') or "Round" in line: continue

            parts = [p.strip() for p in line.split('|') if p.strip()]

            # Parse Vehicle Data
            if current_section == "VEHICLES" and len(parts) == 4:
                r_num, vehicle, loss, acc = int(parts[0]), parts[1], float(parts[2]), float(parts[3].replace('%', ''))
                if vehicle not in vehicle_data: vehicle_data[vehicle] = {'rounds': [], 'loss': [], 'acc': []}
                vehicle_data[vehicle]['rounds'].append(r_num)
                vehicle_data[vehicle]['loss'].append(loss)
                vehicle_data[vehicle]['acc'].append(acc)

            # Parse Global Data
            elif current_section == "GLOBAL" and len(parts) == 2:
                global_rounds.append(int(parts[0]))
                global_acc.append(float(parts[1].replace('%', '')))

            # Parse DML JSD Data
            elif current_section == "JSD" and len(parts) == 3:
                r_num, cluster, jsd = int(parts[0]), parts[1], float(parts[2])
                if cluster not in jsd_data: jsd_data[cluster] = {'rounds': [], 'jsd': []}
                jsd_data[cluster]['rounds'].append(r_num)
                jsd_data[cluster]['jsd'].append(jsd)

    return global_rounds, global_acc, vehicle_data, jsd_data

def main():
    print("Reading logs...")
    global_rounds, global_acc, vehicle_data, jsd_data = parse_logs("training_logs.txt")
    if global_rounds is None: return

    markers = ['s', '^', 'v', 'd', 'o', 'x']

    # 1. Global Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(global_rounds, global_acc, marker='o', color='b', linewidth=2.5)
    plt.title('Global Model Test Accuracy Over Rounds', fontsize=14, fontweight='bold')
    plt.xlabel('Round', fontsize=12); plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7); plt.xticks(global_rounds)
    plt.savefig('global_accuracy.png')

    # 2. Local Loss
    plt.figure(figsize=(10, 6))
    for i, (v, d) in enumerate(vehicle_data.items()):
        plt.plot(d['rounds'], d['loss'], label=v, marker=markers[i % len(markers)], linewidth=2)
    plt.title('Local Devices Training Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Round', fontsize=12); plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7); plt.legend(); plt.xticks(global_rounds)
    plt.savefig('local_loss.png')

    # 3. Local Accuracy
    plt.figure(figsize=(10, 6))
    for i, (v, d) in enumerate(vehicle_data.items()):
        plt.plot(d['rounds'], d['acc'], label=v, marker=markers[i % len(markers)], linewidth=2)
    plt.title('Local Devices Training Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Round', fontsize=12); plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7); plt.legend(); plt.xticks(global_rounds)
    plt.savefig('local_acc.png')

    # 4. NEW: DML Client Drift (JS Divergence)
    if jsd_data:
        plt.figure(figsize=(10, 6))
        for i, (cluster, d) in enumerate(jsd_data.items()):
            plt.plot(d['rounds'], d['jsd'], label=cluster, marker=markers[i % len(markers)], linewidth=2)
        plt.title('DML Client Drift (JS Divergence) Over Rounds', fontsize=14, fontweight='bold')
        plt.xlabel('Communication Round', fontsize=12)
        plt.ylabel('Jensen-Shannon Divergence Score', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(global_rounds)
        plt.savefig('dml_jsd_drift.png')
        print("✅ Added new graph: dml_jsd_drift.png")

if __name__ == "__main__":
    main()