import os
import subprocess
import sys
import time

clusters_list = [2, 4, 6, 8, 10]
vehicles_list = [2, 4, 6, 8, 10]

print("=== STARTING VANET GRID SEARCH ===")

for c in clusters_list:
    for v in vehicles_list:
        print(f"\n>>> Spawning Topology: {c} Clusters | {v} Vehicles per Cluster")

        subprocess.run([sys.executable, "main.py", "--clusters", str(c), "--vehicles", str(v)], check=False)

        if os.path.exists("training_logs.txt"):
            new_name = f"logs_C{c}_V{v}.txt"
            os.replace("training_logs.txt", new_name)
            print(f">>> Saved results to {new_name}")
        else:
            print(f"[ERROR] training_logs.txt not found for C{c}_V{v}")

        print(">>> Cooling down sockets...")
        time.sleep(5)

print("\n=== GRID SEARCH COMPLETE ===")
