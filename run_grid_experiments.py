#!/usr/bin/env python3
# run_grid_experiments.py — Grid experiment runner & visualizer for ProxyFL
"""
Runs ProxyFL across:
  - Datasets: ['vanet', 'mnist'] (default: both sequentially)
  - Clusters: [2, 4, 6, 8, 10]
  - Vehicles per Cluster: [2, 5, 10, 20]

Outputs per dataset:
  - <dataset>_results_summary.csv: Matrix overview (Acc, Loss, Duration)
  - <dataset>_results_detailed_rounds.csv: Round-by-round trajectory
  - <dataset>_grid_heatmap_accuracy.png: 2D Heatmap of Accuracy
  - <dataset>_grid_heatmap_runtime.png: 2D Heatmap of Execution Time
  - <dataset>_grid_scaling_accuracy.png: Node scaling curves
  - <dataset>_grid_convergence_trajectories.png: Convergence curves
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import time
import argparse
import subprocess
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Default grid
DEFAULT_CLUSTERS = [2, 4, 6, 8, 10]
DEFAULT_VEHICLES = [2, 5, 10, 20]
DEFAULT_DATASETS = ["vanet", "mnist"]


def parse_log_metrics(log_filepath):
    """Parse output log to extract global acc, private accs, and losses."""
    if not os.path.exists(log_filepath):
        return {}, {}, {}

    global_proxy_acc = {}
    vehicle_private_acc = {}
    vehicle_train_loss = {}
    current_section = None

    with open(log_filepath, 'r', encoding='utf-8', errors='ignore') as f:
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

            # 1. Vehicle Train Loss
            if current_section == "VEHICLE_TRAIN" and len(parts) == 4:
                try:
                    r_num = int(parts[0])
                    v_name = parts[1]
                    loss = float(parts[2])
                    if v_name not in vehicle_train_loss:
                        vehicle_train_loss[v_name] = {}
                    vehicle_train_loss[v_name][r_num] = loss
                except ValueError:
                    pass

            # 2. Global Proxy Accuracy
            elif current_section == "GLOBAL_PROXY" and len(parts) == 2:
                try:
                    r_num = int(parts[0])
                    acc = float(parts[1].replace('%', ''))
                    global_proxy_acc[r_num] = acc
                except ValueError:
                    pass

            # 3. Private Test Accuracy
            elif current_section == "PRIVATE_TEST" and len(parts) == 3:
                try:
                    r_num = int(parts[0])
                    v_name = parts[1]
                    acc = float(parts[2].replace('%', ''))
                    if v_name not in vehicle_private_acc:
                        vehicle_private_acc[v_name] = {}
                    vehicle_private_acc[v_name][r_num] = acc
                except ValueError:
                    pass

    return global_proxy_acc, vehicle_private_acc, vehicle_train_loss


def plot_grid_results(summary_df, detailed_df, dataset_name="vanet", out_dir="experiment_results"):
    """Generate rich comparison plots for a dataset from collected data."""
    os.makedirs(out_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    ds_label = dataset_name.upper()

    # -------------------------------------------------------------
    # 1. Heatmap: Final Accuracy (Clusters vs Vehicles per cluster)
    # -------------------------------------------------------------
    try:
        pivot_acc = summary_df.pivot(index='vehicles_per_cluster', columns='clusters', values='final_global_proxy_acc')
        plt.figure(figsize=(9, 6))
        plt.imshow(pivot_acc.values, cmap='viridis', aspect='auto', origin='lower')
        plt.colorbar(label='Final Global Proxy Accuracy (%)')
        plt.xticks(ticks=range(len(pivot_acc.columns)), labels=pivot_acc.columns, fontsize=11)
        plt.yticks(ticks=range(len(pivot_acc.index)), labels=pivot_acc.index, fontsize=11)
        plt.xlabel('Number of Clusters (RSUs)', fontsize=12, fontweight='bold')
        plt.ylabel('Vehicles per Cluster', fontsize=12, fontweight='bold')
        plt.title(f'[{ds_label}] Global Proxy Model Accuracy Heatmap (%)', fontsize=14, fontweight='bold')

        for i in range(len(pivot_acc.index)):
            for j in range(len(pivot_acc.columns)):
                val = pivot_acc.values[i, j]
                if not np.isnan(val):
                    plt.text(j, i, f"{val:.2f}%", ha='center', va='center',
                             color='white' if val < (pivot_acc.values.max() * 0.75) else 'black',
                             fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{dataset_name}_grid_heatmap_accuracy.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"[!] Warning plotting accuracy heatmap for {dataset_name}: {e}")

    # -------------------------------------------------------------
    # 2. Heatmap: Runtime (Seconds)
    # -------------------------------------------------------------
    try:
        pivot_time = summary_df.pivot(index='vehicles_per_cluster', columns='clusters', values='duration_sec')
        plt.figure(figsize=(9, 6))
        plt.imshow(pivot_time.values, cmap='magma_r', aspect='auto', origin='lower')
        plt.colorbar(label='Duration (seconds)')
        plt.xticks(ticks=range(len(pivot_time.columns)), labels=pivot_time.columns, fontsize=11)
        plt.yticks(ticks=range(len(pivot_time.index)), labels=pivot_time.index, fontsize=11)
        plt.xlabel('Number of Clusters (RSUs)', fontsize=12, fontweight='bold')
        plt.ylabel('Vehicles per Cluster', fontsize=12, fontweight='bold')
        plt.title(f'[{ds_label}] Simulation Runtime Heatmap (Seconds)', fontsize=14, fontweight='bold')

        for i in range(len(pivot_time.index)):
            for j in range(len(pivot_time.columns)):
                val = pivot_time.values[i, j]
                if not np.isnan(val):
                    plt.text(j, i, f"{val:.1f}s", ha='center', va='center', color='black', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{dataset_name}_grid_heatmap_runtime.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"[!] Warning plotting runtime heatmap for {dataset_name}: {e}")

    # -------------------------------------------------------------
    # 3. Node Scaling Curve: Accuracy vs Total Nodes
    # -------------------------------------------------------------
    try:
        plt.figure(figsize=(10, 6))
        for cluster in sorted(summary_df['clusters'].unique()):
            sub = summary_df[summary_df['clusters'] == cluster].sort_values('total_nodes')
            plt.plot(sub['total_nodes'], sub['final_global_proxy_acc'],
                     marker='o', linewidth=2.2, label=f"{cluster} Clusters")

        plt.xlabel('Total Nodes in Network (Clusters x Vehicles)', fontsize=12, fontweight='bold')
        plt.ylabel('Final Global Proxy Accuracy (%)', fontsize=12, fontweight='bold')
        plt.title(f'[{ds_label}] Accuracy vs Total Network Nodes', fontsize=14, fontweight='bold')
        plt.legend(title="Cluster Configuration", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{dataset_name}_grid_scaling_accuracy.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"[!] Warning plotting scaling curve for {dataset_name}: {e}")

    # -------------------------------------------------------------
    # 4. Convergence Trajectories across Configurations
    # -------------------------------------------------------------
    try:
        if not detailed_df.empty:
            plt.figure(figsize=(11, 7))
            configs = detailed_df.groupby(['clusters', 'vehicles_per_cluster'])
            for (c, v), group in configs:
                lbl = f"C={c}, V={v} (N={c*v})"
                group_sorted = group.sort_values('round')
                plt.plot(group_sorted['round'], group_sorted['global_proxy_acc'],
                         marker='.', linewidth=1.5, alpha=0.8, label=lbl)

            plt.xlabel('Communication Round', fontsize=12, fontweight='bold')
            plt.ylabel('Global Proxy Test Accuracy (%)', fontsize=12, fontweight='bold')
            plt.title(f'[{ds_label}] Round-by-Round Convergence across Grid Configurations', fontsize=14, fontweight='bold')
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'{dataset_name}_grid_convergence_trajectories.png'), dpi=300)
            plt.close()
    except Exception as e:
        print(f"[!] Warning plotting convergence curves for {dataset_name}: {e}")


def run_dataset_grid(dataset, clusters, vehicles, rounds, out_dir, resume):
    """Run full grid search for a single dataset."""
    os.makedirs(out_dir, exist_ok=True)
    summary_csv_path = os.path.join(out_dir, f"{dataset}_results_summary.csv")
    detailed_csv_path = os.path.join(out_dir, f"{dataset}_results_detailed_rounds.csv")

    summary_rows = []
    detailed_rows = []
    completed_configs = set()

    # Load existing if resuming
    if resume and os.path.exists(summary_csv_path):
        try:
            existing_df = pd.read_csv(summary_csv_path)
            for _, row in existing_df.iterrows():
                completed_configs.add((int(row['clusters']), int(row['vehicles_per_cluster'])))
            summary_rows = existing_df.to_dict('records')
            if os.path.exists(detailed_csv_path):
                detailed_rows = pd.read_csv(detailed_csv_path).to_dict('records')
            print(f"[+] [{dataset.upper()}] Resuming: Found {len(completed_configs)} previously completed configurations.")
        except Exception as e:
            print(f"[!] [{dataset.upper()}] Could not load existing CSVs to resume: {e}")

    total_configs = len(clusters) * len(vehicles)
    current_idx = 0

    print("\n" + "=" * 70)
    print(f" ProxyFL Grid Experiment Runner: {dataset.upper()}")
    print(f" Clusters: {clusters}")
    print(f" Vehicles/Cluster: {vehicles}")
    print(f" Total Combinations: {total_configs} | Rounds per run: {rounds}")
    print(f" Output Directory: '{out_dir}'")
    print("=" * 70 + "\n")

    for c in clusters:
        for v in vehicles:
            current_idx += 1
            if (c, v) in completed_configs:
                print(f"[{dataset.upper()} {current_idx}/{total_configs}] Skipping already completed (Clusters={c}, Vehicles={v})")
                continue

            total_nodes = c * v
            print(f"\n>>> [{dataset.upper()} {current_idx}/{total_configs}] Running: {c} Clusters x {v} Vehicles = {total_nodes} Nodes (Rounds={rounds})...")

            start_time = time.time()
            log_file = f"{dataset}_training_logs.txt"

            # Execute simulation via clean isolated subprocess
            cmd = [
                sys.executable, "main.py",
                "--dataset", dataset,
                "--clusters", str(c),
                "--vehicles", str(v),
                "--rounds", str(rounds)
            ]

            try:
                subprocess.run(cmd, check=True)
                duration = round(time.time() - start_time, 2)

                # Parse resulting logs
                g_acc, v_priv_acc, v_loss = parse_log_metrics(log_file)
                all_rounds = sorted(list(g_acc.keys()))
                final_round = all_rounds[-1] if all_rounds else 0

                final_g_acc = g_acc.get(final_round, 0.0)

                # Compute average final private accuracy across vehicles
                priv_accs_final = [
                    v_dict[final_round] for v_dict in v_priv_acc.values() if final_round in v_dict
                ]
                avg_priv_acc = float(np.mean(priv_accs_final)) if priv_accs_final else 0.0

                # Compute average final train loss across vehicles
                train_losses_final = [
                    v_dict[final_round] for v_dict in v_loss.values() if final_round in v_dict
                ]
                avg_train_loss = float(np.mean(train_losses_final)) if train_losses_final else 0.0

                # Store summary row
                summary_rows.append({
                    "dataset": dataset,
                    "clusters": c,
                    "vehicles_per_cluster": v,
                    "total_nodes": total_nodes,
                    "rounds": rounds,
                    "duration_sec": duration,
                    "final_global_proxy_acc": final_g_acc,
                    "avg_final_private_acc": avg_priv_acc,
                    "avg_final_train_loss": avg_train_loss,
                    "status": "SUCCESS"
                })

                # Store detailed round-by-round rows
                for r in all_rounds:
                    priv_r = [v_dict[r] for v_dict in v_priv_acc.values() if r in v_dict]
                    loss_r = [v_dict[r] for v_dict in v_loss.values() if r in v_dict]
                    detailed_rows.append({
                        "dataset": dataset,
                        "clusters": c,
                        "vehicles_per_cluster": v,
                        "total_nodes": total_nodes,
                        "round": r,
                        "global_proxy_acc": g_acc.get(r, 0.0),
                        "avg_private_acc": float(np.mean(priv_r)) if priv_r else 0.0,
                        "avg_train_loss": float(np.mean(loss_r)) if loss_r else 0.0
                    })

                print(f"[+] [{dataset.upper()}] Done in {duration}s | Global Proxy Acc: {final_g_acc:.2f}% | Avg Private Acc: {avg_priv_acc:.2f}%")

            except Exception as e:
                duration = round(time.time() - start_time, 2)
                print(f"[!] Error in run (Clusters={c}, Vehicles={v}): {e}")
                summary_rows.append({
                    "dataset": dataset,
                    "clusters": c,
                    "vehicles_per_cluster": v,
                    "total_nodes": total_nodes,
                    "rounds": rounds,
                    "duration_sec": duration,
                    "final_global_proxy_acc": 0.0,
                    "avg_final_private_acc": 0.0,
                    "avg_final_train_loss": 0.0,
                    "status": f"FAILED: {e}"
                })

            # Save intermediate CSVs after each run
            sum_df = pd.DataFrame(summary_rows)
            sum_df.to_csv(summary_csv_path, index=False)

            det_df = pd.DataFrame(detailed_rows)
            det_df.to_csv(detailed_csv_path, index=False)

            # Re-generate plots dynamically after each run
            plot_grid_results(sum_df, det_df, dataset_name=dataset, out_dir=out_dir)

            # Brief pause to ensure all sockets release cleanly
            time.sleep(1.5)

    print(f"\n[OK] Completed Grid Experiments for {dataset.upper()}!")
    print(f"    - Summary CSV:  {summary_csv_path}")
    print(f"    - Detailed CSV: {detailed_csv_path}")
    print(f"    - Heatmaps & Scaling Plots saved in '{out_dir}/'")


def main():
    parser = argparse.ArgumentParser(description="ProxyFL Experiment Matrix Runner")
    parser.add_argument('--datasets', nargs='+', type=str, default=DEFAULT_DATASETS,
                        choices=["vanet", "mnist", "both"],
                        help="Datasets for simulations (default: vanet mnist)")
    parser.add_argument('--rounds', type=int, default=5,
                        help="Number of communication rounds per simulation (default: 5)")
    parser.add_argument('--clusters', nargs='+', type=int, default=DEFAULT_CLUSTERS,
                        help="List of cluster counts (default: 2 4 6 8 10)")
    parser.add_argument('--vehicles', nargs='+', type=int, default=DEFAULT_VEHICLES,
                        help="List of vehicles per cluster (default: 2 5 10 20)")
    parser.add_argument('--out_dir', type=str, default="experiment_results",
                        help="Directory to save CSVs and plots")
    parser.add_argument('--resume', action="store_true", default=True,
                        help="Skip configurations already recorded in summary CSV")
    args = parser.parse_args()

    # Resolve dataset list
    datasets_to_run = []
    for d in args.datasets:
        if d == "both":
            datasets_to_run.extend(["vanet", "mnist"])
        elif d not in datasets_to_run:
            datasets_to_run.append(d)

    print("\n" + "=" * 70)
    print(" ProxyFL Multi-Dataset Grid Runner Initialized")
    print(f" Target Datasets: {[d.upper() for d in datasets_to_run]}")
    print(f" Clusters: {args.clusters}")
    print(f" Vehicles per Cluster: {args.vehicles}")
    print("=" * 70)

    for dataset in datasets_to_run:
        run_dataset_grid(
            dataset=dataset,
            clusters=args.clusters,
            vehicles=args.vehicles,
            rounds=args.rounds,
            out_dir=args.out_dir,
            resume=args.resume
        )

    print("\n" + "=" * 70)
    print(" ALL Grid Experiments Across All Datasets Complete!")
    print(f" All CSVs and Visualizations are in '{args.out_dir}/'")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
