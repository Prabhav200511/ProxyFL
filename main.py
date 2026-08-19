# main.py — ProxyFL VANET Distributed Simulation Entry Point
#
# Unified command-line interface for running ProxyFL on MNIST, VANET, or Both.
# Automatically logs metrics and produces Accuracy vs. Rounds and Loss vs. Rounds plots.

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import subprocess
import sys
import threading
import time

from config import SERVER_PORT, RSU_BASE_PORT, DEVICE_BASE_PORT, RSU_SPACING, TOTAL_ROUNDS
from server import Server, training_done_event
from device import Device
from rsu import RSU
from vanet_sim import VanetTopology, place_rsu, spawn_vehicle
from shared_logger import logger
from models import VANET_PRIVATE_ARCHITECTURES, MNIST_PRIVATE_ARCHITECTURES


def run_single_simulation(dataset, num_clusters=2, vehicles_per_cluster=2,
                           total_rounds=TOTAL_ROUNDS, heterogeneous=True):
    """Run one full federated learning simulation for a given dataset."""
    total_vehicles = num_clusters * vehicles_per_cluster

    print(f"\n{'=' * 65}")
    print(f" ProxyFL Simulation [{dataset.upper()}]")
    print(f" Topology: {num_clusters} Clusters × {vehicles_per_cluster} Vehicles = {total_vehicles} Nodes")
    print(f" Total Communication Rounds: {total_rounds}")
    print(f" Heterogeneous Private Architectures: {heterogeneous}")
    print(f"{'=' * 65}\n")

    # Reset logging & synchronization
    logger.reset()
    training_done_event.clear()

    # 1. Spatial Topology
    topology = VanetTopology()
    for i in range(num_clusters):
        rsu_name = f"Cluster_{i + 1}"
        place_rsu(topology, rsu_name, center_x=i * RSU_SPACING, center_y=0)

    # Spawn vehicles near assigned RSUs
    vehicle_meta = []  # (name, port, rsu_port, rsu_name, device_id)
    device_id_counter = 0
    for i in range(num_clusters):
        rsu_name = f"Cluster_{i + 1}"
        for j in range(vehicles_per_cluster):
            dev_name = f"C{i + 1}_D{j + 1}"
            dev_port = DEVICE_BASE_PORT + (i * 100) + j
            rsu_port = RSU_BASE_PORT + i
            spawn_vehicle(topology, dev_name, rsu_name)
            vehicle_meta.append((dev_name, dev_port, rsu_port, rsu_name, device_id_counter))
            device_id_counter += 1

    peer_directory = {meta[0]: meta[1] for meta in vehicle_meta}

    # 2. Server
    server = Server(SERVER_PORT, expected_rsus=num_clusters,
                    dataset_type=dataset, total_rounds=total_rounds)

    # 3. RSUs
    rsus = []
    for i in range(num_clusters):
        rsu_name = f"Cluster_{i + 1}"
        rsu_port = RSU_BASE_PORT + i
        cluster_ports = [meta[1] for meta in vehicle_meta if meta[3] == rsu_name]
        cluster_names = [meta[0] for meta in vehicle_meta if meta[3] == rsu_name]
        rsu = RSU(rsu_name, rsu_port, cluster_ports, SERVER_PORT,
                  topology=topology, vehicle_names=cluster_names)
        rsus.append(rsu)

    # 4. Devices (with heterogeneous private models if enabled)
    arch_map = (MNIST_PRIVATE_ARCHITECTURES if dataset == "mnist"
                else VANET_PRIVATE_ARCHITECTURES)
    arch_classes = list(arch_map.values())

    devices = []
    for name, port, rsu_port, rsu_name, dev_id in vehicle_meta:
        priv_cls = (arch_classes[dev_id % len(arch_classes)]
                    if heterogeneous else None)
        device = Device(
            name=name,
            port=port,
            rsu_port=rsu_port,
            rsu_name=rsu_name,
            device_id=dev_id,
            total_vehicles=total_vehicles,
            topology=topology,
            peer_directory=peer_directory,
            dataset_type=dataset,
            private_model_class=priv_cls,
            total_rounds=total_rounds
        )
        devices.append(device)

    # 5. Launch threads
    threads = []
    t_server = threading.Thread(target=server.start, daemon=True)
    t_server.start()
    threads.append(t_server)
    time.sleep(0.5)

    for rsu in rsus:
        t_rsu = threading.Thread(target=rsu.start, daemon=True)
        t_rsu.start()
        threads.append(t_rsu)
    time.sleep(0.5)

    for device in devices:
        device.start()

    print(f"\n[MAIN] All {total_vehicles} vehicles active. Training underway...\n")

    # 6. Await completion
    training_done_event.wait()
    time.sleep(1.0)

    # Clean shutdown of sockets
    server.shutdown()
    for rsu in rsus:
        rsu.shutdown()
    for device in devices:
        device.shutdown()

    log_filename = f"{dataset}_training_logs.txt"
    logger.save_logs(log_filename)
    logger.save_logs("training_logs.txt")
    print(f"\n[OK] Logs saved to '{log_filename}' and 'training_logs.txt'")

    # 7. Generate Plots
    logger.generate_plots(prefix=dataset)

    print("\n" + "=" * 65)
    print(f" Training Complete for [{dataset.upper()}]!")
    print(f" Output Plots: '{dataset}_accuracy_vs_rounds.png', '{dataset}_loss_vs_rounds.png'")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="ProxyFL VANET Federated Learning System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default="vanet",
                        choices=["mnist", "vanet", "both"],
                        help="Dataset to run: 'mnist', 'vanet', or 'both'")
    parser.add_argument('--clusters', type=int, default=2,
                        help="Number of RSU clusters")
    parser.add_argument('--vehicles', type=int, default=2,
                        help="Vehicles per cluster")
    parser.add_argument('--rounds', type=int, default=10,
                        help="Total communication rounds")
    parser.add_argument('--heterogeneous', action="store_true", default=True,
                        help="Enable heterogeneous private architectures across devices")
    parser.add_argument('--homogeneous', dest='heterogeneous', action='store_false',
                        help="Use identical private architectures across all devices")
    args = parser.parse_args()

    if args.dataset in ["mnist", "vanet"]:
        run_single_simulation(
            dataset=args.dataset,
            num_clusters=args.clusters,
            vehicles_per_cluster=args.vehicles,
            total_rounds=args.rounds,
            heterogeneous=args.heterogeneous
        )
    elif args.dataset == "both":
        print("\n>>> Running [1/2]: MNIST Simulation...")
        cmd_mnist = [
            sys.executable, "main.py",
            "--dataset", "mnist",
            "--clusters", str(args.clusters),
            "--vehicles", str(args.vehicles),
            "--rounds", str(args.rounds)
        ]
        if not args.heterogeneous:
            cmd_mnist.append("--homogeneous")
        subprocess.run(cmd_mnist, check=True)

        print("\n>>> Running [2/2]: VANET Simulation...")
        cmd_vanet = [
            sys.executable, "main.py",
            "--dataset", "vanet",
            "--clusters", str(args.clusters),
            "--vehicles", str(args.vehicles),
            "--rounds", str(args.rounds)
        ]
        if not args.heterogeneous:
            cmd_vanet.append("--homogeneous")
        subprocess.run(cmd_vanet, check=True)

        print("\n[DONE] Both MNIST and VANET simulations finished successfully!")
        print("Generated Artifacts:")
        print("  - mnist_accuracy_vs_rounds.png & mnist_loss_vs_rounds.png")
        print("  - vanet_accuracy_vs_rounds.png & vanet_loss_vs_rounds.png")
        print("  - accuracy_vs_rounds.png & loss_vs_rounds.png")


if __name__ == "__main__":
    main()