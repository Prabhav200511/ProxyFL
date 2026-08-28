# main.py — ProxyFL VANET Distributed Simulation Entry Point
#
# Unified command-line interface for running ProxyFL on MNIST, VANET, or Both.
# Automatically logs metrics and produces Accuracy vs. Rounds and Loss vs. Rounds plots.

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import random
import subprocess
import sys
import threading
import time

import numpy as np
import torch

from config import (
    SERVER_PORT, RSU_BASE_PORT, DEVICE_BASE_PORT, TOTAL_ROUNDS,
    SECURITY_ENABLED, BATCH_VERIFICATION_ENABLED, RSU_LAYOUT,
    VEHICLES_PER_CLUSTER_RANGE, SIMULATION_SEED,
)
import config
from server import Server, training_done_event
from device import Device
from rsu import RSU
from vanet_sim import VanetTopology, format_vehicle_id, place_rsu, spawn_vehicle
from shared_logger import logger
from models import VANET_PRIVATE_ARCHITECTURES, MNIST_PRIVATE_ARCHITECTURES
from crypto_protocol import (
    Authority, MIRACL_BRIDGE_AVAILABLE, MIRACL_SYMMETRIC_BACKEND,
)
from metrics import metrics_tracker
from data_utils import prepare_vanet_partitions
from network import WirelessRouter
from routing_sim import RoutingSimulator


def cluster_layout(count):
    """Keep the default layout; extend explicit grid runs on a 1,800m lattice."""
    if count is None:
        return list(RSU_LAYOUT)
    if type(count) is not int or not 1 <= count <= 20:
        raise ValueError("clusters must be 1..20 (bounded by existing TCP port allocation)")
    specs = list(RSU_LAYOUT[:count])
    occupied = {(x, y) for _, _, x, y in specs}
    radius = 1
    while len(specs) < count:
        for x in range(-radius, radius + 1):
            for y in range(-radius, radius + 1):
                position = (x * 1800, y * 1800)
                if position not in occupied and len(specs) < count:
                    specs.append((f"RSU_{len(specs)}_Grid", "Grid", *position))
                    occupied.add(position)
        radius += 1
    return specs


def run_single_simulation(dataset, total_rounds=TOTAL_ROUNDS, heterogeneous=True,
                           security=SECURITY_ENABLED,
                           batch_verify=BATCH_VERIFICATION_ENABLED,
                           seed=SIMULATION_SEED, routing="direct", clusters=None, vehicles=None):
    """Run one full federated learning simulation for a given dataset."""
    if routing not in {"direct", "aodv"}:
        raise ValueError("routing must be direct or aodv")
    if type(total_rounds) is not int or total_rounds < 1:
        raise ValueError("rounds must be positive")
    if vehicles is not None and (type(vehicles) is not int or not 1 <= vehicles <= 99):
        raise ValueError("vehicles per cluster must be 1..99")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config.SECURITY_ENABLED = security
    config.BATCH_VERIFICATION_ENABLED = batch_verify

    cluster_specs = cluster_layout(clusters)
    vehicle_counts = {
        rsu_name: vehicles if vehicles is not None else random.randint(*VEHICLES_PER_CLUSTER_RANGE)
        for rsu_name, _, _, _ in cluster_specs
    }
    total_vehicles = sum(vehicle_counts.values())
    vanet_scaler = None
    vanet_partitions = None
    if dataset == "vanet":
        vanet_scaler, vanet_partitions = prepare_vanet_partitions(
            "Main_data_shuffled.csv", total_vehicles)

    print(f"\n{'=' * 65}")
    print(f" ProxyFL Simulation [{dataset.upper()}]")
    print(f" Topology: {len(cluster_specs)} RSUs | {total_vehicles} Vehicles")
    for rsu_name, direction, x, y in cluster_specs:
        print(f"  - {rsu_name} ({direction}) at ({x}, {y}): "
              f"{vehicle_counts[rsu_name]} vehicles")
    print(f" Total Communication Rounds: {total_rounds}")
    print(f" Heterogeneous Private Architectures: {heterogeneous}")
    print(f" Security Enabled: {security} | Batch Verification: {batch_verify}")
    if security:
        print(f" EC arithmetic: MIRACL Core NIST256 (Python)")
        print(f" Hash / AEAD  : {MIRACL_SYMMETRIC_BACKEND}")
        if not MIRACL_BRIDGE_AVAILABLE:
            print(" [NOTE] Build crypto_protocol/miracl_core.dll to route "
                  "SHA-256 and AES-256-GCM through MIRACL Core C.")
    print(f"{'=' * 65}\n")

    # Reset logging, metrics & synchronization
    logger.reset()
    metrics_tracker.reset()
    metrics_tracker.start_simulation()
    training_done_event.clear()

    # 0. Certificateless Security Bootstrap (TA / KGC + MVD)
    authority = Authority() if security else None
    server_identity = None
    if security and authority is not None:
        authority.enroll_mvd("Server")
        server_identity = authority.register("Server", real_id="Server")

    # 1. Spatial Topology
    topology = VanetTopology(random_seed=seed)
    for rsu_name, _, x, y in cluster_specs:
        place_rsu(topology, rsu_name, center_x=x, center_y=y)

    # Spawn vehicles near assigned RSUs
    vehicle_meta = []  # (name, port, rsu_port, rsu_name, device_id)
    device_id_counter = 0
    for i, (rsu_name, _, _, _) in enumerate(cluster_specs):
        for j in range(vehicle_counts[rsu_name]):
            dev_name = format_vehicle_id(i, j + 1)
            dev_port = DEVICE_BASE_PORT + (i * 100) + j
            rsu_port = RSU_BASE_PORT + i
            spawn_vehicle(topology, dev_name, rsu_name)
            vehicle_meta.append((dev_name, dev_port, rsu_port, rsu_name, device_id_counter))
            device_id_counter += 1

    peer_directory = {meta[0]: meta[1] for meta in vehicle_meta}

    # Register identities with Authority (MVD enrollment + AID issuance)
    rsu_identities = {}
    dev_identities = {}
    if security and authority is not None:
        for rsu_name, _, _, _ in cluster_specs:
            authority.enroll_mvd(rsu_name)
            rsu_identities[rsu_name] = authority.register(rsu_name, real_id=rsu_name)
        for meta in vehicle_meta:
            dev_name = meta[0]
            authority.enroll_mvd(dev_name)
            # Recoverability check: AID must map back to enrolled real ID
            signer = authority.register(dev_name, real_id=dev_name)
            assert authority.recover_identity(signer.aid) == dev_name
            dev_identities[dev_name] = signer

    # 2. Server
    cluster_vehicle_names = {
        rsu_name: [meta[0] for meta in vehicle_meta if meta[3] == rsu_name]
        for rsu_name, _, _, _ in cluster_specs
    }
    rsu_directory = {
        rsu_name: RSU_BASE_PORT + index
        for index, (rsu_name, _, _, _) in enumerate(cluster_specs)
    }
    router = None
    if routing == "aodv":
        router = WirelessRouter(topology, RoutingSimulator(seed=seed))
        for name, port in {**peer_directory, **rsu_directory}.items():
            router.register(name, port)
        print("[ROUTING] Ideal-link Python AODV enabled; host TCP is delivery plumbing.")
    else:
        print("[ROUTING] Direct baseline: routing overhead and NRL are not modeled.")
    server = Server(SERVER_PORT, expected_rsus=len(cluster_specs),
                    dataset_type=dataset, total_rounds=total_rounds,
                    security_authority=authority, security_identity=server_identity,
                    topology=topology, cluster_vehicle_names=cluster_vehicle_names,
                    security_enabled=security,
                    batch_verification_enabled=batch_verify,
                    rsu_directory=rsu_directory,
                    vanet_scaler=vanet_scaler,
                    random_seed=seed, aodv_enabled=router is not None)

    # 3. RSUs
    rsus = []
    for i, (rsu_name, _, _, _) in enumerate(cluster_specs):
        rsu_port = RSU_BASE_PORT + i
        cluster_ports = [meta[1] for meta in vehicle_meta if meta[3] == rsu_name]
        cluster_names = [meta[0] for meta in vehicle_meta if meta[3] == rsu_name]
        rsu = RSU(rsu_name, rsu_port, cluster_ports, SERVER_PORT,
                  topology=topology, vehicle_names=cluster_names,
                  security_authority=authority,
                  security_identity=rsu_identities.get(rsu_name),
                  security_enabled=security,
                  batch_verification_enabled=batch_verify,
                  router=router, total_rounds=total_rounds,
                  initial_global_weights={
                      key: value.detach().cpu().clone()
                      for key, value in server.model.state_dict().items()
                  })
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
            total_rounds=total_rounds,
            security_authority=authority,
            security_identity=dev_identities.get(name),
            security_enabled=security,
            vanet_partition=(
                vanet_partitions[dev_id]
                if vanet_partitions is not None else None
            ),
            random_seed=seed + dev_id,
            router=router,
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
    if router is not None:
        # Silent-round watchdogs can close the server before a disconnected
        # vehicle finishes. Keep receivers alive and wait before exporting.
        for device in devices:
            device.training_thread.join()
    time.sleep(1.0)

    # Clean shutdown of sockets
    server.shutdown()
    for rsu in rsus:
        rsu.shutdown()
    for device in devices:
        device.shutdown()

    metrics_tracker.finish_simulation()

    # Save log text files
    log_filename = f"{dataset}_training_logs.txt"
    logger.save_logs(log_filename)
    logger.save_logs("training_logs.txt")
    print(f"\n[OK] Logs saved to '{log_filename}' and 'training_logs.txt'")

    # Save metrics CSVs
    quality_metrics = {}
    for v, r_dict in logger.vehicle_train_loss.items():
        for r, loss in r_dict.items():
            quality_metrics.setdefault((v, r), {})["train_loss"] = loss
    for v, r_dict in logger.vehicle_train_acc.items():
        for r, acc in r_dict.items():
            quality_metrics.setdefault((v, r), {})["train_accuracy_pct"] = acc
    for v, r_dict in logger.vehicle_private_acc.items():
        for r, acc in r_dict.items():
            quality_metrics.setdefault((v, r), {})["private_test_accuracy_pct"] = acc
    for v, r_dict in logger.vehicle_privacy.items():
        for r, (eps, delta) in r_dict.items():
            quality_metrics.setdefault((v, r), {})["epsilon"] = eps
            quality_metrics.setdefault((v, r), {})["delta"] = delta
    for r, acc in logger.global_proxy_acc.items():
        quality_metrics.setdefault(("Server", r), {})["global_proxy_accuracy_pct"] = acc

    csv_filename = f"{dataset}_metrics.csv"
    metrics_tracker.export_csv(csv_filename, quality_metrics)
    metrics_tracker.export_csv("metrics.csv", quality_metrics)
    metrics_tracker.export_simulation_summary(f"{dataset}_simulation_summary.csv")
    metrics_tracker.export_simulation_summary("simulation_summary.csv")
    print(f"[OK] Metrics saved to '{csv_filename}' and 'metrics.csv'")

    # 7. Generate Plots
    if router is not None:
        router.simulator.ledger.export(dataset, router.simulator.metadata())
    logger.generate_plots(prefix=dataset)
    if router is not None:
        from routing_plots import plot_routing_metrics
        plot_routing_metrics(f"{dataset}_routing_rounds.csv", prefix=dataset)

    print("\n" + "=" * 65)
    print(f" Training Complete for [{dataset.upper()}]!")
    print(f" Output Plots:")
    print(f"  - 'plots/{dataset}_accuracy_vs_rounds.png' & 'plots/{dataset}_loss_vs_rounds.png'")
    print(f"  - 'plots/{dataset}_energy_training_vs_rounds.png' & 'plots/{dataset}_energy_other_vs_rounds.png'")
    print(f"  - 'plots/{dataset}_end_to_end_time_vs_rounds.png' & 'plots/{dataset}_action_to_response_latency.png'")
    print(f"  - 'plots/{dataset}_cryptographic_operations.png' & 'plots/{dataset}_throughput_vs_rounds.png'")
    print(f"  - 'plots/{dataset}_vehicles_in_range_vs_rounds.png'")
    if dataset == "vanet":
        print("  - 'plots/vanet_plot_explanations.md'")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="ProxyFL VANET Federated Learning System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default="vanet",
                        choices=["mnist", "vanet", "both"],
                        help="Dataset to run: 'mnist', 'vanet', or 'both'")
    parser.add_argument('--rounds', type=int, default=5,
                        help="Total communication rounds")
    parser.add_argument('--seed', type=int, default=SIMULATION_SEED,
                        help="Simulation random seed")
    parser.add_argument('--routing', choices=['direct', 'aodv'], default='direct',
                        help="Wireless routing model; AODV uses an ideal-link event simulation")
    parser.add_argument('--clusters', type=int, default=None,
                        help="Explicit experiment RSU count (1..20); default keeps configured layout")
    parser.add_argument('--vehicles', type=int, default=None,
                        help="Fixed vehicles per RSU (1..99); default keeps configured random range")
    parser.add_argument('--heterogeneous', action="store_true", default=True,
                        help="Enable heterogeneous private architectures across devices")
    parser.add_argument('--homogeneous', dest='heterogeneous', action='store_false',
                        help="Use identical private architectures across all devices")
    parser.add_argument('--security', action="store_true", default=True,
                        help="Enable certificateless authentication security layer")
    parser.add_argument('--no-security', dest='security', action='store_false',
                        help="Disable certificateless authentication security layer")
    parser.add_argument('--batch', action="store_true", default=True,
                        help="Enable batch verification for signatures")
    parser.add_argument('--no-batch', dest='batch', action='store_false',
                        help="Disable batch verification and use single verification")
    args = parser.parse_args()

    if args.dataset in ["mnist", "vanet"]:
        run_single_simulation(
            dataset=args.dataset,
            total_rounds=args.rounds,
            heterogeneous=args.heterogeneous,
            security=args.security,
            batch_verify=args.batch,
            seed=args.seed,
            routing=args.routing,
            clusters=args.clusters, vehicles=args.vehicles,
        )
    elif args.dataset == "both":
        print("\n>>> Running [1/2]: MNIST Simulation...")
        cmd_mnist = [
            sys.executable, "main.py",
            "--dataset", "mnist",
            "--routing", args.routing,
            "--rounds", str(args.rounds),
            "--seed", str(args.seed),
        ]
        if not args.heterogeneous:
            cmd_mnist.append("--homogeneous")
        if not args.security:
            cmd_mnist.append("--no-security")
        if not args.batch:
            cmd_mnist.append("--no-batch")
        for option in ("clusters", "vehicles"):
            if getattr(args, option) is not None:
                cmd_mnist.extend(["--" + option, str(getattr(args, option))])
        subprocess.run(cmd_mnist, check=True)

        print("\n>>> Running [2/2]: VANET Simulation...")
        cmd_vanet = [
            sys.executable, "main.py",
            "--dataset", "vanet",
            "--routing", args.routing,
            "--rounds", str(args.rounds),
            "--seed", str(args.seed),
        ]
        if not args.heterogeneous:
            cmd_vanet.append("--homogeneous")
        if not args.security:
            cmd_vanet.append("--no-security")
        if not args.batch:
            cmd_vanet.append("--no-batch")
        for option in ("clusters", "vehicles"):
            if getattr(args, option) is not None:
                cmd_vanet.extend(["--" + option, str(getattr(args, option))])
        subprocess.run(cmd_vanet, check=True)

        print("\n[DONE] Both MNIST and VANET simulations finished successfully!")
        print("Generated Artifacts:")
        print("  - plots/mnist_accuracy_vs_rounds.png & plots/mnist_loss_vs_rounds.png")
        print("  - plots/vanet_accuracy_vs_rounds.png & plots/vanet_loss_vs_rounds.png")
        print("  - plots/*_energy_training_vs_rounds.png & plots/*_energy_other_vs_rounds.png")
        print("  - plots/*_end_to_end_time_vs_rounds.png & plots/*_action_to_response_latency.png")
        print("  - plots/*_cryptographic_operations.png & plots/*_throughput_vs_rounds.png (Mbps)")
        print("  - plots/*_vehicles_in_range_vs_rounds.png")
        print("  - plots/vanet_plot_explanations.md")


if __name__ == "__main__":
    main()
