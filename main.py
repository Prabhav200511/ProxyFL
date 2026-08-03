import argparse
import threading
import time

from server import Server, training_done_event
from device import Device
from rsu import RSU
from shared_logger import logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clusters', type=int, required=True)
    parser.add_argument('--vehicles', type=int, required=True)
    args = parser.parse_args()

    num_clusters = args.clusters
    vehicles_per_cluster = args.vehicles
    total_vehicles = num_clusters * vehicles_per_cluster

    print(f"\n[MAIN] Initializing Topology: {num_clusters} Clusters, {total_vehicles} Total Vehicles...")

    server = Server(8000, expected_rsus=num_clusters)

    rsus = []
    for i in range(num_clusters):
        rsu = RSU(f"Cluster_{i + 1}", 5000 + i, [], 8000)
        rsus.append(rsu)

    devices = []
    device_id_counter = 0
    for i in range(num_clusters):
        target_rsu_port = 5000 + i
        cluster_ports = []
        for j in range(vehicles_per_cluster):
            dev_port = 6000 + (i * 100) + j
            dev_name = f"C{i + 1}_D{j + 1}"
            device = Device(dev_name, dev_port, target_rsu_port, device_id_counter, total_vehicles)
            devices.append(device)
            cluster_ports.append(dev_port)
            device_id_counter += 1

        rsus[i].cluster_ports = cluster_ports

    threads = []
    t_server = threading.Thread(target=server.start)
    t_server.start()
    threads.append(t_server)

    time.sleep(1)

    for rsu in rsus:
        t_rsu = threading.Thread(target=rsu.start)
        t_rsu.start()
        threads.append(t_rsu)

    time.sleep(1)

    for device in devices:
        t_dev = threading.Thread(target=device.start)
        t_dev.start()
        threads.append(t_dev)

    print("[MAIN] Training loop initialized. Processing network streams...")
    training_done_event.wait()

    print("\nTraining Complete! Saving logs to 'training_logs.txt'...")
    logger.save_logs("training_logs.txt")
    print("Logs successfully saved. Exiting.")

    for t in threads:
        t.join(timeout=1)


if __name__ == "__main__":
    main()