# rsu.py — Road-Side Unit (Cluster Head)
#
# Aggregates proxy model weights from vehicles in its cluster using FedAvg,
# then forwards the cluster average to the central server.
# Broadcasts global updates back to vehicles.
# Uses timeout-based aggregation so out-of-range vehicles don't deadlock.

import threading

from models import average_weights
from network import Receiver, send_msg


RSU_ROUND_TIMEOUT = 25  # seconds — generous window for local training before aggregating


class RSU:
    """Road-Side Unit that aggregates a cluster of vehicles.

    Args:
        name:            RSU identifier (e.g. "Cluster_1").
        port:            TCP port this RSU listens on.
        cluster_ports:   List of vehicle TCP ports in this cluster.
        server_port:     TCP port of the central server.
        topology:        Shared VanetTopology for range checks.
        vehicle_names:   List of vehicle names (parallel to cluster_ports).
    """

    def __init__(self, name, port, cluster_ports, server_port,
                 topology=None, vehicle_names=None):
        self.name = name
        self.port = port
        self.cluster_ports = cluster_ports
        self.server_port = server_port
        self.topology = topology
        self.vehicle_names = vehicle_names or []
        self.round_buffers = {}
        self.completed_rounds = set()
        self._round_timers = {}
        self._lock = threading.Lock()  # protects round_buffers, completed_rounds, _round_timers
        self.receiver = Receiver(self.port, self.on_receive)

    def on_receive(self, msg):
        if msg["type"] == "LOCAL_UPDATE":
            r = msg["round"]
            should_aggregate = False

            with self._lock:
                if r in self.completed_rounds:
                    return

                if r not in self.round_buffers:
                    self.round_buffers[r] = []
                    timer = threading.Timer(
                        RSU_ROUND_TIMEOUT, self._force_aggregate, args=[r])
                    timer.daemon = True
                    timer.start()
                    self._round_timers[r] = timer

                self.round_buffers[r].append(msg)

                # If all vehicles reported, aggregate immediately
                if len(self.round_buffers[r]) == len(self.cluster_ports):
                    self._cancel_timer_locked(r)
                    should_aggregate = True

            if should_aggregate:
                self.aggregate(r)

        elif msg["type"] == "GLOBAL_UPDATE":
            # Broadcast global proxy back to cluster vehicles
            for port in self.cluster_ports:
                send_msg(("127.0.0.1", port), msg)

    def aggregate(self, r):
        """FedAvg the received proxy weights and forward to the server."""
        with self._lock:
            if r in self.completed_rounds:
                return
            self.completed_rounds.add(r)
            self._cancel_timer_locked(r)
            data = self.round_buffers.pop(r, None)

        if not data:
            return

        n_received = len(data)
        n_expected = len(self.cluster_ports)
        avg_weights = average_weights([d["weights"] for d in data])

        print(f"[{self.name}] Aggregated {n_received}/{n_expected} vehicles "
              f"(Round {r}) -> forwarding to Server")

        msg = {
            "type": "CLUSTER_UPDATE",
            "rsu_port": self.port,
            "round": r,
            "avg_weights": avg_weights,
        }
        send_msg(("127.0.0.1", self.server_port), msg)

    def _force_aggregate(self, r):
        """Timeout handler: aggregate whatever we have so far."""
        with self._lock:
            if r in self.completed_rounds:
                return
            has_data = r in self.round_buffers and len(self.round_buffers[r]) > 0
            if has_data:
                n = len(self.round_buffers[r])
                print(f"[{self.name}] [!] Timeout! Aggregating {n}/"
                      f"{len(self.cluster_ports)} vehicles for round {r}")

        if has_data:
            self.aggregate(r)

    def _cancel_timer_locked(self, r):
        """Cancel a round timer. Must be called with self._lock held."""
        timer = self._round_timers.pop(r, None)
        if timer:
            timer.cancel()

    def start(self):
        self.receiver.start()

    def shutdown(self):
        self.receiver.shutdown()