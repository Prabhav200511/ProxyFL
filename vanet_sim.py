# vanet_sim.py — Spatial VANET topology simulation
#
# Tracks vehicle/RSU positions, enforces communication range constraints,
# and simulates vehicle mobility across federated rounds.

import math
import random
import threading
import time

from config import V2V_RANGE, V2RSU_RANGE, SPEED_RANGE


def format_vehicle_id(cluster_id, vehicle_id):
    """Return the canonical vehicle identifier used in logs and plots."""
    if not isinstance(cluster_id, int) or cluster_id < 0:
        raise ValueError("cluster_id must be a non-negative integer")
    if not isinstance(vehicle_id, int) or vehicle_id < 1:
        raise ValueError("vehicle_id must be a positive integer")
    return f"C{cluster_id}_V{vehicle_id}"


class VanetTopology:
    """Thread-safe spatial topology manager for the VANET simulation.

    All position and movement operations are protected by a lock so
    multiple Device threads can safely query and update the topology.
    """

    def __init__(self, random_seed=None):
        self._lock = threading.Lock()
        self._master_seed = random_seed
        self._master_rng = random.Random(random_seed)
        self._vehicle_rng = {}
        self._vehicle_pos = {}       # name → (x, y)
        self._vehicle_speed = {}     # name → m/s
        self._vehicle_dir = {}       # name → radians
        self._rsu_pos = {}           # rsu_name → (x, y)
        self._v2v_ready = set()       # (round_num, vehicle_name)
        self._v2v_condition = threading.Condition(self._lock)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_vehicle(self, name, x, y, speed, direction):
        with self._lock:
            if name not in self._vehicle_rng:
                index = len(self._vehicle_rng)
                seed = (
                    None if self._master_seed is None
                    else self._master_seed + index
                )
                self._vehicle_rng[name] = random.Random(seed)
            self._vehicle_pos[name] = (x, y)
            self._vehicle_speed[name] = speed
            self._vehicle_dir[name] = direction

    def register_rsu(self, name, x, y):
        with self._lock:
            self._rsu_pos[name] = (x, y)

    # ------------------------------------------------------------------
    # Range Queries
    # ------------------------------------------------------------------
    @staticmethod
    def _dist(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def get_v2v_neighbors(self, name, all_names):
        """Return list of vehicle names within V2V_RANGE of *name*."""
        with self._lock:
            if name not in self._vehicle_pos:
                return []
            pos = self._vehicle_pos[name]
            neighbors = []
            for other in all_names:
                if other == name or other not in self._vehicle_pos:
                    continue
                if self._dist(pos, self._vehicle_pos[other]) <= V2V_RANGE:
                    neighbors.append(other)
            return neighbors

    def get_v2v_distance(self, first_name, second_name):
        """Return Euclidean distance (meters) between two vehicles."""
        with self._lock:
            first = self._vehicle_pos.get(first_name)
            second = self._vehicle_pos.get(second_name)
            if first is None or second is None:
                return float("inf")
            return self._dist(first, second)

    def mark_v2v_ready(self, name, round_num):
        """Publish that a vehicle has finished local training for a round."""
        with self._v2v_condition:
            self._v2v_ready.add((round_num, name))
            self._v2v_condition.notify_all()

    def wait_for_v2v_ready(
            self, name, round_num, peer_names, timeout):
        """Wait until all current peers are ready, bounded by ``timeout``."""
        del name  # retained in the interface for readable call sites
        required = {(round_num, peer) for peer in peer_names}
        deadline = time.monotonic() + max(float(timeout), 0.0)
        with self._v2v_condition:
            while not required.issubset(self._v2v_ready):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._v2v_condition.wait(remaining)
            return True

    def clear_v2v_ready(self, name, round_num):
        """Remove a round marker once the vehicle leaves the V2V phase."""
        with self._v2v_condition:
            self._v2v_ready.discard((round_num, name))
            self._v2v_condition.notify_all()

    def can_reach_rsu(self, vehicle_name, rsu_name):
        """True if *vehicle_name* is within V2RSU_RANGE of *rsu_name*."""
        with self._lock:
            vp = self._vehicle_pos.get(vehicle_name)
            rp = self._rsu_pos.get(rsu_name)
            if vp is None or rp is None:
                return False
            return self._dist(vp, rp) <= V2RSU_RANGE

    def get_distance_to_rsu(self, vehicle_name, rsu_name):
        """Return Euclidean distance (meters) between vehicle and RSU."""
        with self._lock:
            vp = self._vehicle_pos.get(vehicle_name)
            rp = self._rsu_pos.get(rsu_name)
            if vp is None or rp is None:
                return float('inf')
            return self._dist(vp, rp)

    def get_vehicle_position(self, name):
        with self._lock:
            return self._vehicle_pos.get(name, (0, 0))

    def assigned_rsu_coverage(self, assignments):
        """Return per-RSU and total coverage for assigned vehicle names."""
        with self._lock:
            coverage = {}
            total_in_range = 0
            total_assigned = 0
            for rsu_name, vehicle_names in assignments.items():
                rsu_position = self._rsu_pos.get(rsu_name)
                assigned = len(vehicle_names)
                in_range = 0
                if rsu_position is not None:
                    in_range = sum(
                        1
                        for vehicle_name in vehicle_names
                        if vehicle_name in self._vehicle_pos
                        and self._dist(self._vehicle_pos[vehicle_name], rsu_position)
                        <= V2RSU_RANGE
                    )
                coverage[rsu_name] = {
                    "in_range": in_range,
                    "assigned": assigned,
                }
                total_in_range += in_range
                total_assigned += assigned
            coverage["total"] = {
                "in_range": total_in_range,
                "assigned": total_assigned,
            }
            return coverage

    # ------------------------------------------------------------------
    # Mobility — called once per round
    # ------------------------------------------------------------------
    def move_vehicle(self, name, dt=10.0):
        """Advance *name* by speed × dt, with random steering jitter.

        Args:
            dt: Simulated seconds per federated round.
        """
        with self._lock:
            if name not in self._vehicle_pos:
                return
            x, y = self._vehicle_pos[name]
            speed = self._vehicle_speed[name]
            direction = self._vehicle_dir[name]

            x += speed * math.cos(direction) * dt
            y += speed * math.sin(direction) * dt

            # Random steering jitter (simulates road curves / lane changes)
            direction += self._vehicle_rng[name].uniform(-0.3, 0.3)

            self._vehicle_pos[name] = (x, y)
            self._vehicle_dir[name] = direction


# ------------------------------------------------------------------
# Spawning helpers
# ------------------------------------------------------------------
def place_rsu(topology, rsu_name, center_x, center_y):
    """Register an RSU at a fixed position."""
    topology.register_rsu(rsu_name, center_x, center_y)
    print(f"[VANET] RSU '{rsu_name}' placed at ({center_x:.0f}, {center_y:.0f})")


def spawn_vehicle(topology, vehicle_name, rsu_name, random_seed=None):
    """Spawn a vehicle at a random position within 80 % of V2RSU_RANGE
    of its assigned RSU, with a random speed and heading.
    """
    rsu_x, rsu_y = topology._rsu_pos[rsu_name]  # safe — called before threads start

    rng = (
        random.Random(random_seed)
        if random_seed is not None else topology._master_rng
    )
    angle = rng.uniform(0, 2 * math.pi)
    radius = rng.uniform(0, V2RSU_RANGE * 0.8)  # start inside coverage
    x = rsu_x + radius * math.cos(angle)
    y = rsu_y + radius * math.sin(angle)

    speed = rng.uniform(*SPEED_RANGE)
    direction = rng.uniform(0, 2 * math.pi)

    topology.register_vehicle(vehicle_name, x, y, speed, direction)

    dist = topology._dist((x, y), (rsu_x, rsu_y))
    print(f"[VANET] Vehicle '{vehicle_name}' spawned at ({x:.0f}, {y:.0f}) "
          f"| speed={speed:.1f} m/s | {dist:.0f}m from {rsu_name}")
