# vanet_sim.py — Spatial VANET topology simulation
#
# Tracks vehicle/RSU positions, enforces communication range constraints,
# and simulates vehicle mobility across federated rounds.

import math
import random
import threading

from config import V2V_RANGE, V2RSU_RANGE, SPEED_RANGE


class VanetTopology:
    """Thread-safe spatial topology manager for the VANET simulation.

    All position and movement operations are protected by a lock so
    multiple Device threads can safely query and update the topology.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._vehicle_pos = {}       # name → (x, y)
        self._vehicle_speed = {}     # name → m/s
        self._vehicle_dir = {}       # name → radians
        self._rsu_pos = {}           # rsu_name → (x, y)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_vehicle(self, name, x, y, speed, direction):
        with self._lock:
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
            direction += random.uniform(-0.3, 0.3)

            self._vehicle_pos[name] = (x, y)
            self._vehicle_dir[name] = direction


# ------------------------------------------------------------------
# Spawning helpers
# ------------------------------------------------------------------
def place_rsu(topology, rsu_name, center_x, center_y):
    """Register an RSU at a fixed position."""
    topology.register_rsu(rsu_name, center_x, center_y)
    print(f"[VANET] RSU '{rsu_name}' placed at ({center_x:.0f}, {center_y:.0f})")


def spawn_vehicle(topology, vehicle_name, rsu_name):
    """Spawn a vehicle at a random position within 80 % of V2RSU_RANGE
    of its assigned RSU, with a random speed and heading.
    """
    rsu_x, rsu_y = topology._rsu_pos[rsu_name]  # safe — called before threads start

    angle = random.uniform(0, 2 * math.pi)
    radius = random.uniform(0, V2RSU_RANGE * 0.8)  # start inside coverage
    x = rsu_x + radius * math.cos(angle)
    y = rsu_y + radius * math.sin(angle)

    speed = random.uniform(*SPEED_RANGE)
    direction = random.uniform(0, 2 * math.pi)

    topology.register_vehicle(vehicle_name, x, y, speed, direction)

    dist = topology._dist((x, y), (rsu_x, rsu_y))
    print(f"[VANET] Vehicle '{vehicle_name}' spawned at ({x:.0f}, {y:.0f}) "
          f"| speed={speed:.1f} m/s | {dist:.0f}m from {rsu_name}")
