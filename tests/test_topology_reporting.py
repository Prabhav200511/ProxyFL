"""Regression tests for vehicle naming and RSU coverage reporting."""

from __future__ import annotations

import math
import unittest

import vanet_sim
from vanet_sim import VanetTopology, spawn_vehicle


class TopologyReportingTests(unittest.TestCase):
    def test_seed_reproduces_vehicle_spawn_and_motion(self):
        first = VanetTopology(random_seed=7)
        second = VanetTopology(random_seed=7)
        for topology in [first, second]:
            topology.register_rsu("RSU", 0, 0)
            spawn_vehicle(topology, "C0_V1", "RSU")
            topology.move_vehicle("C0_V1")
        self.assertEqual(
            first.get_vehicle_position("C0_V1"),
            second.get_vehicle_position("C0_V1"),
        )

    def test_v2v_readiness_releases_when_all_peers_mark_ready(self):
        topology = VanetTopology()
        topology.mark_v2v_ready("C0_V1", 2)
        topology.mark_v2v_ready("C0_V2", 2)
        self.assertTrue(topology.wait_for_v2v_ready(
            "C0_V1", 2, ["C0_V2"], timeout=0.01))
        topology.clear_v2v_ready("C0_V1", 2)
        topology.clear_v2v_ready("C0_V2", 2)

    def test_vehicle_identifier_uses_cluster_vehicle_format(self) -> None:
        formatter = getattr(vanet_sim, "format_vehicle_id", None)
        self.assertIsNotNone(formatter, "format_vehicle_id is required")
        self.assertEqual(formatter(0, 1), "C0_V1")
        self.assertEqual(formatter(4, 10), "C4_V10")

    def test_assigned_rsu_coverage_counts_each_cluster_and_total(self) -> None:
        topology = VanetTopology()
        topology.register_rsu("RSU_0_Central", 0, 0)
        topology.register_rsu("RSU_1_North", 0, 1800)
        topology.register_vehicle("C0_V1", 100, 0, 0, 0)
        topology.register_vehicle("C0_V2", 1200, 0, 0, 0)
        topology.register_vehicle("C1_V1", 0, 1750, 0, math.pi)

        coverage_method = getattr(topology, "assigned_rsu_coverage", None)
        self.assertIsNotNone(coverage_method, "assigned_rsu_coverage is required")
        coverage = coverage_method({
            "RSU_0_Central": ["C0_V1", "C0_V2"],
            "RSU_1_North": ["C1_V1"],
        })

        self.assertEqual(coverage["RSU_0_Central"], {"in_range": 1, "assigned": 2})
        self.assertEqual(coverage["RSU_1_North"], {"in_range": 1, "assigned": 1})
        self.assertEqual(coverage["total"], {"in_range": 2, "assigned": 3})


if __name__ == "__main__":
    unittest.main(verbosity=2)
