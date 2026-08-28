import math
import unittest

from aodv import AodvSettings, Route, sequence_newer
from routing_sim import RoutingSimulator, TopologySnapshot


def graph(edges, nodes=("A", "B", "C")):
    return TopologySnapshot.from_edges(nodes, [(a, b, 100.0) for a, b in edges])


class AodvRoutingTests(unittest.TestCase):
    def simulator(self, **settings):
        return RoutingSimulator(AodvSettings(**settings), capacity=lambda d: 1_000_000)

    def test_line_discovers_and_counts_each_hop_once(self):
        sim = self.simulator()
        result = sim.submit("A", "C", 1250, 1250, 1, graph([("A", "B"), ("B", "C")]))
        self.assertTrue(result.delivered)
        self.assertEqual(result.path, ("A", "B", "C"))
        row = sim.ledger.rows()[0]
        self.assertEqual((row["rreq_packets_tx"], row["rrep_packets_tx"]), (2, 2))
        self.assertEqual(row["data_packets_tx"], 4)
        self.assertEqual(row["data_packets_delivered"], 2)
        self.assertEqual(row["routing_control_bytes_tx"], 88)
        self.assertEqual(row["ip_udp_header_bytes_tx"], 224)
        self.assertEqual(row["fl_application_bytes_tx"], 2500)
        self.assertEqual(row["total_wireless_bytes_tx"], 2812)
        self.assertEqual(row["normalized_routing_load"], 2)

    def test_active_route_reused_then_expiry_rediscovers(self):
        sim = self.simulator()
        snapshot = graph([("A", "B"), ("B", "C")])
        sim.submit("A", "C", 100, 100, 1, snapshot)
        sim.submit("A", "C", 100, 100, 1, snapshot)
        self.assertEqual(sim.ledger.rows()[0]["rreq_packets_tx"], 2)
        sim.submit("A", "C", 100, 100, 2, snapshot)
        self.assertEqual(sim.ledger.rows()[1]["rreq_packets_tx"], 2)

    def test_diamond_suppresses_duplicate_requests(self):
        sim = self.simulator()
        snapshot = graph([("A", "B"), ("A", "D"), ("B", "C"), ("D", "C")], ("A", "B", "C", "D"))
        self.assertTrue(sim.submit("A", "C", 100, 100, 1, snapshot).delivered)
        row = sim.ledger.rows()[0]
        self.assertEqual(row["rreq_packets_tx"], 3)
        self.assertEqual(row["rrep_packets_tx"], 2)

    def test_disconnected_retries_are_bounded_and_nrl_undefined(self):
        sim = self.simulator()
        result = sim.submit("A", "C", 100, 100, 1, graph([]))
        self.assertFalse(result.delivered)
        row = sim.ledger.rows()[0]
        self.assertEqual(row["rreq_packets_tx"], 3)
        self.assertEqual(row["total_wireless_bytes_tx"], 156)
        self.assertEqual(row["data_packets_tx"], 0)
        self.assertTrue(math.isnan(row["normalized_routing_load"]))

    def test_hop_limit_prevents_unbounded_flood(self):
        sim = self.simulator(network_diameter=1, rreq_retries=0)
        result = sim.submit("A", "C", 100, 100, 1, graph([("A", "B"), ("B", "C")]))
        self.assertFalse(result.delivered)
        self.assertEqual(sim.ledger.rows()[0]["rreq_packets_tx"], 1)

    def test_broken_link_emits_rerr_then_finds_alternate(self):
        sim = self.simulator()
        nodes = ("A", "B", "C", "D", "E")
        sim.submit("A", "C", 100, 100, 1, graph([("A", "B"), ("B", "C")], nodes))
        result = sim.submit("A", "C", 100, 100, 1, graph([("A", "B"), ("A", "D"), ("D", "E"), ("E", "C")], nodes))
        self.assertTrue(result.delivered)
        self.assertEqual(result.path, ("A", "D", "E", "C"))
        self.assertGreaterEqual(sim.ledger.rows()[0]["rerr_packets_tx"], 1)
        self.assertFalse(sim.protocol.tables["B"]["C"].valid)

    def test_security_partition_and_headers_conserve_bytes(self):
        sim = self.simulator()
        sim.submit("A", "C", 1300, 1000, 1, graph([("A", "B"), ("B", "C")]))
        row = sim.ledger.rows()[0]
        self.assertEqual(row["fl_application_bytes_tx"], 2000)
        self.assertEqual(row["security_bytes_tx"], 600)
        self.assertEqual(row["total_wireless_bytes_tx"], 2912)

    def test_sequence_wrap_and_stale_route_rejection(self):
        self.assertTrue(sequence_newer(0, 0xFFFFFFFF))
        self.assertFalse(sequence_newer(0xFFFFFFFF, 0))
        self.assertFalse(sequence_newer(0x80000000, 0))
        sim = self.simulator()
        sim.protocol.install("A", "C", "B", 2, 0, 3)
        sim.protocol.install("A", "C", "D", 1, 0xFFFFFFFF, 4)
        self.assertEqual(sim.protocol.tables["A"]["C"].next_hop, "B")

    def test_same_order_replays_identical_trace_and_clock_never_rewinds(self):
        traces = []
        for _ in range(2):
            sim = self.simulator()
            snapshot = graph([("A", "B"), ("B", "C")])
            sim.submit("A", "C", 100, 100, 2, snapshot)
            before = sim.now
            sim.submit("C", "A", 100, 100, 1, snapshot)
            self.assertGreater(sim.now, before)
            traces.append(sim.ledger.events)
        self.assertEqual(traces[0], traces[1])

    def test_snapshot_is_immutable_and_invalid_inputs_fail_closed(self):
        snapshot = graph([])
        with self.assertRaises(TypeError):
            snapshot.adjacency["A"]["B"] = 1
        with self.assertRaises(ValueError):
            self.simulator(rreq_retries=-1)
        with self.assertRaises(ValueError):
            self.simulator().submit("missing", "C", 10, 10, 1, snapshot)
        with self.assertRaises(ValueError):
            self.simulator().submit("A", "C", 10, 11, 1, snapshot)

    def test_delivery_latency_does_not_wait_for_unrelated_flood_branch(self):
        sim = self.simulator()
        snapshot = graph([("A", "C"), ("A", "B"), ("B", "D"), ("D", "E"), ("E", "F")],
                         ("A", "B", "C", "D", "E", "F"))
        result = sim.submit("A", "C", 100, 100, 1, snapshot)
        self.assertTrue(result.delivered)
        # RREQ + RREP + DATA: 3*.04 processing + (52+48+128)*8/1e6 serialization.
        self.assertAlmostEqual(result.latency_s, 0.121825, places=5)

    def test_expired_route_does_not_emit_spurious_link_break_error(self):
        sim = self.simulator()
        sim.submit("A", "C", 100, 100, 1, graph([("A", "B"), ("B", "C")]))
        sim.submit("A", "C", 100, 100, 2, graph([("A", "B")]))
        self.assertEqual(sim.ledger.rows()[1]["rerr_packets_tx"], 0)

    def test_discovery_deadline_rejects_reply_that_arrives_too_late(self):
        sim = RoutingSimulator(AodvSettings(network_diameter=1, rreq_retries=0),
                               capacity=lambda d: 1000)
        result = sim.submit("A", "C", 100, 100, 1, graph([("A", "C")]))
        self.assertFalse(result.delivered)
        self.assertAlmostEqual(result.latency_s, 0.08)
        self.assertEqual(sim.ledger.rows()[0]["data_packets_delivered"], 0)

    def test_rerr_to_multiple_precursors_is_one_broadcast(self):
        sim = self.simulator()
        nodes = ("A", "B", "C", "D")
        snapshot = graph([("A", "B"), ("B", "C"), ("D", "B")], nodes)
        sim.submit("A", "C", 100, 100, 1, snapshot)
        sim.submit("D", "C", 100, 100, 1, snapshot)
        sim.submit("A", "C", 100, 100, 1, graph([("A", "B"), ("D", "B")], nodes))
        errors = [event for event in sim.ledger.events if event.get("packet_type") == "RERR"]
        self.assertEqual(len(errors), 1)
        self.assertIsNone(errors[0]["destination"])
        self.assertEqual(errors[0]["recipients"], ["A", "D"])

    def test_rerr_destination_lists_are_split_to_packet_limit(self):
        sim = self.simulator(packet_payload_bytes=24)
        snapshot = graph([("A", "B"), ("B", "C")])
        sim.submit("A", "C", 10, 10, 1, snapshot)
        for destination in ("X", "Y"):
            route = sim.protocol.install("B", destination, "C", 2, 0, sim.now + 3)
            route.precursors.add("A")
        sim.submit("A", "C", 10, 10, 1, graph([("A", "B")]))
        errors = [event for event in sim.ledger.events if event.get("packet_type") == "RERR"]
        self.assertEqual([event["body_bytes"] for event in errors], [20, 12])

    def test_live_source_route_with_expired_relay_rediscovers_before_delivery(self):
        sim = self.simulator()
        snapshot = graph([("A", "B"), ("B", "C")])
        sim.submit("A", "C", 100, 100, 1, snapshot)
        sim.protocol.tables["B"]["C"].expiry = sim.now
        self.assertTrue(sim.submit("A", "C", 100, 100, 1, snapshot).delivered)
        self.assertEqual(sim.ledger.rows()[0]["rreq_packets_tx"], 4)


if __name__ == "__main__":
    unittest.main()
