"""Destination-only, ideal-link AODV subset (RFC 3561).

No intermediate replies, HELLOs, expanding rings, local repair or secure
routing. Control packets are propagated by the event simulator, not by a
global path search. Sequence numbers use serial-number arithmetic.
"""
from dataclasses import dataclass, field
import math


def sequence_newer(value, previous):
    return 0 < ((value - previous) & 0xFFFFFFFF) < 0x80000000


@dataclass(frozen=True)
class AodvSettings:
    active_route_timeout: float = 3.0
    node_traversal_time: float = 0.04
    network_diameter: int = 64
    rreq_retries: int = 2
    packet_payload_bytes: int = 1200

    def __post_init__(self):
        for value in (self.active_route_timeout, self.node_traversal_time):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("routing durations must be finite and positive")
        for value, minimum, maximum in ((self.network_diameter, 1, 255),
                                        (self.rreq_retries, 0, 100),
                                        (self.packet_payload_bytes, 24, 65507)):
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError("invalid routing limit")

    @property
    def net_traversal_time(self):
        return 2 * self.node_traversal_time * self.network_diameter

    @property
    def path_discovery_time(self):
        return 2 * self.net_traversal_time


@dataclass
class Route:
    destination: str
    next_hop: str
    hop_count: int
    sequence: int
    expiry: float
    sequence_valid: bool = True
    valid: bool = True
    precursors: set = field(default_factory=set)


class AodvProtocol:
    def __init__(self, simulator, settings):
        self.sim = simulator
        self.settings = settings
        self.tables = {}
        self.sequences = {}
        self.request_ids = {}
        self.seen = {}

    def install(self, node, destination, next_hop, hops, sequence, expiry):
        table = self.tables.setdefault(node, {})
        old = table.get(destination)
        fresh = old is None or not old.sequence_valid or sequence_newer(sequence, old.sequence)
        equal_better = old is not None and sequence == old.sequence and (
            not old.valid or hops < old.hop_count)
        if fresh or equal_better:
            table[destination] = Route(destination, next_hop, hops, sequence,
                                       expiry, precursors=set() if old is None else old.precursors.copy())
        elif old.sequence == sequence and old.next_hop == next_hop:
            old.expiry = max(old.expiry, expiry)
        return table[destination]

    def active(self, node, destination):
        route = self.tables.get(node, {}).get(destination)
        if route is not None and route.expiry <= self.sim.now:
            route.valid = False
        return route if route is not None and route.valid else None

    def discover(self, source, destination):
        # The source entry may still be live while a relay's entry expired.
        # A failed next-hop traversal must await a NEW reply, not reuse that
        # incomplete route as the discovery completion condition.
        old = self.tables.get(source, {}).get(destination)
        if old is not None:
            old.valid = False
        for attempt in range(self.settings.rreq_retries + 1):
            start = self.sim.now
            self.seen = {key: expiry for key, expiry in self.seen.items() if expiry > start}
            self.sequences[source] = (self.sequences.get(source, 0) + 1) & 0xFFFFFFFF
            self.request_ids[source] = (self.request_ids.get(source, 0) + 1) & 0xFFFFFFFF
            old = self.tables.get(source, {}).get(destination)
            request = dict(origin=source, destination=destination,
                           request_id=self.request_ids[source], origin_sequence=self.sequences[source],
                           destination_sequence=old.sequence if old and old.sequence_valid else None,
                           hops=0, ttl=self.settings.network_diameter)
            self.seen[(source, source, request["request_id"])] = start + self.settings.path_discovery_time
            self.sim.control("RREQ", source, None, 24, request, self.receive_request)
            deadline = start + self.settings.net_traversal_time * (2 ** attempt)
            self.sim.drain(until=lambda: self.active(source, destination) is not None,
                           deadline=deadline)
            if self.active(source, destination):
                return True
            # Discovery backoff is simulated time, never wall-clock sleeping.
            self.sim.now = max(self.sim.now, deadline)
        return False

    def receive_request(self, node, previous, request):
        key = (node, request["origin"], request["request_id"])
        if self.seen.get(key, -1) > self.sim.now:
            return
        self.seen[key] = self.sim.now + self.settings.path_discovery_time
        hops = request["hops"] + 1
        self.install(node, request["origin"], previous, hops, request["origin_sequence"],
                     self.sim.now + max(self.settings.active_route_timeout, self.settings.path_discovery_time))
        if node == request["destination"]:
            sequence = self.sequences.get(node, 0)
            requested = request["destination_sequence"]
            if requested is not None and (requested == sequence or sequence_newer(requested, sequence)):
                sequence = (requested + 1) & 0xFFFFFFFF
            self.sequences[node] = sequence
            reply = dict(origin=request["origin"], destination=node, sequence=sequence, hops=0)
            reverse = self.active(node, request["origin"])
            if reverse:
                self.sim.control("RREP", node, reverse.next_hop, 20, reply, self.receive_reply)
        elif hops < request["ttl"]:
            known = self.tables.get(node, {}).get(request["destination"])
            requested = request["destination_sequence"]
            if known and known.sequence_valid and (requested is None or sequence_newer(known.sequence, requested)):
                requested = known.sequence
            self.sim.control("RREQ", node, None, 24,
                             {**request, "hops": hops, "destination_sequence": requested}, self.receive_request)

    def receive_reply(self, node, previous, reply):
        route = self.install(node, reply["destination"], previous, reply["hops"] + 1,
                             reply["sequence"], self.sim.now + self.settings.active_route_timeout)
        # A stale/worse reply cannot propagate and replace fresher state upstream.
        if not route.valid or route.sequence != reply["sequence"] or route.next_hop != previous:
            return
        if node != reply["origin"]:
            reverse = self.active(node, reply["origin"])
            if reverse:
                route.precursors.add(reverse.next_hop)
                reverse.precursors.add(previous)
                self.sim.control("RREP", node, reverse.next_hop, 20,
                                 {**reply, "hops": reply["hops"] + 1}, self.receive_reply)

    def broken_link(self, node, neighbor):
        unreachable = []
        recipients = set()
        for route in self.tables.get(node, {}).values():
            if route.expiry <= self.sim.now:
                route.valid = False
            if route.valid and route.next_hop == neighbor:
                route.valid = False
                route.sequence = (route.sequence + 1) & 0xFFFFFFFF
                route.sequence_valid = True
                unreachable.append((route.destination, route.sequence))
                recipients.update(route.precursors)
        self.send_errors(node, recipients - {neighbor}, unreachable)

    def send_errors(self, node, recipients, unreachable):
        # 8-bit destination count in RERR; each item is IPv4 address + sequence.
        limit = min(255, (self.settings.packet_payload_bytes - 4) // 8)
        reachable = sorted(recipients & set(self.sim.snapshot.adjacency.get(node, {})))
        if not reachable:
            return
        # RFC 3561: unicast for one precursor, broadcast for multiple.
        recipient = reachable[0] if len(reachable) == 1 else None
        for offset in range(0, len(unreachable), limit):
            chunk = unreachable[offset:offset + limit]
            self.sim.control("RERR", node, recipient, 4 + 8 * len(chunk), chunk, self.receive_error)

    def receive_error(self, node, previous, unreachable):
        affected = []
        recipients = set()
        for destination, sequence in unreachable:
            route = self.tables.get(node, {}).get(destination)
            if route and route.valid and route.next_hop == previous and (
                    sequence == route.sequence or sequence_newer(sequence, route.sequence)):
                route.valid = False
                route.sequence = sequence
                recipients.update(route.precursors)
                affected.append((destination, sequence))
        self.send_errors(node, recipients - {previous}, affected)

    def path(self, source, destination):
        # Follow installed next hops only. This is not a topology path search.
        path = [source]
        while path[-1] != destination:
            node = path[-1]
            route = self.active(node, destination)
            if not route or route.next_hop in path or len(path) > self.settings.network_diameter:
                return ()
            if route.next_hop not in self.sim.snapshot.adjacency.get(node, {}):
                self.broken_link(node, route.next_hop)
                return ()
            path.append(route.next_hop)
        return tuple(path)

    def refresh(self, path):
        for index, node in enumerate(path[:-1]):
            route = self.tables[node][path[-1]]
            route.expiry = self.sim.now + self.settings.active_route_timeout
            if index:
                route.precursors.add(path[index - 1])
        # RFC active traffic refreshes the reverse route as well, if present.
        for node in path[1:]:
            route = self.active(node, path[0])
            if route:
                route.expiry = self.sim.now + self.settings.active_route_timeout
