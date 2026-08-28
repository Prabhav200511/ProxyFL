"""Serialized message-level discrete-event ideal-radio simulation.

Topology is frozen for one message. No wall-clock sleep, contention, fading,
MAC ACK or retransmission is represented. Host TCP remains outside this clock.
"""
from dataclasses import asdict, dataclass
import heapq
import itertools
import math
import threading
from types import MappingProxyType

from aodv import AodvProtocol, AodvSettings
from routing_metrics import RoutingLedger


@dataclass(frozen=True)
class TopologySnapshot:
    adjacency: object

    def __post_init__(self):
        copied = {node: dict(neighbors) for node, neighbors in self.adjacency.items()}
        for node, neighbors in copied.items():
            for other, distance in neighbors.items():
                if node == other or not math.isfinite(distance) or distance < 0:
                    raise ValueError("invalid topology edge")
                if other not in copied or copied[other].get(node) != distance:
                    raise ValueError("topology must be bidirectional")
        object.__setattr__(self, "adjacency", MappingProxyType({
            node: MappingProxyType(neighbors) for node, neighbors in copied.items()}))

    @classmethod
    def from_edges(cls, nodes, edges):
        adjacency = {node: {} for node in nodes}
        for first, second, distance in edges:
            if first not in adjacency or second not in adjacency:
                raise ValueError("edge refers to unregistered topology node")
            adjacency[first][second] = distance
            adjacency[second][first] = distance
        return cls(adjacency)

    def edges(self):
        return [(node, other, distance) for node in sorted(self.adjacency)
                for other, distance in sorted(self.adjacency[node].items()) if node < other]


@dataclass(frozen=True)
class Delivery:
    message_id: int
    round_num: int
    delivered: bool
    path: tuple
    latency_s: float
    wireless_hops: tuple = ()


class RoutingSimulator:
    def __init__(self, settings=None, capacity=None, seed=None):
        self.settings = settings or AodvSettings()
        if capacity is None:
            from vanet_channel import link_capacity_bps
            import config
            capacity = link_capacity_bps
            self.radio_configuration = {name: getattr(config, name) for name in (
                "VANET_BANDWIDTH_HZ", "VANET_NOISE_FIGURE_DB", "VANET_PATH_LOSS_1M_DB",
                "VANET_PATH_LOSS_EXPONENT", "VANET_PHY_MAX_RATE_BPS", "VANET_TX_POWER_DBM")}
        else:
            self.radio_configuration = {"model": "caller supplied; capacities recorded per TX event"}
        self.capacity = capacity
        self.seed = seed
        self.ledger = RoutingLedger()
        self.now = 0.0
        self.snapshot = TopologySnapshot({})
        self.protocol = AodvProtocol(self, self.settings)
        self._queue = []
        self._order = itertools.count()
        self._transmitter_free = {}
        self._message_ids = itertools.count(1)
        self._packet_ids = itertools.count(1)
        self._lock = threading.Lock()

    def schedule(self, when, callback, *args):
        heapq.heappush(self._queue, (when, next(self._order), callback, args))

    def drain(self, until=None, deadline=None):
        while self._queue:
            if until is not None and until():
                return
            if deadline is not None and self._queue[0][0] > deadline:
                self.now = max(self.now, deadline)
                return
            when, _, callback, args = heapq.heappop(self._queue)
            self.now = max(self.now, when)
            callback(*args)
        if deadline is not None and not (until is not None and until()):
            self.now = max(self.now, deadline)

    def transmit(self, kind, source, destination, body_bytes, payload, callback,
                 application=0, security=0, packet_id=None):
        neighbors = self.snapshot.adjacency[source]
        recipients = sorted(neighbors) if destination is None else [destination]
        if destination is not None and destination not in neighbors:
            raise ValueError("attempted transmission over absent link")
        distance = max((neighbors[node] for node in recipients), default=0.0)
        capacity = self.capacity(distance)
        if not math.isfinite(capacity) or capacity <= 0:
            raise ValueError("link capacity must be finite and positive")
        start = max(self.now, self._transmitter_free.get(source, 0))
        finish = start + (body_bytes + 28) * 8 / capacity
        self._transmitter_free[source] = finish
        if packet_id is None:
            packet_id = next(self._packet_ids)
        event = dict(event="tx", message_id=self._message_id, packet_id=packet_id,
                     round=self._round_num, packet_type=kind, source=source,
                     destination=destination, recipients=recipients, body_bytes=body_bytes,
                     header_bytes=28, capacity_bps=capacity, start_s=start, finish_s=finish)
        if kind == "DATA":
            event.update(fl_application_bytes=application, security_bytes=security)
        else:
            event["control"] = payload
        self.ledger.transmission(event, application, security)
        # Even an empty-neighbor broadcast consumes airtime.
        self.schedule(finish, lambda: None)
        for recipient in recipients:
            arrival = finish + neighbors[recipient] / 299_792_458 + self.settings.node_traversal_time
            self.schedule(arrival, callback, recipient, source, payload)

    def control(self, kind, source, destination, size, payload, callback):
        self.transmit(kind, source, destination, size, payload, callback)

    def submit(self, source, destination, wire_bytes, application_bytes, round_num,
               snapshot, arrival_time=None):
        if not isinstance(snapshot, TopologySnapshot):
            raise ValueError("wireless topology is required")
        if source not in snapshot.adjacency or destination not in snapshot.adjacency or source == destination:
            raise ValueError("invalid wireless endpoints")
        if type(wire_bytes) is not int or type(application_bytes) is not int or not 0 <= application_bytes <= wire_bytes or wire_bytes <= 0:
            raise ValueError("invalid message byte partition")
        if type(round_num) is not int or round_num < 1:
            raise ValueError("round must be a positive integer")
        arrival = (round_num - 1) * 10.0 if arrival_time is None else float(arrival_time)
        if not math.isfinite(arrival) or arrival < 0:
            raise ValueError("invalid arrival time")
        with self._lock:
            self.now = max(self.now, arrival)
            start = self.now
            self._message_id = next(self._message_ids)
            self._round_num = round_num
            self.ledger.submission(round_num, dict(event="submission", message_id=self._message_id,
                round=round_num, source=source, destination=destination, wire_bytes=wire_bytes,
                application_bytes=application_bytes, arrival_s=arrival, start_s=start,
                topology_nodes=sorted(snapshot.adjacency), topology_edges=snapshot.edges()))
            old_snapshot = self.snapshot
            self.snapshot = snapshot
            for node in sorted(old_snapshot.adjacency):
                for neighbor in sorted(old_snapshot.adjacency[node]):
                    if neighbor not in snapshot.adjacency.get(node, {}):
                        self.ledger.event(dict(event="link_break", message_id=self._message_id,
                                               round=round_num, source=node, neighbor=neighbor, time_s=self.now))
                        self.protocol.broken_link(node, neighbor)
            self.drain()
            path = self.protocol.path(source, destination)
            if not path:
                self.drain()
                self.protocol.discover(source, destination)
                path = self.protocol.path(source, destination)
            packets = 0
            if path:
                for offset in range(0, wire_bytes, self.settings.packet_payload_bytes):
                    size = min(self.settings.packet_payload_bytes, wire_bytes - offset)
                    # Accounting allocation only, not the physical order of fields:
                    # baseline bytes first, secure-envelope increment second.
                    application = min(size, max(0, application_bytes - offset))
                    packet_id = next(self._packet_ids)
                    for first, second in zip(path, path[1:]):
                        arrived = []
                        self.transmit("DATA", first, second, size, None, lambda *args: arrived.append(True),
                                      application, size - application, packet_id)
                        self.drain(until=lambda: bool(arrived))
                        self.protocol.refresh(path)
                    packets += 1
                    self.ledger.event(dict(event="packet_arrival", message_id=self._message_id,
                                           packet_id=packet_id, destination=destination,
                                           round=round_num, time_s=self.now))
            hops = tuple((node, wire_bytes + packets * 28, self.capacity(snapshot.adjacency[node][other]))
                         for node, other in zip(path, path[1:]))
            result = Delivery(self._message_id, round_num, bool(path), path, self.now - start, hops)
            # Complete residual floods using THIS snapshot, but do not charge
            # their completion time to a message already received at its endpoint.
            self.drain()
            self.ledger.completed(result, packets)
            return result

    def metadata(self, traffic="FL envelopes"):
        return dict(routing_mode="aodv", model="ideal-link destination-only AODV subset",
                    settings=asdict(self.settings), seed=self.seed, traffic=traffic,
                    radio_configuration=self.radio_configuration,
                    boundary="wireless IPv4/UDP packets; host TCP and wired backhaul excluded",
                    units={"time": "simulated seconds", "volume": "bytes", "NRL": "control TX / final data arrivals"},
                    hello_enabled=False, rrep_ack_enabled=False, routing_control_authenticated=False,
                    application_acceptance="not inferred from network arrival or host handoff; endpoint verification unchanged",
                    assumptions=["immutable topology per message; mobility between messages",
                                 "serialized submissions; ordered trace required for deterministic replay",
                                 "per-transmitter serialization, no contention/fading/MAC ACK/retransmission",
                                 "data fragments forwarded sequentially, no pipelining",
                                 "full-network TTL-limited flood; no intermediate replies or local repair",
                                 "IP/UDP virtual packetization; not host TCP measurement"],
                    control_body_bytes={"RREQ": 24, "RREP": 20, "RERR": "4 + 8 * destinations"},
                    ip_udp_header_bytes=28, round_arrival_interval_s=10)
