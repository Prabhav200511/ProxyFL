"""Per-hop IP-boundary accounting; network arrival is not FL acceptance."""
import csv
import json
import math
from pathlib import Path
import threading


COMPONENTS = ("fl_application_bytes_tx", "security_bytes_tx",
              "routing_control_bytes_tx", "ip_udp_header_bytes_tx")


class RoutingLedger:
    def __init__(self):
        self.events = []
        self._rows = {}
        self._lock = threading.RLock()

    def _row(self, round_num):
        if round_num not in self._rows:
            names = (*COMPONENTS, "total_wireless_bytes_tx", "data_packets_tx",
                     "data_packets_delivered", "messages_submitted", "messages_network_delivered",
                     "messages_no_route", "host_handoffs_succeeded", "host_handoffs_failed",
                     "network_latency_sum_s", "successful_network_latency_sum_s",
                     "rreq_packets_tx", "rrep_packets_tx", "rerr_packets_tx",
                     "rreq_bytes_tx", "rrep_bytes_tx", "rerr_bytes_tx")
            self._rows[round_num] = dict.fromkeys(names, 0)
        return self._rows[round_num]

    def event(self, event):
        with self._lock:
            self.events.append(event)

    def transmission(self, event, application=0, security=0):
        with self._lock:
            row = self._row(event["round"])
            kind = event["packet_type"]
            row["ip_udp_header_bytes_tx"] += 28
            row["total_wireless_bytes_tx"] += event["body_bytes"] + 28
            if kind == "DATA":
                row["data_packets_tx"] += 1
                row["fl_application_bytes_tx"] += application
                row["security_bytes_tx"] += security
            else:
                row[kind.lower() + "_packets_tx"] += 1
                row[kind.lower() + "_bytes_tx"] += event["body_bytes"] + 28
                row["routing_control_bytes_tx"] += event["body_bytes"]
            if sum(row[key] for key in COMPONENTS) != row["total_wireless_bytes_tx"]:
                raise ValueError("wireless byte partition does not conserve volume")
            self.events.append(event)

    def submission(self, round_num, event):
        with self._lock:
            self._row(round_num)["messages_submitted"] += 1
            self.events.append(event)

    def completed(self, delivery, packets):
        with self._lock:
            row = self._row(delivery.round_num)
            row["network_latency_sum_s"] += delivery.latency_s
            row["messages_network_delivered" if delivery.delivered else "messages_no_route"] += 1
            if delivery.delivered:
                row["data_packets_delivered"] += packets
                row["successful_network_latency_sum_s"] += delivery.latency_s
            self.events.append(dict(event="network_result", message_id=delivery.message_id,
                                    round=delivery.round_num, delivered=delivery.delivered,
                                    path=delivery.path, latency_s=delivery.latency_s))

    def host_handoff(self, delivery, succeeded):
        with self._lock:
            self._row(delivery.round_num)["host_handoffs_succeeded" if succeeded else "host_handoffs_failed"] += 1
            self.events.append(dict(event="host_handoff", message_id=delivery.message_id,
                                    round=delivery.round_num, succeeded=bool(succeeded)))

    def rows(self):
        with self._lock:
            result = []
            for round_num, original in sorted(self._rows.items()):
                row = {"round": round_num, **original}
                control = sum(row[k + "_packets_tx"] for k in ("rreq", "rrep", "rerr"))
                row["normalized_routing_load"] = control / row["data_packets_delivered"] if row["data_packets_delivered"] else math.nan
                row["network_latency_mean_s"] = row["network_latency_sum_s"] / row["messages_submitted"]
                row["successful_network_latency_mean_s"] = (
                    row["successful_network_latency_sum_s"] / row["messages_network_delivered"]
                    if row["messages_network_delivered"] else math.nan)
                result.append(row)
            return result

    def export(self, prefix, metadata):
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            rows = self.rows()
            with Path(str(prefix) + "_routing_rounds.csv").open("w", newline="", encoding="utf-8") as stream:
                if rows:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            with Path(str(prefix) + "_routing_events.jsonl").open("w", encoding="utf-8") as stream:
                for event in self.events:
                    stream.write(json.dumps(event, sort_keys=True) + "\n")
            Path(str(prefix) + "_routing_metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
