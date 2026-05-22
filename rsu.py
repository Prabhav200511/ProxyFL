from network import Receiver, send_msg
from models import average_weights

class RSU:
    def __init__(self, name, port, cluster_ports, server_port):
        self.name = name
        self.port = port
        self.cluster_ports = cluster_ports
        self.server_port = server_port
        self.round_buffers = {}
        self.receiver = Receiver(self.port, self.on_receive)

    def on_receive(self, msg):
        if msg["type"] == "LOCAL_UPDATE":
            r = msg["round"]
            
            if r not in self.round_buffers:
                self.round_buffers[r] = []
                
            self.round_buffers[r].append(msg)

            if len(self.round_buffers[r]) == len(self.cluster_ports):
                self.aggregate(r)

        elif msg["type"] == "GLOBAL_UPDATE":
            for p in self.cluster_ports:
                send_msg(("127.0.0.1", p), msg)

    def aggregate(self, r):
        data = self.round_buffers[r]
        avg_weights = average_weights([d["weights"] for d in data])

        print(f"[{self.name}] Forwarding aggregated cluster model to Server")
        msg = {
            "type": "CLUSTER_UPDATE",
            "rsu_port": self.port,
            "round": r,
            "avg_weights": avg_weights
        }

        send_msg(("127.0.0.1", self.server_port), msg)
        del self.round_buffers[r]

    def start(self):
        self.receiver.start()