import socket
import threading
import time
import random
import json

TOTAL_ROUNDS = 5


class Server:
    def __init__(self, port):
        self.port = port

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.sock.bind(("127.0.0.1", self.port))

        self.buffer = []
        self.head_addresses = []

    def start(self):

        print(f"\nSERVER running on port {self.port}")

        current_round = 1

        while current_round <= TOTAL_ROUNDS:

            data, addr = self.sock.recvfrom(1024)

            message = json.loads(data.decode())

            print(f"\nSERVER received from HEAD {addr[1]}: {message}")

            if message["type"] == "CLUSTER_UPDATE":

                self.buffer.append(message)

                if addr not in self.head_addresses:
                    self.head_addresses.append(addr)

                if len(self.buffer) == 2:

                    print(f"\nSERVER aggregating global model for ROUND {current_round}")

                    global_gradients = [

                        round(
                            sum(d["avg_gradients"][i] for d in self.buffer)
                            / len(self.buffer),
                            2
                        )

                        for i in range(2)
                    ]

                    global_model = {
                        "type": "GLOBAL_UPDATE",
                        "round": current_round,
                        "global_gradients": global_gradients
                    }

                    print(f"[SERVER] global model: {global_model}")

                    for head in self.head_addresses:

                        self.sock.sendto(
                            json.dumps(global_model).encode(),
                            head
                        )

                    self.buffer = []

                    current_round += 1

        print("\nSERVER TRAINING COMPLETE")



class Device:

    def __init__(
        self,
        name,
        port,
        cluster_ports,
        is_head=False,
        head_port=None,
        server_port=None
    ):

        self.name = name
        self.port = port
        self.cluster_ports = cluster_ports
        self.is_head = is_head
        self.head_port = head_port
        self.server_port = server_port
        self.weight = random.uniform(0, 1)
        self.bias = random.uniform(0, 1)            

        self.buffer = []

        self.round_event = threading.Event()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", self.port))

    def receive(self):

        while True:

            data, addr = self.sock.recvfrom(1024)

            try:
                message = json.loads(data.decode())

            except:
                continue

            print(f"\n{self.name} received from {addr[1]}: {message}")

            if message["type"] == "GLOBAL_UPDATE":

                if self.is_head:

                    print(f"{self.name} forwarding global model to devices")

                    for port in self.cluster_ports:

                        if port != self.port:

                            self.sock.sendto(
                                json.dumps(message).encode(),
                                ("127.0.0.1", port)
                            )

                else:

                    self.weight = message["global_gradients"][0]
                    self.bias = message["global_gradients"][1]

                    print(
                        f"{self.name} updated weight={self.weight}, bias={self.bias}"
                    )

                    self.round_event.set()

            if self.is_head and message["type"] == "LOCAL_UPDATE":

                self.buffer.append(message)

                expected = len(self.cluster_ports) - 1

                if len(self.buffer) == expected:

                    print(f"\n{self.name} aggregating cluster model")

                    avg_gradients = [

                        round(
                            sum(d["gradients"][i] for d in self.buffer)
                            / len(self.buffer),
                            2
                        )

                        for i in range(2)
                    ]

                    avg_loss = round(

                        sum(d["loss"] for d in self.buffer)
                        / len(self.buffer),

                        3
                    )

                    aggregated = {
                        "type": "CLUSTER_UPDATE",
                        "head": self.name,
                        "avg_gradients": avg_gradients,
                        "avg_loss": avg_loss
                    }

                    print(f"{self.name} cluster model: {aggregated}")

                    self.sock.sendto(
                        json.dumps(aggregated).encode(),
                        ("127.0.0.1", self.server_port)
                    )

                    self.buffer = []


    def send(self):

        if self.is_head:
            return

        for round_num in range(1, TOTAL_ROUNDS + 1):

            time.sleep(random.randint(1, 3))

            x = random.randint(1, 10)

            y_true = 2 * x + 1

            y_pred = self.weight * x + self.bias

            loss = (y_true - y_pred) ** 2

            grad_w = -2 * x * (y_true - y_pred)
            grad_b = -2 * (y_true - y_pred)

            data = {
                "type": "LOCAL_UPDATE",
                "sender": self.name,

                "gradients": [
                    round(grad_w, 2),
                    round(grad_b, 2)
                ],

                "loss": round(loss, 3)
            }

            print(f"{self.name} local update: {data}")

            self.sock.sendto(
                json.dumps(data).encode(),
                ("127.0.0.1", self.head_port)
            )

            print(f"{self.name} waiting for global update...")

            self.round_event.wait()
            self.round_event.clear()

        print(f"\n{self.name} TRAINING FINISHED")


    def start(self):

        threading.Thread(
            target=self.receive,
            daemon=True
        ).start()

        threading.Thread(
            target=self.send,
            daemon=True
        ).start()



def main():

    server_port = 9000
    
    cluster1_ports = [5001, 5002, 5003]

    c1_head = 5003

    cluster2_ports = [6001, 6002, 6003]

    c2_head = 6003

    d1 = Device(
        "C1_D1",
        5001,
        cluster1_ports,
        False,
        c1_head,
        server_port
    )

    d2 = Device(
        "C1_D2",
        5002,
        cluster1_ports,
        False,
        c1_head,
        server_port
    )

    d3 = Device(
        "C1_HEAD",
        5003,
        cluster1_ports,
        True,
        None,
        server_port
    )

    d4 = Device(
        "C2_D1",
        6001,
        cluster2_ports,
        False,
        c2_head,
        server_port
    )

    d5 = Device(
        "C2_D2",
        6002,
        cluster2_ports,
        False,
        c2_head,
        server_port
    )

    d6 = Device(
        "C2_HEAD",
        6003,
        cluster2_ports,
        True,
        None,
        server_port
    )


    server = Server(server_port)

    for d in [d1, d2, d3, d4, d5, d6]:

        d.start()

    threading.Thread(
        target=server.start,
        daemon=True
    ).start()

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()