import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from shared_logger import logger
from network import Receiver, send_msg
from models import average_weights, SimpleMNIST

class Server:
    def __init__(self, port, expected_rsus):
        self.port = port
        self.expected_rsus = expected_rsus
        self.round_buffers = {}
        self.rsu_ports = []
        self.receiver = Receiver(self.port, self.on_receive)
        
        self.model = SimpleMNIST()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1000, shuffle=False)

    def on_receive(self, msg):
        if msg["type"] == "CLUSTER_UPDATE":
            r = msg["round"]
            
            if msg["rsu_port"] not in self.rsu_ports:
                self.rsu_ports.append(msg["rsu_port"])

            if r not in self.round_buffers:
                self.round_buffers[r] = []
                
            self.round_buffers[r].append(msg)

            if len(self.round_buffers[r]) == self.expected_rsus:
                self.aggregate(r)

    def evaluate_global_model(self, weights):
        self.model.load_state_dict(weights)
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        return correct / len(self.test_loader.dataset)

    def aggregate(self, r):
        data = self.round_buffers[r]
        global_weights = average_weights([d["avg_weights"] for d in data])

        acc = self.evaluate_global_model(global_weights)
        logger.log_global(r, acc)

        print(f"[SERVER] Aggregated Global Model - Test Accuracy: {round(acc * 100, 2)}%")
        
        msg = {
            "type": "GLOBAL_UPDATE",
            "round": r,
            "global_weights": global_weights
        }

        for p in self.rsu_ports:
            send_msg(("127.0.0.1", p), msg)
            
        del self.round_buffers[r]

    def start(self):
        print(f"[SERVER] Listening on {self.port}")
        self.receiver.start()