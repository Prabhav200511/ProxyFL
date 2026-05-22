import threading
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from config import TOTAL_ROUNDS, TIMEOUT, LR, BATCH_SIZE
from shared_logger import logger
from models import SimpleMNIST
from network import Receiver, send_msg

class Device:
    def __init__(self, name, port, rsu_port, device_id):
        self.name = name
        self.port = port
        self.rsu_port = rsu_port
        
        self.model = SimpleMNIST()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=LR)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        
        subset_size = len(dataset) // 4
        indices = list(range(device_id * subset_size, (device_id + 1) * subset_size))
        self.dataloader = DataLoader(Subset(dataset, indices), batch_size=BATCH_SIZE, shuffle=True)
        
        self.round_event = threading.Event()
        self.receiver = Receiver(self.port, self.on_receive)

    def on_receive(self, msg):
        if msg["type"] == "GLOBAL_UPDATE":
            self.model.load_state_dict(msg["global_weights"])
            self.round_event.set()

    def train_epoch(self):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        
        for data, target in self.dataloader:
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
        return total_loss / len(self.dataloader), correct / total

    def send(self):
        for r in range(1, TOTAL_ROUNDS + 1):
            if self.name.endswith("D1"):
                print(f"\n[{'='*15} ROUND {r} {'='*15}]")

            print(f"[{self.name}] Training local dataset...")
            loss, acc = self.train_epoch()
            logger.log_vehicle(r, self.name, loss, acc)

            msg = {
                "type": "LOCAL_UPDATE",
                "sender": self.name,
                "round": r,
                "weights": self.model.state_dict()
            }
            
            send_msg(("127.0.0.1", self.rsu_port), msg)
            self.round_event.wait(timeout=TIMEOUT)
            self.round_event.clear()

        print(f"[{self.name}] Training Finished")

    def start(self):
        self.receiver.start()
        threading.Thread(target=self.send, daemon=True).start()