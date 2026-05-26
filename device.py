import threading
import copy
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from config import TOTAL_ROUNDS, TIMEOUT, LR, BATCH_SIZE
from shared_logger import logger
from models import SimpleMNIST, PrivateCNN
from network import Receiver, send_msg

class Device:
    def __init__(self, name, port, rsu_port, device_id):
        self.name = name
        self.port = port
        self.rsu_port = rsu_port
        
        # 1. HETEROGENEITY: Initialize dual models
        self.private_model = PrivateCNN()
        self.proxy_model = SimpleMNIST()
        
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # 2. Independent Optimizers
        self.optimizer_private = torch.optim.SGD(self.private_model.parameters(), lr=LR)
        self.optimizer_proxy = torch.optim.SGD(self.proxy_model.parameters(), lr=LR)
        
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
            # 3. GLOBAL SYNC: Only the proxy model is updated by the server
            self.proxy_model.load_state_dict(msg["global_weights"])
            self.round_event.set()

    def train_epoch(self):
        self.private_model.train()
        self.proxy_model.train()
        
        # 1. This accumulator must stay a float
        total_loss_proxy, correct_proxy, total = 0, 0, 0
        
        for data, target in self.dataloader:
            self.optimizer_private.zero_grad()
            self.optimizer_proxy.zero_grad()

            # --- Forward Pass ---
            out_private = self.private_model(data)
            out_proxy = self.proxy_model(data)

            # --- Supervised Learning Loss ---
            loss_ce_private = self.criterion(out_private, target)
            loss_ce_proxy = self.criterion(out_proxy, target)

            # --- Deep Mutual Learning (DML) Loss via KL Divergence ---
            prob_private = F.softmax(out_private, dim=1)
            prob_proxy = F.softmax(out_proxy, dim=1)
            
            log_prob_private = F.log_softmax(out_private, dim=1)
            log_prob_proxy = F.log_softmax(out_proxy, dim=1)

            loss_kl_private = F.kl_div(log_prob_private, prob_proxy.detach(), reduction='batchmean')
            loss_kl_proxy = F.kl_div(log_prob_proxy, prob_private.detach(), reduction='batchmean')

            # 2. FIXED: Use distinct names for the batch loss tensors
            batch_loss_private = loss_ce_private + loss_kl_private
            batch_loss_proxy = loss_ce_proxy + loss_kl_proxy

            # 3. Backward Pass on the batch tensors
            batch_loss_private.backward()
            batch_loss_proxy.backward()

            self.optimizer_private.step()
            self.optimizer_proxy.step()
            
            # --- Metrics (Tracking Proxy) ---
            # 4. Safely accumulate using .item() to keep it a pure float
            total_loss_proxy += loss_ce_proxy.item()
            pred_proxy = out_proxy.argmax(dim=1, keepdim=True)
            correct_proxy += pred_proxy.eq(target.view_as(pred_proxy)).sum().item()
            total += target.size(0)
            
        return total_loss_proxy / len(self.dataloader), correct_proxy / total

    def apply_differential_privacy(self, weights, noise_multiplier=0.01):
        """Adds Gaussian noise to weights to prevent Model Inversion attacks."""
        noisy_weights = copy.deepcopy(weights)
        for key in noisy_weights.keys():
            # Apply noise scaled by the noise_multiplier
            noise = torch.randn_like(noisy_weights[key], dtype=torch.float) * noise_multiplier
            noisy_weights[key] += noise
        return noisy_weights

    def send(self):
        for r in range(1, TOTAL_ROUNDS + 1):
            if self.name.endswith("D1"):
                print(f"\n[{'='*15} ROUND {r} {'='*15}]")

            print(f"[{self.name}] Training local models via DML...")
            loss, acc = self.train_epoch()
            logger.log_vehicle(r, self.name, loss, acc)

            # 4. DIFFERENTIAL PRIVACY: Secure the proxy weights
            raw_proxy_weights = self.proxy_model.state_dict()
            secured_proxy_weights = self.apply_differential_privacy(raw_proxy_weights, noise_multiplier=0.01)

            # 5. TRANSMISSION: Send ONLY the secured proxy to the RSU
            msg = {
                "type": "LOCAL_UPDATE",
                "sender": self.name,
                "round": r,
                "weights": secured_proxy_weights
            }
            
            send_msg(("127.0.0.1", self.rsu_port), msg)
            
            print(f"[{self.name}] Secured Proxy sent to RSU. Waiting for global update...")
            self.round_event.wait(timeout=TIMEOUT)
            self.round_event.clear()

        print(f"\n[{self.name}] TRAINING FINISHED")

    def start(self):
        self.receiver.start()
        threading.Thread(target=self.send, daemon=True).start()