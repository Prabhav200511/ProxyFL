# device.py
import threading
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config import TOTAL_ROUNDS, TIMEOUT, LR, BATCH_SIZE
from shared_logger import logger
from models import VanetIDS
from network import Receiver, send_msg

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# --- CUSTOM PYTORCH DATASET FOR VANET CSV ---
class VANETDataset(Dataset):
    def __init__(self, dataframe):
        # Use only the original 4 features for the stable Deep MLP
        features = dataframe[[
            'velocity_x',
            'velocity_y',
            'constant_offset_check',
            'total_displacement'
        ]].values
        
        targets = dataframe['attacktype'].values
        
        # Convert to PyTorch tensors
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Device:
    def __init__(self, name, port, rsu_port, device_id, total_vehicles):
        self.name = name
        self.port = port
        self.rsu_port = rsu_port
        
        # Load the stable Deep MLP
        self.model = VanetIDS().to(device)

        # Balanced weights: 1.0 for normal traffic, 2.0 for attacks
        weights = torch.tensor([1.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=torch.float32).to(device)
        self.criterion = torch.nn.CrossEntropyLoss(weight=weights)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
             
        # Load original 4-feature datasets
        df = pd.read_csv('Main_data_shuffled.csv')
        
        scaler = StandardScaler()
        feature_cols = [
            'velocity_x', 'velocity_y', 'constant_offset_check', 'total_displacement'
        ]
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        
        subset_size = len(df) // total_vehicles
        start_idx = device_id * subset_size
        if device_id == total_vehicles - 1:
            end_idx = len(df)
        else:
            end_idx = (device_id + 1) * subset_size
        device_df = df.iloc[start_idx:end_idx].copy()
        
        self.dataset = VANETDataset(device_df)
        self.dataloader = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True)
        
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
            data, target = data.to(device), target.to(device)
            self.optimizer.zero_grad()
            
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # (Optional) You can leave DP-SGD clipping/noise here if you kept it!
            
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

            print(f"[{self.name}] Training local VANET telemetry...")
            
            # Each vehicle performs exactly one local training step per federated round
            loss, acc = self.train_epoch()
            avg_loss = float(loss)
            avg_acc = float(acc)
            
            logger.log_vehicle(r, self.name, avg_loss, avg_acc)

            # Step the scheduler: Learning rate decays by 5% for the next round
            self.scheduler.step()

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