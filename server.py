import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader, Dataset
import numpy as np
from shared_logger import logger
from network import Receiver, send_msg
from models import jsd_weighted_average, VanetIDS
from config import TOTAL_ROUNDS
import threading

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
training_done_event = threading.Event()

# Reusing the Dataset class for the Server's testing phase
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
        
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class Server:
    def __init__(self, port, expected_rsus):
        self.port = port
        self.expected_rsus = expected_rsus
        self.round_buffers = {}
        self.rsu_ports = []
        self.receiver = Receiver(self.port, self.on_receive)
        
        self.model = VanetIDS().to(device)
        # --- SERVER TESTING DATA PREPARATION ---
        print("[SERVER] Loading and engineering attack test datasets...")
        test_files = ['attack1_test.csv', 'attack2_test.csv', 'attack3_test.csv', 'attack4_test.csv', 'attack5_test.csv']
        
        dfs = []
        for file in test_files:
            try:
                dfs.append(pd.read_csv(file))
            except FileNotFoundError:
                print(f"[WARNING] Could not find {file}")
                
        if dfs:
            test_df = pd.concat(dfs, ignore_index=True)
            train_df = pd.read_csv('Main_data_shuffled.csv')
            
            # Scale only the original 4 features
            feature_cols = [
                'velocity_x', 'velocity_y', 'constant_offset_check', 'total_displacement'
            ]
            scaler = StandardScaler()
            scaler.fit(train_df[feature_cols])
            test_df[feature_cols] = scaler.transform(test_df[feature_cols])
            
            self.test_dataset = VANETDataset(test_df)
            self.test_loader = DataLoader(self.test_dataset, batch_size=1000, shuffle=False)
        else:
            self.test_loader = None

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
        if not self.test_loader:
            return 0.0, 0.0, 0.0
            
        self.model.load_state_dict(weights)
        self.model.eval()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                
                # Move back to CPU for Scikit-Learn metrics
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
        # Calculate Advanced Cybersecurity Metrics (Weighted handles the class imbalance)
        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        
        return acc, f1, recall

    def aggregate(self, r):
        data = self.round_buffers[r]
        cluster_weights = [d["avg_weights"] for d in data]
        
        # Apply JS Divergence
        global_weights, divergences = jsd_weighted_average(cluster_weights, self.model.state_dict(), alpha=2.0)
        
        for i, div in enumerate(divergences):
            logger.log_jsd(r, f"Cluster_{i+1}", div)

        # Advanced Evaluation
        acc, f1, recall = self.evaluate_global_model(global_weights)
        logger.log_global(r, acc) # Still logging basic accuracy to the old table
        
        print(f"\n[SERVER] --- ROUND {r} GLOBAL METRICS ---")
        print(f"         Test Accuracy : {round(acc * 100, 2)}%")
        print(f"         F1-Score      : {round(f1, 4)}")
        print(f"         Recall        : {round(recall, 4)}\n")
        
        msg = {"type": "GLOBAL_UPDATE", "round": r, "global_weights": global_weights}
        for p in self.rsu_ports:
            send_msg(("127.0.0.1", p), msg)
            
        del self.round_buffers[r]
        
        if r >= TOTAL_ROUNDS:
            global training_done_event
            training_done_event.set()

    def start(self):
        print(f"[SERVER] Listening on {self.port} utilizing {device}")
        self.receiver.start()