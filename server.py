# server.py — Central Server for ProxyFL
#
# Receives cluster-aggregated proxy weights from RSUs, applies
# JSD-weighted global aggregation, evaluates the global proxy model
# on held-out attack test data, and broadcasts the result back.

import threading
import os

import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader

from config import TOTAL_ROUNDS, DEVICE
from data_utils import VANETDataset, get_vanet_scaler
from shared_logger import logger
from network import Receiver, send_msg
from torchvision import datasets, transforms
from models import jsd_weighted_average, ProxyModel, MNISTProxyModel


training_done_event = threading.Event()

SERVER_ROUND_TIMEOUT = 12  # seconds — aggregate whatever RSUs reported


class Server:
    """Central aggregation server.

    Args:
        port:          TCP port to listen on.
        expected_rsus: Number of RSU cluster updates expected per round.
    """

    def __init__(self, port, expected_rsus, dataset_type="mnist", total_rounds=TOTAL_ROUNDS):
        self.port = port
        self.expected_rsus = expected_rsus
        self.dataset_type = dataset_type
        self.total_rounds = total_rounds
        self.round_buffers = {}
        self.completed_rounds = set()
        self.rsu_ports = []
        self._round_timers = {}
        self._lock = threading.Lock()  # protects round_buffers, completed_rounds, rsu_ports, _round_timers
        self.receiver = Receiver(self.port, self.on_receive)

        torch.manual_seed(42)
        if self.dataset_type == "mnist":
            self.model = MNISTProxyModel().to(DEVICE)
            print("[SERVER] Loading MNIST test dataset...")
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
            self.test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        else:
            self.model = ProxyModel().to(DEVICE)
            print("[SERVER] Loading attack test datasets...")
            test_files = ['attack1_test.csv', 'attack2_test.csv', 'attack3_test.csv', 'attack4_test.csv', 'attack5_test.csv']
            dfs = [pd.read_csv(f) for f in test_files if os.path.exists(f)]
            if dfs:
                test_df = pd.concat(dfs, ignore_index=True)
                scaler = get_vanet_scaler()
                test_df[VANETDataset.FEATURE_COLS] = scaler.transform(test_df[VANETDataset.FEATURE_COLS])
                self.test_dataset = VANETDataset(test_df)
                self.test_loader = DataLoader(self.test_dataset, batch_size=1000, shuffle=False)
            else:
                self.test_loader = None

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------
    def on_receive(self, msg):
        if msg["type"] == "CLUSTER_UPDATE":
            r = msg["round"]
            should_aggregate = False

            with self._lock:
                if r in self.completed_rounds:
                    return

                if msg["rsu_port"] not in self.rsu_ports:
                    self.rsu_ports.append(msg["rsu_port"])
                if r not in self.round_buffers:
                    self.round_buffers[r] = []
                    # Start a deadline timer for this round
                    timer = threading.Timer(
                        SERVER_ROUND_TIMEOUT, self._force_aggregate, args=[r])
                    timer.daemon = True
                    timer.start()
                    self._round_timers[r] = timer

                self.round_buffers[r].append(msg)

                if len(self.round_buffers[r]) == self.expected_rsus:
                    self._cancel_timer_locked(r)
                    should_aggregate = True

            if should_aggregate:
                self.aggregate(r)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate_global_model(self, weights):
        if not self.test_loader:
            return 0.0, 0.0, 0.0

        self.model.load_state_dict(weights)
        self.model.eval()

        all_preds, all_targets = [], []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = self.model(data)
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted',
                      zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted',
                              zero_division=0)
        return acc, f1, recall

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def aggregate(self, r):
        with self._lock:
            if r in self.completed_rounds:
                return
            self.completed_rounds.add(r)
            self._cancel_timer_locked(r)
            data = self.round_buffers.pop(r, None)
            rsu_ports_snapshot = list(self.rsu_ports)

        if not data:
            return

        cluster_weights = [d["avg_weights"] for d in data]

        # JSD-weighted average
        global_weights, divergences = jsd_weighted_average(
            cluster_weights, self.model.state_dict(), alpha=2.0)

        for i, div in enumerate(divergences):
            logger.log_jsd(r, f"Cluster_{i + 1}", div)

        # Evaluate
        acc, f1, recall = self.evaluate_global_model(global_weights)
        logger.log_global(r, acc)

        print(f"\n[SERVER] --- ROUND {r} GLOBAL PROXY METRICS ---")
        print(f"         Test Accuracy : {round(acc * 100, 2)}%")
        print(f"         F1-Score      : {round(f1, 4)}")
        print(f"         Recall        : {round(recall, 4)}\n")

        # Broadcast global proxy to all RSUs
        msg = {
            "type": "GLOBAL_UPDATE",
            "round": r,
            "global_weights": global_weights,
        }
        for p in rsu_ports_snapshot:
            send_msg(("127.0.0.1", p), msg)

        if r >= self.total_rounds:
            training_done_event.set()

    def _force_aggregate(self, r):
        with self._lock:
            if r in self.completed_rounds:
                return
            has_data = r in self.round_buffers and len(self.round_buffers[r]) > 0
            if has_data:
                n = len(self.round_buffers[r])
                print(f"[SERVER] [!] Timeout! Aggregating {n}/"
                      f"{self.expected_rsus} RSUs for round {r}")

        if has_data:
            self.aggregate(r)

    def _cancel_timer_locked(self, r):
        """Cancel a round timer. Must be called with self._lock held."""
        timer = self._round_timers.pop(r, None)
        if timer:
            timer.cancel()

    # ------------------------------------------------------------------
    # Startup / Shutdown
    # ------------------------------------------------------------------
    def start(self):
        print(f"[SERVER] Listening on port {self.port} | device={DEVICE}")
        self.receiver.start()

    def shutdown(self):
        """Close the listening socket so the port is freed."""
        try:
            self.receiver.sock.close()
        except Exception:
            pass
