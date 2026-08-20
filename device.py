# device.py — ProxyFL Vehicle Node
#
# Each vehicle maintains TWO models:
#   • Private model  — trained locally, never leaves the device
#   • Proxy model    — shared with RSU / Server for federated aggregation
#
# Training uses Deep Mutual Learning (DML, Eq. 4–5) so both models teach
# each other.  DP-SGD (vectorized per-sample gradient clipping + Gaussian noise,
# Eq. 6–7) is applied ONLY to the proxy model, providing (ε,δ)-differential
# privacy for all shared weights.  Privacy expenditure is tracked via an
# RDP accountant and logged per round.

import time
import threading
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader

from config import (
    TOTAL_ROUNDS, TIMEOUT, BATCH_SIZE, LOCAL_EPOCHS,
    DP_CLIP_NORM, DP_NOISE_MULTIPLIER, DP_DELTA, DP_MAX_EPSILON,
    DML_ALPHA, DML_BETA, DML_TEMPERATURE,
    V2RSU_RANGE, DEVICE, TRAINING_SEMAPHORE,
    SECURITY_ENABLED, V2V_ENABLED, V2V_COLLECT_TIMEOUT,
)
from data_utils import VANETDataset, get_vanet_scaler
from shared_logger import logger
from models import (VanetIDS, ProxyModel,
                    MNISTPrivateModel, MNISTProxyModel,
                    dml_loss, average_weights)
from network import Receiver, send_msg
from privacy import RDPAccountant
from torchvision import datasets, transforms
from crypto_protocol import (
    Authority, CertificatelessSigner, build_envelope, encrypt_payload,
    message_aad, verify_envelope,
)
from model_codec import serialize_weights, deserialize_weights
from metrics import Timer, metrics_tracker


class Device:
    """A VANET vehicle participating in ProxyFL.

    Args:
        name:                Human-readable identifier (e.g. "C1_D1").
        port:                TCP port this vehicle listens on.
        rsu_port:            TCP port of the assigned RSU.
        rsu_name:            Name of the assigned RSU (for spatial lookups).
        device_id:           Global index used to partition the training data.
        total_vehicles:      Total vehicle count (for even data splits).
        topology:            Shared VanetTopology object (thread-safe).
        peer_directory:      Dict mapping vehicle_name → port.
        dataset_type:        "mnist" or "vanet".
        private_model_class: Class to instantiate for the private model.
                             Different devices may use different classes.
        total_rounds:        Number of communication rounds to participate in.
        security_authority:  Optional bootstrap Authority (TA/KGC) instance.
        security_identity:   Optional pre-registered CertificatelessSigner.
    """

    def __init__(self, name, port, rsu_port, rsu_name,
                 device_id, total_vehicles, topology, peer_directory=None,
                 dataset_type="mnist", private_model_class=None,
                 total_rounds=TOTAL_ROUNDS, security_authority=None,
                 security_identity=None):
        self.name = name
        self.port = port
        self.rsu_port = rsu_port
        self.rsu_name = rsu_name
        self.topology = topology
        self.peer_directory = peer_directory or {}
        self.dataset_type = dataset_type
        self.total_rounds = total_rounds
        self.current_round = 0
        self.budget_exhausted = False
        self.security_authority = security_authority
        self.signer = security_identity

        if SECURITY_ENABLED and self.signer is None and self.security_authority is not None:
            with Timer(self.name, 0, "key_generation"):
                self.signer = self.security_authority.register(self.name)

        # ---- Model Selection ----
        torch.manual_seed(42)  # Deterministic proxy init across all devices
        if self.dataset_type == "mnist":
            self.proxy_model = MNISTProxyModel().to(DEVICE)
            torch.seed()  # Random seed for private model heterogeneity
            priv_cls = private_model_class or MNISTPrivateModel
            self.private_model = priv_cls().to(DEVICE)
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.proxy_model = ProxyModel().to(DEVICE)
            torch.seed()
            priv_cls = private_model_class or VanetIDS
            self.private_model = priv_cls().to(DEVICE)
            weights = torch.tensor(
                [1.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=torch.float32
            ).to(DEVICE)
            self.criterion = torch.nn.CrossEntropyLoss(weight=weights)

        priv_params = sum(p.numel() for p in self.private_model.parameters())
        proxy_params = sum(p.numel() for p in self.proxy_model.parameters())
        print(f"[{self.name}] Private: {priv_cls.__name__} "
              f"({priv_params} params) | Proxy: {proxy_params} params")

        self.private_optimizer = torch.optim.Adam(
            self.private_model.parameters(), lr=0.001)
        self.proxy_optimizer = torch.optim.Adam(
            self.proxy_model.parameters(), lr=0.001)

        self.private_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.private_optimizer, gamma=0.95)
        self.proxy_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.proxy_optimizer, gamma=0.95)

        self.dp_clip_norm = DP_CLIP_NORM
        self.dp_noise_multiplier = DP_NOISE_MULTIPLIER

        # ---- Data Partitioning (with 80/20 train/test split) ----
        if self.dataset_type == "mnist":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            datasets.MNIST.urls = [
                "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
                "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
                "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
                "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
            ]
            full_mnist = datasets.MNIST(
                './data', train=True, download=True, transform=transform)

            subset_size = len(full_mnist) // total_vehicles
            start_idx = device_id * subset_size
            end_idx = (len(full_mnist) if device_id == total_vehicles - 1
                       else (device_id + 1) * subset_size)
            subset_indices = list(range(start_idx, end_idx))
            
            # Cap local sample size to realistic vehicle edge limit (max 1000 samples)
            if len(subset_indices) > 1000:
                subset_indices = subset_indices[:1000]

            # 80/20 train/test split for local evaluation
            n_test = max(1, int(len(subset_indices) * 0.2))
            train_indices = subset_indices[:-n_test]
            test_indices = subset_indices[-n_test:]

            self.dataset = torch.utils.data.Subset(full_mnist, train_indices)
            self.dataloader = DataLoader(
                self.dataset, batch_size=BATCH_SIZE, shuffle=True)
            self.test_dataset = torch.utils.data.Subset(
                full_mnist, test_indices)
            self.test_loader = DataLoader(
                self.test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        else:
            df = pd.read_csv('Main_data_shuffled.csv')
            subset_size = len(df) // total_vehicles
            start_idx = device_id * subset_size
            end_idx = (len(df) if device_id == total_vehicles - 1
                       else (device_id + 1) * subset_size)
            device_df = df.iloc[start_idx:end_idx].copy()

            scaler = get_vanet_scaler()
            device_df[VANETDataset.FEATURE_COLS] = scaler.transform(
                device_df[VANETDataset.FEATURE_COLS])

            # 80/20 train/test split for local evaluation
            n_test = max(1, int(len(device_df) * 0.2))
            train_df = device_df.iloc[:-n_test]
            test_df = device_df.iloc[-n_test:]

            self.dataset = VANETDataset(train_df)
            self.dataloader = DataLoader(
                self.dataset, batch_size=BATCH_SIZE, shuffle=True)
            self.test_dataset = VANETDataset(test_df)
            self.test_loader = DataLoader(
                self.test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # ---- Privacy Accountant (RDP-based) ----
        dataset_size = len(self.dataset)
        sample_rate = (min(BATCH_SIZE / dataset_size, 1.0)
                       if dataset_size > 0 else 0.0)
        self.privacy_accountant = RDPAccountant(
            noise_multiplier=DP_NOISE_MULTIPLIER,
            sample_rate=sample_rate,
            delta=DP_DELTA,
        )

        # Pre-compile vectorized per-sample gradient function
        def single_sample_loss(p, b, x, y, soft):
            out = torch.func.functional_call(self.proxy_model, (p, b), x.unsqueeze(0))
            ce = F.cross_entropy(out, y.unsqueeze(0))
            kl = dml_loss(out, soft.unsqueeze(0), DML_TEMPERATURE)
            return (1 - DML_BETA) * ce + DML_BETA * kl

        grad_fn = torch.func.grad(single_sample_loss)
        self.vmap_grad_fn = torch.func.vmap(grad_fn, in_dims=(None, None, 0, 0, 0))

        # ---- Synchronization ----
        self.round_event = threading.Event()
        self.proxy_lock = threading.Lock()
        self._peer_lock = threading.Lock()
        self._peer_buffers = {}  # round → {peer_name: weights}
        self._request_sent_at = {}  # round → perf_counter when LOCAL_UPDATE sent
        self.receiver = Receiver(self.port, self.on_receive, node_name=self.name)

    # ------------------------------------------------------------------
    # Message handling (Device -> RSU -> Server hierarchy + V2V)
    # ------------------------------------------------------------------
    def on_receive(self, msg):
        msg_type = msg.get("type") if isinstance(msg, dict) else None
        if msg_type == "GLOBAL_UPDATE":
            msg_round = msg.get("round", None)
            # Accept global update if matching the current round or if unnumbered
            if msg_round is None or msg_round == self.current_round:
                # Action-to-response latency: LOCAL_UPDATE send → GLOBAL_UPDATE receive
                if isinstance(msg_round, int) and msg_round in self._request_sent_at:
                    metrics_tracker.record_duration(
                        self.name, msg_round, "action_to_response",
                        time.perf_counter() - self._request_sent_at.pop(msg_round),
                    )
                raw = msg.get("global_weights")
                if isinstance(raw, bytes):
                    weights = deserialize_weights(raw)
                else:
                    weights = raw
                if weights is not None:
                    with self.proxy_lock:
                        self.proxy_model.load_state_dict(weights)
                self.round_event.set()

        elif msg_type == "PEER_UPDATE":
            r = msg.get("round")
            sender = msg.get("sender")
            if not isinstance(r, int) or not sender or sender == self.name:
                return
            weights = None
            if SECURITY_ENABLED and self.security_authority is not None and self.signer is not None and "sig" in msg:
                result = verify_envelope(
                    self.security_authority, self.signer, msg, "PEER_UPDATE")
                if result is None:
                    print(f"[{self.name}] [SECURITY] Dropped PEER_UPDATE from {sender}")
                    return
                payload, _ = result
                try:
                    weights = deserialize_weights(payload)
                except Exception:
                    return
            else:
                raw = msg.get("weights")
                if isinstance(raw, bytes):
                    try:
                        weights = deserialize_weights(raw)
                    except Exception:
                        return
                elif isinstance(raw, dict):
                    weights = raw
            if weights is None:
                return
            with self._peer_lock:
                self._peer_buffers.setdefault(r, {})[sender] = weights

    # ------------------------------------------------------------------
    # DML Training with DP-SGD on proxy
    # ------------------------------------------------------------------
    def train_epoch(self):
        """Train both models on one epoch using Deep Mutual Learning.

        Private model: L = (1 − α)·CE + α·KL  (standard optimizer, no DP)
        Proxy model:   L = (1 − β)·CE + β·KL  (vectorized per-sample DP-SGD)
        Throttled via TRAINING_SEMAPHORE to prevent GPU/OpenMP collision.
        """
        with TRAINING_SEMAPHORE:
            self.private_model.train()
            if not self.budget_exhausted:
                self.proxy_model.train()

            priv_loss_sum, priv_correct, proxy_loss_sum, total = 0, 0, 0, 0

            for data, target in self.dataloader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                batch_size = data.size(0)

                # --- Soft targets from each model (detached) ---
                with torch.no_grad():
                    with self.proxy_lock:
                        proxy_soft = F.softmax(
                            self.proxy_model(data) / DML_TEMPERATURE, dim=1)
                    private_soft = F.softmax(
                        self.private_model(data) / DML_TEMPERATURE, dim=1)

                # --- Update private model (no DP, Eq. 4) ---
                private_out = self.private_model(data)
                private_ce = self.criterion(private_out, target)
                private_kl = dml_loss(private_out, proxy_soft, DML_TEMPERATURE)
                private_total = ((1 - DML_ALPHA) * private_ce
                                 + DML_ALPHA * private_kl)

                self.private_optimizer.zero_grad()
                private_total.backward()
                self.private_optimizer.step()

                # --- Update proxy model (Eq. 5, with vectorized DP-SGD if enabled) ---
                if not self.budget_exhausted:
                    with self.proxy_lock:
                        if self.dp_noise_multiplier > 0:
                            self._dp_sgd_step(data, target, private_soft)
                            self.privacy_accountant.step(1)
                        else:
                            # Non-DP batch baseline
                            proxy_out = self.proxy_model(data)
                            proxy_ce = self.criterion(proxy_out, target)
                            proxy_kl = dml_loss(
                                proxy_out, private_soft, DML_TEMPERATURE)
                            proxy_total = ((1 - DML_BETA) * proxy_ce
                                           + DML_BETA * proxy_kl)
                            self.proxy_optimizer.zero_grad()
                            proxy_total.backward()
                            self.proxy_optimizer.step()

                # Track metrics
                priv_loss_sum += private_total.item()
                pred = private_out.argmax(dim=1, keepdim=True)
                priv_correct += pred.eq(target.view_as(pred)).sum().item()
                total += batch_size

        avg_priv_loss = priv_loss_sum / len(self.dataloader)
        avg_priv_acc = priv_correct / total if total > 0 else 0.0
        return avg_priv_loss, avg_priv_acc, 0.0

    def _dp_sgd_step(self, data, target, private_soft):
        """Vectorized DP-SGD on proxy model (Eq. 6–7).

        Computes per-sample gradients in parallel via torch.func.vmap, clips
        each to DP_CLIP_NORM, sums across the batch, and adds Gaussian noise.
        """
        params = dict(self.proxy_model.named_parameters())
        buffers = dict(self.proxy_model.named_buffers())
        batch_size = data.size(0)

        per_sample_grads = self.vmap_grad_fn(
            params, buffers, data, target, private_soft)

        # Compute per-sample gradient L2 norms across all parameters
        flat_grads = []
        for name, g in per_sample_grads.items():
            flat_grads.append(g.reshape(batch_size, -1))
        flat_all = torch.cat(flat_grads, dim=1)
        per_sample_norms = torch.norm(flat_all, p=2, dim=1)

        # Per-sample clipping factor (Eq. 6): min(1, C / ||g_i||_2)
        clip_factors = torch.clamp(self.dp_clip_norm / (per_sample_norms + 1e-6), max=1.0)

        # Scale gradient noise to match average batch update
        noise_std = (self.dp_noise_multiplier * self.dp_clip_norm) / batch_size

        self.proxy_optimizer.zero_grad()
        for name, param in self.proxy_model.named_parameters():
            g = per_sample_grads[name]
            shape = [batch_size] + [1] * (g.dim() - 1)
            clipped_g = g * clip_factors.view(shape)
            avg_g = clipped_g.sum(dim=0) / batch_size
            noise = torch.randn_like(avg_g) * noise_std
            param.grad = avg_g + noise

        self.proxy_optimizer.step()

    # ------------------------------------------------------------------
    # Evaluation (local held-out test split)
    # ------------------------------------------------------------------
    def evaluate_private_model(self):
        """Evaluate private model accuracy on local held-out test split."""
        self.private_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = self.private_model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        return correct / total if total > 0 else 0.0

    def evaluate_proxy_model(self):
        """Evaluate proxy model accuracy on local held-out test split."""
        correct, total = 0, 0
        with torch.no_grad():
            with self.proxy_lock:
                self.proxy_model.eval()
                for data, target in self.test_loader:
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    output = self.proxy_model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
        return correct / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # V2V proxy sharing (Eq. 6)
    # ------------------------------------------------------------------
    def _v2v_share_and_aggregate(self, round_num, local_weights):
        """Broadcast proxy to in-range peers and average received neighbor proxies."""
        if not V2V_ENABLED or not self.peer_directory:
            return local_weights

        all_peers = list(self.peer_directory.keys())
        neighbors = self.topology.get_v2v_neighbors(self.name, all_peers)
        if not neighbors:
            return local_weights

        with self._peer_lock:
            self._peer_buffers.pop(round_num, None)

        # Send PEER_UPDATE to each in-range neighbor
        for peer in neighbors:
            peer_port = self.peer_directory.get(peer)
            if peer_port is None:
                continue
            if SECURITY_ENABLED and self.signer is not None and self.security_authority is not None:
                raw_payload = serialize_weights(local_weights)
                sig = self.signer.sign(raw_payload)
                aad = message_aad("PEER_UPDATE", self.name, peer, round_num)
                peer_info = self.security_authority.public_info(peer)
                shared_secret = self.signer.shared_secret_for(peer_info)
                ciphertext, nonce, tag = encrypt_payload(shared_secret, raw_payload, aad)
                msg = build_envelope(
                    "PEER_UPDATE", self.signer, peer, round_num,
                    sig, ciphertext, nonce, tag,
                )
            else:
                msg = {
                    "type": "PEER_UPDATE",
                    "sender": self.name,
                    "recipient": peer,
                    "round": round_num,
                    "weights": local_weights,
                }
            send_msg(("127.0.0.1", peer_port), msg,
                     sender_name=self.name, round_num=round_num)

        # Collect neighbor proxies for a short window
        deadline = time.time() + V2V_COLLECT_TIMEOUT
        while time.time() < deadline:
            with self._peer_lock:
                received = dict(self._peer_buffers.get(round_num, {}))
            if len(received) >= len(neighbors):
                break
            time.sleep(0.05)

        with self._peer_lock:
            received = dict(self._peer_buffers.pop(round_num, {}))

        if not received:
            print(f"[{self.name}] V2V: no peer proxies received "
                  f"(neighbors={neighbors})")
            return local_weights

        pooled = [local_weights] + list(received.values())
        averaged = average_weights(pooled)
        print(f"[{self.name}] V2V: averaged local + {len(received)} peer "
              f"proxies (neighbors in range: {neighbors})")
        return averaged

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def send(self):
        for r in range(1, self.total_rounds + 1):
          round_t0 = time.perf_counter()
          try:
            self.current_round = r
            self.round_event.clear()

            if self.name.endswith("D1"):
                print(f"\n[{'=' * 15} ROUND {r} {'=' * 15}]")

            # 1. Train with DML
            status = ("(DML + DP-SGD on proxy)"
                      if not self.budget_exhausted
                      else "(private only -- budget exhausted)")
            print(f"[{self.name}] Training {status}...")
            total_priv_loss, total_priv_acc = 0, 0
            with Timer(self.name, r, "training"):
                for _ in range(LOCAL_EPOCHS):
                    priv_loss, priv_acc, proxy_loss = self.train_epoch()
                    total_priv_loss += priv_loss
                    total_priv_acc += priv_acc

            avg_loss = total_priv_loss / max(LOCAL_EPOCHS, 1)
            avg_acc = total_priv_acc / max(LOCAL_EPOCHS, 1)

            # Log training metrics
            logger.log_vehicle(r, self.name, avg_loss, avg_acc)

            # 2. Evaluate both models on local test split
            priv_test_acc = self.evaluate_private_model()
            proxy_test_acc = self.evaluate_proxy_model()
            logger.log_private_accuracy(r, self.name, priv_test_acc)

            print(f"    -> [PRIVATE] Test Acc: {priv_test_acc * 100:.1f}% | "
                  f"[PROXY] Test Acc: {proxy_test_acc * 100:.1f}% | "
                  f"Train Loss: {avg_loss:.4f}")

            # 3. Log privacy spending
            if self.dp_noise_multiplier > 0:
                eps = self.privacy_accountant.get_epsilon()
                logger.log_privacy(
                    r, self.name, eps, self.privacy_accountant.delta)
                print(f"    -> [PRIVACY] eps = {eps:.4f}, "
                      f"delta = {self.privacy_accountant.delta:.1e}")

                # Budget drop-out check
                if (DP_MAX_EPSILON is not None
                        and not self.budget_exhausted
                        and eps >= DP_MAX_EPSILON):
                    self.budget_exhausted = True
                    print(f"    -> [PRIVACY] Budget exhausted! "
                          f"eps={eps:.2f} >= {DP_MAX_EPSILON}. "
                          f"Stopping proxy training & sharing.")

            # 4. Step LR schedulers
            self.private_scheduler.step()
            if not self.budget_exhausted:
                self.proxy_scheduler.step()

            # 5. Check V2RSU range -> send proxy to RSU & wait for global round sync
            dist = self.topology.get_distance_to_rsu(
                self.name, self.rsu_name)
            can_reach = self.topology.can_reach_rsu(
                self.name, self.rsu_name)

            if can_reach and not self.budget_exhausted:
                print(f"[{self.name}] In range of RSU ({dist:.0f}m). "
                      f"Sending proxy to {self.rsu_name}...")
                with self.proxy_lock:
                    proxy_weights = {k: v.cpu() for k, v in self.proxy_model.state_dict().items()}

                # Eq. 6 — V2V neighbor gossip before hierarchical upload
                proxy_weights = self._v2v_share_and_aggregate(r, proxy_weights)

                if SECURITY_ENABLED and self.signer is not None and self.security_authority is not None:
                    raw_payload = serialize_weights(proxy_weights)
                    with Timer(self.name, r, "signature_generation"):
                        sig = self.signer.sign(raw_payload)
                    with Timer(self.name, r, "encryption"):
                        aad = message_aad("LOCAL_UPDATE", self.name, self.rsu_name, r)
                        rsu_info = self.security_authority.public_info(self.rsu_name)
                        shared_secret = self.signer.shared_secret_for(rsu_info)
                        ciphertext, nonce, tag = encrypt_payload(shared_secret, raw_payload, aad)
                    msg = build_envelope(
                        "LOCAL_UPDATE", self.signer, self.rsu_name, r,
                        sig, ciphertext, nonce, tag
                    )
                else:
                    msg = {
                        "type": "LOCAL_UPDATE",
                        "sender": self.name,
                        "round": r,
                        "weights": proxy_weights,
                    }
                send_started = time.perf_counter()
                send_msg(("127.0.0.1", self.rsu_port), msg, sender_name=self.name, round_num=r)
                self._request_sent_at[r] = send_started

                # Wait for global update for this round
                received = self.round_event.wait(timeout=TIMEOUT)
                if not received:
                    self._request_sent_at.pop(r, None)
                    print(f"[{self.name}] [!] Timed out waiting for global update for Round {r}")
            elif self.budget_exhausted and can_reach:
                print(f"[{self.name}] In range but privacy budget exhausted. Waiting for global update...")
                received = self.round_event.wait(timeout=TIMEOUT)
                if not received:
                    print(f"[{self.name}] [!] Timed out waiting for global update for Round {r}")
            else:
                print(f"[{self.name}] [X] OUT OF RANGE ({dist:.0f}m > {V2RSU_RANGE}m). Synchronizing on round barrier...")
                # Wait for round broadcast so out-of-range vehicles do not race ahead
                received = self.round_event.wait(timeout=TIMEOUT)
                if not received:
                    print(f"[{self.name}] [!] Timed out waiting for round {r} synchronization")

            # 6. Move vehicle
            self.topology.move_vehicle(self.name)
            new_pos = self.topology.get_vehicle_position(self.name)
            new_dist = self.topology.get_distance_to_rsu(self.name, self.rsu_name)
            print(f"[{self.name}] Moved -> ({new_pos[0]:.0f}, {new_pos[1]:.0f}) | {new_dist:.0f}m from RSU")

          except Exception as e:
            print(f"[{self.name}] [ERROR] Round {r} failed: {e}")
            import traceback
            traceback.print_exc()
          finally:
            metrics_tracker.record_duration(
                self.name, r, "device_round_execution", time.perf_counter() - round_t0
            )

        print(f"[{self.name}] Training Finished")

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------
    def start(self):
        self.receiver.start()
        threading.Thread(target=self.send, daemon=True).start()

    def shutdown(self):
        self.receiver.shutdown()
