# models.py — Neural network architectures and aggregation functions for ProxyFL
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math


# ==========================================
# PRIVATE MODEL VARIANTS — kept local, never shared
# ==========================================
class VanetIDS(nn.Module):
    """Medium private model for VANET intrusion detection.
    Architecture: 4 → 128 → 64 → 32 → 6 with LayerNorm and Dropout.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 128)
        self.ln1 = nn.LayerNorm(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)

        self.fc3 = nn.Linear(64, 32)
        self.ln3 = nn.LayerNorm(32)

        self.fc4 = nn.Linear(32, 6)

    def forward(self, x):
        x = self.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.ln2(self.fc2(x)))
        x = self.relu(self.ln3(self.fc3(x)))
        return self.fc4(x)


class VanetIDSSmall(nn.Module):
    """Small private model for VANET intrusion detection.
    Architecture: 4 → 64 → 32 → 6 with LayerNorm.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 64)
        self.ln1 = nn.LayerNorm(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.fc2 = nn.Linear(64, 32)
        self.ln2 = nn.LayerNorm(32)

        self.fc3 = nn.Linear(32, 6)

    def forward(self, x):
        x = self.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)


class VanetIDSLarge(nn.Module):
    """Large private model for VANET intrusion detection.
    Architecture: 4 → 256 → 128 → 64 → 32 → 6 with LayerNorm.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 256)
        self.ln1 = nn.LayerNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)

        self.fc3 = nn.Linear(128, 64)
        self.ln3 = nn.LayerNorm(64)

        self.fc4 = nn.Linear(64, 32)
        self.ln4 = nn.LayerNorm(32)

        self.fc5 = nn.Linear(32, 6)

    def forward(self, x):
        x = self.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.ln2(self.fc2(x)))
        x = self.relu(self.ln3(self.fc3(x)))
        x = self.relu(self.ln4(self.fc4(x)))
        return self.fc5(x)


# Registry: name → class.  Only the proxy needs a common architecture;
# each device can independently pick any of these for its private model.
VANET_PRIVATE_ARCHITECTURES = {
    "small":  VanetIDSSmall,
    "medium": VanetIDS,
    "large":  VanetIDSLarge,
}


# ==========================================
# PROXY MODEL — shared via V2V gossip and RSU aggregation
# ==========================================
class ProxyModel(nn.Module):
    """Lightweight proxy model for VANET FL communication.
    Architecture: 4 → 32 → 6 (no LayerNorm, no Dropout).
    Payload size: ~1.4 KB (358 params) — transmits in <5 ms over V2V.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 6)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# ==========================================
# MNIST MODELS
# ==========================================
class MNISTPrivateModel(nn.Module):
    """Medium private model for MNIST (784 → 128 → 64 → 10)."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class MNISTPrivateModelSmall(nn.Module):
    """Small private model for MNIST (784 → 64 → 10)."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class MNISTProxyModel(nn.Module):
    """Lightweight Proxy Model for MNIST (784 → 64 → 10)."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


MNIST_PRIVATE_ARCHITECTURES = {
    "small":  MNISTPrivateModelSmall,
    "medium": MNISTPrivateModel,
}


# ==========================================
# MNIST BASELINE MODEL (for benchmark.py)
# ==========================================
class SimpleMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# ==========================================
# DEEP MUTUAL LEARNING LOSS
# ==========================================
def dml_loss(logits_student, soft_targets, temperature=3.0):
    """KL divergence loss for Deep Mutual Learning.

    The student model's log-softmax is compared against the teacher's
    softmax (both softened by temperature).  The teacher's outputs
    should be pre-computed with torch.no_grad().

    Args:
        logits_student: Raw logits from the model being updated.
        soft_targets: Softmax probabilities from the other model (detached).
        temperature: Softmax temperature for knowledge distillation.
    Returns:
        Scalar KL divergence loss (scaled by T²).
    """
    log_probs = F.log_softmax(logits_student / temperature, dim=1)
    return F.kl_div(log_probs, soft_targets, reduction='batchmean') * (temperature ** 2)


# ==========================================
# AGGREGATION FUNCTIONS
# ==========================================
def average_weights(w_list):
    """Standard FedAvg — arithmetic mean of proxy model state dicts."""
    w_avg = copy.deepcopy(w_list[0])
    for key in w_avg.keys():
        for i in range(1, len(w_list)):
            w_avg[key] += w_list[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w_list))
    return w_avg


def calculate_jsd(weights_p, weights_q):
    """Numerically stable Jensen-Shannon divergence between proxy model weights."""
    p_flat = torch.cat([v.detach().cpu().flatten() for v in weights_p.values()])
    q_flat = torch.cat([v.detach().cpu().flatten() for v in weights_q.values()])

    if not torch.isfinite(p_flat).all() or not torch.isfinite(q_flat).all():
        raise FloatingPointError(
            "Cannot aggregate non-finite model weights; check the local training loss.")

    # LayerNorm biases are initialized to exactly zero.  The previous version
    # evaluated log(0), and KL-divergence then produced 0 * -inf = NaN.  Add a
    # tiny positive mass before normalizing so every probability is valid.
    epsilon = torch.finfo(p_flat.dtype).eps
    p_dist = torch.abs(p_flat) + epsilon
    q_dist = torch.abs(q_flat) + epsilon
    p_dist = p_dist / p_dist.sum()
    q_dist = q_dist / q_dist.sum()

    m_dist = 0.5 * (p_dist + q_dist)

    kl_p_m = torch.sum(p_dist * (p_dist.log() - m_dist.log()))
    kl_q_m = torch.sum(q_dist * (q_dist.log() - m_dist.log()))

    return 0.5 * (kl_p_m + kl_q_m).item()


def jsd_weighted_average(cluster_weights_list, global_weights, alpha=2.0):
    """Server-level aggregation: penalize drifted clusters via JSD."""
    divergences = []
    for cluster_weights in cluster_weights_list:
        jsd = calculate_jsd(cluster_weights, global_weights)
        divergences.append(jsd)

    print(f"[SERVER] Measured JS Divergences: {[round(d, 4) for d in divergences]}")

    penalties = [math.exp(-alpha * d) for d in divergences]
    total_penalty = sum(penalties)
    normalized_weights = [p / total_penalty for p in penalties]

    w_avg = copy.deepcopy(cluster_weights_list[0])
    for key in w_avg.keys():
        device_type = w_avg[key].device
        w_avg[key] = w_avg[key] * normalized_weights[0]
        for i in range(1, len(cluster_weights_list)):
            w_avg[key] += cluster_weights_list[i][key].to(device_type) * normalized_weights[i]

    return w_avg, divergences
