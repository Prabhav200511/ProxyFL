# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

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

class VanetIDS(nn.Module):
    def __init__(self):
        super().__init__()
        # Made the network significantly wider and slightly deeper
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

def average_weights(w_list):
    """Standard FedAvg for RSUs"""
    w_avg = copy.deepcopy(w_list[0])
    for key in w_avg.keys():
        for i in range(1, len(w_list)):
            w_avg[key] += w_list[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w_list))
    return w_avg

def calculate_jsd(weights_p, weights_q):
    """Calculates bounded Jensen-Shannon Divergence using Absolute Normalization"""
    p_flat = torch.cat([v.detach().cpu().flatten() for v in weights_p.values()])
    q_flat = torch.cat([v.detach().cpu().flatten() for v in weights_q.values()])

    # FIX: Use Absolute Normalization instead of Softmax to prevent massive-vector washout
    p_dist = torch.abs(p_flat) / torch.sum(torch.abs(p_flat))
    q_dist = torch.abs(q_flat) / torch.sum(torch.abs(q_flat))

    m_dist = 0.5 * (p_dist + q_dist)

    kl_p_m = F.kl_div(m_dist.log(), p_dist, reduction='sum')
    kl_q_m = F.kl_div(m_dist.log(), q_dist, reduction='sum')

    return 0.5 * (kl_p_m + kl_q_m).item()

def jsd_weighted_average(cluster_weights_list, global_weights, alpha=2.0):
    """Averages cluster models, penalizing those that drifted via JSD"""
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
            
    # FIX: Return both the averaged weights AND the divergences array
    return w_avg, divergences