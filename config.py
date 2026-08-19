# config.py — Centralized configuration for ProxyFL VANET simulation

# ==========================================
# FEDERATED LEARNING
# ==========================================
TOTAL_ROUNDS = 5
TIMEOUT = 45
BATCH_SIZE = 32
LOCAL_EPOCHS = 3

# ==========================================
# VANET SPATIAL SIMULATION
# ==========================================
V2V_RANGE = 350          # meters — vehicle-to-vehicle communication range
V2RSU_RANGE = 1000       # meters — vehicle-to-RSU communication range
SPEED_RANGE = (2, 8)     # m/s (7–28 km/h) — slower city speed bounds
RSU_SPACING = 1000       # meters — distance between RSU centers

# ==========================================
# DIFFERENTIAL PRIVACY (applied to proxy model only)
# ==========================================
DP_CLIP_NORM = 1.0           # Max L2 norm for per-sample gradient clipping
DP_NOISE_MULTIPLIER = 0.05   # Gaussian noise scale (σ = multiplier × clip_norm)
DP_DELTA = 1e-5              # δ parameter for (ε,δ)-DP accounting
DP_MAX_EPSILON = None        # Optional ε budget cap; device stops sharing proxy when exceeded

# ==========================================
# DEEP MUTUAL LEARNING (Eq. 4–5 from ProxyFL paper)
# ==========================================
DML_ALPHA = 0.5              # KL weight for private model loss:  L = (1-α)·CE + α·KL
DML_BETA = 0.5               # KL weight for proxy model loss:    L = (1-β)·CE + β·KL
DML_TEMPERATURE = 3.0        # Softmax temperature for knowledge distillation

# ==========================================
# NETWORK
# ==========================================
SERVER_PORT = 8000
RSU_BASE_PORT = 5000
DEVICE_BASE_PORT = 6000

# ==========================================
# HARDWARE & CONCURRENCY SAFETY (SIGABRT Prevention)
# ==========================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import warnings
import threading

# Prevent OpenMP thread explosion across multi-vehicle simulation threads
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

_device_str = "cpu"
if torch.cuda.is_available():
    _device_str = "cuda"
elif torch.backends.mps.is_available():
    _device_str = "mps"

DEVICE = torch.device(_device_str)

# Concurrency limiter to prevent GPU memory/kernel collisions or thread exhaustion
MAX_CONCURRENT_TRAINING = 4 if _device_str == "cuda" else 8
TRAINING_SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_TRAINING)