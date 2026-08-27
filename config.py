# config.py — Centralized configuration for ProxyFL VANET simulation

# ==========================================
# FEDERATED LEARNING
# ==========================================
TOTAL_ROUNDS = 40
SIMULATION_SEED = 42
# Aggregation deadlines are INACTIVITY windows, not hard round deadlines: the
# window restarts every time a new participant reports, bounded by the MAX_WAIT
# cap measured from the first report of the round.  A fixed 25s deadline
# measured from the first arrival discarded most of every cluster, because
# local training is serialized by TRAINING_SEMAPHORE and the last vehicle in a
# cluster finishes minutes after the first.
RSU_ROUND_TIMEOUT = 45       # seconds — inactivity window for cluster updates
RSU_ROUND_MAX_WAIT = 150     # seconds — hard cap from the round's first report
SERVER_ROUND_TIMEOUT = 45    # seconds — inactivity window for RSU updates
SERVER_ROUND_MAX_WAIT = 180  # seconds — hard cap from the round's first report
# Timeout invariant: device wait timeout must cover the worst-case cascade
# (RSU cap + Server cap) plus a buffer.  Every RSU/Server code path is
# guaranteed to emit a downstream message, so this is a failsafe only.
TIMEOUT = RSU_ROUND_MAX_WAIT + SERVER_ROUND_MAX_WAIT + 30  # 360s
BATCH_SIZE = 32
LOCAL_EPOCHS = 3

# ==========================================
# SECURITY & AUTHENTICATION
# ==========================================
SECURITY_ENABLED = True
BATCH_VERIFICATION_ENABLED = True

# ==========================================
# TRUST SCORE (Eq. 9–10) — drop malicious updates
# ==========================================
TRUST_SCORE_ENABLED = True
# If None: accept updates with L2 deviation ≤ median(deviations) × multiplier
TRUST_L2_THRESHOLD = None
TRUST_MEDIAN_MULTIPLIER = 3.0

# ==========================================
# V2V PROXY SHARING (Eq. 6)
# ==========================================
V2V_ENABLED = True
# Local training is staggered by TRAINING_SEMAPHORE, so in-range peers reach
# their gossip phase tens of seconds apart.  With a 2s rendezvous barrier the
# barrier always timed out and Eq. (6) degenerated into "no peer proxies
# received" every round -- V2V sharing never actually happened.  The barrier
# must be able to span the straggler spread; the collect window only has to
# span one round-trip once the peers are aligned.
V2V_COLLECT_TIMEOUT = 10.0  # seconds to wait for in-range peer proxies
V2V_READY_TIMEOUT = 90.0    # seconds to wait for in-range peers to finish training

# ==========================================
# ENERGY MODEL (OBU power profile)
# E_op (Joules) = OBU_PEAK_POWER_W * x_op * (t_op_ms / 1000)
# ==========================================
OBU_PEAK_POWER_W = 10.88     # Watts — rated peak power draw
X_OP_IDLE = 0.2              # Radio standby, CPU mostly idle
X_OP_TRAIN = 1.0             # Local training (DML + per-sample DP-SGD)
X_OP_COMM = 0.6              # Communication (TX/RX of model payloads)
X_OP_CRYPTO = 0.4            # EC crypto point ops (keygen, sign, verify, batch)

# ==========================================
# VANET SPATIAL SIMULATION
# ==========================================
V2V_RANGE = 350          # meters — vehicle-to-vehicle communication range
V2RSU_RANGE = 1000       # meters — vehicle-to-RSU communication range
SPEED_RANGE = (0,0)     # m/s (7–28 km/h) — slower city speed bounds

# Observational IEEE 802.11p-style link budget used only for capacity metrics.
# These values never add delay, packet loss, or participation constraints.
VANET_BANDWIDTH_HZ = 10_000_000.0
VANET_TX_POWER_DBM = 23.0
VANET_PATH_LOSS_1M_DB = 46.4
VANET_PATH_LOSS_EXPONENT = 2.7
VANET_NOISE_FIGURE_DB = 9.0
VANET_PHY_MAX_RATE_BPS = 27_000_000.0

# Fixed five-RSU layout. Each simulated run assigns an independently random
# number of vehicles (2–10 inclusive) to every RSU.
RSU_LAYOUT = (
    ("RSU_0_Central", "Central", 0, 0),
    ("RSU_1_North", "North", 0, 1800),
    ("RSU_2_East", "East", 1800, 0),
    ("RSU_3_South", "South", 0, -1800),
    ("RSU_4_West", "West", -1800, 0),
)
VEHICLES_PER_CLUSTER_RANGE = (2, 10)

# ==========================================
# DIFFERENTIAL PRIVACY (applied to proxy model only)
# ==========================================
DP_CLIP_NORM = 1.0           # Max L2 norm for per-sample gradient clipping
DP_NOISE_MULTIPLIER = 0.05   # Gaussian noise scale (σ = multiplier × clip_norm)
DP_DELTA = 1e-5              # δ parameter for (ε,δ)-DP accounting
DP_MAX_EPSILON = None        # Optional ε budget cap; device stops sharing proxy when exceeded
DP_EPSILON_WARNING_THRESHOLD = 10.0  # Reporting only; never changes training

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
MAX_NETWORK_MESSAGE_BYTES = 16 * 1024 * 1024

# ==========================================
# HARDWARE & CONCURRENCY SAFETY (SIGABRT Prevention)
# ==========================================
import os
import sys

# Console safety: a single non-ASCII character in a log line used to raise
# UnicodeEncodeError on Windows' cp1252 stdout.  Those prints happen inside
# RSU/Server receiver threads, where the exception aborted an in-flight
# aggregation and silently lost the whole round.  Force a lossy UTF-8 console
# so logging can never affect protocol behaviour.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

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
