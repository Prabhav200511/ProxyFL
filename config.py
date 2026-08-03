# config.py
TOTAL_ROUNDS = 25
SERVER_PORT = 9000

CLUSTER1_PORTS = [5001, 5002]
C1_RSU = 5003

CLUSTER2_PORTS = [6001, 6002]
C2_RSU = 6003

TIMEOUT = 60

LR = 0.01
BATCH_SIZE = 32

# --- DIFFERENTIAL PRIVACY SETTINGS ---
DP_CLIP_NORM = 1.0          # Max allowed limit for gradient updates
DP_NOISE_MULTIPLIER = 0.05   # Amount of statistical noise to inject (Higher = More Private, Less Accurate)