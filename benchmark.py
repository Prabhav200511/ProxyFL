import math
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# Fix MNIST download: the old yann.lecun.com URLs return 404.
# This torchvision version uses a `urls` class attribute (not mirrors/resources).
datasets.MNIST.urls = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
]
# Import custom architectures and aggregation functions from your models.py
from models import SimpleMNIST, VanetIDS, average_weights, jsd_weighted_average

# ==========================================
# BENCHMARK CONFIGURATION
# ==========================================
ROUNDS = 20
NUM_CLUSTERS = 2
VEHICLES_PER_CLUSTER = 2
TOTAL_VEHICLES = NUM_CLUSTERS * VEHICLES_PER_CLUSTER # 4 Vehicles
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"=== RUNNING FEDERATED BENCHMARK ON {DEVICE} ===")
print(f"Topology: {NUM_CLUSTERS} Clusters, {VEHICLES_PER_CLUSTER} Vehicles/Cluster ({TOTAL_VEHICLES} Total Nodes)")
print(f"Rounds: {ROUNDS}\n")


# ==========================================
# 1. MNIST BENCHMARK RUNNER
# ==========================================
def run_mnist_benchmark():
    print("--------------------------------------------------")
    print(" [1/2] Starting MNIST Federated Benchmark Run")
    print("--------------------------------------------------")
    
    # Download and prepare MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Partition dataset evenly across 4 vehicles
    data_split_len = len(train_dataset) // TOTAL_VEHICLES
    client_loaders = []
    for i in range(TOTAL_VEHICLES):
        subset = torch.utils.data.Subset(train_dataset, range(i * data_split_len, (i + 1) * data_split_len))
        client_loaders.append(DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True))
        
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Global Model Initialization
    global_model = SimpleMNIST().to(DEVICE)
    global_weights = global_model.state_dict()
    
    # Optimizers & Loss
    criterion = nn.CrossEntropyLoss()
    mnist_accuracy_history = []

    for r in range(1, ROUNDS + 1):
        cluster_models = []
        
        # Train on each Cluster
        for c in range(NUM_CLUSTERS):
            vehicle_weights = []
            for v in range(VEHICLES_PER_CLUSTER):
                client_idx = (c * VEHICLES_PER_CLUSTER) + v
                local_model = SimpleMNIST().to(DEVICE)
                local_model.load_state_dict(global_weights)
                optimizer = torch.optim.Adam(local_model.parameters(), lr=0.001)
                
                # Local Training (1 epoch)
                local_model.train()
                for images, labels in client_loaders[client_idx]:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    optimizer.zero_grad()
                    output = local_model(images)
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()
                    
                vehicle_weights.append(local_model.state_dict())
                
            # RSU Aggregation (FedAvg across vehicles in cluster)
            cluster_avg = average_weights(vehicle_weights)
            cluster_models.append(cluster_avg)
            
        # Global Server Aggregation (JSD Weighted Average)
        global_weights, _ = jsd_weighted_average(cluster_models, global_weights, alpha=2.0)
        global_model.load_state_dict(global_weights)
        
        # Global Evaluation
        global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = global_model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        accuracy = (correct / total) * 100.0
        mnist_accuracy_history.append(accuracy)
        print(f"Round {r:02d}/{ROUNDS} - MNIST Global Accuracy: {accuracy:.2f}%")
        
    return mnist_accuracy_history


# ==========================================
# 2. VANET IDS BENCHMARK RUNNER
# ==========================================
def run_vanet_benchmark():
    print("\n--------------------------------------------------")
    print(" [2/2] Starting VANET IDS Federated Benchmark Run")
    print("--------------------------------------------------")
    
    # Load Training Data
    df = pd.read_csv('Main_data_shuffled.csv')
    feature_cols = ['velocity_x', 'velocity_y', 'constant_offset_check', 'total_displacement']
    
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df['attacktype'].values, dtype=torch.long)
    
    # Partition dataset evenly across 4 vehicles
    data_split_len = len(df) // TOTAL_VEHICLES
    client_loaders = []
    for i in range(TOTAL_VEHICLES):
        subset_X = X[i * data_split_len : (i + 1) * data_split_len]
        subset_y = y[i * data_split_len : (i + 1) * data_split_len]
        dataset = TensorDataset(subset_X, subset_y)
        client_loaders.append(DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True))
        
    # Load Test Data (attack1_test.csv to attack5_test.csv)
    test_files = sorted(glob.glob('attack*_test.csv'))
    if test_files:
        test_dfs = [pd.read_csv(f) for f in test_files]
        test_df = pd.concat(test_dfs, ignore_index=True)
    else:
        print("[WARN] No attack*_test.csv found. Using a 20% validation split from main dataset.")
        test_df = df.sample(frac=0.2, random_state=42)

    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    X_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32).to(DEVICE)
    y_test = torch.tensor(test_df['attacktype'].values, dtype=torch.long).to(DEVICE)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=1024, shuffle=False)

    # Global Model Initialization
    global_model = VanetIDS().to(DEVICE)
    global_weights = global_model.state_dict()
    
    # Optimizers, Class Weights & Loss
    weights = torch.tensor([1.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    vanet_accuracy_history = []

    for r in range(1, ROUNDS + 1):
        cluster_models = []
        
        # Train on each Cluster
        for c in range(NUM_CLUSTERS):
            vehicle_weights = []
            for v in range(VEHICLES_PER_CLUSTER):
                client_idx = (c * VEHICLES_PER_CLUSTER) + v
                local_model = VanetIDS().to(DEVICE)
                local_model.load_state_dict(global_weights)
                
                optimizer = torch.optim.Adam(local_model.parameters(), lr=0.0001)
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
                
                # Local Training (3 epochs per round)
                local_model.train()
                for _ in range(3):
                    for batch_x, batch_y in client_loaders[client_idx]:
                        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                        optimizer.zero_grad()
                        output = local_model(batch_x)
                        loss = criterion(output, batch_y)
                        loss.backward()
                        optimizer.step()
                
                scheduler.step()
                vehicle_weights.append(local_model.state_dict())
                
            # RSU Aggregation
            cluster_avg = average_weights(vehicle_weights)
            cluster_models.append(cluster_avg)
            
        # Global Server Aggregation (JSD Weighted Average)
        global_weights, _ = jsd_weighted_average(cluster_models, global_weights, alpha=2.0)
        global_model.load_state_dict(global_weights)
        
        # Global Evaluation
        global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = global_model(batch_x)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
        accuracy = (correct / total) * 100.0
        vanet_accuracy_history.append(accuracy)
        print(f"Round {r:02d}/{ROUNDS} - VANET Global Accuracy: {accuracy:.2f}%")
        
    return vanet_accuracy_history


# ==========================================
# 3. PLOT COMPARISON GRAPH
# ==========================================
def plot_results(mnist_acc, vanet_acc):
    rounds = list(range(1, ROUNDS + 1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, mnist_acc, marker='o', color='#1f77b4', linewidth=2.5, label='MNIST Baseline (SimpleMNIST)')
    plt.plot(rounds, vanet_acc, marker='s', color='#ff7f0e', linewidth=2.5, label='VANET IDS (VanetIDS Deep MLP)')
    
    plt.title('Federated Learning Benchmark: MNIST vs. VANET IDS (2 Clusters × 2 Vehicles)', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Federated Round', fontsize=11, labelpad=10)
    plt.ylabel('Global Test Accuracy (%)', fontsize=11, labelpad=10)
    plt.xticks(range(1, ROUNDS + 1))
    plt.ylim(50, 100)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11, loc='lower right')
    
    output_filename = 'mnist_vs_vanet_comparison.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n[SUCCESS] Benchmark complete! Comparison graph saved as '{output_filename}'")
    plt.show()


if __name__ == "__main__":
    # Step 1: Run MNIST
    mnist_acc = run_mnist_benchmark()
    
    # Step 2: Run VANET
    vanet_acc = run_vanet_benchmark()
    
    # Step 3: Generate Plot
    plot_results(mnist_acc, vanet_acc)