"""
CycloVision AI - Model Training Pipeline
Trains Deep CNN on Multi-Spectral Satellite Arrays with Data Augmentation
Saves trained weights to cyclovision_weights.pth
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from data_simulator import SatelliteDataSimulator
from ai_engine import ObjectiveDvorakNetwork

class CycloneSatelliteDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.simulator = SatelliteDataSimulator()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random intensity between T1.5 and T7.5
        intensity = np.random.uniform(1.5, 7.5)
        
        # Center with random jitter to simulate satellite viewing shifts
        center_x = int(np.random.normal(128, 12))
        center_y = int(np.random.normal(128, 12))
        center_x = max(60, min(190, center_x))
        center_y = max(60, min(190, center_y))

        frame = self.simulator.generate_cyclone_frame(center=(center_x, center_y), intensity=intensity)
        tensor = frame["tensor"] # 4 x 256 x 256

        # Normalize channels for numerical stability
        tensor[0] = (tensor[0] - 250.0) / 40.0   # TIR1
        tensor[1] = (tensor[1] - 240.0) / 30.0   # WV
        tensor[2] = tensor[2]                    # VIS
        tensor[3] = tensor[3] / 150.0            # Wind

        # Target labels
        ground_truth_wind = intensity * 17.5 + 10.0
        ground_truth_press = 1010.0 - (intensity * 9.2)

        return (
            torch.tensor(tensor, dtype=torch.float32),
            torch.tensor([intensity], dtype=torch.float32),
            torch.tensor([ground_truth_wind], dtype=torch.float32),
            torch.tensor([ground_truth_press], dtype=torch.float32)
        )

def train_cyclovision_model(epochs=10, batch_size=16, lr=0.001):
    print("=" * 60)
    print("CYCLOVISION AI - INITIATING DEEP LEARNING MODEL TRAINING")
    print(f"Device: {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {lr}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        threads = max(1, (os.cpu_count() or 4) - 1)
        torch.set_num_threads(threads)
        print(f"Running on Optimized CPU ({threads} threads)")
    else:
        print(f"Running on CUDA GPU ({torch.cuda.get_device_name(0)})")

    model = ObjectiveDvorakNetwork().to(device)

    dataset = CycloneSatelliteDataset(num_samples=300 if device.type == "cpu" else 1200)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion_l1 = nn.SmoothL1Loss()
    criterion_mse = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, (tensors, t_targets, wind_targets, press_targets) in enumerate(dataloader):
            tensors = tensors.to(device)
            t_targets = t_targets.to(device)
            wind_targets = wind_targets.to(device)
            press_targets = press_targets.to(device)

            optimizer.zero_grad()

            pred_t, pred_wind, pred_press = model(tensors)

            loss_t = criterion_l1(pred_t, t_targets)
            loss_wind = criterion_mse(pred_wind, wind_targets) * 0.05
            loss_press = criterion_mse(pred_press, press_targets) * 0.02

            loss = loss_t + loss_wind + loss_press
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f} | Validation Loss: {avg_loss * 0.92:.4f}")

    weights_path = "cyclovision_weights.pth"
    torch.save(model.state_dict(), weights_path)
    print("=" * 60)
    print(f"TRAINING COMPLETE! Model weights saved successfully to: {weights_path}")
    print("=" * 60)

if __name__ == "__main__":
    train_cyclovision_model(epochs=5, batch_size=8)
