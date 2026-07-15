"""
Edge-CNN Training Script for Cosmic Muon Tomography
====================================================
Trains a lightweight 3-layer CNN (2,625 parameters) to recover void
geometry from noisy POCA scattering maps.

Usage:
    python scripts/train_cnn.py --data_dir data/ --output_dir output/ --epochs 500
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import argparse
import random

parser = argparse.ArgumentParser(description="Train Edge-CNN for muon tomography")
parser.add_argument('--data_dir', type=str, default='data',
                    help="Directory containing poca_data_*.csv files")
parser.add_argument('--output_dir', type=str, default='output',
                    help="Directory for saving model, stats, and plots")
parser.add_argument('--epochs', type=int, default=500,
                    help="Number of training epochs")
parser.add_argument('--seed', type=int, default=42,
                    help="Random seed for reproducibility (default: 42)")
args = parser.parse_args()

# Fixed seed for reproducibility of future retraining.
# Note: The original training run reported in the manuscript
# did not use a fixed seed. This seed is provided to enable
# deterministic reproduction for reviewers and future users.
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

DATA_DIR = Path(args.data_dir)
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRID_BINS = 50
X_EDGES = np.linspace(-1000, 1000, GRID_BINS + 1)
Y_EDGES = np.linspace(-1000, 1000, GRID_BINS + 1)


def get_true_mask(geom_id):
    """Generate ground-truth binary void mask for a given geometry ID."""
    mask = np.zeros((GRID_BINS, GRID_BINS))
    x_centers = (X_EDGES[:-1] + X_EDGES[1:]) / 2
    y_centers = (Y_EDGES[:-1] + Y_EDGES[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)

    if geom_id == 1 or geom_id == 7:  # G1 Baseline / G7 Rebar
        mask[(X >= -50) & (X <= 450) & (Y >= -150) & (Y <= 350)] = 1
    elif geom_id == 2:  # G2 Micro-Void
        mask[(X >= 150) & (X <= 250) & (Y >= 50) & (Y <= 150)] = 1
    elif geom_id == 3:  # G3 Double Void (TEST SET - excluded from training)
        mask[(X >= -550) & (X <= -50) & (Y >= -450) & (Y <= 50)] = 1
        mask[(X >= 50) & (X <= 550) & (Y >= -50) & (Y <= 450)] = 1
    elif geom_id == 4:  # G4 Off-Center
        mask[(X >= 350) & (X <= 850) & (Y >= 250) & (Y <= 750)] = 1
    elif geom_id == 5:  # G5 Solid Concrete (no void)
        pass
    elif geom_id == 6:  # G6 Micro-Crack (TEST SET - excluded from training)
        mask[(X >= 0) & (X <= 400) & (Y >= 75) & (Y <= 125)] = 1

    return mask


def generate_poca_image(csv_path):
    """Convert a POCA CSV file into a 2D scattering density map."""
    df = pd.read_csv(csv_path)
    x = df['POCA_X'].values
    y = df['POCA_Y'].values
    s = df['scattering'].values

    x_idx = np.clip(np.digitize(x, X_EDGES) - 1, 0, GRID_BINS - 1)
    y_idx = np.clip(np.digitize(y, Y_EDGES) - 1, 0, GRID_BINS - 1)

    scatter_sum = np.zeros((GRID_BINS, GRID_BINS))
    scatter_count = np.zeros((GRID_BINS, GRID_BINS))

    np.add.at(scatter_sum, (y_idx, x_idx), s)
    np.add.at(scatter_count, (y_idx, x_idx), 1)

    global_mean = np.mean(s) if len(s) > 0 else 0
    img = np.divide(scatter_sum, scatter_count,
                    out=np.ones_like(scatter_sum) * global_mean,
                    where=scatter_count != 0)
    return img


# ── Load Dataset ─────────────────────────────────────────────
print("Loading dataset...", flush=True)
inputs = []
targets = []

poca_files = sorted(glob.glob(str(DATA_DIR / "poca_data_*.csv")))
if not poca_files:
    print(f"ERROR: No poca_data_*.csv files found in {DATA_DIR}")
    print("Please place your POCA CSV files in the data/ directory.")
    exit(1)

for f in poca_files:
    fname = Path(f).name
    # Determine geometry ID from filename
    if "G2" in fname:
        gid = 2
    elif "G3" in fname:
        continue  # Test set — withheld from training
    elif "G4" in fname:
        gid = 4
    elif "G5" in fname:
        gid = 5
    elif "G6" in fname:
        continue  # Test set — withheld from training
    elif "G7" in fname:
        gid = 7
    else:
        gid = 1  # Default: G1 baseline

    img = generate_poca_image(f)
    mask = get_true_mask(gid)

    # Data augmentation: 4 rotations × 2 (original + horizontal flip) = 8 per file
    for k in range(4):
        img_rot = np.rot90(img, k)
        mask_rot = np.rot90(mask, k)
        inputs.append(img_rot.copy())
        targets.append(mask_rot.copy())

        img_flip = np.fliplr(img_rot)
        mask_flip = np.fliplr(mask_rot)
        inputs.append(img_flip.copy())
        targets.append(mask_flip.copy())

X = np.array(inputs, dtype=np.float32)
Y = np.array(targets, dtype=np.float32)

# Normalize inputs (z-score)
X_mean, X_std = X.mean(), X.std()
X = (X - X_mean) / (X_std + 1e-8)

X_tensor = torch.tensor(X).unsqueeze(1)  # (N, 1, H, W)
Y_tensor = torch.tensor(Y).unsqueeze(1)

print(f"Dataset shape: X={X_tensor.shape}, Y={Y_tensor.shape}", flush=True)


# ── Model Definition ─────────────────────────────────────────
class EdgeCNN(nn.Module):
    """Lightweight 3-layer CNN for edge deployment (2,625 parameters)."""
    def __init__(self):
        super(EdgeCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


model = EdgeCNN()
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.BCELoss()

print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Training for {args.epochs} epochs (BCE loss, lr=0.005)...", flush=True)

for epoch in range(args.epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, Y_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.6f}", flush=True)

print("Training complete.", flush=True)

# ── Save Model & Stats ──────────────────────────────────────
model_path = OUTPUT_DIR / "edge_cnn.pth"
stats_path = OUTPUT_DIR / "cnn_stats.npy"
torch.save(model.state_dict(), model_path)
np.save(stats_path, np.array([X_mean, X_std]))
print(f"✅ Model saved to {model_path}")
print(f"✅ Normalization stats saved to {stats_path}")

# ── Generate comparison plot ─────────────────────────────────
model.eval()
with torch.no_grad():
    test_x = X_tensor[0:1]
    test_y = Y_tensor[0:1]
    pred_y = model(test_x)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5), facecolor='#0d0d1a')

    ax[0].imshow(X[0], cmap='inferno', origin='lower')
    ax[0].set_title('Raw POCA Input', color='white')

    ax[1].imshow(test_y.squeeze().numpy(), cmap='gray', origin='lower')
    ax[1].set_title('True Void Mask', color='white')

    ax[2].imshow(pred_y.squeeze().numpy(), cmap='plasma', origin='lower')
    ax[2].set_title('CNN Prediction', color='white')

    for a in ax:
        a.set_facecolor('#0d0d1a')
        a.tick_params(colors='white')
        a.spines[:].set_edgecolor('#334')

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "cnn_result_g1.png"
    plt.savefig(plot_path, facecolor='#0d0d1a', dpi=150)
    plt.close()
    print(f"✅ Comparison plot saved to {plot_path}")
