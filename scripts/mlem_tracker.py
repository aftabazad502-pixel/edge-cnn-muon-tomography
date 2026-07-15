"""
MLEM Reconstruction Script for Cosmic Muon Tomography
======================================================
Implements a 20-iteration quadratic Maximum Likelihood Expectation
Maximization (MLEM) solver on a 25×25×10 voxel grid.

Usage:
    python scripts/mlem_tracker.py --data_dir data/raw/ --output_dir output/
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import glob
import sys
import argparse
from scipy.sparse import coo_matrix

parser = argparse.ArgumentParser(description="MLEM reconstruction for muon tomography")
parser.add_argument('--data_dir', type=str, default='data/raw',
                    help="Directory containing Geant4 MuonData CSV files")
parser.add_argument('--output_dir', type=str, default='output',
                    help="Directory for saving reconstruction output")
parser.add_argument('--iters', type=int, default=20,
                    help="Number of MLEM iterations")
parser.add_argument('--samples', type=int, default=50,
                    help="Number of uniform samples per muon trajectory")
args = parser.parse_args()

DATA_DIR = Path(args.data_dir)
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load Data ─────────────────────────────────────────────
print(f"Loading data from: {DATA_DIR}", flush=True)
csv_files = sorted(glob.glob(str(DATA_DIR / "MuonData_nt_MuonHits_t*.csv")))
if not csv_files:
    print(f"ERROR: No MuonData CSV files found in {DATA_DIR}")
    print("Please run the Geant4 simulation first, then point --data_dir to the output.")
    sys.exit(1)

dfs = []
for f in csv_files:
    df = pd.read_csv(f, comment='#', header=None,
                     names=['EventID', 'Sensor', 'X', 'Y', 'Z', 'Theta', 'Phi'])
    dfs.append(df)
combined = pd.concat(dfs, ignore_index=True)

top = combined[combined['Sensor'] == 'TopSensor'].copy()
bot = combined[combined['Sensor'] == 'BottomSensor'].copy()
top = top.drop_duplicates('EventID').rename(
    columns={'X': 'X_in', 'Y': 'Y_in', 'Z': 'Z_in', 'Theta': 'Theta_in', 'Phi': 'Phi_in'})
bot = bot.drop_duplicates('EventID').rename(
    columns={'X': 'X_out', 'Y': 'Y_out', 'Z': 'Z_out', 'Theta': 'Theta_out', 'Phi': 'Phi_out'})
matched = pd.merge(top, bot, on='EventID')
print(f"Matched {len(matched)} trajectories", flush=True)

# ── 2. Compute Directions and Scattering Angles ─────────────
def angles_to_vec(theta, phi):
    ux = np.sin(theta) * np.cos(phi)
    uy = np.sin(theta) * np.sin(phi)
    uz = np.cos(theta)
    return np.stack([ux, uy, uz], axis=1)

D1 = angles_to_vec(matched['Theta_in'].values, matched['Phi_in'].values)
D2 = angles_to_vec(matched['Theta_out'].values, matched['Phi_out'].values)
dot = np.clip(np.sum(D1 * D2, axis=1), -1.0, 1.0)
scattering_angle = np.arccos(dot)

P_in = matched[['X_in', 'Y_in', 'Z_in']].values
P_out = matched[['X_out', 'Y_out', 'Z_out']].values

# ── 3. Ray Sampling Along Trajectories ──────────────────────
V = P_out - P_in
t1 = (500 - P_in[:, 2]) / V[:, 2]
t2 = (-500 - P_in[:, 2]) / V[:, 2]

t_entry = np.minimum(t1, t2)
t_exit = np.maximum(t1, t2)

N_samples = args.samples
t_vals = np.linspace(0, 1, N_samples)
t_matrix = t_entry[:, None] + t_vals[None, :] * (t_exit - t_entry)[:, None]

pts = P_in[:, None, :] + t_matrix[:, :, None] * V[:, None, :]

# Voxel grid: 25 × 25 × 10
nx, ny, nz = 25, 25, 10
x_min, x_max = -1000, 1000
y_min, y_max = -1000, 1000
z_min, z_max = -500, 500

dx = (x_max - x_min) / nx
dy = (y_max - y_min) / ny
dz = (z_max - z_min) / nz

L_inside = np.linalg.norm(
    P_in + t_exit[:, None] * V - (P_in + t_entry[:, None] * V), axis=1)
dL = (L_inside / N_samples)[:, None]

ix = np.floor((pts[:, :, 0] - x_min) / dx).astype(int)
iy = np.floor((pts[:, :, 1] - y_min) / dy).astype(int)
iz = np.floor((pts[:, :, 2] - z_min) / dz).astype(int)

valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)

muon_indices = np.repeat(np.arange(len(matched)), N_samples)[valid.flatten()]
flat_ix = ix[valid]
flat_iy = iy[valid]
flat_iz = iz[valid]

voxel_indices = flat_ix * (ny * nz) + flat_iy * nz + flat_iz
flat_dL = np.repeat(dL, N_samples, axis=1)[valid]

M = len(matched)
J = nx * ny * nz
L_matrix = coo_matrix((flat_dL, (muon_indices, voxel_indices)), shape=(M, J)).tocsr()
L_sum_j = np.array(L_matrix.sum(axis=0)).flatten()
L_sum_j[L_sum_j == 0] = 1e-10

lambda_j = np.ones(J)

# ── 4. MLEM Iterations ──────────────────────────────────────
print(f"Starting {args.iters} MLEM iterations...", flush=True)
history = []

for it in range(args.iters):
    E_i = L_matrix.dot(lambda_j)
    E_i = np.maximum(E_i, 1e-10)

    # Quadratic MLEM update for scattering variance (Schultz et al., 2007)
    numerator = L_matrix.T.dot((scattering_angle ** 2) / (E_i ** 2))
    denominator = L_matrix.T.dot(1.0 / E_i)
    update_factor = np.sqrt(numerator / (denominator + 1e-10))
    lambda_j = lambda_j * update_factor

    if it + 1 in [5, 10, 20]:
        history.append((it + 1, lambda_j.copy().reshape((nx, ny, nz))))

    print(f"Iteration {it+1}/{args.iters} - Mean lambda: {np.mean(lambda_j):.6f}", flush=True)

final_lambda = lambda_j.reshape((nx, ny, nz))
npy_path = OUTPUT_DIR / "mlem_voxels.npy"
np.save(npy_path, final_lambda)
print(f"✅ Voxel data saved to {npy_path}")

# ── 5. Plotting ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor='#0d0d1a')
axes = axes.flatten()

slice_idx = nz // 2

for i, (it, lam) in enumerate(history):
    ax = axes[i]
    ax.set_facecolor('#0d0d1a')
    img = lam[:, :, slice_idx].T

    im = ax.imshow(img, extent=[x_min, x_max, y_min, y_max],
                   origin='lower', cmap='inferno')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Scattering Density')

    void_x, void_y = 200 - 250, 100 - 250
    rect = Rectangle((void_x, void_y), 500, 500, linewidth=2,
                      edgecolor='#00FFFF', facecolor='none', linestyle='--',
                      label='True Void Location')
    ax.add_patch(rect)

    ax.set_title(f'MLEM Iteration {it} (Z-slice)', color='white',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('X (mm)', color='white')
    ax.set_ylabel('Y (mm)', color='white')
    ax.tick_params(colors='white')
    if i == 0:
        ax.legend(loc='upper left', fontsize=8, facecolor='#1a1a2e',
                  edgecolor='#444', labelcolor='white')

ax = axes[3]
ax.set_facecolor('#0d0d1a')
img = final_lambda[:, ny // 2, :].T
im = ax.imshow(img, extent=[x_min, x_max, z_min, z_max],
               origin='lower', cmap='inferno', aspect='auto')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Scattering Density')

void_x, void_z = 200 - 250, 0 - 250
rect = Rectangle((void_x, void_z), 500, 500, linewidth=2,
                  edgecolor='#00FFFF', facecolor='none', linestyle='--')
ax.add_patch(rect)
ax.set_title('Final MLEM (Side View Y-slice)', color='white',
             fontsize=12, fontweight='bold')
ax.set_xlabel('X (mm)', color='white')
ax.set_ylabel('Z (mm)', color='white')
ax.tick_params(colors='white')

plt.tight_layout()
plot_path = OUTPUT_DIR / "mlem_reconstruction.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
print(f"✅ MLEM reconstruction plot saved to {plot_path}")
