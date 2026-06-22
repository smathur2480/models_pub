"""
Train script — loads train/val simulation outputs, trains NPE with a CNN
embedding net (suited for long BOLD time series ~14k timepoints), saves
the posterior + embeddings for downstream testing and sensitivity analysis.
"""

import torch
import numpy as np

from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import CNNEmbedding
from torch.distributions import Independent, Uniform

# ── paths ─────────────────────────────────────────────────────────────────────
LOAD_PATH      = "/work/sm222/sim_outputs_train_val.pt"
POSTERIOR_PATH = "/work/sm222/posterior.pt"
TRAIN_LOG_PATH = "/work/sm222/training_log.pt"

# ── device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# ── parameter bounds (must match simulate.py) ─────────────────────────────────
theta_lower_alpha = 0.24
theta_upper_alpha = 0.40
theta_lower_tau   = 1.6
theta_upper_tau   = 2.4
theta_lower_eo    = 0.2
theta_upper_eo    = 0.55

# ── load train/val simulation outputs ────────────────────────────────────────
print(f"\nLoading train/val simulation outputs from {LOAD_PATH} ...")
sim_outputs = torch.load(LOAD_PATH)

raw_theta_tensor = sim_outputs['input'].to(device)     # (n_train_val, 3)
x_tensor         = sim_outputs['features'].to(device)  # (n_train_val, n_timepoints)

n_samples    = x_tensor.shape[0]
n_timepoints = x_tensor.shape[1]

print(f"n_train_val samples: {n_samples}")
print(f"n_timepoints:        {n_timepoints}")

# ── prior ─────────────────────────────────────────────────────────────────────
prior = Independent(
    Uniform(
        low  = torch.tensor([theta_lower_alpha, theta_lower_tau, theta_lower_eo], device=device),
        high = torch.tensor([theta_upper_alpha, theta_upper_tau, theta_upper_eo], device=device)
    ),
    reinterpreted_batch_ndims=1
)

# ── CNN embedding net ─────────────────────────────────────────────────────────
# Compresses n_timepoints → EMBEDDING_OUTPUT_DIM before the normalizing flow.
# pool_kernel_size=8 aggressively downsamples 14k-length autocorrelated signals.
# Sized dynamically from n_timepoints so it adapts if signal length changes.
EMBEDDING_OUTPUT_DIM = 20

embedding_net = CNNEmbedding(
    input_shape=(n_timepoints,),
    out_channels_per_layer=[6, 12],
    num_conv_layers=2,
    num_linear_layers=2,
    output_dim=EMBEDDING_OUTPUT_DIM,
    pool_kernel_size=8,
).to(device)

print(f"\nCNN embedding net: {n_timepoints} -> {EMBEDDING_OUTPUT_DIM} dims")
print(embedding_net)

# ── density estimator builder with the embedding net wired in ─────────────────
density_estimator_builder = posterior_nn(
    model="maf",
    embedding_net=embedding_net,
)

# ── NPE training ──────────────────────────────────────────────────────────────
# Start with batch_size=128 on A100 (40GB) — raise to 256/512 if memory allows.
# Watch nvidia-smi on first full 50k run to tune this.
TRAINING_BATCH_SIZE = 128

print(f"\nTraining NPE (batch_size={TRAINING_BATCH_SIZE})...")
inference = NPE(
    prior,
    density_estimator=density_estimator_builder,
    device=str(device),
)

density_estimator = inference.append_simulations(raw_theta_tensor, x_tensor).train(
    training_batch_size=TRAINING_BATCH_SIZE,
    show_train_summary=True,
)

# ── pull training loss curves ─────────────────────────────────────────────────
train_losses = inference.summary.get("training_loss", [])
val_losses   = inference.summary.get("validation_loss", [])

print(f"\nFinal train loss: {train_losses[-1] if train_losses else 'N/A'}")
print(f"Final val loss:   {val_losses[-1] if val_losses else 'N/A'}")

# ── build and save posterior ──────────────────────────────────────────────────
observed_bold = x_tensor.mean(dim=0)
posterior     = inference.build_posterior(density_estimator)
posterior.set_default_x(observed_bold)

# ── extract learned embeddings for train/val set ──────────────────────────────
with torch.no_grad():
    embeddings = density_estimator.embedding_net(x_tensor).cpu()

print(f"\nExtracted train/val embeddings shape: {embeddings.shape}")

# ── save ──────────────────────────────────────────────────────────────────────
torch.save({
    'posterior':         posterior,
    'density_estimator': density_estimator,
}, POSTERIOR_PATH)

torch.save({
    'train_losses':      train_losses,
    'val_losses':        val_losses,
    'embeddings':        embeddings,              # (n_train_val, EMBEDDING_OUTPUT_DIM)
    'theta':             raw_theta_tensor.cpu(),  # (n_train_val, 3)
    'theta_bounds': {
        'alpha': (theta_lower_alpha, theta_upper_alpha),
        'tau':   (theta_lower_tau,   theta_upper_tau),
        'eo':    (theta_lower_eo,    theta_upper_eo),
    },
}, TRAIN_LOG_PATH)

print(f"Saved posterior    → {POSTERIOR_PATH}")
print(f"Saved training log → {TRAIN_LOG_PATH}")
