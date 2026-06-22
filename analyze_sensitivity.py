"""
Sensitivity analysis script — loads the trained posterior, runs active
subspace analysis to rank parameter importance, produces pairplot and
activity score visualizations.

Completely independent of test data — uses the posterior only.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sbi.analysis import ActiveSubspace, pairplot

# ── paths ─────────────────────────────────────────────────────────────────────
POSTERIOR_PATH      = "/work/sm222/posterior.pt"
TRAIN_LOG_PATH      = "/work/sm222/training_log.pt"
LOSS_PLOT_PATH      = "/work/sm222/training_loss.png"
EMBEDDING_PLOT_PATH = "/work/sm222/embedding_space.png"
PAIRPLOT_PATH       = "/work/sm222/pairplot.png"
SENSITIVITY_PATH    = "/work/sm222/sensitivity.png"

param_names = ['alpha', 'tau', 'Eo']

# ── load ──────────────────────────────────────────────────────────────────────
print("Loading posterior ...")
posterior_data = torch.load(POSTERIOR_PATH, weights_only=False)
posterior      = posterior_data['posterior']

print("Loading training log ...")
train_log    = torch.load(TRAIN_LOG_PATH, weights_only=False)
train_losses = train_log['train_losses']
val_losses   = train_log['val_losses']
embeddings   = train_log['embeddings']    # (n_train_val, EMBEDDING_OUTPUT_DIM)
theta        = train_log['theta']         # (n_train_val, 3)
theta_bounds = train_log['theta_bounds']

# ── 1. training loss curve ────────────────────────────────────────────────────
if len(train_losses) > 0:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="train loss",      color="steelblue")
    ax.plot(val_losses,   label="validation loss", color="darkorange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (negative log-likelihood)")
    ax.set_title("NPE Training Progress")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved training loss curve → {LOSS_PLOT_PATH}")

# ── 2. embedding space visualization (PCA + t-SNE) ───────────────────────────
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

embeddings_np = embeddings.numpy()
theta_np      = theta.numpy()
n_samples     = embeddings_np.shape[0]

MAX_TSNE_POINTS = 5000
if n_samples > MAX_TSNE_POINTS:
    idx = np.random.choice(n_samples, MAX_TSNE_POINTS, replace=False)
else:
    idx = np.arange(n_samples)

emb_sub   = embeddings_np[idx]
theta_sub = theta_np[idx]

print(f"\nRunning PCA on embeddings ({emb_sub.shape}) ...")
pca     = PCA(n_components=2)
emb_pca = pca.fit_transform(emb_sub)
print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

print(f"Running t-SNE on embeddings ({emb_sub.shape}) ...")
tsne     = TSNE(n_components=2, perplexity=min(30, n_samples // 4), random_state=42)
emb_tsne = tsne.fit_transform(emb_sub)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

for col, name in enumerate(param_names):
    color_vals = theta_sub[:, col]

    ax = axes[0, col]
    sc = ax.scatter(emb_pca[:, 0], emb_pca[:, 1], c=color_vals, cmap="viridis", s=8, alpha=0.7)
    ax.set_title(f"PCA — colored by {name}")
    ax.set_xlabel("PC1");  ax.set_ylabel("PC2")
    plt.colorbar(sc, ax=ax, label=name)

    ax = axes[1, col]
    sc = ax.scatter(emb_tsne[:, 0], emb_tsne[:, 1], c=color_vals, cmap="viridis", s=8, alpha=0.7)
    ax.set_title(f"t-SNE — colored by {name}")
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    plt.colorbar(sc, ax=ax, label=name)

plt.tight_layout()
plt.savefig(EMBEDDING_PLOT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved embedding space visualization → {EMBEDDING_PLOT_PATH}")

# ── 3. posterior pairplot ─────────────────────────────────────────────────────
print("\nDrawing posterior samples for pairplot ...")
posterior_samples = posterior.sample((5000,)).cpu()

_ = pairplot(
    posterior_samples,
    limits=[list(theta_bounds['alpha']),
            list(theta_bounds['tau']),
            list(theta_bounds['eo'])],
    figsize=(5, 5)
)
plt.savefig(PAIRPLOT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved pairplot → {PAIRPLOT_PATH}")

# ── 4. active subspace sensitivity ───────────────────────────────────────────
print("\nRunning active subspace analysis ...")
sensitivity    = ActiveSubspace(posterior)
e_vals, e_vecs = sensitivity.find_directions(posterior_log_prob_as_property=True)

e_vals = e_vals.cpu()
e_vecs = e_vecs.cpu()

print("Eigenvalues:\n",  e_vals, "\n")
print("Eigenvectors:\n", e_vecs)

# ── 5. activity scores ────────────────────────────────────────────────────────
activity_scores = torch.zeros(3)
for i in range(3):
    for j in range(3):
        activity_scores[i] += e_vals[j] * e_vecs[i, j] ** 2

print("\nActivity Scores:")
for name, score in zip(param_names, activity_scores):
    print(f"  {name}: {score.item():.6e}")

ranked = sorted(zip(param_names, activity_scores.tolist()), key=lambda x: x[1], reverse=True)
print("\nRanked by importance:")
for rank, (name, score) in enumerate(ranked, 1):
    print(f"  {rank}. {name}: {score:.6e}")

# ── 6. sensitivity bar chart ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# activity scores bar
ax = axes[0]
scores = [activity_scores[i].item() for i in range(3)]
colors = ['steelblue' if s < max(scores) else 'darkorange' for s in scores]
ax.bar(param_names, scores, color=colors, edgecolor='white', alpha=0.85)
ax.set_ylabel("Activity Score")
ax.set_title("Parameter Sensitivity (Active Subspace)")
ax.grid(axis='y', alpha=0.3)

# eigenvector heatmap
ax = axes[1]
im = ax.imshow(e_vecs.numpy() ** 2, cmap='viridis', vmin=0, vmax=1)
ax.set_xticks(range(3)); ax.set_xticklabels([f"λ{j+1}" for j in range(3)])
ax.set_yticks(range(3)); ax.set_yticklabels(param_names)
ax.set_title("Eigenvector² Components\n(contribution of each param to each direction)")
plt.colorbar(im, ax=ax)

# annotate heatmap cells
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{e_vecs[i, j].item()**2:.2f}",
                ha='center', va='center', color='white', fontsize=10)

plt.tight_layout()
plt.savefig(SENSITIVITY_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved sensitivity visualization → {SENSITIVITY_PATH}")
