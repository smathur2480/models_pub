"""
Model testing script — loads the trained posterior and held-out test data,
evaluates how accurately the posterior can recover the true parameters
from unseen BOLD signals.

Metrics reported per parameter (alpha, tau, Eo):
  - Posterior mean vs true value (bias)
  - Posterior std (sharpness / uncertainty)
  - RMSE of posterior mean predictions
  - Coverage: what fraction of true values fall within the posterior CI
  - Rank statistic (simulation-based calibration check)
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── paths ─────────────────────────────────────────────────────────────────────
POSTERIOR_PATH    = "/work/sm222/posterior.pt"
TRAIN_LOG_PATH    = "/work/sm222/training_log.pt"
TEST_PATH         = "/work/sm222/sim_outputs_test.pt"
RECOVERY_PLOT     = "/work/sm222/test_recovery.png"
COVERAGE_PLOT     = "/work/sm222/test_coverage.png"
CALIBRATION_PLOT  = "/work/sm222/test_calibration.png"

N_POSTERIOR_SAMPLES = 1000   # samples to draw per test observation for CI estimation

param_names = ['alpha', 'tau', 'Eo']

# ── load ──────────────────────────────────────────────────────────────────────
print("Loading posterior ...")
posterior_data    = torch.load(POSTERIOR_PATH, weights_only=False)
posterior         = posterior_data['posterior']
density_estimator = posterior_data['density_estimator']

print("Loading training log (for theta bounds) ...")
train_log    = torch.load(TRAIN_LOG_PATH, weights_only=False)
theta_bounds = train_log['theta_bounds']

print(f"Loading test data from {TEST_PATH} ...")
test_data    = torch.load(TEST_PATH)
x_test       = test_data['features']   # (n_test, n_timepoints)
theta_true   = test_data['input']      # (n_test, 3)

n_test = x_test.shape[0]
print(f"Test samples: {n_test}")

# ── per-test-observation posterior inference ───────────────────────────────────
print(f"\nRunning posterior inference on {n_test} test observations ...")
print(f"Drawing {N_POSTERIOR_SAMPLES} posterior samples per observation ...")

all_samples = []   # (n_test, N_POSTERIOR_SAMPLES, 3)

for i in range(n_test):
    x_obs     = x_test[i]
    samples_i = posterior.sample(
        (N_POSTERIOR_SAMPLES,),
        x=x_obs,
        show_progress_bars=False,
    ).cpu()
    all_samples.append(samples_i)
    if (i + 1) % max(1, n_test // 10) == 0:
        print(f"  {i+1}/{n_test} done")

all_samples = torch.stack(all_samples)   # (n_test, N_POSTERIOR_SAMPLES, 3)
theta_true  = theta_true.cpu()

# ── posterior mean and std per test observation ───────────────────────────────
posterior_mean = all_samples.mean(dim=1)   # (n_test, 3)
posterior_std  = all_samples.std(dim=1)    # (n_test, 3)

# ── metrics ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RECOVERY METRICS")
print("=" * 60)

rmse_per_param  = []
bias_per_param  = []
coverage_90     = []
coverage_50     = []

for p_idx, name in enumerate(param_names):
    true_vals  = theta_true[:, p_idx].numpy()
    pred_means = posterior_mean[:, p_idx].numpy()
    pred_stds  = posterior_std[:, p_idx].numpy()
    samples_p  = all_samples[:, :, p_idx].numpy()   # (n_test, N_POSTERIOR_SAMPLES)

    rmse = np.sqrt(np.mean((pred_means - true_vals) ** 2))
    bias = np.mean(pred_means - true_vals)

    # coverage: fraction of true values within posterior percentile intervals
    lo_90 = np.percentile(samples_p, 5,  axis=1)
    hi_90 = np.percentile(samples_p, 95, axis=1)
    lo_50 = np.percentile(samples_p, 25, axis=1)
    hi_50 = np.percentile(samples_p, 75, axis=1)

    cov_90 = np.mean((true_vals >= lo_90) & (true_vals <= hi_90))
    cov_50 = np.mean((true_vals >= lo_50) & (true_vals <= hi_50))

    rmse_per_param.append(rmse)
    bias_per_param.append(bias)
    coverage_90.append(cov_90)
    coverage_50.append(cov_50)

    print(f"\n  {name}:")
    print(f"    RMSE:            {rmse:.6f}")
    print(f"    Bias:            {bias:.6f}")
    print(f"    Mean post. std:  {pred_stds.mean():.6f}")
    print(f"    90% CI coverage: {cov_90*100:.1f}%  (ideal: 90%)")
    print(f"    50% CI coverage: {cov_50*100:.1f}%  (ideal: 50%)")

# ── 1. recovery plot: posterior mean vs true value ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for p_idx, name in enumerate(param_names):
    ax         = axes[p_idx]
    true_vals  = theta_true[:, p_idx].numpy()
    pred_means = posterior_mean[:, p_idx].numpy()
    pred_stds  = posterior_std[:, p_idx].numpy()

    lo, hi = theta_bounds[name.lower()]

    ax.errorbar(
        true_vals, pred_means,
        yerr=pred_stds,
        fmt='o', alpha=0.6, markersize=4,
        color='steelblue', ecolor='lightsteelblue', elinewidth=1,
        label="posterior mean ± std"
    )
    # perfect recovery diagonal
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label="perfect recovery")

    ax.set_xlabel(f"True {name}")
    ax.set_ylabel(f"Predicted {name}")
    ax.set_title(f"{name}  (RMSE={rmse_per_param[p_idx]:.4f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle("Posterior Mean Recovery vs True Parameters (test set)", fontsize=13)
plt.tight_layout()
plt.savefig(RECOVERY_PLOT, dpi=150, bbox_inches="tight")
print(f"\nSaved recovery plot → {RECOVERY_PLOT}")

# ── 2. coverage plot: CI coverage bar chart ───────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

x      = np.arange(len(param_names))
width  = 0.35

bars_90 = ax.bar(x - width/2, [c*100 for c in coverage_90], width,
                 label="90% CI", color='steelblue', alpha=0.8)
bars_50 = ax.bar(x + width/2, [c*100 for c in coverage_50], width,
                 label="50% CI", color='darkorange', alpha=0.8)

ax.axhline(90, color='steelblue', linestyle='--', linewidth=1, alpha=0.6, label="ideal 90%")
ax.axhline(50, color='darkorange', linestyle='--', linewidth=1, alpha=0.6, label="ideal 50%")

ax.set_xticks(x)
ax.set_xticklabels(param_names)
ax.set_ylabel("Coverage (%)")
ax.set_ylim(0, 110)
ax.set_title("Posterior Credible Interval Coverage (test set)")
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(COVERAGE_PLOT, dpi=150, bbox_inches="tight")
print(f"Saved coverage plot → {COVERAGE_PLOT}")

# ── 3. rank calibration plot (simulation-based calibration) ───────────────────
# For a well-calibrated posterior, ranks of the true value among posterior
# samples should be uniformly distributed. Deviations indicate over/under-
# confidence or systematic bias.
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for p_idx, name in enumerate(param_names):
    true_vals = theta_true[:, p_idx].numpy()
    samples_p = all_samples[:, :, p_idx].numpy()   # (n_test, N_POSTERIOR_SAMPLES)

    # rank of true value among posterior samples for each test obs
    ranks = np.array([
        np.sum(samples_p[i] < true_vals[i])
        for i in range(n_test)
    ])

    ax = axes[p_idx]
    ax.hist(ranks, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
    # flat line = ideal uniform distribution
    ax.axhline(n_test / 20, color='red', linestyle='--', linewidth=1.5, label="ideal (uniform)")
    ax.set_xlabel("Rank of true value")
    ax.set_ylabel("Count")
    ax.set_title(f"Rank calibration — {name}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle("Simulation-Based Calibration (uniform = well-calibrated)", fontsize=13)
plt.tight_layout()
plt.savefig(CALIBRATION_PLOT, dpi=150, bbox_inches="tight")
print(f"Saved calibration plot → {CALIBRATION_PLOT}")
