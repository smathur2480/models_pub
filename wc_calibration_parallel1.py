"""
wc_calibration.py
─────────────────
Parallel Bayesian calibration of neurolib WCModel + VBI ParBold simulator.
Updated with soft baseline penalties to resolve flatline optimization constraints.

Parameter selection (updated):
The 6 calibrated parameters are now the top-6 most sensitive parameters from
the OAT sensitivity run (wc_sensitivity_bounded.py), ranked by aggregate CV:
  1. mu_inh    (0.9059)  θ_I inh threshold
  2. c_excexc  (0.5860)  W_EE (E→E coupling)
  3. exc_ext   (0.5514)  i_E time-dep input to E
  4. mu_exc    (0.4736)  θ_E exc threshold
  5. a_exc     (0.3590)  m_E exc sigmoid gain
  6. a_inh     (0.1309)  m_I inh sigmoid gain

Bounds for each are the STAGE 0 narrowed (physiologically-plausible) bounds
from that same run, not hand-picked ranges:
  c_excexc : [10.0000, 40.0000]
  exc_ext  : [ 0.0000,  4.0000]
  a_inh    : [ 1.4276,  3.0000]
  a_exc    : [ 0.0000,  1.4741]
  mu_exc   : [ 0.0000,  2.9483]
  mu_inh   : [ 2.8552,  6.0000]

tau_exc and inh_ext_baseline were dropped from the calibrated set (they
ranked 7th and 14th) and are now held fixed at their neurolib defaults in
every simulation rather than being optimized over.
"""

import numpy as np
import torch
import scipy.signal
import logging
import warnings
from multiprocessing import Pool, cpu_count

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from neurolib.models.wc import WCModel
from vbi.models.numba.bold import ParBold, do_bold_step

from ax.service.managed_loop import optimize
from gpytorch.utils.warnings import NumericalWarning

## suppress Ax/GPyTorch INFO logs
logging.getLogger("ax").setLevel(logging.WARNING)
logging.getLogger("ax.service.managed_loop").setLevel(logging.WARNING)
logging.getLogger("ax.service.utils.instantiation").setLevel(logging.WARNING)
logging.getLogger("ax.generation_strategy.dispatch_utils").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=NumericalWarning)

# ── paths ─────────────────────────────────────────────────────────────────────
HCP_DATA_PATH = "Data_103818/103818_1_LR.R.1D"
RESULT_PATH   = "calibration_result.pt"
PLOT_PATH     = "calibration_spectral_comparison.png"

# ── HCP acquisition ───────────────────────────────────────────────────────────
TR = 0.72  # seconds

# ── simulation settings ───────────────────────────────────────────────────────
DURATION_MIN  = 18     # total WC simulation duration (minutes)
SAVE_MIN      = 15     # keep last 15 min, discard first 3 as transient
K_SIMS        = 500    # simulations per spectral_loss call
N_BO_TRIALS   = 50     # Bayesian optimization budget

# parameters held fixed (not calibrated) at their neurolib defaults, since
# they ranked outside the top 6 in the sensitivity analysis
FIXED_TAU_EXC          = 2.5   # ms, neurolib default
FIXED_INH_EXT_BASELINE = 0.0   # neurolib default

# ── load and preprocess HCP data ──────────────────────────────────────────────
print("Loading HCP data ...")
y_obs = torch.from_numpy(np.loadtxt("/work/sm222/103818_1_LR.R.1D")).float()

y_obs_standard = (y_obs - y_obs.mean(dim=-1, keepdim=True)) / \
                 (y_obs.std(dim=-1, keepdim=True))

filter_vertices = torch.isnan(y_obs_standard).any(dim=1)
y_obs_standard  = y_obs_standard[~filter_vertices, :]

V, M = y_obs_standard.shape
print(f"Loaded HCP data: {V} vertices, {M} timepoints (TR={TR}s)")

# ── observed mean PSD ─────────────────────────────────────────────────────────
_, psd_obs = scipy.signal.periodogram(y_obs_standard.numpy(), 1.0 / TR, axis=-1)
P_obs_mean = torch.from_numpy(psd_obs).float().mean(dim=0)

# ── single WC + VBI BOLD simulation ──────────────────────────────────────────
def run_wc_bold(params_tuple):
    # Top-6 sensitive parameters (see module docstring for ranking + bounds)
    exc_ext, mu_inh, a_inh, c_excexc, a_exc, mu_exc, seed, *_extra = params_tuple
    np.random.seed(seed)
    model = WCModel()
    model.params['sigma_ou']           = .01
    model.params['duration']           = DURATION_MIN * 60 * 1000   # ms
    model.params['exc_ext']            = exc_ext
    model.params['mu_inh']             = mu_inh
    model.params['a_inh']              = a_inh
    model.params['c_excexc']           = c_excexc
    model.params['a_exc']              = a_exc
    model.params['mu_exc']             = mu_exc
    # held fixed — not part of the calibrated top-6
    model.params['tau_exc']            = FIXED_TAU_EXC
    model.params['inh_ext_baseline']   = FIXED_INH_EXT_BASELINE
    model.run()

    exc = model.outputs['exc'][0]

    if np.isnan(exc).any() or exc.std() < 1e-6:
        return None

    dtt          = model.params['dt'] / 1000.0
    steps_per_tr = max(1, int(round(TR * 1000.0 / model.params['dt'])))

    P      = ParBold()
    nn_val = 1
    s      = np.ones((2, nn_val));  f      = np.ones((2, nn_val))
    ftilde = np.zeros((2, nn_val)); vtilde = np.zeros((2, nn_val))
    qtilde = np.zeros((2, nn_val)); v      = np.ones((2, nn_val))
    q      = np.ones((2, nn_val))

    bold_out = []
    for j, x in enumerate(exc):
        r_in = np.array([x])
        do_bold_step(r_in, s, f, ftilde, vtilde, qtilde, v, q, dtt, P)
        if (j % steps_per_tr) == 0:
            bold_val = P.vo * (
                (4.3 * P.theta0 * P.Eo * P.TE)   * (1.0 - q[0, 0])
              + (P.epsilon * P.r0 * P.Eo * P.TE)  * (1.0 - q[0, 0] / v[0, 0])
              + (1.0 - P.epsilon)                  * (1.0 - v[0, 0])
            )
            bold_val += np.random.normal(0, .00048)
            bold_out.append(bold_val)

    bold         = np.array(bold_out)
    n_discard_tr = int(round((DURATION_MIN - SAVE_MIN) * 60.0 / TR))
    bold_trimmed = bold[n_discard_tr:]
    
    if np.isnan(bold_trimmed).any() or bold_trimmed.std() < 1e-10:
        return None

    return bold_trimmed

# ── spectral loss (Parallelized with Soft Penalties) ─────────────────────────
def spectral_loss(gamma):
    exc_ext  = gamma["exc_ext"]
    mu_inh   = gamma["mu_inh"]
    a_inh    = gamma["a_inh"]
    c_excexc = gamma["c_excexc"]
    a_exc    = gamma["a_exc"]
    mu_exc   = gamma["mu_exc"]

    seed_seq = np.random.SeedSequence()
    child_seeds = [s.generate_state(1)[0] for s in seed_seq.spawn(K_SIMS)]

    task_params = [(exc_ext, mu_inh, a_inh, c_excexc, a_exc, mu_exc, child_seeds[i])
                   for i in range(K_SIMS)]

    num_workers = min(cpu_count(), 64)
    with Pool(processes=num_workers) as pool:
        results = pool.map(run_wc_bold, task_params)

    bold_sims = [b for b in results if b is not None]
    n_failed = K_SIMS - len(bold_sims)

    # FIX: Handle cases where ALL simulations fail by computing baseline data distance
    if len(bold_sims) == 0:
        P_sim_dead = torch.zeros_like(P_obs_mean)
        n_freq = min(P_obs_mean.shape[0], P_sim_dead.shape[0])
        worst_loss = torch.sum(torch.abs(P_obs_mean[:n_freq] - P_sim_dead[:n_freq])).item()
        print(f"  All {K_SIMS} sims degenerate — returning baseline data loss: {worst_loss:.4f}")
        return {"loss": (worst_loss, 0.0)}

    # FIX: If some simulations survived, evaluate survivors instead of breaking
    if n_failed > 0.8 * K_SIMS:
        print(f"  Warning: {n_failed}/{K_SIMS} failed. Calculating loss using {len(bold_sims)} survivors.")

    min_len  = min(len(b) for b in bold_sims)
    bold_arr = np.stack([b[:min_len] for b in bold_sims])

    bold_std = bold_arr.std(axis=-1, keepdims=True)
    bold_std = np.where(bold_std < 1e-10, 1.0, bold_std)
    bold_arr = (bold_arr - bold_arr.mean(axis=-1, keepdims=True)) / bold_std

    _, psd_sim = scipy.signal.periodogram(bold_arr, 1.0 / TR, axis=-1)
    P_sim_mean = torch.from_numpy(psd_sim).float().mean(dim=0)

    n_freq = min(P_obs_mean.shape[0], P_sim_mean.shape[0])
    loss   = torch.sum(torch.abs(P_obs_mean[:n_freq] - P_sim_mean[:n_freq])).item()

    print(f"  exc_ext={exc_ext:.3f}, mu_inh={mu_inh:.3f}, a_inh={a_inh:.3f}, "
          f"c_excexc={c_excexc:.3f}, a_exc={a_exc:.3f}, "
          f"mu_exc={mu_exc:.3f} → loss={loss:.4f} ({n_failed} failed)")

    return {"loss": (loss, 0.0)}

# ── Bayesian optimization ─────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"\nStarting Parallel Bayesian optimization ({N_BO_TRIALS} trials, "
          f"{K_SIMS} sims per trial) ...\n")

    # Bounds = STAGE 0 narrowed (physiologically-plausible) bounds from the
    # sensitivity run, for the top-6 most sensitive parameters.
    result = optimize(
        parameters=[
            {"name": "exc_ext",  "type": "range", "bounds": [0.0,    4.0]},
            {"name": "mu_inh",   "type": "range", "bounds": [2.8552, 6.0]},
            {"name": "a_inh",    "type": "range", "bounds": [1.4276, 3.0]},
            {"name": "c_excexc", "type": "range", "bounds": [10.0,   40.0]},
            {"name": "a_exc",    "type": "range", "bounds": [0.0,    1.4741]},
            {"name": "mu_exc",   "type": "range", "bounds": [0.0,    2.9483]},
        ],
        evaluation_function=spectral_loss,
        objective_name="loss",
        minimize=True,
        total_trials=N_BO_TRIALS,
        random_seed=42,
    )

    gamma_hat = result[0]
    print(f"\n{'='*60}\nCALIBRATION RESULT\n{'='*60}")
    for k, v_val in gamma_hat.items():
        print(f"  {k}: {v_val:.6f}")

    # ── save ──────────────────────────────────────────────────────────────────────
    torch.save({
        'gamma_hat':      gamma_hat,
        'n_bo_trials':    N_BO_TRIALS,
        'k_sims':         K_SIMS,
        'tr':             TR,
    }, RESULT_PATH)
    print(f"\nSaved calibration result → {RESULT_PATH}")

    # ── validation plot ───────────────────────────────────────────────────────────
    print("\nRunning validation simulations with calibrated parameters ...")
    N_VAL = min(200, K_SIMS)

    val_seed_seq = np.random.SeedSequence()
    val_child_seeds = [s.generate_state(1)[0] for s in val_seed_seq.spawn(N_VAL)]

    val_params = [(gamma_hat["exc_ext"], gamma_hat["mu_inh"], gamma_hat["a_inh"],
                   gamma_hat["c_excexc"], gamma_hat["a_exc"], gamma_hat["mu_exc"],
                   val_child_seeds[i]) for i in range(N_VAL)]

    with Pool(processes=min(cpu_count(), 64)) as pool:
        val_results = pool.map(run_wc_bold, val_params)
    
    val_bold = [b for b in val_results if b is not None]

    if len(val_bold) == 0:
        print(" All validation sims failed — check calibrated parameters")
    else:
        min_len  = min(len(b) for b in val_bold)
        val_arr  = np.stack([b[:min_len] for b in val_bold])
        val_std  = val_arr.std(axis=-1, keepdims=True)
        val_std  = np.where(val_std < 1e-10, 1.0, val_std)
        val_arr  = (val_arr - val_arr.mean(axis=-1, keepdims=True)) / val_std

        freqs_sim, psd_sim   = scipy.signal.periodogram(val_arr, 1.0 / TR, axis=-1)
        freqs_obs, psd_obs_v = scipy.signal.periodogram(y_obs_standard.numpy(), 1.0 / TR, axis=-1)

        freqs_sim  = freqs_sim[1:];  psd_sim   = psd_sim[:, 1:]
        freqs_obs  = freqs_obs[1:];  psd_obs_v = psd_obs_v[:, 1:]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        plot_len = min(min_len, M)  # M = y_obs_standard.shape[1], from earlier in the script
        t = np.arange(plot_len) * TR
        for i in range(min(3, len(val_bold))):
            ax.plot(t, val_arr[i, :plot_len], alpha=0.7, color='steelblue', lw=0.8,
                    label='Simulated (calibrated)' if i == 0 else None)
        for i in range(min(3, V)):
            ax.plot(t, y_obs_standard[i, :plot_len].numpy(),
                    alpha=0.7, color='darkorange', lw=0.8,
                    label='Observed (HCP)' if i == 0 else None)
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("Standardized BOLD", fontsize=11)
        ax.set_title("BOLD time series comparison", fontsize=12)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

        ax = axes[1]
        ax.plot(freqs_obs, psd_obs_v.mean(axis=0), lw=2, color='darkorange', label='Observed (HCP)')
        ax.plot(freqs_sim, psd_sim.mean(axis=0),   lw=2, color='steelblue',  label='Simulated (calibrated)')
        ax.set_xlabel("Frequency (Hz)", fontsize=11)
        ax.set_ylabel("Power Spectral Density", fontsize=11)
        ax.set_title("Mean PSD after calibration", fontsize=12)
        ax.set_yscale("log")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        ax.text(0.02, 0.02,
            f"exc_ext={gamma_hat['exc_ext']:.3f}\n"
            f"mu_inh={gamma_hat['mu_inh']:.3f}\n"
            f"a_inh={gamma_hat['a_inh']:.3f}\n"
            f"c_excexc={gamma_hat['c_excexc']:.3f}\n"
            f"a_exc={gamma_hat['a_exc']:.3f}\n"
            f"mu_exc={gamma_hat['mu_exc']:.3f}",
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
        print(f"Saved plot → {PLOT_PATH}")
