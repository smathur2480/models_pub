import os


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import glob
import os
import torch

from neurolib.models.wc import WCModel
import neurolib.utils.loadData as ld
import neurolib.utils.functions as func
from vbi.models.numba.bold import ParBold, do_bold_step




import argparse
parser = argparse.ArgumentParser()
parser.add_argument("seed", type=str)
args = parser.parse_args()
print(args.seed)


SAVE_PATH = f"/work/sm222/simouts/sim_output_{args.seed}.pt"

# ── calibrated wc parameters ─────
EXC_EXT = 1.004
TAU_EXC = 6.976
A_INH = 2.066
INH_EXT_BASELINE = 1.000
A_EXC = 2.366
MU_EXC = 3.658

# ── parameter bounds ─────
theta_lower_alpha = 0.24
theta_upper_alpha = 0.40
theta_lower_tau   = 1.6
theta_upper_tau   = 2.4
theta_lower_eo    = 0.2
theta_upper_eo    = 0.55
theta_lower_kappa = 0.65
theta_upper_kappa = 1.5

batch_size = 100
tr_sec = .72
rng = np.random.default_rng(seed= int(args.seed))

# sample each parameter independently from uniform distributions
alpha_inputs = rng.uniform(theta_lower_alpha, theta_upper_alpha, batch_size)
tau_inputs   = rng.uniform(theta_lower_tau,   theta_upper_tau,   batch_size)
eo_inputs    = rng.uniform(theta_lower_eo,    theta_upper_eo,    batch_size)
kappa_inputs = rng.uniform(theta_lower_kappa, theta_upper_kappa, batch_size)


print("batches were set up!")


# stack into (batch_size, 3) tensor for SBI — [alpha, tau, Eo]
raw_theta_tensor = torch.tensor(
    np.stack([alpha_inputs, tau_inputs, eo_inputs,  kappa_inputs], axis=1),
    dtype=torch.float32
)

print("torch conversion done of theta inputs!")

outputs_exc = []
outputs_bold_clean   = []
outputs_bold_noisy   = []

bold_noise_variance = 0.00048



for i in range(batch_size):
    model = WCModel()
    model.params['duration'] = 18 * 60000
    model.params['exc_ext']  = EXC_EXT
    model.params['tau_exc']  = TAU_EXC
    model.params['a_inh']  = A_INH
    model.params['inh_ext_baseline']  = INH_EXT_BASELINE
    model.params['a_exc']  = A_EXC
    model.params['mu_exc']  = MU_EXC
    
    
    model.run()
    
    

    exc = model.outputs['exc'][0]

    nn_val           = 1
    dtt          = model.params['dt'] / 1000.0
    steps_per_tr = max(1, int(round(tr_sec * 1000.0 / model.params['dt'])))

    P = ParBold(
        alpha = alpha_inputs[i],
        tau   = tau_inputs[i],
        Eo    = eo_inputs[i],
        kappa = kappa_inputs[i],
    )

    s      = np.ones((2, nn_val))
    f      = np.ones((2, nn_val))
    ftilde = np.zeros((2, nn_val))
    vtilde = np.zeros((2, nn_val))
    qtilde = np.zeros((2, nn_val))
    v      = np.ones((2, nn_val))
    q      = np.ones((2, nn_val))

    bold_out_clean = []
    for j, x in enumerate(exc):
        r_in = np.array([x])
        do_bold_step(r_in, s, f, ftilde, vtilde, qtilde, v, q, dtt, P)
        if (j % steps_per_tr) == 0:
            bold_val = P.vo * ((4.3 * P.theta0 * P.Eo * P.TE)   * (1.0 - q[0, 0])
            + (P.epsilon * P.r0 * P.Eo * P.TE) * (1.0 - q[0, 0] / v[0, 0])
            + (1.0 - P.epsilon)  * (1.0 - v[0, 0])
            )
            bold_out_clean.append(bold_val)
            
            
    bold_clean = np.array(bold_out_clean)
    bold_noisy = bold_clean + rng.normal(0, (bold_noise_variance/2), size=bold_clean.shape)


    #cutoff_exc  = len(exc)  // 2
    cutoff_bold = int(round(15 * 60.0 / tr_sec))
   
    outputs_bold_clean.append(bold_clean[-cutoff_bold:])
    outputs_bold_noisy.append(bold_noisy[-cutoff_bold:])
    #outputs_exc.append(exc[-cutoff_exc:])
    
    
    print("simulation", i, "done")
# ── unpack results ────────────────────────────────────────────────────────────
# if saving exc too, swap the line below with:
# outputs_exc, outputs_bold = zip(*results)
# outputs_exc  = list(outputs_exc)

print("simulations are done")

# ── build output tensor ───────────────────────────────────────────────────────
array_output_bold_clean = np.array(outputs_bold_clean)                    # (batch_size, n_bold_timepoints)
array_output_bold_noisy = np.array(outputs_bold_noisy)                    # (batch_size, n_bold_timepoints)
bold_tensor_clean       = torch.from_numpy(array_output_bold_clean).float()
bold_tensor_noisy       = torch.from_numpy(array_output_bold_noisy).float()
 
# ── save dictionary ───────────────────────────────────────────────────────────
sim_outputs = {
    'input':          raw_theta_tensor,    # (batch_size, 4)  — [alpha, tau, Eo, kappa]
    'features':       bold_tensor_noisy,   # (batch_size, n_bold_timepoints) — with noise, kept as 'features' for drop-in compatibility with downstream SBI code
    'features_clean': bold_tensor_clean,   # (batch_size, n_bold_timepoints) — noise-free, so you can re-apply different noise structures without rerunning sims
 
    # ── uncomment to also save exc outputs ────────────────────────────────────
    # 'exc': torch.from_numpy(np.array(outputs_exc)).float(),
}
 
torch.save(sim_outputs, SAVE_PATH)
print(f"\nSaved to {SAVE_PATH}")
 
# ── verification ──────────────────────────────────────────────────────────────
print(f"Created dictionary with keys: {list(sim_outputs.keys())}")
print(f"Shape of 'input'          tensor: {sim_outputs['input'].shape}")
print(f"Shape of 'features'       tensor: {sim_outputs['features'].shape}")
print(f"Shape of 'features_clean' tensor: {sim_outputs['features_clean'].shape}")
print(f"First 5 input values:                    {sim_outputs['input'][:5].tolist()}")
