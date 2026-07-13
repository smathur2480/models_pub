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


# ── parameter bounds ──────────────────────────────────────────────────────────
theta_lower_alpha = 0.24
theta_upper_alpha = 0.40
theta_lower_tau   = 1.6
theta_upper_tau   = 2.4
theta_lower_eo    = 0.2
theta_upper_eo    = 0.55

batch_size = 100
tr_sec = .72
rng = np.random.default_rng(seed= int(args.seed))

# sample each parameter independently from uniform distributions
alpha_inputs = rng.uniform(theta_lower_alpha, theta_upper_alpha, batch_size)
tau_inputs   = rng.uniform(theta_lower_tau,   theta_upper_tau,   batch_size)
eo_inputs    = rng.uniform(theta_lower_eo,    theta_upper_eo,    batch_size)

sigma_ou_low,  sigma_ou_high  = 0.005, 0.05
exc_ext_low,   exc_ext_high   = 0.5,   1.5

sigma_ou_inputs = rng.uniform(sigma_ou_low,  sigma_ou_high,  batch_size)
exc_ext_inputs  = rng.uniform(exc_ext_low,   exc_ext_high,   batch_size)
print("batches were set up!")

# stack into (batch_size, 3) tensor for SBI — [alpha, tau, Eo]
raw_theta_tensor = torch.tensor(
    np.stack([alpha_inputs, tau_inputs, eo_inputs], axis=1),
    dtype=torch.float32
)

print("torch conversion done of theta inputs!")

outputs_exc  = []
outputs_bold = []

bold_noise_variance = 0.00048
bold_noise = rng.normal(0,  bold_noise_variance,  1250)


for i in range(batch_size):
    model = WCModel()
    model.params['duration'] = 18 * 60000
    model.params['sigma_ou'] = sigma_ou_inputs[i]
    model.params['exc_ext']  = exc_ext_inputs[i]
    model.run()
    
    

    exc = model.outputs['exc'][0]

    nn_val           = 1
    dtt          = model.params['dt'] / 1000.0
    steps_per_tr = max(1, int(round(tr_sec * 1000.0 / model.params['dt'])))

    P = ParBold(
        alpha = alpha_inputs[i],
        tau   = tau_inputs[i],
        Eo    = eo_inputs[i],
    )

    s      = np.ones((2, nn_val))
    f      = np.ones((2, nn_val))
    ftilde = np.zeros((2, nn_val))
    vtilde = np.zeros((2, nn_val))
    qtilde = np.zeros((2, nn_val))
    v      = np.ones((2, nn_val))
    q      = np.ones((2, nn_val))

    bold_out = []
    for j, x in enumerate(exc):
        r_in = np.array([x])
        do_bold_step(r_in, s, f, ftilde, vtilde, qtilde, v, q, dtt, P)
        if (j % steps_per_tr) == 0:
            bold_val = P.vo * ((4.3 * P.theta0 * P.Eo * P.TE)   * (1.0 - q[0, 0])
            + (P.epsilon * P.r0 * P.Eo * P.TE) * (1.0 - q[0, 0] / v[0, 0])
            + (1.0 - P.epsilon)  * (1.0 - v[0, 0])
            )
            bold_val+= (bold_noise[j])
            bold_out.append(bold_val)

    bold = np.array(bold_out)

    #cutoff_exc  = len(exc)  // 2
    cutoff_bold = int(round(15 * 60.0 / tr_sec))
   
    outputs_bold.append(bold[-cutoff_bold:])
    #outputs_exc.append(exc[-cutoff_exc:])
    
    
    print("simulation", i, "done")
# ── unpack results ────────────────────────────────────────────────────────────
# if saving exc too, swap the line below with:
# outputs_exc, outputs_bold = zip(*results)
# outputs_exc  = list(outputs_exc)

print("simulations are done")

# ── build output tensor ───────────────────────────────────────────────────────
array_output_bold = np.array(outputs_bold)                          # (batch_size, n_bold_timepoints)
bold_tensor       = torch.from_numpy(array_output_bold).float()    # (batch_size, n_bold_timepoints)

# ── save dictionary ───────────────────────────────────────────────────────────
sim_outputs = {
    'input':    raw_theta_tensor,   # (batch_size, 3)  — [alpha, tau, Eo]
    'features': bold_tensor,        # (batch_size, n_bold_timepoints)

    # ── uncomment to also save exc outputs ────────────────────────────────────
    # 'exc': torch.from_numpy(np.array(outputs_exc)).float(),
}

torch.save(sim_outputs, SAVE_PATH)
print(f"\nSaved to {SAVE_PATH}")

# ── verification ──────────────────────────────────────────────────────────────
print(f"Created dictionary with keys: {list(sim_outputs.keys())}")
print(f"Shape of 'input'    tensor: {sim_outputs['input'].shape}")
print(f"Shape of 'features' tensor: {sim_outputs['features'].shape}")
print(f"First 5 input values:                    {sim_outputs['input'][:5].tolist()}")
