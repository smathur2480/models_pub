import argparse

import numpy as np
import torch
import torch.nn as nn

from neurolib.models.wc import WCModel

parser = argparse.ArgumentParser()
parser.add_argument("seed", type=str)
args = parser.parse_args()
print(args.seed)

SAVE_PATH = f"/work/sm222/simouts/sim_output_{args.seed}.pt"


# ══════════════════════════════════════════════════════════════════════════════
#  Balloon-Windkessel Nonlinear Hemodynamic Forward Model
# ══════════════════════════════════════════════════════════════════════════════

KERNEL_EPS = 1e-3
DT = 0.05  # only used to build the (unused here) kernel time axis; the class referenced it but never defined it


class BalloonWindkessel(nn.Module):

    is_lti = False

    def __init__(self, t_r=0.72, T=1300, L=30, onset=0.0, dt=0.05,
                 alpha=0.32, rho=0.34, tau=0.98, epsilon=1.0,
                 V0=0.02, TE=0.04, bold_model='revised',
                 nu0=40.3, r0=25.0, dtype=torch.float64):
        super().__init__()
        self.t_r = float(t_r); self.T = int(T); self.L = int(L)
        self.onset = float(onset); self.dtype = dtype
        self.dur = self.L * self.t_r
        t_fine = torch.linspace(0, self.dur, int(float(self.dur) / DT)) - float(self.onset) / DT
        self.t_kernel = t_fine[::int(self.t_r / DT)].to(dtype)
        self.n_sub = max(1, int(round(self.t_r / float(dt))))
        self.dt = self.t_r / self.n_sub          # dt divides TR exactly
        self.alpha, self.rho, self.tau = alpha, rho, tau
        self.epsilon, self.V0 = epsilon, V0
        self.inv_alpha = 1.0 / alpha
        self.log1mrho = float(np.log1p(-rho))    # log(1-rho), for a stable E(f)
        if bold_model == 'classic':              # Buxton/Friston, 1.5T
            self.k1, self.k2, self.k3 = 7*rho, 2.0, 2*rho - 0.2
        else:                                    # Obata/Stephan revised, 3T
            self.k1 = 4.3*nu0*rho*TE
            self.k2 = epsilon*r0*rho*TE
            self.k3 = 1.0 - epsilon

    def _deriv(self, s, xf, xv, xq, kappa, gamma, u):
        emxf = torch.exp(-xf)                            # 1/f
        f = 1.0 / emxf
        E = -torch.expm1(self.log1mrho * emxf)           # 1 - (1-rho)^(1/f)
        outflow = torch.exp((self.inv_alpha - 1.0)*xv)   # v^(1/alpha)/v
        ds  = self.epsilon*u - kappa*s - gamma*(f - 1.0)
        dxf = s*emxf
        dxv = (torch.exp(xf - xv) - outflow) / self.tau
        dxq = (torch.exp(xf - xq)*E/self.rho - outflow) / self.tau
        return ds, dxf, dxv, dxq

    def _bold(self, xv, xq):
        v, q = torch.exp(xv), torch.exp(xq)
        return self.V0*(self.k1*(1.0 - q) + self.k2*(1.0 - q/v) + self.k3*(1.0 - v))

    def _as_tensor(self, x, device=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(np.ascontiguousarray(x))
        elif not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.to(self.dtype)
        return x if device is None else x.to(device)

    def forward(self, theta, neural_signals):
        theta = self._as_tensor(theta)
        u_tr = self._as_tensor(neural_signals, device=theta.device)
        if u_tr.dim() == 1:
            u_tr = u_tr.unsqueeze(0)
        B, T = u_tr.shape                        # T from the drive, NOT self.T
        if theta.shape[0] != B:
            raise ValueError("theta has %d rows but neural_signals has %d" % (theta.shape[0], B))
        kappa = theta[:, 0].clamp_min(1e-3); gamma = theta[:, 1].clamp_min(1e-3)
        s  = torch.zeros(B, dtype=self.dtype, device=theta.device)
        xf = torch.zeros_like(s); xv = torch.zeros_like(s); xq = torch.zeros_like(s)
        y = torch.empty(B, T, dtype=self.dtype, device=theta.device)
        dt = self.dt
        for t in range(T):
            y[:, t] = self._bold(xv, xq)
            u = u_tr[:, t]
            for _ in range(self.n_sub):
                a1 = self._deriv(s, xf, xv, xq, kappa, gamma, u)
                a2 = self._deriv(s+0.5*dt*a1[0], xf+0.5*dt*a1[1],
                                 xv+0.5*dt*a1[2], xq+0.5*dt*a1[3], kappa, gamma, u)
                a3 = self._deriv(s+0.5*dt*a2[0], xf+0.5*dt*a2[1],
                                 xv+0.5*dt*a2[2], xq+0.5*dt*a2[3], kappa, gamma, u)
                a4 = self._deriv(s+dt*a3[0], xf+dt*a3[1],
                                 xv+dt*a3[2], xq+dt*a3[3], kappa, gamma, u)
                s  = s  + dt/6*(a1[0]+2*a2[0]+2*a3[0]+a4[0])
                xf = xf + dt/6*(a1[1]+2*a2[1]+2*a3[1]+a4[1])
                xv = xv + dt/6*(a1[2]+2*a2[2]+2*a3[2]+a4[2])
                xq = xq + dt/6*(a1[3]+2*a2[3]+2*a3[3]+a4[3])
        if not torch.isfinite(y).all():
            raise FloatingPointError("Balloon-Windkessel integration diverged; "
                                     "check theta ranges and drive amplitude.")
        return y

    def _compute_kernel(self, theta, amplitude=KERNEL_EPS):
        theta = self._as_tensor(theta)
        A = float(amplitude)
        u = torch.zeros(theta.shape[0], self.L, dtype=self.dtype, device=theta.device)
        u[:, 0] = A
        kernel = self.forward(theta, u) / A
        return kernel, self.t_kernel.to(theta.device)


# ══════════════════════════════════════════════════════════════════════════════
#  Simulation
# ══════════════════════════════════════════════════════════════════════════════

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

# ── fixed (non-inferred) BW parameters ─────
# gamma is not sampled; set it to whatever you used before (0.41 = vbi ParBold default, worth double-checking).
GAMMA   = 0.41
EPSILON = 1.0     # NOTE: in this class epsilon scales the neural drive AND sets k2/k3
V0      = 0.02    # class default (vbi ParBold's vo differs)
TE      = 0.04

batch_size = 100
tr_sec = .72
rng = np.random.default_rng(seed=int(args.seed))

# sample each parameter independently from uniform distributions
alpha_inputs = rng.uniform(theta_lower_alpha, theta_upper_alpha, batch_size)
tau_inputs   = rng.uniform(theta_lower_tau,   theta_upper_tau,   batch_size)
eo_inputs    = rng.uniform(theta_lower_eo,    theta_upper_eo,    batch_size)
kappa_inputs = rng.uniform(theta_lower_kappa, theta_upper_kappa, batch_size)

print("batches were set up!")

# stack into (batch_size, 4) tensor for SBI — [alpha, tau, Eo, kappa]
raw_theta_tensor = torch.tensor(
    np.stack([alpha_inputs, tau_inputs, eo_inputs, kappa_inputs], axis=1),
    dtype=torch.float32
)

print("torch conversion done of theta inputs!")

outputs_bold_clean = []
outputs_bold_noisy = []

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

    # ── downsample the neural drive to one value per TR (mean over each TR window) ──
    steps_per_tr = max(1, int(round(tr_sec * 1000.0 / model.params['dt'])))
    n_tr = len(exc) // steps_per_tr
    exc_tr = exc[:n_tr * steps_per_tr].reshape(n_tr, steps_per_tr).mean(axis=1)

    # ── Balloon-Windkessel: alpha, tau, Eo(=rho) are constructor args, kappa/gamma go in theta ──
    bw = BalloonWindkessel(
        t_r=tr_sec, T=n_tr,
        alpha=float(alpha_inputs[i]),
        rho=float(eo_inputs[i]),
        tau=float(tau_inputs[i]),
        epsilon=EPSILON, V0=V0, TE=TE,
        bold_model='revised',
    )
    theta_bw = np.array([[kappa_inputs[i], GAMMA]])       # (1, 2): [kappa, gamma]

    with torch.no_grad():
        bold_clean = bw(theta_bw, exc_tr[None, :]).squeeze(0).cpu().numpy()   # (n_tr,)

    # ── acquisition noise (unchanged) ──
    bold_noisy = bold_clean + rng.normal(0, (bold_noise_variance/2), size=bold_clean.shape)

    cutoff_bold = int(round(15 * 60.0 / tr_sec))
    outputs_bold_clean.append(bold_clean[-cutoff_bold:])
    outputs_bold_noisy.append(bold_noisy[-cutoff_bold:])

    print("simulation", i, "done")

print("simulations are done")

# ── build output tensor ───────────────────────────────────────────────────────
array_output_bold_clean = np.array(outputs_bold_clean)   # (batch_size, n_bold_timepoints)
array_output_bold_noisy = np.array(outputs_bold_noisy)   # (batch_size, n_bold_timepoints)
bold_tensor_clean       = torch.from_numpy(array_output_bold_clean).float()
bold_tensor_noisy       = torch.from_numpy(array_output_bold_noisy).float()

# ── save dictionary ───────────────────────────────────────────────────────────
sim_outputs = {
    'input':          raw_theta_tensor,    # (batch_size, 4)  — [alpha, tau, Eo, kappa]
    'features':       bold_tensor_noisy,   # noisy BOLD, kept as 'features' for drop-in compatibility with downstream SBI code
    'features_clean': bold_tensor_clean,   # noise-free BOLD
}

torch.save(sim_outputs, SAVE_PATH)
print(f"\nSaved to {SAVE_PATH}")

# ── verification ──────────────────────────────────────────────────────────────
print(f"Created dictionary with keys: {list(sim_outputs.keys())}")
print(f"Shape of 'input'          tensor: {sim_outputs['input'].shape}")
print(f"Shape of 'features'       tensor: {sim_outputs['features'].shape}")
print(f"Shape of 'features_clean' tensor: {sim_outputs['features_clean'].shape}")
print(f"First 5 input values:                    {sim_outputs['input'][:5].tolist()}")
