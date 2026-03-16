"""
THz pulse - water interaction simulation
========================================
Models the evolution of electron density in liquid water under
an intense terahertz (THz) pulse, considering:
  1. Energy-dependent collision frequency  ν_c(ε) ∝ √ε  (Drude model)
  2. Collisional (impact) ionisation:  ν_i(ε) = A_imp · exp(-ε_i / ε)
  3. Field-driven (tunnel) ionisation:  W_tun(E) = A_tun · exp(-β / E)
  4. Electron attachment / recombination for post-pulse density decay

THz pulse parameters:
  - Centre frequency  f₀ = 0.2 THz
  - Pulse width (FWHM of E-field envelope)  τ_FWHM = 1.8 ps
  - Field strengths:  0.9 / 2.1 / 3.0 MV/cm

Reference: peak electron densities are on the order of 10²¹ m⁻³ and the
temporal profile is characterised by a sharp rise during the pulse and an
exponential decay set by the attachment time constant.

Usage:
    python thz_water_simulation.py
Outputs:
    electron_density.png  –  density evolution for three field strengths
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────
e_charge = 1.602e-19   # elementary charge  [C]
m_e      = 9.109e-31   # electron mass      [kg]
n_mol    = 3.34e28     # water molecule number density  [m⁻³]

# ─────────────────────────────────────────────────────────────────────────────
# THz pulse parameters
# ─────────────────────────────────────────────────────────────────────────────
f0       = 0.2e12           # centre frequency           [Hz]
omega0   = 2 * np.pi * f0   # angular frequency          [rad s⁻¹]
tau_fwhm = 1.8e-12          # E-field envelope FWHM      [s]
# Gaussian width σ so that exp(-t²/2σ²) has FWHM = tau_fwhm
sigma_E  = tau_fwhm / (2 * np.sqrt(2 * np.log(2)))   # ≈ 0.764 ps


def E_envelope(t, E0):
    """Gaussian envelope of the THz pulse (cycle-averaged field amplitude)."""
    return E0 * np.exp(-0.5 * (t / sigma_E) ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Material / plasma model parameters  (liquid water)
# ─────────────────────────────────────────────────────────────────────────────
# Collision frequency  ν_c(ε) = ν_c0 · √(ε / ε_ref)
nu_c0   = 100.0e12          # collision-frequency prefactor at ε_ref  [s⁻¹]
eps_ref = 1.0 * e_charge    # reference electron energy (1 eV)         [J]

# Electron energy relaxation
tau_eps = 0.3e-12           # energy relaxation time                    [s]
nu_loss = 1.0 / tau_eps     # energy loss rate                          [s⁻¹]

# Impact (collisional) ionisation:  ν_i = A_imp · exp(−ε_i / ε)
eps_i  = 6.5 * e_charge     # impact ionisation threshold               [J]
A_imp  = 1.0e11             # impact ionisation prefactor               [s⁻¹]

# Tunnel (field) ionisation:  W_tun = A_tun · exp(−β_tun / E)
A_tun    = 6.5e6            # tunnel ionisation prefactor               [s⁻¹]
beta_tun = 3.0e8            # characteristic tunnel field               [V m⁻¹]

# Electron attachment (post-pulse density decay)
tau_att = 4.0e-12           # attachment time constant                  [s]
nu_att  = 1.0 / tau_att     # attachment rate                           [s⁻¹]

# Initial conditions
n_seed  = 1.0e16            # seed electron density                     [m⁻³]
eps_ini = 0.05 * e_charge   # initial mean electron energy (0.05 eV)    [J]


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def collision_freq(eps):
    """
    Energy-dependent electron–molecule collision frequency.
    ν_c(ε) = ν_c0 · √(ε / ε_ref)   [s⁻¹]
    """
    return nu_c0 * np.sqrt(max(eps, 1e-4 * e_charge) / eps_ref)


def impact_ionisation_rate(eps):
    """
    Townsend-type impact (collisional) ionisation rate.
    ν_i(ε) = A_imp · exp(−ε_i / ε)   [s⁻¹]
    """
    if eps <= 0.0:
        return 0.0
    return A_imp * np.exp(-eps_i / max(eps, 1e-4 * e_charge))


def tunnel_ionisation_rate(E_env):
    """
    Field-driven tunnel ionisation rate per molecule.
    W_tun(E) = A_tun · exp(−β_tun / E)   [s⁻¹]
    Vanishes when the field is negligible.
    """
    if E_env < 1.0e4:   # effectively zero below 0.01 MV/m
        return 0.0
    return A_tun * np.exp(-beta_tun / E_env)


# ─────────────────────────────────────────────────────────────────────────────
# Rate equations
# ─────────────────────────────────────────────────────────────────────────────
def odes(t, y, E0):
    """
    Coupled ODEs for electron density n [m⁻³] and mean energy ε [J].

    Equations
    ---------
    dε/dt = P_abs(E_env, ε) − ν_loss · ε − ν_i(ε) · ε_i
    dn/dt = [ν_i(ε) − ν_att] · n  +  W_tun(E_env) · n_mol

    where P_abs is the cycle-averaged Drude power absorption per electron:
        P_abs = e² E_env² ν_c(ε) / [m_e (ω₀² + ν_c²(ε))]
    (reduces to e²E²/(m_e ν_c) in the overdamped limit ν_c >> ω₀).
    """
    n   = max(y[0], 0.0)
    eps = max(y[1], 0.0)

    # THz field envelope at time t
    E_env = E_envelope(t, E0)

    # Energy-dependent collision frequency
    nu_c = collision_freq(eps)

    # Cycle-averaged Drude power absorption per electron [J s⁻¹]
    P_abs = (e_charge ** 2 * E_env ** 2 * nu_c) / (m_e * (omega0 ** 2 + nu_c ** 2))

    # Impact ionisation rate  [s⁻¹]
    nu_i = impact_ionisation_rate(eps)

    # Tunnel ionisation rate per molecule  [s⁻¹]
    W_tun = tunnel_ionisation_rate(E_env)

    # Electron mean-energy equation
    d_eps = P_abs - nu_loss * eps - nu_i * eps_i

    # Electron density equation  (avalanche + tunnel source − attachment)
    d_n = (nu_i - nu_att) * n + W_tun * n_mol

    return [d_n, d_eps]


# ─────────────────────────────────────────────────────────────────────────────
# Solve for three field strengths
# ─────────────────────────────────────────────────────────────────────────────
E0_values = [3.0e8, 2.1e8, 0.9e8]   # V/m  (3.0 / 2.1 / 0.9 MV/cm)
labels    = ['3.0 MV/cm', '2.1 MV/cm', '0.9 MV/cm']
colors    = ['#0a2a5e', '#2878c8', '#90c4e8']  # dark → light blue

t_start = -5.0e-12
t_end   = 15.0e-12
t_eval  = np.linspace(t_start, t_end, 40000)

y0 = [n_seed, eps_ini]

print("Running THz-water interaction simulation …")
print(f"  THz centre frequency : {f0 / 1e12:.1f} THz")
print(f"  Pulse FWHM           : {tau_fwhm * 1e12:.1f} ps")
print(f"  Gaussian σ           : {sigma_E * 1e12:.3f} ps")
print(f"  Seed density         : {n_seed:.0e} m⁻³")
print()

solutions = {}
for E0, label in zip(E0_values, labels):
    sol = solve_ivp(
        fun      = lambda t, y: odes(t, y, E0),
        t_span   = [t_start, t_end],
        y0       = y0,
        t_eval   = t_eval,
        method   = 'RK45',
        rtol     = 1e-7,
        atol     = [1e8, 1e-26],
    )
    solutions[E0] = sol
    n_peak = sol.y[0].max() / 1e21
    t_peak = sol.t[sol.y[0].argmax()] * 1e12
    print(f"  {label}: peak n = {n_peak:.1f} × 10²¹ m⁻³  at t = {t_peak:.1f} ps")

# ─────────────────────────────────────────────────────────────────────────────
# Plot: electron density vs time
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

for E0, label, color in zip(E0_values, labels, colors):
    sol = solutions[E0]
    t_ps = sol.t * 1e12
    n21  = np.maximum(sol.y[0], 0.0) / 1e21
    ax.plot(t_ps, n21, color=color, linewidth=2.0, label=label)

ax.set_xlim(-4, 14)
ax.set_ylim(0, 110)
ax.set_xlabel('t (ps)', fontsize=13)
ax.set_ylabel(r'$n_f$ ($10^{21}$ m$^{-3}$)', fontsize=13)
ax.legend(loc='upper right', fontsize=11, frameon=True)
ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig('electron_density.png', dpi=200, bbox_inches='tight')
print("\nSaved: electron_density.png")
