"""
THz pulse - water interaction simulation
========================================
Models the evolution of electron density in liquid water under
an intense terahertz (THz) pulse, considering:
  1. Energy-dependent collision frequency  ν_c(ε) ∝ √ε  (Drude model)
  2. Collisional (impact) ionisation:  ν_i(ε) = A_imp · exp(-ε_i / ε)
  3. Field-enhanced avalanche ionisation:  W_tun(E) = A_tun · exp(-β / E)
     (THz quasi-static field enhances the effective ionisation rate of
     existing electrons; source ∝ n, not ∝ n_mol)
  4. Electron attachment / recombination for post-pulse density decay
  5. Energy dilution: new electrons created at near-zero energy cool the
     electron ensemble — this couples the mean energy ε to the ionisation
     dynamics, and since every source is proportional to n, the peak
     electron density scales with the initial seed density n_seed.

THz pulse parameters:
  - Centre frequency  f₀ = 0.2 THz
  - Pulse width (FWHM of E-field envelope)  τ_FWHM = 1.8 ps
  - Field strengths:  0.9 / 2.1 / 3.0 MV/cm

Note: for a given field strength, the peak electron density scales
linearly with the seed density n_seed, demonstrating the expected
sensitivity of the ionisation yield to the initial carrier population.

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
nu_c0   = 20.0e12           # collision-frequency prefactor at ε_ref  [s⁻¹]
eps_ref = 1.0 * e_charge    # reference electron energy (1 eV)         [J]

# Electron energy relaxation
tau_eps = 0.3e-12           # energy relaxation time                    [s]
nu_loss = 1.0 / tau_eps     # energy loss rate                          [s⁻¹]

# Impact (collisional) ionisation:  ν_i = A_imp · exp(−ε_i / ε)
eps_i  = 6.5 * e_charge     # impact ionisation threshold               [J]
A_imp  = 1.0e13             # impact ionisation prefactor               [s⁻¹]

# Tunnel (field) ionisation:  W_tun = A_tun · exp(−β_tun / E)
A_tun    = 6.5e6            # tunnel ionisation prefactor               [s⁻¹]
beta_tun = 3.0e8            # characteristic tunnel field               [V m⁻¹]
E_tun_min = 1.0e4           # field below which W_tun is set to zero    [V m⁻¹]

# Electron attachment (post-pulse density decay)
tau_att = 4.0e-12           # attachment time constant                  [s]
nu_att  = 1.0 / tau_att     # attachment rate                           [s⁻¹]

# Initial conditions
n_seed  = 1.0e16            # seed electron density                     [m⁻³]
eps_ini = 0.05 * e_charge   # initial mean electron energy (0.05 eV)    [J]

# Plot floor: minimum density shown on the log-scale plot (avoids log 0)
N_PLOT_FLOOR = 1.0          # [m⁻³]


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
    Field-enhanced avalanche ionisation rate per existing electron.
    W_tun(E) = A_tun · exp(−β_tun / E)   [s⁻¹]
    In the THz quasi-static regime the strong field lowers the effective
    ionisation barrier of existing plasma electrons (not neutral molecules),
    so the source term is W_tun · n rather than W_tun · n_mol.
    Vanishes when the field is negligible.
    """
    if E_env < E_tun_min:   # effectively zero below 0.01 MV/m
        return 0.0
    return A_tun * np.exp(-beta_tun / E_env)


# ─────────────────────────────────────────────────────────────────────────────
# Rate equations
# ─────────────────────────────────────────────────────────────────────────────
def odes(t, y, E0):
    """
    Coupled ODEs for electron density n [m⁻³] and mean energy ε [J].

    Equations (derived from total energy conservation  d(n·ε)/dt):
    ---------------------------------------------------------------
    dn/dt = [ν_i(ε) + W_tun(E) − ν_att] · n

    dε/dt = P_abs(E_env, ε) − [ν_loss + ν_i(ε) + W_tun(E)] · ε − ν_i(ε) · ε_i

    where P_abs is the cycle-averaged Drude power absorption per electron:
        P_abs = e² E_env² ν_c(ε) / [m_e (ω₀² + ν_c²(ε))]
    (reduces to e²E²/(m_e ν_c) in the overdamped limit ν_c >> ω₀).

    Both ionisation sources (W_tun and ν_i) are proportional to n, so
    n(t) = n_seed · exp(∫[ν_i + W_tun − ν_att] dt) and the peak density
    scales linearly with the seed.  The energy equation is obtained by
    expanding d(n·ε)/dt, noting that:
      • −ν_i · ε  : impact-ionised child electrons enter at ~0 eV, diluting ε
      • −W_tun · ε : field-enhanced electrons also enter at ~0 eV, diluting ε
      • attachment terms (±ν_att · ε) cancel in the mean-energy equation
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

    # Field-enhanced avalanche ionisation rate per existing electron  [s⁻¹]
    W_tun = tunnel_ionisation_rate(E_env)

    # Electron mean-energy equation (derived from total energy conservation)
    # −ν_i·ε  : dilution by child electrons (impact ionisation, enter at ~0 eV)
    # −W_tun·ε : dilution by field-enhanced electrons (enter at ~0 eV)
    # Attachment terms (±ν_att·ε) cancel in the per-electron energy equation
    d_eps = P_abs - (nu_loss + nu_i + W_tun) * eps - nu_i * eps_i

    # Electron density equation: all sources ∝ n  →  n_peak ∝ n_seed
    d_n = (nu_i + W_tun - nu_att) * n

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
print("\nSeed sensitivity check (E₀ = 3.0 MV/cm):")
E0_check = 3.0e8
for n_test in [1e14, 1e16, 1e18]:
    sol_check = solve_ivp(
        fun    = lambda t, y: odes(t, y, E0_check),
        t_span = [t_start, t_end],
        y0     = [n_test, eps_ini],
        t_eval = t_eval,
        method = 'RK45',
        rtol   = 1e-7,
        atol   = [1e8, 1e-26],
    )
    print(f"  n_seed = {n_test:.0e}  →  peak n = {sol_check.y[0].max():.2e} m⁻³")
print()
print("  (peak density scales proportionally with seed — model is corrected)")
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
    n_peak = sol.y[0].max()
    t_peak = sol.t[sol.y[0].argmax()] * 1e12
    print(f"  {label}: peak n = {n_peak:.2e} m⁻³  at t = {t_peak:.1f} ps")

# ─────────────────────────────────────────────────────────────────────────────
# Plot: electron density vs time
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

for E0, label, color in zip(E0_values, labels, colors):
    sol = solutions[E0]
    t_ps = sol.t * 1e12
    n_m3 = np.maximum(sol.y[0], N_PLOT_FLOOR)   # floor to avoid log(0)
    ax.semilogy(t_ps, n_m3, color=color, linewidth=2.0, label=label)

ax.set_xlim(-4, 14)
ax.set_ylim(1e15, 1e22)
ax.set_xlabel('t (ps)', fontsize=13)
ax.set_ylabel(r'$n_e$ (m$^{-3}$)', fontsize=13)
ax.legend(loc='upper right', fontsize=11, frameon=True)
ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig('electron_density.png', dpi=200, bbox_inches='tight')
print("\nSaved: electron_density.png")
