"""
IMM-KF: CA → CV → CT Mode-Switching Demo
=========================================
Scenario — three sequential motion phases:
  Phase 1 (0 – 1 s)   Constant Acceleration: straight line, 5 → 15 m/s
  Phase 2 (1 – 2 s)   Constant Velocity:     straight line at 15 m/s
  Phase 3 (2 – 5 s)   Constant Turn-rate:    left turn at 45 deg/s (~135°)

One IMM-KF with three models — all in the shared 6-D state
[x, vx, ax, y, vy, ω] — tracks noisy position + velocity measurements.

CV : ax and ω zeroed each step (different F from CA)
CA : ax propagated as constant state (different F from CV)
CT : exact constant-turn-rate kinematics, ax zeroed
"""

import numpy as np
import matplotlib.pyplot as plt
from imm_kf import IMMKF
from models import make_2d_cv_model, make_2d_ca_model, make_2d_ct_model


Ts     = 0.01
N      = 500           # total steps (5 s)
CA_END = 100           # last CA step  (1 s)
CV_END = 200           # last CV step  (2 s), CT begins after
OMEGA  = np.deg2rad(45.0)   # turn rate [rad/s]

SIGMA_OBS_POS = 0.1    # position measurement noise [m]
SIGMA_OBS_VEL = 0.5    # velocity measurement noise [m/s]


# ---------------------------------------------------------------------------
# Ground-truth trajectory
# ---------------------------------------------------------------------------

def ground_truth() -> tuple:
    x  = np.zeros(N);  vx = np.zeros(N)
    y  = np.zeros(N);  vy = np.zeros(N)
    vx[0] = 5.0
    ax_ca = (15.0 - 5.0) / (CA_END * Ts)

    # Process noise std-devs for each phase
    sigma_jerk   = 2.0    # jerk [m/s³] — perturbs acceleration during CA
    sigma_accel  = 0.3    # spurious acceleration [m/s²] — perturbs velocity during CV
    sigma_omega  = 0.05   # turn-rate jitter [rad/s] — perturbs heading during CT

    for k in range(1, N):
        if k <= CA_END:                             # --- CA ---
            jerk = np.random.randn() * sigma_jerk
            ax   = ax_ca + jerk * Ts               # perturbed instantaneous accel
            vx[k] = vx[k-1] + Ts * ax
            vy[k] = vy[k-1] + np.random.randn() * sigma_accel * Ts
            x[k]  = x[k-1] + Ts * vx[k-1] + 0.5 * Ts**2 * ax
            y[k]  = y[k-1]  + Ts * vy[k-1]
        elif k <= CV_END:                           # --- CV ---
            dvx = np.random.randn() * sigma_accel * Ts
            dvy = np.random.randn() * sigma_accel * Ts
            vx[k] = vx[k-1] + dvx
            vy[k] = vy[k-1] + dvy
            x[k]  = x[k-1] + Ts * vx[k-1]
            y[k]  = y[k-1]  + Ts * vy[k-1]
        else:                                       # --- CT with turn-rate jitter ---
            omega_k  = OMEGA + np.random.randn() * sigma_omega
            theta_k  = omega_k * Ts
            sk, ck   = np.sin(theta_k), np.cos(theta_k)
            x[k]  = x[k-1] + sk/omega_k * vx[k-1] - (1 - ck)/omega_k * vy[k-1]
            y[k]  = y[k-1] + (1 - ck)/omega_k * vx[k-1] + sk/omega_k * vy[k-1]
            vx[k] = ck * vx[k-1] - sk * vy[k-1]
            vy[k] = sk * vx[k-1] + ck * vy[k-1]

    return x, y, vx, vy


# ---------------------------------------------------------------------------
# IMM-KF — three models from models.py, all in 6D state [x, vx, ax, y, vy, ω]
# ---------------------------------------------------------------------------

def build_imm() -> IMMKF:
    turn_deg = np.rad2deg(OMEGA)
    ca_model = make_2d_ca_model(Ts, sigma_j=3.0,  sigma_obs_pos=SIGMA_OBS_POS, sigma_obs_vel=SIGMA_OBS_VEL)
    cv_model = make_2d_cv_model(Ts, sigma_cv=0.5, sigma_obs_pos=SIGMA_OBS_POS, sigma_obs_vel=SIGMA_OBS_VEL)
    ct_model = make_2d_ct_model(Ts, turn_rate_deg=turn_deg, sigma_obs_pos=SIGMA_OBS_POS, sigma_obs_vel=SIGMA_OBS_VEL)

    pi = np.array([
        [0.95, 0.025, 0.025],
        [0.025, 0.95, 0.025],
        [0.025, 0.025, 0.95],
    ])
    return IMMKF([ca_model, cv_model, ct_model], pi,
                 mu0=np.array([0.8, 0.1, 0.1]))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def run_demo():
    np.random.seed(0)
    t = np.arange(N) * Ts

    x_true, y_true, vx_true, vy_true = ground_truth()
    meas_x  = x_true  + np.random.randn(N) * SIGMA_OBS_POS
    meas_y  = y_true  + np.random.randn(N) * SIGMA_OBS_POS
    meas_vx = vx_true + np.random.randn(N) * SIGMA_OBS_VEL
    meas_vy = vy_true + np.random.randn(N) * SIGMA_OBS_VEL

    imm   = build_imm()
    # 6D initial state: [x, vx, ax=0, y, vy, ω=0]
    state = imm.init(
        x0=np.array([x_true[0], vx_true[0], 0., y_true[0], vy_true[0], 0.]),
        P0=np.diag([1., 1., 1., 1., 1., 0.01]),
    )

    est     = np.zeros((N, 6))
    mu_hist = np.zeros((N, 3))
    for k in range(N):
        # Measurement order matches H rows: [x, vx, y, vy]
        y_k        = np.array([meas_x[k], meas_vx[k], meas_y[k], meas_vy[k]])
        state      = imm.step(state, y_k)
        est[k]     = state.x_est
        mu_hist[k] = state.mu

    # 25-step prediction from the final state
    T_PRED = 25
    x_pred, P_pred, best_i = imm.predict(state, T=T_PRED)
    t_pred = t[-1] + np.arange(T_PRED + 1) * Ts

    # ------------------------------------------------------------------
    # Console summary at phase boundaries
    # ------------------------------------------------------------------
    names = [m.name for m in imm.models]
    print(f"\n{'Phase':<12}" + "".join(f"{n:>10}" for n in names))
    for step, label in [(0, 'init'), (CA_END, 'CA end'), (CV_END, 'CV end'), (N-1, 'final')]:
        print(f"{label:<12}" + "".join(f"{mu_hist[step, i]:>10.3f}" for i in range(3)))
    print(f"\nMost probable model at end: {imm.models[best_i].name}")

    # ------------------------------------------------------------------
    # Plot  (y is at state index 3 in 6D, not 2)
    # ------------------------------------------------------------------
    t_ca = CA_END * Ts
    t_cv = CV_END * Ts

    fig, axes = plt.subplots(4, 1, figsize=(11, 15))
    fig.suptitle(
        "IMM-KF: CA → CV → CT Mode-Switching Tracking\n"
        "State: [x, vx, ax, y, vy, ω]  |  Models: distinct F per mode  |  Obs: [x, vx, y, vy]",
        fontsize=11, fontweight='bold',
    )

    phase_fill = [
        (0,     t_ca,   '#ffe0e0'),
        (t_ca,  t_cv,   '#e0e8ff'),
        (t_cv,  t[-1],  '#e0ffe0'),
    ]
    model_colors = ['tab:red', 'tab:blue', 'tab:green']

    def shade(ax):
        for t0, t1, col in phase_fill:
            ax.axvspan(t0, t1, color=col, alpha=0.55)
        for tv in (t_ca, t_cv):
            ax.axvline(tv, color='gray', lw=0.8, ls='--')

    def phase_labels(ax):
        for txt, xmid in [('CA', 0.5 * t_ca),
                           ('CV', 0.5 * (t_ca + t_cv)),
                           ('CT', 0.5 * (t_cv + t[-1]))]:
            ax.text(xmid, 0.97, txt, transform=ax.get_xaxis_transform(),
                    ha='center', va='top', fontsize=9, fontweight='bold', color='#444')

    # -- (0) x position --
    ax = axes[0]
    shade(ax);  phase_labels(ax)
    ax.plot(t, x_true,    'k-',  lw=2,   label='True')
    ax.plot(t, meas_x,    '.', color='silver', ms=3, alpha=0.6, label='Meas')
    ax.plot(t, est[:, 0], 'b--', lw=1.5, label='IMM-KF')
    ax.plot(t_pred, x_pred[:, 0], 'r--', lw=1.5, label='Pred')
    ax.fill_between(t_pred,
                    x_pred[:, 0] - 2*np.sqrt(P_pred[:, 0, 0]),
                    x_pred[:, 0] + 2*np.sqrt(P_pred[:, 0, 0]),
                    color='red', alpha=0.15, label='±2σ')
    ax.set_ylabel('x [m]');  ax.legend(fontsize=7, ncol=5);  ax.grid(True, alpha=0.3)

    # -- (1) y position  (index 3 in 6D state) --
    ax = axes[1]
    shade(ax)
    ax.plot(t, y_true,    'k-',  lw=2,   label='True')
    ax.plot(t, meas_y,    '.', color='silver', ms=3, alpha=0.6, label='Meas')
    ax.plot(t, est[:, 3], 'b--', lw=1.5, label='IMM-KF')
    ax.plot(t_pred, x_pred[:, 3], 'r--', lw=1.5, label='Pred')
    ax.fill_between(t_pred,
                    x_pred[:, 3] - 2*np.sqrt(P_pred[:, 3, 3]),
                    x_pred[:, 3] + 2*np.sqrt(P_pred[:, 3, 3]),
                    color='red', alpha=0.15)
    ax.set_ylabel('y [m]');  ax.legend(fontsize=7, ncol=3);  ax.grid(True, alpha=0.3)

    # -- (2) model probabilities --
    ax = axes[2]
    shade(ax)
    for i, (m, c) in enumerate(zip(imm.models, model_colors)):
        ax.plot(t, mu_hist[:, i], lw=2, color=c, label=f'P({m.name})')
    ax.set_ylabel('Model probability');  ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)

    # -- (3) bird's-eye trajectory --
    ax = axes[3]
    ax.plot(x_true, y_true,       'k-',  lw=2,   label='True path')
    ax.plot(meas_x, meas_y,       '.', color='black', ms=3, alpha=0.4, label='Measurements')
    ax.plot(est[:, 0], est[:, 3], 'b--', lw=1.5, label='IMM-KF')
    ax.plot(x_pred[:, 0], x_pred[:, 3], 'r--', lw=1.5, label='Prediction')
    for step, label, col in [(0,      'CA start', 'tab:red'),
                               (CA_END, 'CV start', 'tab:blue'),
                               (CV_END, 'CT start', 'tab:green')]:
        ax.plot(x_true[step], y_true[step], 'o', color=col, ms=9, zorder=5)
        ax.annotate(label, (x_true[step], y_true[step]),
                    xytext=(6, 6), textcoords='offset points',
                    fontsize=8, color=col, fontweight='bold')
    ax.set_xlabel('x [m]');  ax.set_ylabel('y [m]')
    ax.legend(fontsize=7, ncol=2);  ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    out = 'imm_kf_ca_cv_ct.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    run_demo()
