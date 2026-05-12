"""
IMM-KF Lane-Change Demo
=======================
Joint longitudinal + lateral IMM-KF demonstration using the lane-change
scenario from Carvalho et al., AVEC 2014.

Usage example at the bottom of this file.
"""

import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from imm_kf import KalmanModel, IMMState, IMMKF
from models import make_lateral_models


# ---------------------------------------------------------------------------
# Ground-truth lateral trajectory generator
# ---------------------------------------------------------------------------

def simulate_lane_change(
    N_steps: int,
    Ts: float,
    change_start: int,
    change_duration: int,
    lane_width: float = 3.6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a smooth lateral lane-change trajectory using a sigmoid.

    Returns
    -------
    ey_true  : (N_steps,) lateral position relative to original lane centre
    eydot_true : (N_steps,) lateral velocity
    """
    ey   = np.zeros(N_steps)
    eyd  = np.zeros(N_steps)

    for k in range(N_steps):
        # Sigmoid scaled to [0, lane_width] over change_duration steps
        tau = (k - change_start) / change_duration
        if k < change_start:
            ey[k]  = 0.0
            eyd[k] = 0.0
        else:
            sigmoid  = 1.0 / (1.0 + np.exp(-10.0 * (tau - 0.5)))
            sigmoid0 = 1.0 / (1.0 + np.exp(-10.0 * (0.0  - 0.5)))
            sigmoid1 = 1.0 / (1.0 + np.exp(-10.0 * (1.0  - 0.5)))
            # Normalise so it goes exactly 0 → lane_width
            ey[k] = lane_width * (sigmoid - sigmoid0) / (sigmoid1 - sigmoid0)
            # Derivative via finite difference
            if k > 0:
                eyd[k] = (ey[k] - ey[k-1]) / Ts
    return ey, eyd


# ---------------------------------------------------------------------------
# Demonstration / usage example
# ---------------------------------------------------------------------------

def run_demo():
    """
    Joint longitudinal + lateral IMM-KF demo.

    Scenario
    --------
    A vehicle drives straight at 10 m/s for 3 s, then simultaneously:
      - longitudinally: accelerates at 2 m/s² (CV → CA maneuver)
      - laterally:      performs a left lane change over ~2 s  (LK → LCL)

    Two decoupled IMM-KFs run in parallel — exactly as in both papers.

    Longitudinal IMM-KF  (2 models):  CV, CA
    Lateral IMM-KF       (3 models):  LaneKeep, LaneChangeLeft, LaneChangeRight

    We plot:
      Row 1 — longitudinal position
      Row 2 — longitudinal velocity
      Row 3 — lateral position (with lane boundaries)
      Row 4 — longitudinal model probabilities
      Row 5 — lateral model probabilities
      Row 6 — bird's-eye vehicle trajectory
    """
    np.random.seed(42)

    Ts            = 0.1
    N_steps       = 80
    t             = np.arange(N_steps) * Ts
    LANE_WIDTH    = 3.6
    maneuver_step = 30          # index at which both maneuvers begin
    lc_duration   = 20          # lateral lane change completes over 20 steps

    # -----------------------------------------------------------------------
    # Ground truth
    # -----------------------------------------------------------------------

    # Longitudinal
    lon_pos = np.zeros(N_steps)
    lon_vel = np.zeros(N_steps)
    lon_vel[0] = 10.0
    for k in range(1, N_steps):
        a = 0.0 if k < maneuver_step else 2.0
        lon_vel[k] = lon_vel[k-1] + Ts * a
        lon_pos[k] = lon_pos[k-1] + Ts * lon_vel[k-1] + 0.5 * Ts**2 * a

    # Lateral (sigmoid lane change to the left)
    lat_pos, lat_vel = simulate_lane_change(
        N_steps, Ts,
        change_start=maneuver_step,
        change_duration=lc_duration,
        lane_width=LANE_WIDTH,
    )

    # -----------------------------------------------------------------------
    # Noisy measurements
    # -----------------------------------------------------------------------
    lon_obs_noise = 0.3
    lat_obs_noise = 0.15

    y_lon = np.column_stack([
        lon_pos + np.random.randn(N_steps) * lon_obs_noise,
        lon_vel + np.random.randn(N_steps) * lon_obs_noise,
    ])
    y_lat = lat_pos + np.random.randn(N_steps) * lat_obs_noise   # shape (N,)

    # -----------------------------------------------------------------------
    # Build longitudinal IMM-KF  (CV and CA embedded in 3-state space)
    # -----------------------------------------------------------------------
    F_cv = np.array([[1, Ts, 0   ],
                     [0,  1, 0   ],
                     [0,  0, 0   ]])
    H_lon = np.array([[1, 0, 0],
                      [0, 1, 0]])
    Q_cv  = np.diag([0.25*Ts**4, Ts**2, 4.0])
    R_lon = lon_obs_noise**2 * np.eye(2)
    cv_model = KalmanModel("CV", F_cv, H_lon, Q_cv, R_lon)

    F_ca = np.array([[1, Ts, 0.5*Ts**2],
                     [0,  1,        Ts],
                     [0,  0,         1]])
    Q_ca = np.diag([Ts**4/36, Ts**2/4, 1.0])
    ca_model = KalmanModel("CA", F_ca, H_lon, Q_ca, R_lon)

    pi_lon = np.array([[0.95, 0.05],
                       [0.05, 0.95]])
    imm_lon = IMMKF([cv_model, ca_model], pi_lon, mu0=np.array([0.5, 0.5]))
    state_lon = imm_lon.init(
        x0=np.array([lon_pos[0], lon_vel[0], 0.0]),
        P0=np.diag([1.0, 1.0, 1.0]),
    )

    # -----------------------------------------------------------------------
    # Build lateral IMM-KF  (LaneKeep, LaneChangeLeft, LaneChangeRight)
    # -----------------------------------------------------------------------
    lat_models = make_lateral_models(Ts)

    # Transition matrix: lane changes are rare; once started, likely to persist
    p_stay   = 0.9
    p_switch = (1 - p_stay) / 2
    pi_lat = np.array([
        [p_stay,   p_switch, p_switch],
        [p_switch, p_stay,   p_switch],
        [p_switch, p_switch, p_stay  ],
    ])
    imm_lat = IMMKF(lat_models, pi_lat, mu0=np.array([0.8, 0.1, 0.1]))
    state_lat = imm_lat.init(
        x0=np.array([lat_pos[0], lat_vel[0]]),
        P0=np.diag([0.1, 0.1]),
    )

    # -----------------------------------------------------------------------
    # Run both filters
    # -----------------------------------------------------------------------
    lon_est    = []
    lat_est    = []          # stored in absolute coordinates for plotting
    mu_lon_hist = []
    mu_lat_hist = []

    # Lane reference tracker — the IMM-KF always works in lane-relative
    # coordinates (e_y = y_absolute - y_ref).  When a lane change is detected
    # as complete we shift y_ref and re-centre the filter state.
    y_ref = 0.0
    lc_done = False  # guard against re-triggering on the same maneuver

    # Completion thresholds
    PROB_THRESH = 0.55          # dominant model probability
    DIST_THRESH = LANE_WIDTH * 0.75  # vehicle is ≥75 % across the lane gap
    VEL_THRESH  = 0.25          # lateral velocity nearly zero [m/s]

    for k in range(N_steps):
        state_lon = imm_lon.step(state_lon, y_lon[k])

        # Feed lane-relative measurement to the lateral filter
        e_y_meas = y_lat[k] - y_ref
        state_lat = imm_lat.step(state_lat, np.array([e_y_meas]))

        mu         = state_lat.mu
        e_y_est    = state_lat.x_est[0]
        e_ydot_est = state_lat.x_est[1]

        # ---- Lane-change completion detection --------------------------------
        # When LCL/LCR is dominant, the vehicle has nearly reached the new lane
        # centre, and lateral velocity has settled, shift the reference and
        # re-centre the filter so LaneKeep becomes the correct model again.
        if not lc_done:
            completed_left  = (mu[1] > PROB_THRESH
                               and e_y_est  >  DIST_THRESH
                               and abs(e_ydot_est) < VEL_THRESH)
            completed_right = (mu[2] > PROB_THRESH
                               and e_y_est  < -DIST_THRESH
                               and abs(e_ydot_est) < VEL_THRESH)
            if completed_left or completed_right:
                delta = LANE_WIDTH if completed_left else -LANE_WIDTH
                y_ref += delta
                shift = np.array([delta, 0.0])
                # Re-express all filter states in the new lane-relative frame
                new_x_models = [x - shift for x in state_lat.x_models]
                new_x_est    = state_lat.x_est - shift
                state_lat = IMMState(
                    x_est    = new_x_est,
                    P_est    = state_lat.P_est,
                    mu       = np.array([0.7, 0.15, 0.15]),  # reset toward LK
                    x_models = new_x_models,
                    P_models = state_lat.P_models,
                )
                lc_done = True  # one lane change per demo
        # ----------------------------------------------------------------------

        lon_est.append(state_lon.x_est.copy())
        # Convert back to absolute coordinates for recording / plotting
        lat_est.append(state_lat.x_est + np.array([y_ref, 0.0]))
        mu_lon_hist.append(state_lon.mu.copy())
        mu_lat_hist.append(state_lat.mu.copy())

    lon_est     = np.array(lon_est)
    lat_est     = np.array(lat_est)
    mu_lon_hist = np.array(mu_lon_hist)
    mu_lat_hist = np.array(mu_lat_hist)

    # -----------------------------------------------------------------------
    # Predictions from final state (20 steps = 2 s ahead)
    # -----------------------------------------------------------------------
    PRED_STEPS = 20
    x_pred_lon, P_pred_lon, best_lon = imm_lon.predict(state_lon, T=PRED_STEPS)
    x_pred_lat, P_pred_lat, best_lat = imm_lat.predict(state_lat, T=PRED_STEPS)
    # Predictions are in lane-relative coords; shift back to absolute
    x_pred_lat = x_pred_lat.copy()
    x_pred_lat[:, 0] += y_ref
    P_pred_lat = P_pred_lat.copy()  # variances are unaffected by the shift
    t_pred = t[-1] + np.arange(PRED_STEPS + 1) * Ts

    print(f"\n--- Final model probabilities ---")
    print(f"Longitudinal:  CV={mu_lon_hist[-1,0]:.3f}  CA={mu_lon_hist[-1,1]:.3f}")
    print(f"Lateral:       LK={mu_lat_hist[-1,0]:.3f}  "
          f"LCL={mu_lat_hist[-1,1]:.3f}  LCR={mu_lat_hist[-1,2]:.3f}")
    print(f"Most probable longitudinal model: {imm_lon.models[best_lon].name}")
    print(f"Most probable lateral model:      {imm_lat.models[best_lat].name}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    t_maneuver = maneuver_step * Ts

    fig, axes = plt.subplots(6, 1, figsize=(11, 18))
    fig.suptitle(
        "IMM-KF: Joint Longitudinal + Lateral Estimation\n"
        "Vehicle accelerates AND changes lane left at t = 3 s",
        fontsize=13, fontweight='bold',
    )

    def vline(ax):
        ax.axvline(t_maneuver, color='orange', ls=':', lw=1.5,
                   label='Maneuver onset')

    # -- (0) Longitudinal position --
    ax = axes[0]
    ax.plot(t, lon_pos, 'k-', lw=2, label='True position')
    ax.plot(t, y_lon[:, 0], '.', color='gray', ms=3, alpha=0.5, label='Measurement')
    ax.plot(t, lon_est[:, 0], 'b--', lw=1.5, label='IMM-KF estimate')
    ax.plot(t_pred, x_pred_lon[:, 0], 'r--', lw=1.5, label='Prediction')
    ax.fill_between(t_pred,
                    x_pred_lon[:, 0] - 2*np.sqrt(P_pred_lon[:, 0, 0]),
                    x_pred_lon[:, 0] + 2*np.sqrt(P_pred_lon[:, 0, 0]),
                    color='red', alpha=0.15, label='±2σ')
    vline(ax)
    ax.set_ylabel('Lon. Position [m]')
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

    # -- (1) Longitudinal velocity --
    ax = axes[1]
    ax.plot(t, lon_vel, 'k-', lw=2, label='True velocity')
    ax.plot(t, y_lon[:, 1], '.', color='gray', ms=3, alpha=0.5, label='Measurement')
    ax.plot(t, lon_est[:, 1], 'b--', lw=1.5, label='IMM-KF estimate')
    ax.plot(t_pred, x_pred_lon[:, 1], 'r--', lw=1.5, label='Prediction')
    ax.fill_between(t_pred,
                    x_pred_lon[:, 1] - 2*np.sqrt(P_pred_lon[:, 1, 1]),
                    x_pred_lon[:, 1] + 2*np.sqrt(P_pred_lon[:, 1, 1]),
                    color='red', alpha=0.15)
    vline(ax)
    ax.set_ylabel('Lon. Velocity [m/s]')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # -- (2) Lateral position with lane markings --
    ax = axes[2]
    # Lane boundaries (original lane: 0 ± LANE_WIDTH/2; left lane: LANE_WIDTH ± LW/2)
    for y_boundary in [-LANE_WIDTH/2, LANE_WIDTH/2, 3*LANE_WIDTH/2]:
        ax.axhline(y_boundary, color='goldenrod', lw=1.2, ls='--', alpha=0.7)
    ax.axhline(0,          color='goldenrod', lw=0.7, ls=':',  alpha=0.5,
               label='Lane centres / boundaries')
    ax.axhline(LANE_WIDTH, color='goldenrod', lw=0.7, ls=':',  alpha=0.5)

    ax.plot(t, lat_pos, 'k-', lw=2, label='True lateral pos')
    ax.plot(t, y_lat,   '.', color='gray', ms=3, alpha=0.5, label='Measurement')
    ax.plot(t, lat_est[:, 0], 'b--', lw=1.5, label='IMM-KF estimate')
    ax.plot(t_pred, x_pred_lat[:, 0], 'r--', lw=1.5, label='Prediction')
    ax.fill_between(t_pred,
                    x_pred_lat[:, 0] - 2*np.sqrt(P_pred_lat[:, 0, 0]),
                    x_pred_lat[:, 0] + 2*np.sqrt(P_pred_lat[:, 0, 0]),
                    color='red', alpha=0.15, label='±2σ')
    vline(ax)
    ax.set_ylabel('Lateral Position [m]')
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

    # -- (3) Longitudinal model probabilities --
    ax = axes[3]
    ax.plot(t, mu_lon_hist[:, 0], 'b-',  lw=2, label='P(CV)')
    ax.plot(t, mu_lon_hist[:, 1], 'r-',  lw=2, label='P(CA)')
    vline(ax)
    ax.set_ylabel('Lon. Model Prob.')
    ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # -- (4) Lateral model probabilities --
    colors_lat = ['steelblue', 'forestgreen', 'crimson']
    ax = axes[4]
    for i, (m, c) in enumerate(zip(lat_models, colors_lat)):
        ax.plot(t, mu_lat_hist[:, i], lw=2, color=c, label=f'P({m.name})')
    vline(ax)
    ax.set_ylabel('Lat. Model Prob.')
    ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # -- (5) Bird's-eye view trajectory --
    ax = axes[5]
    # Lane fill
    ax.axhspan(-LANE_WIDTH/2,   LANE_WIDTH/2,   color='lightgray', alpha=0.4,
               label='Original lane')
    ax.axhspan( LANE_WIDTH/2, 3*LANE_WIDTH/2,   color='lightblue', alpha=0.4,
               label='Left lane')
    for y_boundary in [-LANE_WIDTH/2, LANE_WIDTH/2, 3*LANE_WIDTH/2]:
        ax.axhline(y_boundary, color='goldenrod', lw=1.5, ls='--')

    ax.plot(lon_pos, lat_pos,     'k-',  lw=2,   label='True path')
    ax.plot(lon_est[:, 0], lat_est[:, 0], 'b--', lw=1.5, label='IMM-KF estimate')
    ax.plot(x_pred_lon[:, 0], x_pred_lat[:, 0], 'r--', lw=1.5, label='Prediction')

    # Mark maneuver start
    ax.plot(lon_pos[maneuver_step], lat_pos[maneuver_step],
            'o', color='orange', ms=8, zorder=5, label='Maneuver onset')

    ax.set_xlabel('Longitudinal Position [m]')
    ax.set_ylabel('Lateral Position [m]')
    ax.set_ylim([-LANE_WIDTH, 2 * LANE_WIDTH])
    ax.legend(fontsize=7, ncol=3, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('auto')

    plt.tight_layout()
    plt.savefig('imm_kf_lc.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved to imm_kf_lc.png")


if __name__ == "__main__":
    run_demo()
