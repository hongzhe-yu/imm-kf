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
from models import make_lateral_models, make_constant_velocity_model, make_constant_acceleration_model
from params_lc import (
    RANDOM_SEED, Ts, N_STEPS, LANE_WIDTH, MANEUVER_STEP, LC_DURATION,
    LON_VEL_INIT, LON_ACCEL,
    LON_OBS_NOISE, LAT_OBS_NOISE,
    SIGMA_A_CV, SIGMA_J_CA, PI_LON, MU0_LON, P0_LON,
    K1_LAT, K2_LAT, K1_LAT2, K2_LAT2, SIGMA_W_LAT, SIGMA_OBS_LAT_VEL,
    P_STAY_LAT, P_SWITCH_LAT_LC, MU0_LAT, P0_LAT,
    DIST_THRESH_FRAC, VEL_THRESH, MU_RESET_LAT,
    PRED_STEPS,
)


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

def run_demo(direction: str = "left"):
    """
    Joint longitudinal + lateral IMM-KF demo.

    Parameters
    ----------
    direction : "left" or "right" — which lane the vehicle changes into.
    """
    assert direction in ("left", "right"), "direction must be 'left' or 'right'"
    sign = 1 if direction == "left" else -1

    np.random.seed(RANDOM_SEED)

    t             = np.arange(N_STEPS) * Ts
    DIST_THRESH   = LANE_WIDTH * DIST_THRESH_FRAC

    # -----------------------------------------------------------------------
    # Ground truth
    # -----------------------------------------------------------------------

    # Longitudinal
    lon_pos = np.zeros(N_STEPS)
    lon_vel = np.zeros(N_STEPS)
    lon_vel[0] = LON_VEL_INIT
    for k in range(1, N_STEPS):
        a = 0.0 if k < MANEUVER_STEP else LON_ACCEL
        lon_vel[k] = lon_vel[k-1] + Ts * a
        lon_pos[k] = lon_pos[k-1] + Ts * lon_vel[k-1] + 0.5 * Ts**2 * a

    # Lateral (sigmoid lane change in the requested direction)
    lat_pos, lat_vel = simulate_lane_change(
        N_STEPS, Ts,
        change_start=MANEUVER_STEP,
        change_duration=LC_DURATION,
        lane_width=sign * LANE_WIDTH,
    )

    # -----------------------------------------------------------------------
    # Noisy measurements
    # -----------------------------------------------------------------------
    y_lon = np.column_stack([
        lon_pos + np.random.randn(N_STEPS) * LON_OBS_NOISE,
        lon_vel + np.random.randn(N_STEPS) * LON_OBS_NOISE,
    ])
    y_lat     = lat_pos + np.random.randn(N_STEPS) * LAT_OBS_NOISE      # position obs
    y_lat_vel = lat_vel + np.random.randn(N_STEPS) * SIGMA_OBS_LAT_VEL  # velocity obs

    # -----------------------------------------------------------------------
    # Build longitudinal IMM-KF  (CV and CA, both in 3-state space)
    # -----------------------------------------------------------------------
    # CA: factory produces a 3-state [pos, vel, accel] model observing [pos, vel]
    ca_model = make_constant_acceleration_model(Ts, sigma_j=SIGMA_J_CA, sigma_obs=LON_OBS_NOISE)

    # CV: factory produces a 2-state [pos, vel] model observing [pos].
    # Embed into 3-state space by padding F and Q with zeros for the unused
    # acceleration dimension, and expand H/R to observe [pos, vel] to match CA.
    _cv2 = make_constant_velocity_model(Ts, sigma_a=SIGMA_A_CV, sigma_obs=LON_OBS_NOISE)
    F_cv = np.zeros((3, 3));  F_cv[:2, :2] = _cv2.F   # accel row zeroed → vel held constant
    Q_cv = np.zeros((3, 3));  Q_cv[:2, :2] = _cv2.Q
    H_lon = np.array([[1, 0, 0], [0, 1, 0]])
    R_lon = LON_OBS_NOISE**2 * np.eye(2)
    cv_model = KalmanModel("CV", F_cv, H_lon, Q_cv, R_lon)

    imm_lon = IMMKF([cv_model, ca_model], PI_LON, mu0=MU0_LON)
    state_lon = imm_lon.init(
        x0=np.array([lon_pos[0], lon_vel[0], 0.0]),
        P0=P0_LON,
    )

    # -----------------------------------------------------------------------
    # Build lateral IMM-KF  (LaneKeep, LaneChangeLeft, LaneChangeRight)
    # -----------------------------------------------------------------------
    lat_models = make_lateral_models(
        Ts, K1=K1_LAT, K2=K2_LAT,
        sigma_w=SIGMA_W_LAT, sigma_obs=LAT_OBS_NOISE,
        sigma_obs_vel=SIGMA_OBS_LAT_VEL,
        extra_specs=[
            ("LCL-Gentle",  LANE_WIDTH, K1_LAT2, K2_LAT2),
            ("LCR-Gentle", -LANE_WIDTH, K1_LAT2, K2_LAT2),
        ],
    )

    # Transition matrix: within-direction transitions use a larger probability.
    # Models sharing the same direction keyword ("Left" / "Right") form a group.
    n_lat  = len(lat_models)
    groups = [[i for i, m in enumerate(lat_models) if kw in m.name]
              for kw in ("Left", "Right")]
    groups = [g for g in groups if len(g) > 1]   # keep only multi-member groups

    pi_lat = np.zeros((n_lat, n_lat))
    for i in range(n_lat):
        pi_lat[i, i] = P_STAY_LAT
        my_group = next((g for g in groups if i in g), None)
        if my_group:
            peers    = [j for j in my_group if j != i]
            others   = [j for j in range(n_lat) if j != i and j not in my_group]
            p_others = (1 - P_STAY_LAT - P_SWITCH_LAT_LC * len(peers)) / len(others)
            for j in peers:
                pi_lat[i, j] = P_SWITCH_LAT_LC
            for j in others:
                pi_lat[i, j] = p_others
        else:
            p_generic = (1 - P_STAY_LAT) / (n_lat - 1)
            for j in range(n_lat):
                if j != i:
                    pi_lat[i, j] = p_generic
    imm_lat = IMMKF(lat_models, pi_lat, mu0=MU0_LAT)
    state_lat = imm_lat.init(
        x0=np.array([lat_pos[0], lat_vel[0]]),
        P0=P0_LAT,
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

    for k in range(N_STEPS):
        state_lon = imm_lon.step(state_lon, y_lon[k])

        # Feed lane-relative measurement to the lateral filter
        e_y_meas = y_lat[k] - y_ref
        z_lat = np.array([e_y_meas, y_lat_vel[k]]) if SIGMA_OBS_LAT_VEL > 0 else np.array([e_y_meas])
        state_lat = imm_lat.step(state_lat, z_lat)

        e_y_est    = state_lat.x_est[0]
        e_ydot_est = state_lat.x_est[1]

        # ---- Lane-change completion detection --------------------------------
        # When LCL/LCR is dominant, the vehicle has nearly reached the new lane
        # centre, and lateral velocity has settled, shift the reference and
        # re-centre the filter so LaneKeep becomes the correct model again.
        if not lc_done:
            completed_left  = (e_y_est >  DIST_THRESH and abs(e_ydot_est) < VEL_THRESH)
            completed_right = (e_y_est < -DIST_THRESH and abs(e_ydot_est) < VEL_THRESH)
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
                    mu       = MU_RESET_LAT.copy(),
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
    # Predictions from final state
    # -----------------------------------------------------------------------
    x_pred_lon, P_pred_lon, best_lon = imm_lon.predict(state_lon, T=PRED_STEPS)
    x_pred_lat, P_pred_lat, best_lat = imm_lat.predict(state_lat, T=PRED_STEPS)
    # Predictions are in lane-relative coords; shift back to absolute
    x_pred_lat = x_pred_lat.copy()
    x_pred_lat[:, 0] += y_ref
    P_pred_lat = P_pred_lat.copy()  # variances are unaffected by the shift
    t_pred = t[-1] + np.arange(PRED_STEPS + 1) * Ts

    print(f"\n--- Final model probabilities ---")
    print(f"Longitudinal:  CV={mu_lon_hist[-1,0]:.3f}  CA={mu_lon_hist[-1,1]:.3f}")
    lat_prob_str = "  ".join(
        f"{m.name}={mu_lat_hist[-1, i]:.3f}" for i, m in enumerate(lat_models)
    )
    print(f"Lateral:       {lat_prob_str}")
    print(f"Most probable longitudinal model: {imm_lon.models[best_lon].name}")
    print(f"Most probable lateral model:      {imm_lat.models[best_lat].name}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    t_maneuver = MANEUVER_STEP * Ts

    fig, axes = plt.subplots(5, 1, figsize=(11, 15))
    fig.suptitle(
        f"IMM-KF: Joint Longitudinal + Lateral Estimation\n"
        f"Vehicle accelerates AND changes lane {direction} at t = {MANEUVER_STEP*Ts:.0f} s",
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
    target_lane = sign * LANE_WIDTH
    inner = min(0, target_lane); outer = max(0, target_lane)
    for y_boundary in [inner - LANE_WIDTH/2, inner + LANE_WIDTH/2, outer + sign * LANE_WIDTH/2]:
        ax.axhline(y_boundary, color='goldenrod', lw=1.2, ls='--', alpha=0.7)
    ax.axhline(0,           color='goldenrod', lw=0.7, ls=':', alpha=0.5,
               label='Lane centres / boundaries')
    ax.axhline(target_lane, color='goldenrod', lw=0.7, ls=':', alpha=0.5)

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
    colors_lat = ['steelblue', 'forestgreen', 'crimson', 'darkorange', 'mediumpurple']
    ax = axes[4]
    for i, (m, c) in enumerate(zip(lat_models, colors_lat)):
        ax.plot(t, mu_lat_hist[:, i], lw=2, color=c, label=f'P({m.name})')
    vline(ax)
    ax.set_ylabel('Lat. Model Prob.')
    ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    axes[4].set_xlabel('Time [s]')

    plt.tight_layout()
    fname = f"imm_kf_lc_{direction}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {fname}")


if __name__ == "__main__":
    run_demo("left")
    run_demo("right")
