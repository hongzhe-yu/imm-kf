"""
Hyperparameters for the IMM-KF lane-change demo (imm_kf_lc.py).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
RANDOM_SEED   = 42
Ts            = 0.1         # [s] sample period
N_STEPS       = 200         # total simulation steps
LANE_WIDTH    = 3.6         # [m]
MANEUVER_STEP = 30          # step index at which both maneuvers begin
LC_DURATION   = 100         # lateral lane-change completes over this many steps

# ---------------------------------------------------------------------------
# Ground-truth scenario
# ---------------------------------------------------------------------------
LON_VEL_INIT  = 10.0        # [m/s] initial longitudinal speed
LON_ACCEL     = 2.0         # [m/s²] longitudinal acceleration after maneuver onset

# ---------------------------------------------------------------------------
# Measurement noise (std dev)
# ---------------------------------------------------------------------------
LON_OBS_NOISE = 0.5        # [m] or [m/s] — applied to both pos and vel
LAT_OBS_NOISE = 0.5        # [m]

# ---------------------------------------------------------------------------
# Longitudinal IMM-KF
# ---------------------------------------------------------------------------
SIGMA_A_CV    = 1.0         # [m/s²] acceleration noise std for CV model
SIGMA_J_CA    = 1.0        # [m/s³] jerk noise std for CA model

PI_LON        = np.array([[0.95, 0.05],
                           [0.05, 0.95]])   # mode-transition matrix (CV/CA)
MU0_LON       = np.array([0.5, 0.5])       # initial model probabilities
P0_LON        = 0.1 * np.diag([1.0, 1.0, 1.0])  # initial state covariance

# ---------------------------------------------------------------------------
# Lateral IMM-KF
# ---------------------------------------------------------------------------
# Controller gains for the PD lateral model.
# Stability (overdamped, no oscillation) requires K2 < K1**2 / 4.
# K2 also scales the E term — the ONLY thing that distinguishes the three
# models — so a larger K2 gives faster, cleaner probability discrimination.
K1_LAT        = 1.5    # damping gain
K2_LAT        = 0.5    # stiffness gain  (max overdamped: K1**2/4 = 0.5625)
K1_LAT2       = 1.0    # gentle LCL variant — slower settling
K2_LAT2       = 0.2    # weaker pull  (max overdamped: K1**2/4 = 0.25)
SIGMA_W_LAT   = 0.1    # [m/s²] lateral process noise std-dev
SIGMA_OBS_LAT_VEL = 0.1  # [m/s] lateral velocity observation noise (0 = position-only)

P_STAY_LAT       = 0.85    # probability of staying in the same lateral mode
P_SWITCH_LAT_LC  = 0.1  # transition prob between left-LC variants (LCL ↔ LCL-Gentle)
MU0_LAT       = np.array([0.5, 0.125, 0.125, 0.125, 0.125])  # initial model probs (LK/LCL/LCR/LCL-Gentle/LCR-Gentle)
P0_LAT        = np.diag([0.1, 0.1])                    # initial state covariance

# ---------------------------------------------------------------------------
# Lane-change completion detection
# ---------------------------------------------------------------------------
PROB_THRESH      = 0.55     # dominant model probability required
DIST_THRESH_FRAC = 0.95     # vehicle must be ≥ this fraction across the lane gap
VEL_THRESH       = 0.25     # [m/s] lateral velocity considered "settled"

# Post-completion mode probability reset (back toward LaneKeep)
MU_RESET_LAT  = np.array([0.8, 0.05, 0.05, 0.05, 0.05])

# ---------------------------------------------------------------------------
# Prediction horizon
# ---------------------------------------------------------------------------
PRED_STEPS    = 20          # steps (= PRED_STEPS * Ts seconds) ahead to predict
