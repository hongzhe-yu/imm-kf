"""
Pre-built KalmanModel factories for common motion models.
"""

import math
import numpy as np
from typing import List

from imm_kf import KalmanModel


def make_constant_velocity_model(
    Ts: float,
    sigma_a: float = 0.5,
    sigma_obs: float = 1.0,
) -> KalmanModel:
    """
    Constant-velocity model (longitudinal).
    State:  [position, velocity]
    Obs:    [position]

    sigma_a   : std-dev of acceleration noise (process noise)
    sigma_obs : std-dev of position measurement noise
    """
    F = np.array([[1, Ts],
                  [0,  1]])
    H = np.array([[1, 0]])
    # Discrete process noise from continuous white-noise acceleration
    Q = sigma_a**2 * np.array([[Ts**4/4, Ts**3/2],
                                [Ts**3/2, Ts**2  ]])
    R = np.array([[sigma_obs**2]])
    return KalmanModel(name="ConstantVelocity", F=F, H=H, Q=Q, R=R)


def make_constant_acceleration_model(
    Ts: float,
    sigma_j: float = 1.0,
    sigma_obs: float = 1.0,
) -> KalmanModel:
    """
    Constant-acceleration model (longitudinal).
    State:  [position, velocity, acceleration]
    Obs:    [position, velocity]

    sigma_j   : std-dev of jerk noise (process noise)
    sigma_obs : std-dev of measurement noise
    """
    F = np.array([[1, Ts, Ts**2/2],
                  [0,  1,      Ts],
                  [0,  0,       1]])
    H = np.array([[1, 0, 0],
                  [0, 1, 0]])
    Q = sigma_j**2 * np.array([
        [Ts**6/36, Ts**5/12, Ts**4/6],
        [Ts**5/12, Ts**4/4,  Ts**3/2],
        [Ts**4/6,  Ts**3/2,  Ts**2  ],
    ])
    R = sigma_obs**2 * np.eye(2)
    return KalmanModel(name="ConstantAcceleration", F=F, H=H, Q=Q, R=R)


def make_lane_keeping_model(
    Ts: float,
    K1: float = 1.5,
    K2: float = 2.0,
    p_ref: float = 0.0,
    sigma_w: float = 0.1,
    sigma_obs: float = 0.2,
) -> KalmanModel:
    """
    Lateral lane-keeping model as in Carvalho et al. Section 3.1.2.
    State:  [lateral error e_y, lateral error rate ė_y]
    Obs:    [e_y]

    The vehicle tracks lateral position p_ref using a PD-like controller.
    K1, K2 : feedback gains (affect settling time / damping)
    p_ref  : reference lane lateral position
    """
    F = np.array([[1,         Ts        ],
                  [-Ts * K2, 1 - Ts * K1]])
    H = np.array([[1, 0]])
    E = np.array([0, Ts * K2 * p_ref])
    Q = sigma_w**2 * np.array([[Ts**4/4, Ts**3/2],
                                [Ts**3/2, Ts**2  ]])
    R = np.array([[sigma_obs**2]])
    return KalmanModel(name=f"LaneKeeping(ref={p_ref:.1f})", F=F, H=H, Q=Q, R=R, E=E)


def make_lateral_models(
    Ts: float,
    K1: float = 1.5,
    K2: float = 0.5,
    sigma_w: float = 0.15,
    sigma_obs: float = 0.15,
    sigma_obs_vel: float = 0.0,
    extra_specs: list = None,   # [(name, p_ref, K1, K2), ...]
) -> List[KalmanModel]:
    """
    Build the three lateral IMM-KF models from Carvalho et al. Section 3.1.2.

    State:  [e_y, ė_y]  — lateral error and its rate relative to the CURRENT
                          lane centre (y_ref), NOT absolute lateral position.
    Obs:    [e_y]        — lateral error only (sigma_obs_vel == 0)
         or [e_y, ė_y]  — lateral error + rate (sigma_obs_vel > 0)

    Three models, all sharing the same second-order feedback dynamics but with
    different reference offsets (affine term E):

      m0 — LaneKeep        : p_ref =  0.0         → pulls e_y toward 0
      m1 — LaneChangeLeft  : p_ref = +LANE_WIDTH  → pulls e_y toward +LANE_WIDTH
      m2 — LaneChangeRight : p_ref = -LANE_WIDTH  → pulls e_y toward -LANE_WIDTH

    Parameters
    ----------
    K1 : damping gain — higher = faster settling, less overshoot.
    K2 : stiffness gain — higher = stronger pull toward p_ref = stronger
         model discrimination.  Must satisfy K2 < K1²/4 for overdamped
         (non-oscillatory) closed-loop eigenvalues.
    sigma_w       : process noise std-dev [m/s²].
    sigma_obs     : lateral position observation noise std-dev [m].
    sigma_obs_vel : lateral velocity observation noise std-dev [m/s].
                    When > 0, ė_y is added to the observation vector.
                    E = [0, Ts·K2·p_ref] then has H@E = [0, Ts·K2·p_ref],
                    giving direct per-step model discrimination (no delay).
    """
    LANE_WIDTH = 3.6   # metres — standard highway lane

    if sigma_obs_vel > 0:
        H = np.array([[1.0, 0.0],
                      [0.0, 1.0]])
        R = np.diag([sigma_obs**2, sigma_obs_vel**2])
    else:
        H = np.array([[1.0, 0.0]])
        R = np.array([[sigma_obs**2]])

    def _make(name, p_ref, k1, k2):
        F = np.array([[1.0,       Ts       ],
                      [-Ts * k2, 1 - Ts*k1]])
        E = np.array([0.0, Ts * k2 * p_ref])
        Q = sigma_w**2 * np.array([[Ts**4 / 4, Ts**3 / 2],
                                    [Ts**3 / 2, Ts**2    ]])
        return KalmanModel(name=name, F=F, H=H, Q=Q, R=R, E=E)

    models = [
        _make("LaneKeep",        0.0,        K1, K2),
        _make("LaneChangeLeft",  +LANE_WIDTH, K1, K2),
        _make("LaneChangeRight", -LANE_WIDTH, K1, K2),
    ]
    for name, p_ref, k1, k2 in (extra_specs or []):
        models.append(_make(name, p_ref, k1, k2))
    return models


def make_constant_turn_model(
    Ts: float,
    turn_rate_deg: float = 15.0,
    sigma_process: float = 1.0,
    sigma_obs: float = math.sqrt(150),
) -> KalmanModel:
    """
    Constant turn-rate model.
    State:  [x, vx, y, vy, omega]  — 2-D position, velocity, and turn rate
    Obs:    [x, vx, y, vy]

    turn_rate_deg : nominal turn rate [deg/s]
    sigma_process : std-dev of process noise (isotropic across all 5 states)
    sigma_obs     : std-dev of measurement noise (isotropic across 4 obs)
    """
    omega = math.pi / 180.0 * turn_rate_deg   # rad/s
    theta = omega * Ts                          # angle turned in one timestep

    F = np.array([
        [1., math.sin(theta) / omega,          0., -(1. - math.cos(theta)) / omega, 0.],
        [0., math.cos(theta),                  0., -math.sin(theta),                0.],
        [0., (1. - math.cos(theta)) / omega,   1.,  math.sin(theta) / omega,        0.],
        [0., math.sin(theta),                  0.,  math.cos(theta),                0.],
        [0., 0.,                               0.,  0.,                             1.],
    ])
    H = np.array([
        [1., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0.],
        [0., 0., 1., 0., 0.],
        [0., 0., 0., 1., 0.],
    ])
    Q = sigma_process**2 * np.eye(5)
    R = sigma_obs**2 * np.eye(4)
    return KalmanModel(name=f"ConstantTurn({turn_rate_deg:.1f}deg/s)", F=F, H=H, Q=Q, R=R)


# ---------------------------------------------------------------------------
# 2D motion models — shared 6D state [x, vx, ax, y, vy, ω]
#
# All three models use the same H (observe [x, vx, y, vy]) and the same R,
# so they can run together inside one IMM-KF.
#
# CV : ax → 0 each step, ω → 0 each step  (straight, constant speed)
# CA : ax propagated as nearly-constant, ω → 0  (straight, accelerating)
# CT : exact constant-turn-rate kinematics, ax → 0  (turning at fixed ω)
# ---------------------------------------------------------------------------

def _2d_obs(sigma_obs_pos: float, sigma_obs_vel: float):
    """Shared H and R for the 6D model suite."""
    H = np.array([
        [1., 0., 0., 0., 0., 0.],   # x
        [0., 1., 0., 0., 0., 0.],   # vx
        [0., 0., 0., 1., 0., 0.],   # y
        [0., 0., 0., 0., 1., 0.],   # vy
    ])
    R = np.diag([sigma_obs_pos**2, sigma_obs_vel**2,
                 sigma_obs_pos**2, sigma_obs_vel**2])
    return H, R


def make_2d_cv_model(
    Ts: float,
    sigma_cv: float = 0.5,
    sigma_obs_pos: float = 0.1,
    sigma_obs_vel: float = 0.5,
) -> KalmanModel:
    """
    2D constant-velocity model.
    State:  [x, vx, ax, y, vy, ω]
    Obs:    [x, vx, y, vy]

    ax and ω are zeroed each step — they play no role in CV dynamics.
    """
    F = np.array([
        [1., Ts, 0., 0.,  0., 0.],
        [0.,  1., 0., 0.,  0., 0.],
        [0.,  0., 0., 0.,  0., 0.],   # ax → 0
        [0.,  0., 0., 1., Ts, 0.],
        [0.,  0., 0., 0.,  1., 0.],
        [0.,  0., 0., 0.,  0., 0.],   # ω → 0
    ])
    Q = np.diag([sigma_cv**2 * Ts**4/4, sigma_cv**2 * Ts**2, 0.,
                 sigma_cv**2 * Ts**4/4, sigma_cv**2 * Ts**2, 0.])
    H, R = _2d_obs(sigma_obs_pos, sigma_obs_vel)
    return KalmanModel(name="CV", F=F, H=H, Q=Q, R=R)


def make_2d_ca_model(
    Ts: float,
    sigma_j: float = 3.0,
    sigma_cv: float = 0.5,
    sigma_obs_pos: float = 0.1,
    sigma_obs_vel: float = 0.5,
) -> KalmanModel:
    """
    2D constant-acceleration model (CA in x, CV in y).
    State:  [x, vx, ax, y, vy, ω]
    Obs:    [x, vx, y, vy]

    ax is propagated as a nearly-constant state (jerk noise sigma_j).
    The y-axis uses CV dynamics (no lateral acceleration in this mode).
    ω is zeroed each step.
    """
    F = np.array([
        [1., Ts, 0.5*Ts**2, 0.,  0., 0.],
        [0.,  1.,        Ts, 0.,  0., 0.],
        [0.,  0.,         1., 0.,  0., 0.],   # ax constant
        [0.,  0.,         0., 1., Ts, 0.],
        [0.,  0.,         0., 0.,  1., 0.],
        [0.,  0.,         0., 0.,  0., 0.],   # ω → 0
    ])
    Q = np.diag([
        sigma_j**2 * Ts**6/36, sigma_j**2 * Ts**4/4, sigma_j**2 * Ts**2,
        sigma_cv**2 * Ts**4/4, sigma_cv**2 * Ts**2,  0.,
    ])
    H, R = _2d_obs(sigma_obs_pos, sigma_obs_vel)
    return KalmanModel(name="CA", F=F, H=H, Q=Q, R=R)


def make_2d_ct_model(
    Ts: float,
    turn_rate_deg: float = 15.0,
    sigma_pos: float = 0.01,
    sigma_vel: float = 0.02,
    sigma_obs_pos: float = 0.1,
    sigma_obs_vel: float = 0.5,
) -> KalmanModel:
    """
    2D constant-turn-rate model with exact CTR kinematics.
    State:  [x, vx, ax, y, vy, ω]
    Obs:    [x, vx, y, vy]

    ax is zeroed each step (no longitudinal acceleration during the turn).
    ω is propagated as a constant state.

    sigma_pos / sigma_vel: small process noise reflecting that a CT model
    closely follows the arc — large values let the model track arbitrary
    trajectories and destroy mode discrimination.
    """
    omega = math.pi / 180.0 * turn_rate_deg
    theta = omega * Ts
    s, c = math.sin(theta), math.cos(theta)
    c1 = 1. - c

    # Columns: x(0), vx(1), ax(2), y(3), vy(4), ω(5)
    F = np.array([
        [1., s/omega,  0., 0., -c1/omega, 0.],   # x
        [0., c,        0., 0., -s,        0.],   # vx
        [0., 0.,       0., 0.,  0.,       0.],   # ax → 0
        [0., c1/omega, 0., 1.,  s/omega,  0.],   # y
        [0., s,        0., 0.,  c,        0.],   # vy
        [0., 0.,       0., 0.,  0.,       1.],   # ω constant
    ])
    Q = np.diag([sigma_pos**2, sigma_vel**2, 0.,
                 sigma_pos**2, sigma_vel**2, 1e-4])
    H, R = _2d_obs(sigma_obs_pos, sigma_obs_vel)
    return KalmanModel(name=f"CT({turn_rate_deg:.0f}°/s)", F=F, H=H, Q=Q, R=R)
