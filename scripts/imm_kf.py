"""
Interacting Multiple Model Kalman Filter (IMM-KF)
==================================================
Implementation based on:
  [1] Lefkopoulos et al., "Interaction-Aware Motion Prediction for
      Autonomous Driving: A Multiple Model Kalman Filtering Scheme",
      IEEE RA-L 2021.
  [2] Carvalho et al., "Stochastic Predictive Control of Autonomous
      Vehicles in Uncertain Environments", AVEC 2014.

The IMM-KF runs M Kalman filters in parallel, each corresponding to
a different motion model (e.g. constant velocity, constant acceleration,
lane change). At every timestep it:
  1. Mixes filter states weighted by transition probabilities  (interaction)
  2. Runs each KF independently on its mixed initial condition  (filtering)
  3. Updates model probabilities using innovation likelihoods   (prob update)
  4. Fuses all filter outputs into one estimate                 (combination)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class KalmanModel:
    """
    One motion model for the IMM-KF.

    All matrices follow the linear (possibly time-varying) system:
        x_{k+1} = F @ x_k + E + w_k,   w_k ~ N(0, Q)
        y_k     = H @ x_k + v_k,        v_k ~ N(0, R)

    Attributes
    ----------
    name : str
        Human-readable label, e.g. "ConstantVelocity".
    F : (nx, nx) array
        State transition matrix.
    H : (ny, nx) array
        Observation matrix.
    Q : (nx, nx) array
        Process noise covariance.
    R : (ny, ny) array
        Measurement noise covariance.
    E : (nx,) array, optional
        Deterministic input offset (affine term). Defaults to zeros.
    """
    name: str
    F: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    E: Optional[np.ndarray] = None

    def __post_init__(self):
        nx = self.F.shape[0]
        if self.E is None:
            self.E = np.zeros(nx)
        # Validate shapes
        assert self.F.shape == (nx, nx), "F must be (nx, nx)"
        assert self.H.shape[1] == nx,    "H must be (ny, nx)"
        assert self.Q.shape == (nx, nx), "Q must be (nx, nx)"
        ny = self.H.shape[0]
        assert self.R.shape == (ny, ny), "R must be (ny, ny)"
        assert self.E.shape == (nx,),    "E must be (nx,)"


@dataclass
class IMMState:
    """
    Full state of the IMM-KF at one timestep.

    Attributes
    ----------
    x_est : (nx,) array
        Fused (combined) state estimate.
    P_est : (nx, nx) array
        Fused estimate covariance.
    mu : (M,) array
        Model probabilities (sum to 1).
    x_models : list of (nx,) arrays
        Per-model state estimates.
    P_models : list of (nx, nx) arrays
        Per-model covariance estimates.
    """
    x_est:    np.ndarray
    P_est:    np.ndarray
    mu:       np.ndarray
    x_models: List[np.ndarray]
    P_models: List[np.ndarray]


# ---------------------------------------------------------------------------
# Core IMM-KF class
# ---------------------------------------------------------------------------

class IMMKF:
    """
    Interacting Multiple Model Kalman Filter.

    Parameters
    ----------
    models : list of KalmanModel
        The M motion models to run in parallel.
    pi : (M, M) array
        Markov transition matrix. pi[i, j] = P(switch from i to j).
        Rows must sum to 1.
    mu0 : (M,) array, optional
        Initial model probabilities. Defaults to uniform.
    """

    def __init__(
        self,
        models: List[KalmanModel],
        pi: np.ndarray,
        mu0: Optional[np.ndarray] = None,
    ):
        self.models = models
        self.M = len(models)
        self.nx = models[0].F.shape[0]

        # Validate transition matrix
        pi = np.array(pi, dtype=float)
        assert pi.shape == (self.M, self.M), "pi must be (M, M)"
        assert np.allclose(pi.sum(axis=1), 1.0), "Each row of pi must sum to 1"
        self.pi = pi

        if mu0 is None:
            mu0 = np.ones(self.M) / self.M
        mu0 = np.array(mu0, dtype=float)
        assert np.isclose(mu0.sum(), 1.0), "mu0 must sum to 1"
        self.mu0 = mu0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
    ) -> IMMState:
        """
        Initialise the filter state.

        Parameters
        ----------
        x0 : (nx,) array  — initial state mean
        P0 : (nx, nx) array — initial state covariance

        Returns
        -------
        IMMState
        """
        x_models = [x0.copy() for _ in range(self.M)]
        P_models = [P0.copy() for _ in range(self.M)]
        return IMMState(
            x_est=x0.copy(),
            P_est=P0.copy(),
            mu=self.mu0.copy(),
            x_models=x_models,
            P_models=P_models,
        )

    def step(
        self,
        state: IMMState,
        y: np.ndarray,
    ) -> IMMState:
        """
        Run one full IMM-KF recursion given a new measurement.

        Implements equations (4)-(6) from Lefkopoulos et al. /
        Steps 1-4 from Carvalho et al.

        Parameters
        ----------
        state : IMMState  — filter state from previous timestep
        y     : (ny,) array — new measurement vector

        Returns
        -------
        IMMState — updated filter state
        """
        # ---- Step 1: Interaction (mixing) --------------------------------
        x_mixed, P_mixed = self._interaction(state)

        # ---- Step 2: Mode-conditioned filtering --------------------------
        x_upd, P_upd, innovations, S_list = self._filtering(x_mixed, P_mixed, y)

        # ---- Step 3: Model probability update ----------------------------
        mu_new = self._probability_update(state.mu, innovations, S_list)

        # ---- Step 4: Combination -----------------------------------------
        x_fused, P_fused = self._combination(x_upd, P_upd, mu_new)

        return IMMState(
            x_est=x_fused,
            P_est=P_fused,
            mu=mu_new,
            x_models=x_upd,
            P_models=P_upd,
        )

    def predict(
        self,
        state: IMMState,
        T: int,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Generate multi-step predictions using the most probable model.

        Parameters
        ----------
        state : IMMState — current filter state
        T     : int — number of steps to predict ahead

        Returns
        -------
        x_pred : (T+1, nx) array — predicted state means (index 0 = current)
        P_pred : (T+1, nx, nx) array — predicted covariances
        best_model_idx : int — index of the most probable model used
        """
        best_i = int(np.argmax(state.mu))
        m = self.models[best_i]

        x_pred = np.zeros((T + 1, self.nx))
        P_pred = np.zeros((T + 1, self.nx, self.nx))
        x_pred[0] = state.x_est
        P_pred[0] = state.P_est

        x = state.x_models[best_i].copy()
        P = state.P_models[best_i].copy()

        for t in range(1, T + 1):
            x = m.F @ x + m.E
            P = m.F @ P @ m.F.T + m.Q
            x_pred[t] = x
            P_pred[t] = P

        return x_pred, P_pred, best_i

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _interaction(
        self,
        state: IMMState,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Step 1 — Interaction / mixing.

        Equations (4a)-(4d) in Lefkopoulos et al.
        For each target model i, compute a mixed initial condition by
        weighting the estimates of all source models j by the conditional
        probability of having come from j given we're now in i.
        """
        mu = state.mu
        pi = self.pi
        M  = self.M

        # Predicted model probabilities: c^(i) = sum_j pi_{ji} mu^(j)
        # (probability of being in model i at next step)
        c = pi.T @ mu                          # shape (M,)
        c = np.where(c < 1e-12, 1e-12, c)     # numerical safety

        # Conditional mixing weights: mu^{j|i} = pi_{ji} mu^(j) / c^(i)
        # mu_mix[j, i] = probability of having come from j given now in i
        mu_mix = (pi.T * mu[np.newaxis, :]) / c[:, np.newaxis]  # (M, M)
        # mu_mix[i, j] = mu^{j|i}  (rows = target model, cols = source model)
        # Rewrite for clarity:
        mu_mix = mu_mix.T   # now mu_mix[j, i] = mu^{j|i}

        x_mixed = []
        P_mixed = []

        for i in range(M):
            # Mixed state estimate: x̄^(i) = sum_j mu^{j|i} x̂^(j)
            x_bar = sum(mu_mix[j, i] * state.x_models[j] for j in range(M))

            # Mixed covariance: P̄^(i) = sum_j mu^{j|i} [P^(j) + spread]
            P_bar = np.zeros((self.nx, self.nx))
            for j in range(M):
                diff = (x_bar - state.x_models[j])[:, np.newaxis]
                P_bar += mu_mix[j, i] * (state.P_models[j] + diff @ diff.T)

            x_mixed.append(x_bar)
            P_mixed.append(P_bar)

        return x_mixed, P_mixed

    def _filtering(
        self,
        x_mixed: List[np.ndarray],
        P_mixed: List[np.ndarray],
        y: np.ndarray,
    ) -> Tuple[List, List, List, List]:
        """
        Step 2 — Mode-conditioned Kalman filtering.

        Runs a standard KF predict+update for each model m^(i),
        starting from the mixed initial conditions.
        Returns updated estimates, innovations, and innovation covariances.
        """
        x_upd, P_upd, innovations, S_list = [], [], [], []

        for i, m in enumerate(self.models):
            x, P = x_mixed[i], P_mixed[i]

            # -- Predict --
            x_pred = m.F @ x + m.E
            P_pred = m.F @ P @ m.F.T + m.Q

            # -- Update --
            y_tilde = y - m.H @ x_pred               # innovation
            S = m.H @ P_pred @ m.H.T + m.R            # innovation covariance
            K = P_pred @ m.H.T @ np.linalg.inv(S)     # Kalman gain

            x_new = x_pred + K @ y_tilde
            P_new = (np.eye(self.nx) - K @ m.H) @ P_pred

            # Joseph form for numerical stability:
            # P_new = (I-KH)P(I-KH)^T + KRK^T
            IKH = np.eye(self.nx) - K @ m.H
            P_new = IKH @ P_pred @ IKH.T + K @ m.R @ K.T

            x_upd.append(x_new)
            P_upd.append(P_new)
            innovations.append(y_tilde)
            S_list.append(S)

        return x_upd, P_upd, innovations, S_list

    def _probability_update(
        self,
        mu_prev: np.ndarray,
        innovations: List[np.ndarray],
        S_list: List[np.ndarray],
    ) -> np.ndarray:
        """
        Step 3 — Model probability update.

        Equations (5a)-(5b) in Lefkopoulos et al. / Eq (10) in Carvalho et al.
        Each model's likelihood is the Gaussian PDF of its innovation.
        Bayes' rule then updates the model probabilities.
        """
        # Predicted mixing normaliser c^(i)
        c = self.pi.T @ mu_prev
        c = np.where(c < 1e-12, 1e-12, c)

        likelihoods = np.zeros(self.M)
        for i in range(self.M):
            likelihoods[i] = self._gaussian_pdf(innovations[i], S_list[i])

        # Unnormalised: c^(i) * L^(i)
        unnorm = c * likelihoods
        total  = unnorm.sum()
        if total < 1e-300:
            # All models have near-zero likelihood — keep previous probs
            return mu_prev.copy()

        return unnorm / total

    def _combination(
        self,
        x_upd: List[np.ndarray],
        P_upd: List[np.ndarray],
        mu: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 4 — Combination.

        Equations (6a)-(6b) in Lefkopoulos et al. / Eqs (11)-(12) in Carvalho.
        Fuse per-model estimates into a single Gaussian using model probs.
        """
        # Fused mean
        x_fused = sum(mu[i] * x_upd[i] for i in range(self.M))

        # Fused covariance (within-model + between-model spread)
        P_fused = np.zeros((self.nx, self.nx))
        for i in range(self.M):
            diff = (x_fused - x_upd[i])[:, np.newaxis]
            P_fused += mu[i] * (P_upd[i] + diff @ diff.T)

        return x_fused, P_fused

    @staticmethod
    def _gaussian_pdf(y: np.ndarray, S: np.ndarray) -> float:
        """
        Evaluate the multivariate Gaussian PDF N(y; 0, S).
        Used for the likelihood computation in Step 3.
        """
        ny = len(y)
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            return 1e-300
        exponent = -0.5 * y @ np.linalg.inv(S) @ y
        log_pdf  = -0.5 * (ny * np.log(2 * np.pi) + logdet) + exponent
        # Clamp to avoid underflow
        return float(np.exp(np.clip(log_pdf, -700, 0)))
