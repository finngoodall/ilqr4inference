from numpy.typing import NDArray
from typing import List, Tuple

import autograd.numpy as np

from lib.base import Gaussian
from lib.models.dynamics import LinearDynamics
from lib.models.measurements import LinearGaussianMeasurement
from lib.models.priors import GaussianPrior



class KalmanSmoother():
    """Class implementation of the Kalman smoother. The linear system dynamics
    are passed through `system`."""

    def __init__(
            self,
            dynamics: LinearDynamics,
            meas_model: LinearGaussianMeasurement,
            input_prior: GaussianPrior,
            x0_prior: GaussianPrior
        ) -> None:
        self.dynamics = dynamics
        self.meas_model = meas_model
        self.input_prior = input_prior
        self.x0_prior = x0_prior

        linear_model = (isinstance(dynamics, LinearDynamics)
                        and isinstance(meas_model, LinearGaussianMeasurement)
                        and isinstance(input_prior, GaussianPrior))
        if not linear_model:
            print("Warning: LQR can only infer over linear Gaussian generative models correctly.")

    def _A(self, t: int) -> NDArray:
        return self.dynamics.df_dx(
            np.zeros(self.dynamics.Nx),
            np.zeros(self.dynamics.Nu),
            t
        )

    def _B(self, t: int) -> NDArray:
        return self.dynamics.df_du(
            np.zeros(self.dynamics.Nx),
            np.zeros(self.dynamics.Nu),
            t
        )

    def _C(self, t: int) -> NDArray:
        return self.meas_model.dh_dx(np.zeros(self.dynamics.Nx), t)

    def _predict(self, x: Gaussian, t: int) -> Tuple[Gaussian]:
        """Kalman filter predict step. Predicts the next state and measurement
        from the current state `x`."""

        A = self._A(t)
        B = self._B(t)
        C = self._C(t)

        x_pred = Gaussian(
            A @ x.mean,
            A @ x.cov @ A.T + B @ self.input_prior.cov @ B.T
        )
        y_pred = Gaussian(
            C @ x_pred.mean,
            C @ x_pred.cov @ C.T + self.meas_model.cov
        )

        return x_pred, y_pred

    def _update(self, x_pred: Gaussian, y_pred: Gaussian, y: NDArray,
                t: int) -> Gaussian:
        """Kalman filter update step. Computes the best estimate for the state
        given the predicted state `x_pred`, predicted measurement `y_pred` and
        true measurement `y`."""
        
        C = self._C(t)

        K = x_pred.cov @ C.T @ np.linalg.inv(y_pred.cov)

        x = Gaussian(
            x_pred.mean + K @ (y - y_pred.mean),
            (np.eye(self.dynamics.Nx) - K @ C) @ x_pred.cov
        )

        return x

    def _smooth(self, x_s_next: Gaussian, x_f: Gaussian,
                t: int) -> Tuple[Gaussian]:
        """Evaluates the smoothed state and input estimates from the next smooth
        state `x_s_next` and filter state `x_f`."""

        A = self._A(t)
        B = self._B(t)
        B_pinv = np.linalg.pinv(B)

        # Smoother gain `G`
        pred_cov = A @ x_f.cov @ A.T + B @ self.input_prior.cov @ B.T
        G = x_f.cov @ A.T @ np.linalg.inv(pred_cov)

        x_s = Gaussian(
            x_f.mean + G @ (x_s_next.mean - A @ x_f.mean),
            x_f.cov + G @ (x_s_next.cov - pred_cov) @ G.T
        )
        u_s = Gaussian(
            B_pinv @ (x_s_next.mean - A @ x_s.mean),
            B_pinv @ (x_s_next.cov + A @ x_s.cov @ A.T - \
                A @ G @ x_s_next.cov - x_s_next.cov @ G.T @ A.T) @ B_pinv.T
        )

        return x_s, u_s

    def __call__(self, ys: List[NDArray]) -> Tuple[List[Gaussian]]:
        """Runs the smoother to estimate the latent states and control inputs
        from the observed measurements `ys` and the prior over the initial state
        `x1_prior`."""

        T = len(ys)

        filter_xs = []
        for t in range(T):
            if t == 0:
                C = self._C(0)
                x_pred = Gaussian(self.x0_prior.mean, self.x0_prior.cov)
                y_pred = Gaussian(
                    C @ self.x0_prior.mean,
                    C @ self.x0_prior.cov @ C.T + self.meas_model.cov
                )
            else:
                x_pred, y_pred = self._predict(x, t)
            
            x = self._update(x_pred, y_pred, ys[t], t)
            filter_xs.append(x)

        # Set final input to prior as it cannot be determined
        smooth_xs = [filter_xs[-1]]
        smooth_us = [Gaussian(self.input_prior.mean, self.input_prior.cov)]
        for t in range(T-2, -1, -1):
            x, u = self._smooth(x, filter_xs[t], t)
            smooth_xs.append(x)
            smooth_us.append(u)

        smooth_xs.reverse()
        smooth_us.reverse()

        return smooth_xs, smooth_us


