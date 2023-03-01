from numpy.typing import NDArray
from typing import List, Tuple

from autograd import jacobian
import autograd.numpy as np
from functools import cached_property

from lib.base import Gaussian, Dynamics, InputPrior, MeasurementModel



class KalmanSmoother():
    """Class implementation of the Kalman smoother. The linear system dynamics
    are passed through `system`."""

    def __init__(self, A: NDArray, B: NDArray, C: NDArray, u_cov: NDArray,
                 y_cov: NDArray):
        self.A = A
        self.B = B
        self.C = C
        self.u_cov = u_cov
        self.y_cov = y_cov
        self.Nx, self.Nu = self.B.shape
        self.Ny = self.C.shape[0]

    def _predict(self, x: Gaussian) -> Tuple[Gaussian]:
        """Kalman filter predict step. Predicts the next state and measurement
        from the current state `x`."""

        x_pred = Gaussian(
            self.A @ x.mean,
            self.A @ x.cov @ self.A.T + self.B @ self.u_cov @ self.B.T
        )
        y_pred = Gaussian(
            self.C @ x_pred.mean,
            self.C @ x_pred.cov @ self.C.T + self.y_cov
        )

        return x_pred, y_pred

    def _update(self, x_pred: Gaussian, y_pred: Gaussian,
                y: NDArray) -> Gaussian:
        """Kalman filter update step. Computes the best estimate for the state
        given the predicted state `x_pred`, predicted measurement `y_pred` and
        true measurement `y`."""
        
        # Kalman gain
        K = x_pred.cov @ self.C.T @ np.linalg.inv(y_pred.cov)

        x = Gaussian(
            x_pred.mean + K @ (y - y_pred.mean),
            (np.eye(self.Nx) - K @ self.C) @ x_pred.cov
        )

        return x

    def _smooth(self, x_s_next: Gaussian, x_f: Gaussian) -> Tuple[Gaussian]:
        """Evaluates the smoothed state and input estimates from the next smooth
        state `x_s_next` and filter state `x_f`."""

        B_pinv = np.linalg.pinv(self.B)

        # Smoother gain `G`
        pred_cov = self.A @ x_f.cov @ self.A.T + self.B @ self.u_cov @ self.B.T
        G = x_f.cov @ self.A.T @ np.linalg.inv(pred_cov)

        x_s = Gaussian(
            x_f.mean + G @ (x_s_next.mean - self.A @ x_f.mean),
            x_f.cov + G @ (x_s_next.cov - pred_cov) @ G.T
        )
        u_s = Gaussian(
            B_pinv @ (x_s_next.mean - self.A @ x_s.mean),
            B_pinv @ (x_s_next.cov + self.A @ x_s.cov @ self.A.T - \
                self.A @ G @ x_s_next.cov - \
                x_s_next.cov @ G.T @ self.A.T) @ B_pinv.T
        )

        return x_s, u_s

    def __call__(self, ys: List[NDArray], x1_prior: Gaussian) -> Tuple[List[Gaussian]]:
        """Runs the smoother to estimate the latent states and control inputs
        from the observed measurements `ys` and the prior over the initial state
        `x1_prior`."""

        # Posterior over x1
        y1_pred = Gaussian(
            self.C @ x1_prior.mean,
            self.C @ x1_prior.cov @ self.C.T + self.y_cov
        )
        x = self._update(x1_prior, y1_pred, ys[0])

        filter_xs = [x]
        for y in ys[1:]:
            x_pred, y_pred = self._predict(x)
            x = self._update(x_pred, y_pred, y)
            filter_xs.append(x)

        # Final input set to 0 as it cannot be determined
        xs = [filter_xs[-1]]
        us = [Gaussian(np.zeros(self.Nu), np.eye(self.Nu))]
        for x_f in reversed(filter_xs[:-1]):
            x, u = self._smooth(x, x_f)
            xs.append(x)
            us.append(u)

        xs.reverse()
        us.reverse()

        return xs, us


