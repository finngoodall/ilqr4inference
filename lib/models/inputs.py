from numpy.typing import NDArray

import autograd.numpy as np
from scipy.stats import t as student

from lib.base import AGInputPrior, InputPrior



class GaussianPrior(InputPrior):
    """Class for a Gaussian prior over the control inputs in a generative
    model with mean 0 and covariance `cov`. `t0_weight` multiplies the
    covariance at timestep 0 to allow for cheaper setting of the initial
    conditions."""

    def __init__(self, Nu: int, cov: NDArray, t0_weight: float = 10.0):
        self.Nu = Nu
        self.cov = cov
        self.w = t0_weight
        # Store the precision matrix to use in calculations
        self._P = np.linalg.inv(self.cov)

    def sample(self, t: int) -> NDArray:
        if t == 0:
            return np.random.multivariate_normal(
                mean=np.zeros(self.Nu),
                cov=self.w*self.cov
            )
        else:
            return np.random.multivariate_normal(
                mean=np.zeros(self.Nu),
                cov=self.cov
            )
    
    def ll(self, u: NDArray, t: int) -> float:
        if t == 0:
            return -0.5 * u.T @ self._P @ u / self.w
        else:
            return -0.5 * u.T @ self._P @ u

    def dll(self, u: NDArray, t: int) -> NDArray:
        if t == 0:
            return -self._P @ u / self.w
        else:
            return -self._P @ u
    
    def d2ll(self, u: NDArray, t: int) -> NDArray:
        if t == 0:
            return -self._P/self.w
        else:
            return -self._P



class StudentPrior(AGInputPrior):
    """Class for a Student t-distribution prior over the control inputs in a
    generative model with mean 0, `nu` degrees of freedom and shape vector `S`.
    `t0_weight` multiplies the shape at timestep 0 to allow for cheaper setting
    of the initial conditions.
    
    Inherits methods from `MeasDistribution` but overwrites the signatures of
    the the methods so that only the control input and time are needed."""

    def __init__(self, Nu: int, nu: float, S: NDArray, t0_weight: float = 10.0):
        self.Nu = Nu
        self.nu = nu
        self.S = S
        self.w = t0_weight

    def sample(self, t: int) -> NDArray:
        if t == 0:
            return np.random.multivariate_normal(
                np.zeros(self.Nu),
                self.w*np.diag(self.S)
            )
        else:
            return student.rvs(df=self.nu, scale=self.S)
    
    def ll(self, u: NDArray, t: int) -> float:
        if t == 0:
            return -0.5 * u.T/(self.S*self.w) @ u
        else:
            return -np.log(1 + (u.T/self.S @ u) / self.nu)
