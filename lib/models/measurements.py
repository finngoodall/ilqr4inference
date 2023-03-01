from numpy.typing import NDArray
from typing import Callable

import autograd.numpy as np

from lib.base import AGMeasurementModel, MeasurementModel



class GaussianMeasurement(AGMeasurementModel):
    """Class for a multivariate normal distribution with zero mean and
    covariance `cov`.
    
    `h` maps the latent state `x` at time `t` to the mean of the distribution in
    the observation domain, i.e. `h(x, t)` and `cov` have compatible
    dimensions."""

    def __init__(self, Ny: int, cov: NDArray,
                 h: Callable[[NDArray], NDArray] = lambda x, t: x):
        self.Ny = Ny
        self.cov = cov
        self.h = h
        # Store the precision matrix to use in calculations
        self._P = np.linalg.inv(self.cov)

    def sample(self, x: NDArray, t: int) -> NDArray:
        return np.random.multivariate_normal(self.h(x, t), self.cov)
    
    def ll(self, x: NDArray, y: NDArray, t: int) -> float:
        v = self.h(x, t) - y
        return -0.5 * v.T @ self._P @ v



class PoissonMeasurement(AGMeasurementModel):
    """Class for a Poisson distribution with `Ny` observation dimensions.
    
    `h` maps the latent state `x` at time `t` to the mean of the  observations'
    Poisson distribution."""

    def __init__(self, Ny: int,
                 h: Callable[[NDArray], NDArray] = lambda x, t: x):
        self.Ny = Ny
        self.h = h

    def sample(self, x: NDArray, t: int) -> NDArray:
        return np.random.poisson(lam=self.h(x, t))
    
    def ll(self, x: NDArray, y: NDArray, t: int) -> float:
        m = self.h(x, t)
        return np.sum(y*np.log(m) - m)
    


class GSMExpMeasurement(MeasurementModel):
    """Class for a Gaussian scale mixture (GSM) model with `Ny` observation
    dimensions, feature matrix `A` and noise covariance `cov`.
    
    This implementation uses hand-written derivatves."""

    def __init__(self, Ny: int, A: NDArray, cov: NDArray):
        self.Ny = Ny
        self.A = A
        self.h = lambda x, t: np.exp(x[-1]) * self.A @ x[:-1]
        self.cov = cov
        # Store the precision matrix
        self._P = np.linalg.inv(self.cov)

    def sample(self, x: NDArray, t: int) -> NDArray:
        return np.random.multivariate_normal(self.h(x, t), self.cov)
    
    def ll(self, x: NDArray, y: NDArray, t: int) -> float:
        v = self.h(x, t) - y
        return -0.5 * v.T @ self._P @ v

    def dll(self, x: NDArray, y: NDArray, t: int) -> NDArray:
        m = self.h(x, t)
        e_c = np.exp(x[-1])
        dll_dx = -e_c * self.A.T @ self._P @ (m - y)
        dll_dc = -m.T @ self._P @ (m - y)
        return np.hstack((dll_dx, dll_dc))

    def d2ll(self, x: NDArray, y: NDArray, t: int) -> NDArray:
        m = self.h(x, t)
        e_c = np.exp(x[-1])
        d2ll_dx2 = -e_c**2 * self.A.T @ self._P @ self.A
        d2ll_dc2 = -m.T @ self._P @ (2*m - y)
        d2ll_dcx = -e_c * self.A.T @ self._P @ (2*m - y)
        d2ll_dcx = np.reshape(d2ll_dcx, (len(x)-1, 1))
        return np.block([[d2ll_dx2, d2ll_dcx], [d2ll_dcx.T, d2ll_dc2]])



