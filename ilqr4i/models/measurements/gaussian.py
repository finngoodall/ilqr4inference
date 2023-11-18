from typing import Callable
from numpy.typing import NDArray

import autograd.numpy as np

from .base import MeasurementModel, AGMeasurementModel



class LinearGaussianMeasurement(MeasurementModel):
    """Class for a Gaussian distributed measurements with a linear mapping from
    the latent state to the measurement mean.
    """
    
    def __init__(
            self,
            Ny: int,
            cov: NDArray,
            C: NDArray
        ) -> None:
        """Construct the measurement model.
        
        Parameters
        - `Ny`:
            Number of measurement dimensions
        - `cov`:
            Covariance matrix of the measurements
        - `C`:
            Matrix that maps the latent state to the mean of the Gaussian
            distribution.
        """

        self.Ny = Ny
        self.cov = cov
        self.C = C
        # Store the precision matrix to use in calculations
        self._P = np.linalg.inv(self.cov)

    def h(self, x: NDArray, t: int) -> NDArray:
        return self.C @ x

    def dh_dx(self, x: NDArray, t: int) -> NDArray:
        return self.C

    def sample(self, x: NDArray, t: int) -> NDArray:
        return np.random.multivariate_normal(self.C@x, self.cov)
    
    def ll(self, x: NDArray, y: NDArray, t: int) -> float:
        v = self.C@x - y
        return -0.5 * v.T @ self._P @ v
    
    def dll(self, x: NDArray, y: NDArray, t: int) -> float:
        v = self.C@x - y
        return -self.C.T @ self._P @ v
    
    def d2ll(self, x: NDArray, y: NDArray, t: int) -> float:
        return -self.C.T @ self._P @ self.C



class GaussianMeasurement(AGMeasurementModel):
    """Class for a Gaussian distributed measurements with a nonlinear mapping
    from the latent state to the measurement mean.
    """
    
    def __init__(
            self,
            Ny: int,
            cov: NDArray,
            mean_func: Callable[[NDArray, int], NDArray] = lambda x, t: x
        ) -> None:
        """Construct the measurement model.
        
        Parameters
        - `Ny`:
            Number of measurement dimensions
        - `cov`:
            Covariance matrix of the measurements
        - `h`:
            Function that maps the latent state `x` at time `t` to the mean of
            the Gaussian distribution.
        """

        self.Ny = Ny
        self.cov = cov
        self.mean_func = mean_func
        # Store the precision matrix to use in calculations
        self._P = np.linalg.inv(self.cov)

    def h(self, x: NDArray, t: int) -> NDArray:
        return self.mean_func(x, t)

    def sample(self, x: NDArray, t: int) -> NDArray:
        return np.random.multivariate_normal(self.h(x, t), self.cov)

    def ll(self, x: NDArray, y: NDArray, t: int) -> float:
        v = self.h(x, t) - y
        return -0.5 * v.T @ self._P @ v