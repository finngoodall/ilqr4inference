from numpy.typing import NDArray
from typing import Callable

import autograd.numpy as np

from lib.base import AGMeasurementModel, MeasurementModel



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



class PoissonMeasurement(AGMeasurementModel):
    """Class for Poisson distributed measurements."""

    def __init__(
            self,
            Ny: int,
            mean_func: Callable[[NDArray], NDArray] = lambda x, t: x
        ) -> None:
        """Construct the measurement model.
        
        Parameters
        - `Ny`:
            Number of measurement dimensions
        - `h`:
            Function that maps the latent state `x` at time `t` to the mean of
            the Gaussian distribution.
        """

        self.Ny = Ny
        self.mean_func = mean_func

    def h(self, x: NDArray, t: int) -> NDArray:
        return self.mean_func(x, t)

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
        self.cov = cov
        # Store the precision matrix
        self._P = np.linalg.inv(self.cov)

    def h(self, x: NDArray, t: int) -> NDArray:
        return np.exp(x[-1]) * self.A @ x[:-1]
    
    def dh_dx(self, x: NDArray, t: int) -> NDArray:
        top = x[-1] * self.A
        bot = self.A @ x[:-1]
        return np.hstack((top, bot))

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



class BernoulliMeasurement(AGMeasurementModel):
    def __init__(self, Nx: int, Ny: int):
        self.Nx = Nx
        self.Ny = Ny

    def h(self, x: NDArray, t: int) -> NDArray:
        return np.exp(-0.75*np.sum(x**2))
    
    def sample(self, x: NDArray, t: int) -> NDArray:
        ys = np.random.uniform(size=self.Ny) <= self.h(x, t)
        return ys.astype(int)
    
    def ll(self, x: NDArray, y: NDArray, t: int) -> NDArray:
        m = self.h(x, t)
        return np.sum(y)*np.log(m) + np.sum(1 - y)*np.log(1 - m)