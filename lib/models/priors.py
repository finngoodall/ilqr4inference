from numpy.typing import NDArray

import autograd.numpy as np
from scipy.stats import t as student

from lib.base import AGPrior, Prior



class GaussianPrior(Prior):
    """Class for a Gaussian prior over the inital state or control inputs."""

    def __init__(self, Nu: int, mean: NDArray, cov: NDArray):
        """Construct the Gaussian prior.
        
        Parameters
        - `Nu`:
            Number of control input dimensions
        - `mean`:
            Mean of the Gaussian distribution
        - `cov`:
            Covariance of the Gaussian distribution
        """

        self.Nu = Nu
        self.mean = mean
        self.cov = cov
        # Store the precision matrix to use in calculations
        self._P = np.linalg.inv(self.cov)

    def sample(self, t: int) -> NDArray:
        return np.random.multivariate_normal(self.mean, self.cov)
    
    def ll(self, z: NDArray, t: int) -> float:
        v = z - self.mean
        return -0.5 * v.T @ self._P @ v

    def dll(self, z: NDArray, t: int) -> NDArray:
        return -self._P @ (z - self.mean)
    
    def d2ll(self, z: NDArray, t: int) -> NDArray:
        return -self._P



class StudentPrior(AGPrior):
    """Class for a Student prior over the initial state or control inputs."""

    def __init__(self, Nu: int, nu: float, mean: NDArray, S: NDArray):
        """Construct the Gaussian prior.
        
        Parameters
        - `Nu`:
            Number of control input dimensions
        - `dof`:
            Degrees of freedeom of the distribution
        - `mean`:
            Mean of the distribution
        - `S`:
            Shape vector of the distribution
        """

        self.Nu = Nu
        self.mean = mean
        self.nu = nu
        self.S = S

    def sample(self, t: int) -> NDArray:
        rv = student.rvs(df=self.nu, loc=self.mean, scale=self.S)
        
        if self.Nu == 1:
            return np.array([rv])
            
        return rv

    def ll(self, z: NDArray, t: int) -> float:
        v = z - self.mean
        return -np.log(1 + (v.T/self.S @ v) / self.nu)
