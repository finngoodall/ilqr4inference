from numpy.typing import NDArray

import autograd.numpy as np

from .base import Prior



class GaussianPrior(Prior):
    """Class for a Gaussian prior over the inital state or control inputs."""

    def __init__(self, Ndims: int, mean: NDArray, cov: NDArray):
        """Construct the Gaussian prior.
        
        Parameters
        - `Ndims`:
            Number of state or control input dimensions
        - `mean`:
            Mean of the Gaussian distribution
        - `cov`:
            Covariance of the Gaussian distribution
        """

        self.Ndims = Ndims
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
