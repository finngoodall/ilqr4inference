from numpy.typing import NDArray

import autograd.numpy as np
from scipy.stats import t as student

from lib.base import AGInputPrior, InputPrior



class GaussianPrior(InputPrior):
    """Class for a Gaussian prior over the control inputs to a generative model.
    """

    def __init__(self, Nu: int, mean:NDArray, cov: NDArray):
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
        """Sample from the control input prior at timestep `t`."""

        return np.random.multivariate_normal(self.mean, self.cov)
    
    def ll(self, u: NDArray, t: int) -> float:
        z = u - self.mean
        return -0.5 * z.T @ self._P @ z

    def dll(self, u: NDArray, t: int) -> NDArray:
        return -self._P @ (u - self.mean)
    
    def d2ll(self, u: NDArray, t: int) -> NDArray:
        return -self._P



class StudentPrior(AGInputPrior):
    """Class for a Student t-distribution prior over the control inputs in a
    generative model.
    """

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
        """Sample from the control input prior at timestep `t`."""
        
        return student.rvs(df=self.nu, loc=self.mean, scale=self.S)
    
    def ll(self, u: NDArray, t: int) -> float:
        z = u - self.mean
        return -np.log(1 + (z.T/self.S @ z) / self.nu)
