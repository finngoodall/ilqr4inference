from numpy.typing import NDArray

import autograd.numpy as np
from scipy.stats import t as student

from lib.base import AGInputPrior, InputPrior



class GaussianPrior(InputPrior):
    """Class for a zero-mean Gaussian prior over the control inputs to a
    generative model.
    """

    def __init__(self, Nu: int, cov: NDArray):
        """Construct the Gaussian prior.
        
        Parameters
        - `Nu`:
            Number of control input dimensions
        - `cov`:
            Covariance of the Gaussian distribution
        """

        self.Nu = Nu
        self.cov = cov
        # Store the precision matrix to use in calculations
        self._P = np.linalg.inv(self.cov)

    def sample(self, t: int) -> NDArray:
        """Sample from the control input prior at timestep `t`."""

        return np.random.multivariate_normal(np.zeros(self.Nu), self.cov)
    
    def ll(self, u: NDArray, t: int) -> float:
        return -0.5 * u.T @ self._P @ u

    def dll(self, u: NDArray, t: int) -> NDArray:
        return -self._P @ u
    
    def d2ll(self, u: NDArray, t: int) -> NDArray:
        return -self._P



class StudentPrior(AGInputPrior):
    """Class for a zero-mean Student t-distribution prior over the control
    inputs in a generative model.
    """

    def __init__(self, Nu: int, nu: float, S: NDArray):
        """Construct the Gaussian prior.
        
        Parameters
        - `Nu`:
            Number of control input dimensions
        - `dof`:
            Degrees of freedeom of the distribution
        - `S`:
            Shape vector of the distribution
        """

        self.Nu = Nu
        self.nu = nu
        self.S = S

    def sample(self, t: int) -> NDArray:
        """Sample from the control input prior at timestep `t`."""
        
        return student.rvs(df=self.nu, scale=self.S)
    
    def ll(self, u: NDArray, t: int) -> float:
        return -np.log(1 + (u.T/self.S @ u) / self.nu)
