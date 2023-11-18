from numpy.typing import NDArray

import autograd.numpy as np
from scipy.stats import t as student

from .base import AGPrior



class StudentPrior(AGPrior):
    """Class for a Student prior over the initial state or control inputs."""

    def __init__(self, Ndims: int, dof: float, mean: NDArray, S: NDArray):
        """Construct the Gaussian prior.
        
        Parameters
        - `Ndims`:
            Number of state or control input dimensions
        - `dof`:
            Degrees of freedeom of the distribution
        - `mean`:
            Mean of the distribution
        - `S`:
            Shape vector of the distribution
        """

        self.Ndims = Ndims
        self.mean = mean
        self.dof = dof
        self.S = S

    def sample(self, t: int) -> NDArray:
        rv = student.rvs(df=self.dof, loc=self.mean, scale=self.S)
        
        if self.Ndims == 1:
            return np.array([rv])
            
        return rv

    def ll(self, z: NDArray, t: int) -> float:
        v = z - self.mean
        return -0.5*(1 + self.dof)*np.log(1 + (v.T/self.S @ v) / self.dof)
