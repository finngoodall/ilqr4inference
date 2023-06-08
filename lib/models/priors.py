from numpy.typing import NDArray

import autograd.numpy as np
from scipy.stats import t as student
from scipy.stats import hypsecant

from lib.base import AGPrior, Prior



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



class HypSecantPrior(Prior):
    """Class for a hyperbolic secant prior over the intial state or control
    inputs.
    """

    def __init__(self, Ndims: int, mean: NDArray, scale: NDArray):
        """Construct the prior.
        
        Parameters
        - `Ndims`:
            Number of state or control input dimensions
        - `mean`:
            Mean of the distribution
        - `scale`:
            Shape vector of the distribution
        """
        
        self.Ndims = Ndims
        self.mean = mean
        self.scale = scale

    def sample(self, t: int) -> NDArray:
        rv = hypsecant.rvs(loc=self.mean, scale=self.scale)
        if self.Ndims == 1:
            return np.array([rv])
            
        return rv
    
    def ll(self, z: NDArray, t: int) -> float:
        v = (z - self.mean)/self.scale
        return np.sum(-np.log(np.cosh(v)))
    
    def dll(self, z: NDArray, t: int) -> float:
        v = (z - self.mean)/self.scale
        return (np.exp(v) - np.exp(-v))/(np.exp(v) + np.exp(-v))
    
    def d2ll(self, z: NDArray, t: int) -> NDArray:
        v = (z - self.mean)/self.scale
        M = np.diag((np.exp(v) - np.exp(-v))/(np.exp(v) + np.exp(-v)))
        return np.eye(self.Ndims) - M**2