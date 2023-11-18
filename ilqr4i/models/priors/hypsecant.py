from numpy.typing import NDArray

import autograd.numpy as np

from scipy.stats import hypsecant

from .base import AGPrior, Prior



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