from numpy.typing import NDArray

from autograd import jacobian
from functools import cached_property



class Prior():
    """Base class for prior distributions over initial states or control inputs
    in the generative model.
    """

    def sample(self, t: int) -> NDArray:
        raise NotImplementedError
    
    def ll(self, z: NDArray, t: int) -> float:
        """Evaluate the log-likelihood of the sample `z` at time `t`."""
        raise NotImplementedError
    
    def dll(self, z: NDArray, t: int) -> NDArray:
        """Evaluate the derivative of the log-likelihood at the sample `z` and
        time `t`.
        """
        raise NotImplementedError
    
    def d2ll(self, z: NDArray, t: int) -> NDArray:
        """Evaluate the second derivative of the log-likelihood at the sample
        `z` and time `t`.
        """
        raise NotImplementedError



class AGPrior(Prior):
    """Class for `Prior` objects that uses autograd to automatically compute
    the derivatives of the log-likelihood.
    """

    # Use private properties for derivatives because `dll = jacobian(.)` makes
    # Pylance falsely think code is unreachable
    @cached_property
    def _dll(self):
        return jacobian(self.ll)

    @cached_property
    def _d2ll(self):
        return jacobian(self._dll)
    
    def dll(self, z: NDArray, t: int) -> NDArray:
        return self._dll(z, t)
    
    def d2ll(self, z: NDArray, t: int) -> NDArray:
        return self._d2ll(z, t)