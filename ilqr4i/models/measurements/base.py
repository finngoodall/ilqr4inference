from numpy.typing import NDArray

from autograd import jacobian
from functools import cached_property



class MeasurementModel():
    """Base class for the measurement distributions in generative models."""

    def h(self, x: NDArray, t: int) -> NDArray:
        """Evaluate the measurement mean fuction at the state `x` and time
        `t`.
        """
        raise NotImplementedError

    def dh_dx(self, x: NDArray, t: int) -> NDArray:
        """Evaluate the derivative of the measurement mean function at the state
        `x` and time `t`.
        """
        raise NotImplementedError

    def sample(self, x: NDArray, t: int) -> NDArray:
        """Sample a measurement from the latent state `x` at time `t`."""
        raise NotImplementedError
    
    def ll(self, x: NDArray, y: NDArray, t: int) -> float:
        """Evaluate the log-likelihood of the observation `y` from the state `x` 
        at time `t`.
        """
        raise NotImplementedError
    
    def dll(self, x: NDArray, y: NDArray, t: int) -> NDArray:
        """Evaluate the derivative of the log-likelihood with respect to the
        state at the state `x`, observation `y` and time `t`.
        """
        raise NotImplementedError
    
    def d2ll(self, x: NDArray, y: NDArray, t: int) -> NDArray:
        """Evaluate the second derivative of the log-likelihood with respect to
        the state at the state `x`, observation `y` and time `t`.
        """
        raise NotImplementedError



class AGMeasurementModel(MeasurementModel):
    """Class for `Measurement` objects that uses autograd to automatically
    compute the derivatives of the log-likelihood.
    """

    # Use private properties for derivatives because `dc_dx = jacobian(.)` makes
    # Pylance falsely think code is unreachable
    @cached_property
    def _dh_dx(self):
        return jacobian(self.h)
    
    @cached_property
    def _dll(self):
        return jacobian(self.ll)

    @cached_property
    def _d2ll(self):
        return jacobian(self._dll)
    
    def dh_dx(self, x: NDArray, t: int) -> NDArray:
        return self._dh_dx(x, t)

    def dll(self, x: NDArray, y: NDArray, t: int) -> NDArray:
        return self._dll(x, y, t)
    
    def d2ll(self, x: NDArray, y: NDArray, t: int) -> NDArray:
        return self._d2ll(x, y, t)