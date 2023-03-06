from numpy.typing import NDArray

from autograd import jacobian
from functools import cached_property

from lib.utils import is_pos_def



class Gaussian():
    """Class for a Gaussian distribuion over an inferred state, described by the
    given `mean` and covaraince matrix `cov`."""

    def __init__(self, mean: NDArray , cov: NDArray):
        self.mean = mean
        self.cov = cov
        if self.mean.shape:
            self.Ndims = self.mean.shape[0]
        else:
            # An empty shape tuple means a size of 1
            self.Ndims = 1

        if len(self.mean.shape) > 1:
            raise ValueError(f"State mean must be 1 dimensional")
        if not is_pos_def(self.cov):
            raise ValueError("Covariance matrix must be positive definite")
        if self.Ndims != self.cov.shape[0]:
            raise ValueError(f"Mean and covariance must have equal dimensions")
        
    # Used for ease in commparing to None
    def __bool__(self):
        return True



class MeasurementModel():
    """Base class for the measurement distributions in generative models."""

    def sample(self, x: NDArray, t: int) -> NDArray:
        """Sample a measurement from the latent state `x` at time `t`."""
        raise NotImplementedError
    
    def ll(self, x: NDArray, y: NDArray, t: int) -> float:
        """Computes the log-likelihood of seeing the observation `y` from the
        state `x` at time `t`."""
        raise NotImplementedError
    
    def dll(self, x: NDArray, y: NDArray, t: int) -> NDArray:
        """Computes the derivative of `ll` with respect to the latent state `x`
        at time `t`."""
        raise NotImplementedError
    
    def d2ll(self, x: NDArray, y: NDArray, t: int) -> NDArray:
        """Computes the second derivative of `ll` with respect to the latent
        state `x` at time `t`."""
        raise NotImplementedError



class AGMeasurementModel(MeasurementModel):
    """Class for `Measurement` objects that uses autograd to automatically
    compute `dll` and `d2ll`."""

    # Use private properties for derivatives because `dc_dx = jacobian(.)` makes
    # Pylance falsely think code is unreachable
    @cached_property
    def _dll(self):
        return jacobian(self.ll)

    @cached_property
    def _d2ll(self):
        return jacobian(self._dll)
    
    def dll(self, x: NDArray, y: NDArray, t: int) -> NDArray:
        """Computes the derivative of `ll` with respect to the latent state `x`
        at time `t`."""
        return self._dll(x, y, t)
    
    def d2ll(self, x: NDArray, y: NDArray, t: int) -> NDArray:
        """Computes the second derivative of `ll` with respect to the latent
        state `x` at time `t`."""
        return self._d2ll(x, y, t)




class Dynamics():
    """Base class to represent the system dynamics in generative models."""

    def f(self, x: NDArray, u: NDArray, t: int) -> NDArray:
        """Computes the next state given the current state `x`, input `u` and
        time `t`."""
        raise NotImplementedError

    def df_dx(self, x: NDArray, u: NDArray, t: int) -> NDArray:
        """Computes the derivative of the transition function `f` with respect
        to the state `x`."""
        raise NotImplementedError
    
    def df_du(self, x: NDArray, u: NDArray, t: int) -> NDArray:
        """Computes the derivative of the transition function `f` with respect
        to the input `u`."""
        raise NotImplementedError



class AGDynamics(Dynamics):
    """Class for `Dynamics` objects that uses autograd to automatically compute
    `df_dx` and `df_du`."""

    # Use private properties for derivatives because `df_dx = jacobian(.)` makes
    # Pylance falsely think code is unreachable
    @cached_property
    def _df_dx(self):
        return jacobian(self.f)

    @cached_property
    def _df_du(self):
        return jacobian(self.f, argnum=1)
    
    def df_dx(self, x: NDArray, u: NDArray, t: int) -> NDArray:
        """Computes the derivative of the transition function `f` with respect
        to the state `x`."""
        return self._df_dx(x, u, t)
    
    def df_du(self, x: NDArray, u: NDArray, t: int) -> NDArray:
        """Computes the derivative of the transition function `f` with respect
        to the input `u`."""
        return self._df_du(x, u, t)
    


class InputPrior():
    """Base class for prior distributions over control inputs."""

    def sample(self, t: int) -> NDArray:
        raise NotImplementedError
    
    def ll(self, u: NDArray, t: int) -> float:
        """Computes the log-likelihood of the input `u` at time `t`."""
        raise NotImplementedError
    
    def dll(self, u: NDArray, t: int) -> NDArray:
        """Computes the derivative of `ll` with respect to the latent state `x`
        at time `t`."""
        raise NotImplementedError
    
    def d2ll(self, u: NDArray, t: int) -> NDArray:
        """Computes the second derivative of `ll` with respect to the latent
        state `x` at time `t`."""
        raise NotImplementedError



class AGInputPrior(InputPrior):
    """Class for `InputPrior` objects that uses autograd to automatically
    compute `dll` and `d2ll`."""

    # Use private properties for derivatives because `df_dx = jacobian(.)` makes
    # Pylance falsely think code is unreachable
    @cached_property
    def _dll(self):
        return jacobian(self.ll)

    @cached_property
    def _d2ll(self):
        return jacobian(self._dll)
    
    def dll(self, u: NDArray, t: int) -> NDArray:
        """Computes the derivative of `ll` with respect to the latent state `x`
        at time `t`."""
        return self._dll(u, t)
    
    def d2ll(self, u: NDArray, t: int) -> NDArray:
        """Computes the second derivative of `ll` with respect to the latent
        state `x` at time `t`."""
        return self._d2ll(u, t)