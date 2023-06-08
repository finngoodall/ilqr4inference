from numpy.typing import NDArray

from autograd import jacobian
from functools import cached_property

from lib.utils import is_pos_def



class Gaussian():
    """Class for a Gaussian distribuion over an inferred state."""

    def __init__(self, mean: NDArray , cov: NDArray):
        """Construct the Gaussian object.
        
        Parameters
        - `mean`: NDArray
            The mean of the distribution
        - `cov`: NDArray
            The covariance of the distribution
        """

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
        
    # Used for ease in commparing to NoneTypes
    def __bool__(self):
        return True



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




class Dynamics():
    """Base class to represent the system dynamics in generative models."""

    def f(self, x: NDArray, u: NDArray, t: int) -> NDArray:
        """Evaluate the next state from the current state `x`, input `u` and
        time `t`.
        """
        raise NotImplementedError

    def df_dx(self, x: NDArray, u: NDArray, t: int) -> NDArray:
        """Evaluate the derivative of the transition function with repsect to
        the state at the state `x`, input `u` and time `t`.
        """
        raise NotImplementedError
    
    def df_du(self, x: NDArray, u: NDArray, t: int) -> NDArray:
        """Evaluate the derivative of the transition function with repsect to
        the input at the state `x`, input `u` and time `t`.
        """
        raise NotImplementedError



class AGDynamics(Dynamics):
    """Class for `Dynamics` objects that uses autograd to automatically compute
    the derivatives of the state transition function.
    """

    # Use private properties for derivatives because `df_dx = jacobian(.)` makes
    # Pylance falsely think code is unreachable
    @cached_property
    def _df_dx(self):
        return jacobian(self.f)

    @cached_property
    def _df_du(self):
        return jacobian(self.f, argnum=1)
    
    def df_dx(self, x: NDArray, u: NDArray, t: int) -> NDArray:
        return self._df_dx(x, u, t)
    
    def df_du(self, x: NDArray, u: NDArray, t: int) -> NDArray:
        return self._df_du(x, u, t)
    


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