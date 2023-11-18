from autograd import jacobian
from functools import cached_property

from numpy.typing import NDArray



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