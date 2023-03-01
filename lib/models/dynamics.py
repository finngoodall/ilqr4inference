from numpy.typing import NDArray

import numpy as np

from lib.base import AGDynamics, Dynamics



class LinearDynamics(Dynamics):
    """Class for a linear dynamical system with state transition matrix `A` and
    control matrix `B`."""

    def __init__(self, Nx: int, Nu: int, A: NDArray, B: NDArray):
        self.Nx = Nx
        self.Nu = Nu
        self.A = A
        self.B = B

    def f(self, x: NDArray, u: NDArray, t: int) -> NDArray:
        return self.A @ x + self.B @ u
    
    def df_dx(self, x: NDArray, u: NDArray, t: int) -> NDArray:
        return self.A

    def df_du(self, x: NDArray, u: NDArray, t: int) -> NDArray:
        return self.B



class GSMDynamics(Dynamics):

    def __init__(self, Nx: int, Nu: int, B: NDArray, dt: float = 0.1,
                 tau_xs: float = 1.0, tau_c: float = 10.0):
        self.Nx = Nx
        self.Nu = Nu
        # Compute the transition and control terms
        self._T = np.diag(np.hstack((np.ones(Nx-1)/tau_xs, 1/tau_c)))
        self._Xmat = np.eye(self.Nx) - dt*self._T
        self._Umat = dt*self._T@B

    def f(self, x: NDArray, u: NDArray, t: int) -> NDArray:
        return self._Xmat@x + self._Umat@u
    
    def df_dx(self, x: NDArray, u: NDArray, t: int) -> NDArray:
        return self._Xmat

    def df_du(self, x: NDArray, u: NDArray, t: int) -> NDArray:
        return self._Umat
