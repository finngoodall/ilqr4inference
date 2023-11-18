from numpy.typing import NDArray

import autograd.numpy as np

from .base import Dynamics



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