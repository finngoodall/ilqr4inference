import autograd.numpy as np

from .base import Dynamics



class VanDerPolDynamics(Dynamics):
    """Class for 2D dynamics following a discretised Van der Pol oscillator."""

    def __init__(self, mu: float, dt: float = 0.1) -> None:
        self.mu = mu
        self.dt = dt
        self.Nx = 2
        self.Nu = 2

    def _A(self, x):
        return np.array([[0, 1], [-1 - self.mu*x[0]*x[1], self.mu]])

    def _B(self):
        return np.eye(self.Nu)

    def f(self, x, u, t):
        return (np.eye(self.Nx) + self.dt*self._A(x))@x + self._B()@u

    def df_dx(self, x, u, t):
        return np.eye(self.Nx) + self.dt*self._A(x)
    
    def df_du(self, x, u, t):
        return self._B()