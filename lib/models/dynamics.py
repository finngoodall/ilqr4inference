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
    """Class for Gaussian scale mixture (GSM) model dynamics, where the contrast
    is an unknown state of the system.
    
    The dynamics follow a discretised Wiener process.
    """

    def __init__(self, Nx: int, Nu: int, B: NDArray, dt: float = 0.1,
                 tau_x: float | NDArray = 0.5, tau_c: float = 5.0):
        """Constructs the GSM dynamics model.
        
        Parameters:
        - `Nx`: Number of state dimensions
        - `Nu`: Number of input dimensions
        - `B`: Control matrix, equivalent to the standard deviation of the
            process
        - `dt`: Length of the time steps in seconds
        - `tau_x`: Time constant(s) of the latent features in seconds
        - `tau_c`: Time constant of the contrast coefficient in seconds
        """

        self.Nx = Nx
        self.Nu = Nu
        self.B = B
        self.dt = dt
        self.tau_x = tau_x
        self.tau_c = tau_c
        # Store the transition and control matrices
        self._tau = np.ones(self.Nx)*self.tau_x
        self._Xmat = np.diag(1 - self.dt/self._tau)
        self._Umat = np.diag((2*self.dt/self._tau)**0.5) @ self.B

    def f(self, x: NDArray, u: NDArray, t: int) -> NDArray:
        return self._Xmat@x + self._Umat@u
    
    def df_dx(self, x: NDArray, u: NDArray, t: int) -> NDArray:
        return self._Xmat

    def df_du(self, x: NDArray, u: NDArray, t: int) -> NDArray:
        return self._Umat




class GSMDynamicsKnownContrast(Dynamics):
    """Class for Gaussian scale mixture (GSM) model dynamics, where the contrast
    is a known function of time. This function is given as part of `h` in a
    measurement model.
    
    The dynamics follow a discretised Wiener process.
    """

    def __init__(self, Nx: int, Nu: int, B: NDArray, dt: float = 0.1,
                 tau_x: float | NDArray = 0.5, tau_c: float = 5.0):
        """Constructs the GSM dynamics model.
        
        Parameters:
        - `Nx`: Number of state dimensions
        - `Nu`: Number of input dimensions
        - `B`: Control matrix, equivalent to the standard deviation of the
            process
        - `dt`: Length of the time steps in seconds
        - `tau_x`: Time constant(s) of the latent features in seconds
        """

        self.Nx = Nx
        self.Nu = Nu
        self.B = B
        self.dt = dt
        self.tau_x = tau_x
        # Store the transition and control matrices
        self._tau = np.ones(self.Nx)*self.tau_x
        self._Xmat = np.diag(1 - self.dt/self._tau)
        self._Umat = np.diag((2*self.dt/self._tau)**0.5) @ self.B

    def f(self, x: NDArray, u: NDArray, t: int) -> NDArray:
        return self._Xmat@x + self._Umat@u
    
    def df_dx(self, x: NDArray, u: NDArray, t: int) -> NDArray:
        return self._Xmat

    def df_du(self, x: NDArray, u: NDArray, t: int) -> NDArray:
        return self._Umat



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