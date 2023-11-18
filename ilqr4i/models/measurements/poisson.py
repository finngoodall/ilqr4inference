from typing import Callable
from numpy.typing import NDArray

import autograd.numpy as np

from .base import AGMeasurementModel



class PoissonMeasurement(AGMeasurementModel):
    """Class for Poisson distributed measurements."""

    def __init__(
            self,
            Ny: int,
            mean_func: Callable[[NDArray], NDArray] = lambda x, t: x
        ) -> None:
        """Construct the measurement model.
        
        Parameters
        - `Ny`:
            Number of measurement dimensions
        - `h`:
            Function that maps the latent state `x` at time `t` to the mean of
            the Gaussian distribution.
        """

        self.Ny = Ny
        self.mean_func = mean_func

    def h(self, x: NDArray, t: int) -> NDArray:
        return self.mean_func(x, t)

    def sample(self, x: NDArray, t: int) -> NDArray:
        return np.random.poisson(lam=self.h(x, t))
    
    def ll(self, x: NDArray, y: NDArray, t: int) -> float:
        m = self.h(x, t)
        return np.sum(y*np.log(m) - m)