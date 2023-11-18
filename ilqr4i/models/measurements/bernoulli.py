from numpy.typing import NDArray

import autograd.numpy as np

from .base import AGMeasurementModel



class BernoulliMeasurement(AGMeasurementModel):
    def __init__(self, Nx: int, Ny: int):
        self.Nx = Nx
        self.Ny = Ny

    def h(self, x: NDArray, t: int) -> NDArray:
        return np.exp(-0.75*np.sum(x**2))
    
    def sample(self, x: NDArray, t: int) -> NDArray:
        ys = np.random.uniform(size=self.Ny) <= self.h(x, t)
        return ys.astype(int)
    
    def ll(self, x: NDArray, y: NDArray, t: int) -> NDArray:
        m = self.h(x, t)
        return np.sum(y)*np.log(m) + np.sum(1 - y)*np.log(1 - m)