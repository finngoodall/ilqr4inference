from numpy.typing import NDArray

import numpy as np

from lib.base import Dynamics, MeasurementModel, InputPrior



def sample_trajectory(dynamics: Dynamics, meas_model: MeasurementModel,
                      input_prior: InputPrior, T: int, x1: NDArray = None):
    """Samples a sequence of corresponding states, control inputs and
    observations from the generative model up to time `T`.
    
    `x1` can be given as the first state in the sequence, else an input is
    sampled from the input prior at time 0 with a hidden initial state `x0` of
    zero.
    
    The returned sequences will only contain timesteps 1 to `T`."""

    if isinstance(x1, np.ndarray):
        x = x1
    else:
        u0 = input_prior.sample(0)
        x = dynamics.f(np.zeros(dynamics.Nx), u0, 0)

    xs = []
    us = []
    ys = []

    for t in range(1, T+1):
        u = input_prior.sample(t)
        y = meas_model.sample(x, t)
        
        xs.append(x)
        us.append(u)
        ys.append(y)

        x = dynamics.f(x, u, t)

    return xs, us, ys